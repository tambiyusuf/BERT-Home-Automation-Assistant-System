import torch
from transformers import BertForSequenceClassification, BertTokenizer
from firebase_admin import credentials, firestore
import firebase_admin
import time
import json
import pygame
import time

cred = credentials.Certificate("serviceAccountKey.json") #comes from google firebase
firebase_admin.initialize_app(cred)
db = firestore.client()  

def parse_command(command):
    parts = command.split('_')
    
    
    location = parts[0] 
    function = '_'.join(parts[1:])  

    
    function = function.replace('()', '')

    return location, function

def model(command):
    
    model_path = 'models/bert_model_output/saved_model.pth'   
    tokenizer_path = 'models/bert_model_output/saved_tokenizer'   
    
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    
    label_map = {
        'home_open()': 0, 'home_close()': 1, 'balcony_open()': 2, 'balcony_close()': 3, 
        'kitchen_open()': 4, 'kitchen_close()': 5, 'bedroom_open()': 6, 'bedroom_close()': 7, 
        'bathroom_open()': 8, 'bathroom_close()': 9, 'studyroom_open()': 10, 'studyroom_close()': 11, 
        'livingroom_open()': 12, 'livingroom_close()': 13, 'hall_open()': 14, 'hall_close()': 15
    }
    
    
    inputs = tokenizer(command, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  
    
    predicted_class = torch.argmax(logits, dim=1).item()  
    
    predicted_label = [label for label, idx in label_map.items() if idx == predicted_class][0]
    return predicted_label

def get_unprocessed_commands():
    
    commands_ref = db.collection("commands")
    
    while True:
        unprocessed_docs = commands_ref.where("processed", "==", False).stream()

        commands_by_home = {}
        command_id = None  

        for doc in unprocessed_docs:
            data = doc.to_dict()
            home_number = data["homeNumber"]

            if home_number not in commands_by_home:
                commands_by_home[home_number] = []

            commands_by_home[home_number].append((doc.id, data))
            command_id = doc.id  

        print("LOG: commands_by_home", commands_by_home)
        
        if commands_by_home:
            return commands_by_home, command_id
        else:
            print("🔄 İşlenmemiş komut bulunamadı, 5 saniye sonra tekrar kontrol ediliyor...")
            time.sleep(5)  

def process_commands(commands_by_home, command_id):
    
    for doc_id, values in commands_by_home.items():
        for doc in values:
            doc_data = doc[1]

            command = doc_data.get('command')
            home_number = doc_data.get('homeNumber')
            processed = doc_data.get('processed')

            print(f"Command: {command}, Home Number: {home_number}, Processed: {processed}")

    predict = model(command)
    location, function = parse_command(predict)
    update_home_status(home_number, location, function) 
    
    command_ref = db.collection('commands').document(command_id)
    command_ref.update({
        'processed': True  
    })
    print(f"Command ID {command_id} için 'processed' alanı False olarak güncellendi.")

def update_home_status(home_number, location, function):
    
    home_status_ref = db.collection("homeStatus").document(home_number)
    home_status_doc = home_status_ref.get()

    if not home_status_doc.exists:
        print(f"⚠️ {home_number} için homeStatus verisi bulunamadı!")
        return

    home_status = home_status_doc.to_dict()
    

    rooms = ["bathroom", "bedroom", "kitchen", "studyroom", "livingroom", "hall", "balcony"]
    update_data = {}
    if location in rooms:
        loc = f"{location}_light"
        value = True if function == 'open' else False

        if home_status[loc] != value :
            update_data[f"{loc}"] =  value 
    elif location == "home": 
        for room in rooms:
            loc = f"{room}_light"
            value = True if function == 'open' else False 
            if home_status[loc] != value :
                update_data[f"{loc}"] =  value

    print(f"📌 Güncellenmesi gereken veriler: {update_data}")

    if update_data:
        home_status_ref.update(update_data)  
        print(f"✅ {home_number} için Firestore güncellendi: {update_data}")
    else:
        print(f"🔵 {home_number} için değişiklik yapılmadı, mevcut durum zaten aynı.")

def get_document_data(collection_name, document_id):
    
    try:
        doc_ref = db.collection(collection_name).document(document_id)
        
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            print(f"No document found with ID: {document_id}")
            return None
    except Exception as e:
        print(f"Error fetching document: {e}")
        return None


def scrap_status():
    collection_name = "homeStatus"
    document_id = "10001"
    data = get_document_data(collection_name, document_id)
    
    light_data = {key: value for key, value in data.items() if 'light' in key}
    processed_data = {key.replace('_light', ''): value for key, value in light_data.items()}
    return processed_data

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600


BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

ALPHA = 50

def draw_room_polygons(screen, room_status, room_locations):
   
    for room, polygon in room_locations.items():
        if room in room_status:  
            points = [(int(coord[0]), int(coord[1])) for coord in polygon]
            
            
            if room_status[room]:  
                overlay_color = YELLOW
            else:  
                overlay_color = BLACK

            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(overlay, (*overlay_color, ALPHA), points)
            screen.blit(overlay, (0, 0))


def main(room_locations):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Room Visualization")

    home_image = pygame.image.load('home.png').convert_alpha()

    room_status = scrap_status()

    
    clock = pygame.time.Clock()

    last_update_time = time.time()  
    update_interval = 5  

    running = True
    while running:
        screen.fill((255, 255, 255))  
        screen.blit(home_image, (0, 0))  

        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            room_status = scrap_status()  
            last_update_time = current_time  

        draw_room_polygons(screen, room_status, room_locations)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(60)  

    pygame.quit()