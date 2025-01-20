import torch
from transformers import BertForSequenceClassification, BertTokenizer


model_path = 'models/bert_model_output/saved_model.pth'   
tokenizer_path = 'models/bert_model_output/saved_tokenizer'   

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)

# CPU'ya yükleme
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

label_map = {
    'home_open': 0, 'home_close': 1, 'balcony_open': 2, 'balcony_close': 3, 
    'kitchen_open': 4, 'kitchen_close': 5, 'bedroom_open': 6, 'bedroom_close': 7, 
    'bathroom_open': 8, 'bathroom_close': 9, 'studyroom_open': 10, 'studyroom_close': 11, 
    'livingroom_open': 12, 'livingroom_close': 13, 'hall_open': 14, 'hall_close': 15
}

example_sentence = "open the bedroom and turn off the bathroom"  
inputs = tokenizer(example_sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  


predicted_class = torch.argmax(logits, dim=1).item()  


predicted_label = [label for label, idx in label_map.items() if idx == predicted_class][0]

print(f"Örnek cümle: {example_sentence}")
print(f"Modelin tahmin ettiği sınıf: {predicted_class} -> {predicted_label}")