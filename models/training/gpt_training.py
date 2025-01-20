import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from utils import *




df = pd.read_csv('BERT-Home-Automation-System/data/processed_data.csv')


print(df.head())  

texts = df['command'].tolist()
labels = df['location_function'].astype(int).tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)

class CommandDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = CommandDataset(train_texts, train_labels, tokenizer, max_len=64)
val_dataset = CommandDataset(val_texts, val_labels, tokenizer, max_len=64)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f'Using device: {device}')

for epoch in range(4):  
    print(f'Epoch {epoch + 1}/{4}')
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f'Train Loss: {train_loss}')
    
    print('Evaluating...')
    eval_report = eval_model(model, val_dataloader, device)
    print(eval_report)