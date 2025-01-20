import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("BERT-Home-Automation-System/data/dataset.csv", delimiter=";")

df['location_function'] = df['location'] + "_" + df['function']
df.drop(['location', 'function'], axis=1, inplace=True)
df['location_function'] = df['location_function'].str.strip()
df['location_function'] = df['location_function'].str.replace(" ", "")

label_map = {
    'home_open': 0, 'home_close': 1, 'balcony_open': 2, 'balcony_close': 3, 
    'kitchen_open': 4, 'kitchen_close': 5, 'bedroom_open': 6, 'bedroom_close': 7, 
    'bathroom_open': 8, 'bathroom_close': 9, 'studyroom_open': 10, 'studyroom_close': 11, 
    'livingroom_open': 12, 'livingroom_close': 13, 'hall_open': 14, 'hall_close': 15
}

# Veri setine label_mapping'i uygulamak
df['location_function'] = df['location_function'].map(label_map)

df.to_csv('BERT-Home-Automation-System/data/processed_data.csv', index=False)

print("'processed_data.csv' saved.")