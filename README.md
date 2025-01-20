# BERT-based Home Automation System

This project utilizes the **BERT (Bidirectional Encoder Representations from Transformers)** model to create an AI application for controlling lights in a home automation system. 
BERT is a powerful language model in natural language processing (NLP), and in this project, it is used to understand voice commands and control the lights in a home. 
The model has been retrained with 20k unique data and is integrated with voice commands. 
Data is retrieved through Firebase and processed in the Python backend, with the model outputs visualized using pygame.

**Project Overview**:
- **BERT Model**: Contains the BERT model trained to control the lights.
- **Firebase Integration**: Voice data from the user is sent to Firebase and processed in the Python backend.
- **Pygame Simulation**: Model outputs are visualized using pygame.
- **Screenshots**: Screenshots of the mobile app's simulated outputs are provided.

## Features
- **BERT-based Light Control**: The model analyzes voice commands to control the light states.
- **Firebase Integration**: Data from the user is sent to Firebase and processed in the Python backend.
- **Pygame Simulation**: Outputs from the model are visualized using pygame.
- **Flexible Dataset**: The dataset is suitable for multi-class classification, but it can be converted into a single class if needed. Additionally, the dataset can be augmented for more detailed analysis.


-----------------------------------------------------------------------------------------------------------------------------------------------
# BERT-based Home Automation System
Bu proje, **BERT (Bidirectional Encoder Representations from Transformers)** 
modelini kullanarak ev otomasyon sistemi üzerinde ışık kontrolü sağlayan bir yapay zeka uygulamasını içermektedir. 
BERT, doğal dil işleme (NLP) alanında güçlü bir dil modelidir ve bu proje, sesli komutları anlamak ve evdeki ışıkları kontrol etmek için kullanılmıştır. 
Model, 20k benzersiz veri ile yeniden eğitilmiştir ve sesli komutlar ile entegre edilmiştir. 
Firebase üzerinden veri alınarak Python backend’inde model çıktıları işlenir ve pygame ile görselleştirilir.

**Proje İçeriği**:
- **BERT Modeli**: Işıkları kontrol etmek için eğitilmiş BERT modelini içerir.
- **Firebase Entegrasyonu**: Kullanıcıdan alınan ses verileri Firebase'e gönderilir ve Python backend'inde işlenir.
- **Pygame Simülasyonu**: Model çıktıları pygame kullanılarak görselleştirilir.
- **Ekran Görüntüleri**: Mobil uygulamanın simüle edilmiş çıktılarının ekran görüntüleri yer almaktadır.

## Özellikler
- **BERT Tabanlı Işık Kontrolü**: Model, sesli komutları analiz ederek ışıkların durumunu kontrol eder.
- **Firebase Entegrasyonu**: Kullanıcıdan alınan veriler Firebase’e gönderilir ve arka planda Python ile işlenir.
- **Pygame Simülasyonu**: Modelin verdiği çıktılar pygame ile görselleştirilir.
- **Esnek Veri Seti**: Veri seti çoklu sınıf sınıflandırmasına uygundur ancak istenirse tek sınıfa dönüştürülebilir. Ayrıca, veri seti artırılarak daha detaylı analiz yapılabilir.


## Usage
### **1. Installing Dependencies**
To install the necessary Python dependencies for the project, run the following command in your terminal or command line:


### model.pth

- **https://drive.google.com/file/d/1EOzG1wVFQm2FlAKR1fm3Qpg_GCu_yiB9/view?usp=drive_link on this link because I couldn't upload to github.
--------------------------------------------------------------------------------------------------------------------------------------------

```bash
pip install -r requirements.txt


