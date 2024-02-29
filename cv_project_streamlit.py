import streamlit as st
import torch
from PIL import  Image
import cv2
import numpy as np
######
import requests
from io import BytesIO


model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.torchscript', trust_repo=True) 
model.eval()
model.conf = 0.8

def prediction(input, url_reques = False):
    if url_reques:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = Image.open(input)
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    
    image_cv2 = cv2.resize(image_cv2, (540, 540))
        # Выполните детекцию объектов с помощью модели YOLOv5
    results = model(image_cv2)

    results_image = Image.fromarray(results.render()[0])

    # Отобразите оригинальное изображение и результаты детекции
    st.subheader('Загруженное изображение')
    st.image(image)

    st.subheader('Результат детекций')
    st.image(results_image)

    detections = results.xyxy[0]
    for detection in detections:
        class_id = int(detection[5])
        class_name = model.names[class_id]
        confidence = float(detection[4])
        st.write('='*88)  
        st.write(f'Обнаруженный класс {class_name}') 
        st.write(f'С вероятностью {confidence:.2f} %')
        st.write('='*88)  


uploaded_file2 = st.file_uploader("Загрузите изображение", type=["jpg",'png'], accept_multiple_files=True)
if uploaded_file2 is not None:
    for i in uploaded_file2:
        prediction(i)
        
                        
url = st.text_input("Введите URL изображения:")

if url:
    prediction(url, url_reques= True)
        