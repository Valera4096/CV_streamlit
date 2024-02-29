import streamlit as st
import torch
from PIL import  Image
import cv2
import numpy as np



model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.torchscript') 
model.eval()
model.conf = 0.8



uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Преобразуйте изображение в формат OpenCV
    image = Image.open(uploaded_file)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
        st.write(f'Обнаруженный класс {class_name}: С вероятностью {confidence:.2f} %')