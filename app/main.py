import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model', 'trained_fashion_mnist_model.h5')

model = tf.keras.models.load_model(model_path)

st.title("Fashion MNIST Image Classifier")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L') # Convert to grayscale
    img_array = np.array(img)/255.0
    img_array = np.reshape(img_array, (1, 28, 28, 1))
    return img_array

uploaded_file = st.file_uploader("Choose a Fashion MNIST image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1,col2 = st.columns(2)

    with col1:
        resized_image = image.resize((140,140))
        st.image(resized_image)

    with col2:
        if st.button('Predict'):
            processed_image = preprocess_image(uploaded_file)
            predictions = model.predict(processed_image)
            predicted_class = class_names[np.argmax(predictions)]
            # confidence = np.max(predictions) * 100

            st.write(f"Predicted Class: {predicted_class}")
            # st.write(f"Confidence: **{confidence:.2f}%**")