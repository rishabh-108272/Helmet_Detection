import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved model
loaded_model = tf.keras.models.load_model("./helmet_detection_model.h5")

# Function to preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Function to make predictions on images
def predict_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Function to interpret the prediction
def interpret_prediction(prediction, threshold=0.6):
    if prediction > threshold:
        return "With Helmet"
    else:
        return "Without Helmet"

# Function to display image with prediction
def display_image_with_prediction(image, prediction_text):
    plt.imshow(image)
    plt.title(prediction_text)
    plt.axis('off')
    st.pyplot(plt)

st.title("Helmet Detection for Bike Riders")
st.write("Upload an image of a bike rider to check if they are wearing a helmet.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Predict
    prediction = predict_image(image, loaded_model)
    result = interpret_prediction(prediction[0][0])  # Assuming the model output is in the form [[probability]]
    
    # Display image and prediction
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {result}")
    
    # Display image with prediction
    display_image_with_prediction(image, result)
