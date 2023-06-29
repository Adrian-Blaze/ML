import os
import streamlit as st
import pandas as pd
import pickle
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tempfile

path = f'C:/Users/user/Desktop/Python_lessons/ML/ML/Portfolio/face_detection/family_img_classification2.pickle'
path2 = f'C:/Users/user/Desktop/Python_lessons/ML/ML/Portfolio/face_detection/family_img_classification3.h5'

def generate_decorated_title(title):
    title_length = len(title)
    line = '-' * (title_length + 4)
    decorated_title = f"\n{line}\n| {title} |\n{line}\n"
    return decorated_title

def predict():
    # Load the model from a pickle file
    with tf.keras.utils.CustomObjectScope({'KerasLayer': hub.KerasLayer}):
      #with open('family_img_classification2.pickle', 'rb') as f:
        #model = pickle.load(f)
        model= tf.keras.models.load_model(path2)

    



    #feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    #pretrained_model_without_top_layer = hub.KerasLayer(
    #feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    #model.add( pretrained_model_without_top_layer)

    images_dict_label2 = { 0 : 'Dad', 
               1:'Fj', 
               2:'Kwubeche',
               3:'Mum', 
               4:'Shuu', 
               5:'Sinko',
               6:'TonyBlaze'}
    
    
    

  # Create a predict button
    predict_button = st.button("Predict")

  # Check if the predict button is clicked
    if predict_button:
    # Perform prediction or any desired action
        st.write("Prediction in progress...")
        prediction = model.predict(pred_img1)
        prediction = np.argmax(prediction)
      
        st.write("Prediction complete!")
        st.write('This should be', images_dict_label2[prediction])


    


def run_app():
    app_title = "Resemblometer App"
    decorated_title = generate_decorated_title(app_title)
    st.markdown(decorated_title)
    
     # File uploader for a single picture file
    uploaded_file = st.file_uploader("Upload a picture file", type=["jpg", "jpeg", "png"])
    # Display the uploaded picture
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
      st.image(uploaded_file)  
      with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name
        
        # Do something with the file path
        st.write('Uploaded file path:', file_path)
        predict_path = file_path
        predict_img = cv2.imread(predict_path)
        pred_img = cv2.resize(predict_img, (224, 224))
        
        
        pred_img = np.array(pred_img)
        pred_img = pred_img/255

  # Expand dimensions to create batch size of 1
        global pred_img1
        pred_img1 = np.expand_dims(pred_img, axis=0)

    predict()
    return
# Run the app
print('Here2')
if __name__ == '__main__':
    run_app()
print('Here3')
