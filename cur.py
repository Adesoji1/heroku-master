import tensorflow as tf
import h5py as h5py
import streamlit as st
#from tensorflow.keras.models import Model 

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #To disable run on GPU

#for relative path
import os
from pathlib import Path
current_directory = Path(__file__).parent #Get current directory
file = open(os.path.join(current_directory, 'vehicle_detection.h5')
model = tf.keras.models.load_model.load(file) 

#model = tf.keras.models.load_model('/home/adesoji/hh/curacel/vehicle_detection.h5')



st.write("""
         # Bad or Good cars Prediction
         """
         )
st.write("This is a simple image classification web app to predict Bad or Good cars")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (192,192)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(96, 96)))
        #interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Damaged car!")
    else:
         np.argmax(prediction) == 1
         st.write("It is a Good car!")
    
    
    st.text("Probability (0:Damaged Car, 1: Good Car")
    st.write(prediction)