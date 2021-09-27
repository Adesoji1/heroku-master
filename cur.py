import tensorflow as tf
import h5py as h5py
import streamlit as st


#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #To disable run on GPU

#for relative path
import os
from pathlib import Path
current_directory = Path(__file__).parent #Get current directory
file = open(os.path.join(current_directory, 'vehicle_detection.h5'))
model = tf.keras.models.load_model('vehicle_detection.h5')


st.write("""
         # Bad or Good cars Prediction
         """
         )
st.write("This is a simple image classification web app to predict Bad or Good cars")
fille = st.file_uploader("Please upload an image file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (192,192)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(96, 96)))
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)

        
        return prediction




if fille is None:
    st.text("Please upload an image file")
else:
    image = Image.open(fille)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    
     if np.argmax(prediction) <=0.01:
        st.write("If is a Bad  car!")

    
     else:
          np.argmax(prediction) >=0.02
          st.write("It is a Good Car !")
         
          st.text("Probability <0.01: Damaged Car, >0.02: It is a Good car!")
          st.write(prediction)
    
        
    

        
      
