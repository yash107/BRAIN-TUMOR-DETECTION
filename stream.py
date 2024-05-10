import streamlit as st
import tensorflow as tf
import streamlit as st

def load_model():
  model=tf.keras.models.load_model('brain_tumor_detection.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Brain Tumor Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
import PIL


def import_and_predict(image, model):
        image = cv2.resize(image,(150,150))
        img_array = np.array(image)
        img_array=img_array.reshape(1,150,150,3)
        prediction = model.predict(img_array)
        indices=prediction.argmax()
        return indices

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    size = (150,150)    
    image = ImageOps.fit(image, size, PIL.Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img_array = np.array(image)
    img_array=image.reshape(1,150,150,3)
    a=model.predict(img_array)
    indices=a.argmax()
    labels=['Glioma Tumor','Meningioma Tumor','No Tumor','Pituitary Tumor']

    st.write("""The above patient has""",labels[indices])

