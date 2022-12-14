import numpy as np
import pandas as pd
from operator import mod
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras.utils import load_img, img_to_array

# @st.cache
def load_model():
  loaded_model = tf.keras.models.load_model('./cnn_model_exported')
  return loaded_model

model = load_model()

st.title('Land, Sea, or City?')

st.subheader('Is your image a landscape, a cityscape, or an oceanscape?')


uploaded_file = st.file_uploader(label='Upload your image here')
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(150, 150), color_mode='rgb')
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image /= 255.0


if st.button('Submit'):
    st.image(uploaded_file, width=200)
    pred = model.predict(image)
    prediction = pred[0].argmax()
    if prediction == 0:
        classification = 'Building'
    elif prediction == 1:
        classification = 'Forest'
    elif prediction == 2:
        classification = 'Glacier'
    elif prediction == 3:
        classification = 'Mountain'
    elif prediction == 4:
        classification = 'Sea'
    else:
        classification = 'Street'

    st.write('Your image prediction is:   ', classification)
    st.write('Probability:   ', str(pred.max()))
