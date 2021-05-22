from io import UnsupportedOperation
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import models

model = models.load_model('my_model')


def encoder(y_pred):
    if y_pred == 0:
        return 'Kertas'
    elif y_pred == 1:
        return 'Batu'
    elif y_pred == 2:
        return 'Gunting'


st.markdown('''
<style>
    .e19lei0e1, .etr89bj1 {
        margin: 0 auto;
    }
    .etr89bj0 {
        text-align: center;
    }
</style>
''', unsafe_allow_html=True)

st.title('Rock Paper Scissors')
st.write('This is Machine Learning App that help you to predict wheter are rock, paper or scissors')
image = st.file_uploader(label='Upload your image')

if image is not None:
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image_cv2 = cv2.imdecode(file_bytes, 1)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    image_cv2 = cv2.resize(image_cv2, (125, 125))

    x = np.array(image_cv2).reshape(-1, 125, 125, 1).astype('float32') / 255.0

    prob = model.predict(x)
    y_pred = np.argmax(prob)
    prob_batu, prob_kertas, prob_gunting = prob[0][1], prob[0][0], prob[0][2]

    df_prob = pd.DataFrame({
        'Batu': [prob_batu],
        'Kertas': [prob_kertas],
        'Gunting': [prob_gunting],
    })

    st.image(image, width=250, caption=f'Hasil : {encoder(y_pred)}')
    st.write(df_prob)
