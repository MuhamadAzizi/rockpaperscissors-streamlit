import cv2
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st


def decoder(y_pred):
    if y_pred == 0:
        return 'Kertas'
    elif y_pred == 1:
        return 'Batu'
    elif y_pred == 2:
        return 'Gunting'


st.markdown('''
    <style>
        .etr89bj1, .e10os9ge0 {
            margin: 0 auto;
        }
    </style>
''', unsafe_allow_html=True)

model = models.load_model('my_model')

st.title('Rock Paper Scissors')
st.write('Website ini merupakan sebuah website untuk mendeteksi permainan batu kertas gunting \
    apakah gambar yang diupload berbentuk batu, kertas, atau gunting.')

image = st.file_uploader('Upload gambar anda')
if image is not None:
    image_byte = np.asarray(bytearray(image.read()), dtype=np.uint8)
    cv_image = cv2.imdecode(image_byte, 1)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv_image = cv2.resize(cv_image, (125, 125))

    x = np.array(cv_image).reshape(-1, 125, 125, 1).astype('float32') / 255.0
    prob = model.predict(x)
    y_pred = np.argmax(prob)

    batu, kertas, gunting = prob[0][1], prob[0][0], prob[0][2]
    prob_table = pd.DataFrame(
        {
            'Batu': [batu],
            'Kertas': [kertas],
            'Gunting': [gunting]
        }
    )

    st.image(image, caption=f'Hasilnya adalah : {decoder(y_pred)}')
    prob_table

st.sidebar.write('Catatan untuk mendapatkan prediksi yang diinginkan')
st.sidebar.warning('Gambar harus memiliki background polos')
st.sidebar.warning('Posisi harus tangan kanan menghadap ke kiri')
