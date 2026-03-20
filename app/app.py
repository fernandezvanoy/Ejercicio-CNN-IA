import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import load_model

digits = load_digits()
X = digits.images
y = digits.target

model = load_model("../models/cnn_model.h5")

st.title("Clasificador de dígitos")

idx = st.slider("Selecciona imagen", 0, len(X)-1)

img = X[idx] / 16.0

st.image(img, width=150)

pred = model.predict(img.reshape(1,8,8,1))
pred_class = np.argmax(pred)

st.write(f"Predicción: {pred_class}")
st.write(f"Real: {y[idx]}")