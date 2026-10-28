
import os
import tensorflow as tf
import streamlit as st
from urllib.request import urlretrieve
MODEL_URL = "https://huggingface.co/ankitha1215/brain-tumor-detection-model/resolve/main/brain_tumor_model.keras"
MODEL_PATH = "/tmp/brain_tumor_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urlretrieve(MODEL_URL, MODEL_PATH)

    return tf.keras.models.load_model(MODEL_PATH)
