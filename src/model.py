

import os
import tensorflow as tf
import streamlit as st
from urllib.request import urlretrieve

MODEL_URL = "https://huggingface.co/ankitha1215/brain-tumor-detection-model/resolve/main/brain_tumor_model.keras"
MODEL_PATH = "/tmp/brain_tumor_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        try:
            urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Model download failed: {e}") from e

    if os.path.getsize(MODEL_PATH) < 1000:
        raise RuntimeError("Downloaded model file is too small. Check the Hugging Face file.")

    return tf.keras.models.load_model(MODEL_PATH)
