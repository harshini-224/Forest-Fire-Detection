import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model and class names
model = load_model("FFD.keras")
class_names = ["fire", "non_fire"]  # Adjust if needed

# Prediction function
def predict_fire(model, img, class_names):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    label = class_names[1] if prediction > 0.5 else class_names[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Streamlit UI
st.set_page_config(page_title="Forest Fire Detection", layout="centered")
st.title("🌲🔥 Forest Fire Detection App")
st.write("Upload an image to check if it shows signs of wildfire.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            label, confidence = predict_fire(model, img, class_names)
        st.markdown(f"### 🔍 Prediction: `{label}`")
        st.markdown(f"### 📊 Confidence: `{confidence:.2%}`")
