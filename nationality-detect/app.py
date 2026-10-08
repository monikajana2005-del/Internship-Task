import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="Multi-Attribute Analysis System",
    page_icon="👤",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        color: #1E88E5;
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>👤 Multi-Attribute Facial Analysis System</h1>", unsafe_allow_html=True)


# 0: United States, 1: African, 2: Indian, 3: Other
NAT_CLASSES = ['United States', 'African', 'Indian', 'Other']


# LOAD TRAINED MODEL

@st.cache_resource
def load_trained_model():
   
    for path in ["nationality_attribute_model.keras", "nationality_attribute_model.keras"]:
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            continue
    st.error("Model file not found! Ensure 'nationality_attribute_model.keras' is in the project folder.")
    return None

model = load_trained_model()

#  HELPER FUNCTIONS (EMOTION & DRESS)

def detect_emotion_heuristic(img_np):
    
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Load cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return "Neutral"
            
        for (x, y, w, h) in faces:
            roi_gray = gray[y + int(h/2): y + h, x: x + w] 
            
            # Detect smile 
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.4, minNeighbors=12)
            
            if len(smiles) > 0:
                return "Happy"
            
            # Variance check on mouth region to detect open mouth 
            std_dev = np.std(roi_gray)
            if std_dev > 55:
                return "Surprise"
            elif std_dev < 30:
                return "Sad"
                
        return "Neutral"
    except Exception:
        return "Neutral"
"""Detects clothing/shirt color of the image using HSV analysis."""
 
def detect_dress_color(img_np):
   
    h, w, _ = img_np.shape
    clothing_crop = img_np[int(h * 0.65):, :, :]
    
    if clothing_crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(clothing_crop, cv2.COLOR_RGB2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))
    hue, sat, val = mean_hsv[0], mean_hsv[1], mean_hsv[2]

    if val < 50:
        return "Black"
    elif val > 200 and sat < 30:
        return "White"
    elif sat < 40:
        return "Grey"
    elif hue < 10 or hue > 160:
        return "Red"
    elif 10 <= hue < 35:
        return "Orange/Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 130:
        return "Blue"
    elif 130 <= hue < 160:
        return "Purple/Pink"
    return "Multi-color"

#  PREDICTION ENGINE

def run_inference(img_np):
    
    resized_img = cv2.resize(img_np, (128, 128)).astype('float32') / 255.0
    input_tensor = np.expand_dims(resized_img, axis=0)

    
    predictions = model.predict(input_tensor, verbose=0)

    if isinstance(predictions, list):
        nat_preds = predictions[0][0]
        age_preds = predictions[1][0][0]
    elif isinstance(predictions, dict):
        nat_preds = predictions['nationality_output'][0]
        age_preds = predictions['age_output'][0][0]

    nat_idx = np.argmax(nat_preds)
    predicted_nat = NAT_CLASSES[nat_idx]
    estimated_age = max(1, int(round(age_preds)))
    
    
    predicted_emotion = detect_emotion_heuristic(img_np)
    predicted_dress_color = detect_dress_color(img_np)

    return predicted_nat, predicted_emotion, estimated_age, predicted_dress_color

# STREAMLIT INTERFACE

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Input Image")
    uploaded_file = st.file_uploader("Choose a facial image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)

with col2:
    st.subheader("📊 Prediction Results")

    if uploaded_file is not None and model is not None:
        with st.spinner("Processing image tensor..."):
            nat, emotion, age, dress_color = run_inference(img_np)

            st.success("Analysis Complete!")
            st.metric(label="Predicted Nationality", value=nat)
            st.divider()

            st.markdown("### 📋 Conditional Outputs")

            # Project Conditional Output Requirements
            if nat == "Indian":
                c1, c2, c3 = st.columns(3)
                c1.metric("Emotion", emotion)
                c2.metric("Estimated Age", f"{age} yrs")
                c3.metric("Dress Colour", dress_color)

            elif nat == "United States":
                c1, c2 = st.columns(2)
                c1.metric("Emotion", emotion)
                c2.metric("Estimated Age", f"{age} yrs")

            elif nat == "African":
                c1, c2 = st.columns(2)
                c1.metric("Emotion", emotion)
                c2.metric("Dress Colour", dress_color)

            else:  
                st.metric("Emotion", emotion)

    else:
        st.info("Upload an image on the left to show prediction metrics.")