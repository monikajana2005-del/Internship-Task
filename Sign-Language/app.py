import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# CONFIGURATION & MODEL LOADING

st.set_page_config(page_title="Sign Language Recognition", layout="wide", page_icon="🤟")

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'DELETE', 'NOTHING', 'SPACE'
]

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("sign_language_cnn_stable.h5")

# TIME WINDOW VALIDATION (6 PM to 10 PM)

def is_system_operational(start_hour=18, end_hour=22):
    current_hour = datetime.now().hour
    return start_hour <= current_hour < end_hour


# PREPROCESSING 

def preprocess_frame(img_array, target_size=(64, 64), enable_center_crop=True):
    
    h, w, _ = img_array.shape

    # Center crop square box 
    if enable_center_crop:
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        img_cropped = img_array[start_y:start_y + min_dim, start_x:start_x + min_dim]
    else:
        img_cropped = img_array

    # Resize image
    resized_img = cv2.resize(img_cropped, target_size)

    normalized_img = resized_img.astype('float32') / 255.0
    
    return np.expand_dims(normalized_img, axis=0), resized_img

#  STREAMLIT INTERFACE

st.title("🤟 Real-Time Sign Language Detection")
st.markdown("CNN-Powered Gesture Classifier with Upload & Live Webcam Modes")

# Sidebar - Operational Status & Time Settings
st.sidebar.header(" Operational Controls")
override_time = st.sidebar.checkbox("Bypass Time Window Restriction (Testing)", value=False)
enable_crop = st.sidebar.checkbox("Enable Center Cropping (Recommended)", value=True)

START_HOUR = 18  # 6 PM
END_HOUR = 22    # 10 PM

operational = is_system_operational(START_HOUR, END_HOUR) or override_time

# Display Status Banner
if operational:
    st.sidebar.success(f"🟢 System Status: Active ({START_HOUR}:00 - {END_HOUR}:00)")
else:
    st.sidebar.error(f"🔴 System Status: Inactive (Active window: 6:00 PM - 10:00 PM)")

if not operational:
    st.warning("⚠️ The Sign Language Detection system is currently offline. System operation is restricted to 6:00 PM – 10:00 PM.")
    st.info("Check 'Bypass Time Window Restriction' in the sidebar to run out-of-schedule tests.")
else:
    model = load_trained_model()
    
    app_mode = st.radio("Select Operating Mode:", ["Upload Image", "Real-Time Webcam Video"])

    #  IMAGE UPLOAD

    if app_mode == "Upload Image":
        st.subheader("📷 Upload Sign Language Image")
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(image)
            
            # Preprocess frame
            processed_input, processed_preview = preprocess_frame(img_np, enable_center_crop=enable_crop)
            
            # Predict
            predictions = model.predict(processed_input)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Sign Gesture", use_container_width=True)
                st.image(processed_preview, caption="Resized Model Input (64x64)", width=150)

            with col2:
                st.markdown("### Prediction Result")
                st.metric(label="Recognized Sign", value=predicted_class)
                st.progress(int(confidence))
                st.write(f"**Confidence Level:** {confidence:.2f}%")

    #  REAL-TIME WEBCAM VIDEO
   
    elif app_mode == "Real-Time Webcam Video":
        st.subheader("🎥 Real-Time Sign Language Recognition")
        st.info(" Place your hand in the center of the frame for best detection accuracy.")

        camera_image = st.camera_input("Take a picture for real-time gesture recognition")

        if camera_image is not None:
            image = Image.open(camera_image).convert('RGB')
            img_np = np.array(image)
            
            # Preprocess frame
            processed_input, processed_preview = preprocess_frame(img_np, enable_center_crop=enable_crop)
            
            # Predict
            predictions = model.predict(processed_input)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Captured Webcam Frame", use_container_width=True)
                st.image(processed_preview, caption="Resized Model Input (64x64)", width=150)

            with col2:
                st.markdown("### Real-Time Output")
                st.metric(label="Predicted Sign", value=predicted_class)
                st.progress(int(confidence))
                st.write(f"**Confidence:** {confidence:.2f}%")