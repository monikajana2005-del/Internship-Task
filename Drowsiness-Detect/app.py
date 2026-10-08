import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="In-Vehicle Sleep & Age Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Multi-Person In-Vehicle Sleep & Age Detector")
st.markdown(
    "Upload an image or video to analyze passenger drowsiness status and estimated age in real-time."
)

# Load OpenCV's YuNet ONNX Face Detector 
@st.cache_resource(show_spinner=False)
def load_face_detector():
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    model_file = "face_detection_yunet.onnx"

    # Download ONNX model file locally 
    if not os.path.exists(model_file):
        urllib.request.urlretrieve(model_url, model_file)

    detector = cv2.FaceDetectorYN.create(
        model=model_file,
        config="",
        input_size=(300, 300),
        score_threshold=0.5,
        nms_threshold=0.3
    )
    return detector

# Load Trained Keras Models
@st.cache_resource(show_spinner=False)
def load_models():
    sleep_model = tf.keras.models.load_model("sleep_detector_cnn.h5", compile=False)
    age_model = tf.keras.models.load_model("age_estimator_cnn.h5", compile=False)
    return sleep_model, age_model

face_detector = load_face_detector()
sleep_model, age_model = load_models()

def detect_faces_yunet(frame):
    """Detects face bounding boxes using YuNet ONNX detector."""
    h, w, _ = frame.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    
    boxes = []
    if faces is not None:
        for face in faces:
            box = face[:4].astype(int)
            x, y, bw, bh = box[0], box[1], box[2], box[3]
            x, y = max(0, x), max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)
            if bw > 0 and bh > 0:
                boxes.append((x, y, bw, bh))
    return boxes

def process_frame(frame):
    annotated_frame = frame.copy()
    faces = detect_faces_yunet(frame)
    
    total_people = len(faces)
    sleeping_count = 0
    sleeping_ages = []

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        # Convert BGR crop to RGB format 
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(face_crop_rgb, (64, 64)) / 255.0
        input_tensor = np.expand_dims(resized_face.astype('float32'), axis=0)

        # Predict Sleep & Age
        sleep_pred = sleep_model.predict(input_tensor, verbose=0)[0][0]
        age_pred = int(np.round(age_model.predict(input_tensor, verbose=0)[0][0]))
        age_pred = max(1, age_pred)

        # Inverted threshold condition 
        if sleep_pred < 0.5:  # Asleep Threshold
            sleeping_count += 1
            sleeping_ages.append(age_pred)
            color = (0, 0, 255)  # RED for Sleeping 
            label = f"ASLEEP | Age: {age_pred}"
        else:
            color = (0, 255, 0)  # GREEN for Awake 
            label = f"AWAKE | Age: {age_pred}"

        # Draw Rectangle and Text Overlay
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y - 10, label_size[1] + 10)
        
        cv2.rectangle(
            annotated_frame, 
            (x, label_y - label_size[1] - 5), 
            (x + label_size[0] + 5, label_y + 5), 
            color, 
            -1
        )
        cv2.putText(
            annotated_frame, 
            label, 
            (x + 2, label_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
        )

    return annotated_frame, total_people, sleeping_count, sleeping_ages

# Custom Streamlit Pop-up Dialog
@st.dialog(" Drowsiness Alert Notice")
def show_alert_popup(sleeping_count, ages):
    st.error(f" **Warning:** {sleeping_count} sleeping passenger(s) detected in the vehicle!")
    st.markdown("### Passenger Details:")
    for idx, age in enumerate(ages, 1):
        st.write(f"- **Passenger {idx}:** State = `ASLEEP` | Estimated Age = **{age} years old**")
    st.warning("Immediate action required: Driver/Vehicle alert system activated.")
    if st.button("Acknowledge & Close"):
        st.rerun()

# Sidebar Configuration
st.sidebar.title("Configuration")
option = st.sidebar.selectbox("Select Mode", ["Image Mode", "Video Mode"])


if option == "Image Mode":
    uploaded_file = st.file_uploader("Upload an Image file", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        processed_frame, total, sleeping, ages = process_frame(frame)
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Input Image Preview")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("#### Detection Preview")
            st.image(processed_rgb, use_container_width=True)

        st.subheader("Results Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total People Detected", total)
        m2.metric("Awake Count", total - sleeping)
        m3.metric("Sleeping Count", sleeping)

        if sleeping > 0:
            st.error(f" ALERT: {sleeping} Sleeping Person(s) Detected! Ages: {ages}")
            st.toast(f" Pop-up Alert: {sleeping} person(s) sleeping (Ages: {ages})", icon="⚠️")
            
            if st.button("View Alert Details Pop-up"):
                show_alert_popup(sleeping, ages)

# VIDEO MODE 
elif option == "Video Mode":
    uploaded_video = st.file_uploader("Upload a Video file", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original Video Preview")
            st.video(uploaded_video)

        with col2:
            st.markdown("#### Inference Output Stream")
            st_frame = st.empty()
            metrics_placeholder = st.empty()

        if st.button("Start Processing Stream"):
            cap = cv2.VideoCapture(temp_video_path)
            has_sleeping_occurred = False
            final_sleeping_count = 0
            final_ages = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame, total, sleeping, ages = process_frame(frame)
                st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                with metrics_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total People", total)
                    m2.metric("Awake Count", total - sleeping)
                    m3.metric("Sleeping Count", sleeping)

                if sleeping > 0:
                    has_sleeping_occurred = True
                    final_sleeping_count = sleeping
                    final_ages = ages
                    st.toast(f" {sleeping} Sleeping Person(s) Detected! Ages: {ages}")
                    
            cap.release()
            
            if has_sleeping_occurred:
                show_alert_popup(final_sleeping_count, final_ages)