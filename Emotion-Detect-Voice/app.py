import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st

from gender_verifier import is_female_voice

# Page configuration
st.set_page_config(
    page_title="Female Voice Emotion Detector",
    page_icon="🎙️",
    layout="centered"
)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("emotion_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_artifacts()
except Exception as e:
    st.error("Model artifacts not found. Please ensure 'emotion_model.h5', 'scaler.pkl', and 'label_encoder.pkl' exist in the workspace directory.")
    st.stop()

# Helper function for MFCC feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Process audio and make predictions
def process_and_predict(audio_path):
    with st.spinner("Analyzing voice parameters..."):
        #  Gender Verification
        is_female, pitch = is_female_voice(audio_path)
        
        if not is_female:
            st.error(
                f"**Voice Verification Failed:** Non-female voice detected "
                f"(Average Pitch: {pitch:.1f} Hz). Please record or upload a female voice."
            )
            return

        #  Emotion Classification
        features = extract_features(audio_path)
        scaled_features = scaler.transform([features])
        
        preds = model.predict(scaled_features)
        pred_idx = np.argmax(preds)
        emotion = label_encoder.classes_[pred_idx].upper()
        confidence = preds[0][pred_idx] * 100

    # Display Results
    st.success("Analysis Complete!")
    st.metric(label="Detected Emotion", value=f"{emotion}", delta=f"{confidence:.1f}% Confidence")
    
    # Visual confidence distribution across all classes
    st.subheader("Emotion Confidence Breakdown")
    conf_dict = {
        label_encoder.classes_[i].capitalize(): float(preds[0][i])
        for i in range(len(label_encoder.classes_))
    }
    st.bar_chart(conf_dict)

# UI Header
st.title("🎙️ Female Voice Emotion Recognition")
st.write("Upload an audio note or use your microphone to detect emotions from female speech.")

st.divider()


tab1, tab2 = st.tabs([" Upload Voice Note", " Record Live Voice"])

with tab1:
    st.subheader("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(temp_path)
        if st.button("Analyze Uploaded Audio", key="btn_upload"):
            process_and_predict(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)

with tab2:
    st.subheader("Record Audio from Microphone")
    recorded_audio = st.audio_input("Record a voice clip")
    
    if recorded_audio is not None:
        temp_rec_path = "temp_recorded_audio.wav"
        with open(temp_rec_path, "wb") as f:
            f.write(recorded_audio.getbuffer())
            
        if st.button("Analyze Recorded Audio", key="btn_record"):
            process_and_predict(temp_rec_path)
            if os.path.exists(temp_rec_path):
                os.remove(temp_rec_path)