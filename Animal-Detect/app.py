import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

st.set_page_config(page_title="Animal Detection & Carnivore Identifier", layout="wide")

# Set of Carnivorous species present in the 90-class dataset
CARNIVORES = {
    "bear", "cheetah", "coyote", "crocodile", "fox", "hyena", 
    "jaguar", "leopard", "lion", "panther", "tiger", "wolf"
}

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("animal_cnn_model.h5")
    class_names = np.load("classes.npy")
    return model, class_names

model, class_names = load_resources()

#streamlit pop-up dialog function
@st.dialog("🚨 Carnivore Alert")
def show_carnivore_popup(carnivore_count, detected_carnivores):
    st.warning(f"**Attention Needed!**")
    st.write(f"The system detected **{carnivore_count}** carnivorous animal(s) in the frame.")
    st.markdown("**Detected Species:**")
    for species in detected_carnivores:
        st.write(f" {species.capitalize()}")
    if st.button("Acknowledge & Close"):
        st.rerun()
# Multi-window sliding detection engine
def detect_and_classify(image, confidence_thresh=0.70):
    h, w, _ = image.shape
    detections = []
    
    # Scale sliding windows according to frame dimensions
    window_scales = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
    stride = 64

    for scale_w, scale_h in window_scales:
        win_w, win_h = int(w * scale_w), int(h * scale_h)
        for y in range(0, h - win_h + 1, stride):
            for x in range(0, w - win_w + 1, stride):
                crop = image[y:y+win_h, x:x+win_w]
                resized = cv2.resize(crop, (128, 128)) / 255.0
                inp = np.expand_dims(resized, axis=0)
                
                preds = model.predict(inp, verbose=0)[0]
                class_idx = np.argmax(preds)
                conf = preds[class_idx]
                
                if conf >= confidence_thresh:
                    label = class_names[class_idx]
                    detections.append({
                        'box': (x, y, win_w, win_h),
                        'label': label,
                        'conf': conf
                    })

    filtered_detections = []
    for d in sorted(detections, key=lambda k: k['conf'], reverse=True):
        box_a = d['box']
        overlap = False
        for f in filtered_detections:
            box_b = f['box']
            xA = max(box_a[0], box_b[0])
            yA = max(box_a[1], box_b[1])
            xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
            yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = box_a[2] * box_a[3]
            
            if interArea / float(boxAArea) > 0.3:
                overlap = True
                break
        if not overlap:
            filtered_detections.append(d)

    return filtered_detections

def process_frame(frame):
    # Ensure standard BGR format for OpenCV drawing operations
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
    
    detections = detect_and_classify(frame)
    carnivore_count = 0
    detected_carnivores = set()
    
    for det in detections:
        x, y, w, h = det['box']
        label = det['label']
        is_carnivore = label.lower() in CARNIVORES
        
        if is_carnivore:
            carnivore_count += 1
            detected_carnivores.add(label)
            #  Red  (0, 0, 255)
            color = (0, 0, 255)  
        else:
            #  Green  (0, 255, 0)
            color = (0, 255, 0)  

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            frame_bgr, f"{label} ({det['conf']:.2f})", 
            (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
    # Convert frame back to RGB for Streamlit rendering
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, carnivore_count, list(detected_carnivores)

#  STREAMLIT DASHBOARD UI 
st.title("🦁 Animal Detection & Carnivore Identifier")

mode = st.sidebar.radio("Select Processing Mode", ["Image Mode", "Video Mode"])

if mode == "Image Mode":
    st.subheader("Image Detection Preview")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Image", use_container_width=True)
            
        with st.spinner("Detecting and classifying animals..."):
            processed_img, carnivore_count, carnivore_list = process_frame(img_np.copy())
            
        with col2:
            st.image(processed_img, caption="Detection Preview", use_container_width=True)

        if carnivore_count > 0:
            st.error(f" ALERT: Detected {carnivore_count} Carnivorous Animal(s) in the frame!")
            # Trigger Streamlit Modal Pop-Up
            show_carnivore_popup(carnivore_count, carnivore_list)
        else:
            st.success("No carnivorous animals detected.")

elif mode == "Video Mode":
    st.subheader("Video Detection Preview")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        alert_placeholder = st.empty()
        
        max_carnivores = 0
        all_detected_carnivores = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, count, species_list = process_frame(frame_rgb)
            
            if count > max_carnivores:
                max_carnivores = count
            for s in species_list:
                all_detected_carnivores.add(s)
                
            st_frame.image(processed_frame, channels="RGB", use_container_width=True)
            
            if count > 0:
                alert_placeholder.error(f" ALERT: {count} Carnivore(s) currently visible in frame!")
            else:
                alert_placeholder.success("Frame clear of carnivorous animals.")
                
        cap.release()
        
        # Trigger pop-up summary dialog after video ends if carnivores were found
        if max_carnivores > 0:
            show_carnivore_popup(max_carnivores, list(all_detected_carnivores))