import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


 #  STREAMLIT PAGE CONFIGURATION

st.set_page_config(
    page_title="Traffic Vision: Car Color & Pedestrian Detector",
    page_icon="🚦",
    layout="wide",
)


#  Model loading

@st.cache_resource
def load_cnn_model():
    
    model_path = "cnn_model.h5"

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None



#  HSV COLOR ANALYSIS FOR BLUE CARS

def is_blue_car(car_crop_rgb):
    
    if car_crop_rgb is None or car_crop_rgb.size == 0:
        return False

    # Convert RGB to HSV
    hsv_crop = cv2.cvtColor(car_crop_rgb, cv2.COLOR_RGB2HSV)

    # Defined HSV range for blue hues
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Calculate blue pixel ratio in the car crop
    blue_mask = cv2.inRange(hsv_crop, lower_blue, upper_blue)
    blue_ratio = np.count_nonzero(blue_mask) / max(
        1, (car_crop_rgb.shape[0] * car_crop_rgb.shape[1])
    )

    
    return blue_ratio >= 0.12 # Classified as Blue if >12% of the crop area is blue


 # OPENCV DETECTOR 

def detect_cars_and_people(img_np, confidence_threshold):
    """Detects people and car bounding regions within the traffic scene."""
    height, width, _ = img_np.shape
    detections = []

    
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        ped_boxes, weights = hog.detectMultiScale(
            img_np, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        for (x, y, w, h), weight in zip(ped_boxes, weights):
            if float(weight) >= (confidence_threshold * 0.35):
                detections.append((
                    "person",
                    max(0, x),
                    max(0, y),
                    min(width, x + w),
                    min(height, y + h),
                ))
    except Exception:
        
        if hasattr(cv2, "CascadeClassifier"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
                body_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                bodies = body_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 60)
                )
                for x, y, w, h in bodies:
                    detections.append((
                        "person",
                        max(0, x),
                        max(0, y),
                        min(width, x + w),
                        min(height, y + h),
                    ))
            except Exception:
                pass 

    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    dilated = cv2.dilate(
        edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2
    )

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / max(1, h)

        if (
            (width * height) * 0.015 < area < (width * height) * 0.45
            and 0.5 < aspect_ratio < 2.8
        ):
            
            overlap = any(
                d[0] == "person"
                and abs(d[1] - x) < 20
                and abs(d[2] - y) < 20
                for d in detections
            )
            if not overlap:
                detections.append((
                    "car",
                    max(0, x),
                    max(0, y),
                    min(width, x + w),
                    min(height, y + h),
                ))

    return detections


#  USER INTERFACE ARCHITECTURE

st.title("🚦 Traffic Signal Vision System")
st.markdown("""
### Detection Criteria Enforcement:
*  **Red Rectangle**: Drawn around **Blue Cars**.
*  **Blue Rectangle**: Drawn around **Other Color Cars**.
*  **People Counter**: Banner on the image displaying the count of people at the signal.
""")

model = load_cnn_model()

if model is not None:
    st.sidebar.success("`cnn_model.h5` loaded successfully!")
else:
    st.sidebar.warning(
        "cnn_model.h5` not found in root directory."
    )

# Sidebar Controls
confidence_thresh = st.sidebar.slider(
    "Sensitivity Threshold", 0.2, 1.0, 0.4, 0.05
)
uploaded_file = st.sidebar.file_uploader(
    "Upload Traffic Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    input_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(input_pil)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Traffic Scene")
        st.image(input_pil, use_container_width=True)

    if st.button(" Analyze Traffic Image", type="primary"):
        with st.spinner("Analyzing cars and counting pedestrians..."):

            detections = detect_cars_and_people(img_np, confidence_thresh)
            annotated_img = img_np.copy()

            people_count = 0
            blue_car_count = 0
            other_car_count = 0

            for obj_type, x1, y1, x2, y2 in detections:
                #  Count Pedestrians
                if obj_type == "person":
                    people_count += 1
                    cv2.rectangle(
                        annotated_img,
                        (x1, y1),
                        (x2, y2),
                        color=(255, 255, 0),
                        thickness=2,
                    )

                # Color-based Car Rectangles
                elif obj_type == "car":
                    car_crop = img_np[y1:y2, x1:x2]

                    if is_blue_car(car_crop):
                        blue_car_count += 1
                        box_color = (255, 0, 0)  # RGB: Red for Blue Cars
                        label = "Blue Car"
                    else:
                        other_car_count += 1
                        box_color = (0, 0, 255)  # RGB: Blue for Other Cars
                        label = "Car"

                    # Draw Bounding Rectangle & Label
                    cv2.rectangle(
                        annotated_img,
                        (x1, y1),
                        (x2, y2),
                        color=box_color,
                        thickness=3,
                    )
                    cv2.putText(
                        annotated_img,
                        label,
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

            #  High-Contrast People Counter Overlay Banner on Traffic Image
            banner_text = f"People at Signal: {people_count}"
            cv2.rectangle(annotated_img, (10, 10), (360, 55), (0, 0, 0), -1)
            cv2.putText(
                annotated_img,
                banner_text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        with col2:
            st.subheader(" Analyzed Signal Frame")
            st.image(annotated_img, use_container_width=True)

        st.divider()

        # Summary Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("🚶 People at Signal", value=people_count)
        m2.metric("🔴 Blue Cars (Red Rectangle)", value=blue_car_count)
        m3.metric("🔵 Other Cars (Blue Rectangle)", value=other_car_count)

else:
    st.info(
        "👈 upload a traffic image in the sidebar to start analysis."
    )