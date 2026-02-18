"""
Streamlit web application for interactive cattle muzzle segmentation and blur detection.
Loads YOLO model, allows image upload, detects blur using Laplacian variance,
runs segmentation, and displays original, overlay, masked crop, and tight cropped muzzle.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO mode
@st.cache_resource
def load_model():
    model = YOLO("G:/tamal/my tasks/cattle identification V2/muzzle detection yolo/YOLO Segmentation (Mask)/runs/segment/train5/weights/best.pt")
    return model

model = load_model()

# CONFIG
BLUR_THRESHOLD = 120.0  # Tune if needed

# FUNCTIONS
def blur_score_laplacian(image_bgr) -> float:
    """Higher score = sharper image"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_blur(image_bgr, threshold: float):
    score = blur_score_laplacian(image_bgr)
    is_blur = score < threshold
    return is_blur, score

# Streamlit app
st.title("🐄 Cattle Muzzle Segmentation Viewer with Blur Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO segmentation
    results = model(image_np)
    result = results[0]

    # Check for blur detection
    is_blurry, blur_score = detect_blur(image_bgr, BLUR_THRESHOLD)

    # Display blur result
    st.subheader(f"Blur Score: {blur_score:.2f}")
    st.subheader(f"Threshold: {BLUR_THRESHOLD}")
    if is_blurry:
        st.warning("Result: BLURRY ❌")
    else:
        st.success("Result: SHARP ✅")

    # Run YOLO segmentation
    if result.masks is not None:

        # Get first mask
        mask = result.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        # Resize mask to original image size
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))


        # Create overlay
        overlay = image_np.copy()
        overlay[mask > 0] = [0, 255, 0]


        # Tight Bounding Box
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            st.warning("Mask detected but no valid region found.")
            st.stop()

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # Crop image and mask FIRST
        cropped_image = image_np[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # Apply mask AFTER cropping (this removes extra black border)
        masked_cropped = cv2.bitwise_and(
            cropped_image,
            cropped_image,
            mask=cropped_mask
        )


        # COLUMN LAYOUT
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.markdown("### Original")
            st.image(image_np, use_container_width=True)

        with col2:
            st.markdown("### Segmentation Overlay")
            st.image(overlay, use_container_width=True)

        with col3:
            st.markdown("### Masked (Tight Crop)")
            st.image(masked_cropped, use_container_width=True)

        with col4:
            st.markdown("### Tight Cropped Muzzle")
            st.image(cropped_image, use_container_width=True)

    else:
        st.warning("No muzzle detected.")
