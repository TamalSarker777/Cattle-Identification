import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# FUNCTIONS
def blur_score_laplacian(image_bgr) -> float:
    """Higher score = sharper image"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_blur(image_bgr, threshold: float):
    score = blur_score_laplacian(image_bgr)
    is_blur = score < threshold
    return is_blur, score

def detect_lighting(image_bgr, threshold: float):
    """Check average light intensity"""
    # Convert to grayscale to get intensity
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    is_dark = avg_intensity < threshold
    return is_dark, avg_intensity

def detect_reflection(image_bgr, threshold: float):
    """Detect reflection by looking for high-intensity regions"""
    reflection_mask = image_bgr > threshold
    reflection_intensity = np.sum(reflection_mask) / (image_bgr.size)
    return reflection_intensity, reflection_mask

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("G:/tamal/my tasks/cattle identification V2/Cattle-Identification/muzzle detection yolo/YOLO Segmentation (Mask)/runs/segment/train5/weights/best.pt")
    return model

model = load_model()

# Streamlit app
st.title("🐄 Cattle Muzzle Segmentation Viewer with Intensity and Blur Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Add sliders for manual threshold setting
st.sidebar.title("Adjust Thresholds")
BLUR_THRESHOLD = st.sidebar.slider("Blur Threshold", min_value=0, max_value=200, value=120, step=1)
LIGHT_THRESHOLD = st.sidebar.slider("Lighting Intensity Threshold", min_value=0, max_value=255, value=100, step=1)
REFLECTION_THRESHOLD = st.sidebar.slider("Reflection Intensity Threshold", min_value=0, max_value=255, value=200, step=1)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Check for blur detection
    is_blurry, blur_score = detect_blur(image_bgr, BLUR_THRESHOLD)
    st.subheader(f"Blur Score: {blur_score:.2f}")
    st.subheader(f"Threshold: {BLUR_THRESHOLD}")
    if is_blurry:
        st.warning("Result: The Image is BLURRY ❌")
    else:
        st.success("Result: The Image is SHARP ✅")

    # Check for lighting intensity (too dark?)
    is_dark, avg_intensity = detect_lighting(image_bgr, LIGHT_THRESHOLD)
    st.subheader(f"Average Intensity: {avg_intensity:.2f}")
    if is_dark:
        st.warning("Warning: The image is too DARK. Consider retaking the picture!")
    else:
        st.success("Lighting is good.")

    # Check for reflections
    reflection_intensity, reflection_mask = detect_reflection(image_bgr, REFLECTION_THRESHOLD)
    st.subheader(f"Reflection Intensity: {reflection_intensity:.2f}")
    if reflection_intensity > 0.1:  # If more than 10% of the image has reflections
        st.warning("Warning: Reflections detected! Consider retaking the picture!")
    else:
        st.success("No significant reflections detected.")

    # Run YOLO segmentation
    results = model(image_np)
    result = results[0]

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
