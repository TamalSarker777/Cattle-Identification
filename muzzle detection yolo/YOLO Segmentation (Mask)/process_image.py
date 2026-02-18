import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# CONFIG
BLUR_THRESHOLD = 120.0  # Tune if needed
LIGHT_THRESHOLD = 75.0  # Average pixel intensity threshold for lighting
REFLECTION_THRESHOLD = 200  # Pixel intensity threshold to detect reflections

# FUNCTIONS
def blur_score_laplacian(image_bgr) -> float:
    """Higher score = sharper image"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_blur(image_bgr, threshold: float):
    score = blur_score_laplacian(image_bgr)
    is_blur = score < threshold
    return is_blur, score

def detect_lighting(image_bgr, mask, threshold: float):
    """Check average light intensity of the muzzle region"""
    # Ensure the mask is binary and of type uint8
    mask = mask.astype(np.uint8)

    # Ensure mask size matches image size
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))

    # Use the mask to only consider the muzzle region
    masked_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    is_dark = avg_intensity < threshold
    return is_dark, avg_intensity

def detect_reflection(image_bgr, threshold: float):
    """Detect reflection by looking for high-intensity regions"""
    reflection_mask = image_bgr > threshold
    reflection_intensity = np.sum(reflection_mask) / (image_bgr.size)
    return reflection_intensity, reflection_mask

# Load YOLO model
def load_model():
    model = YOLO("G:/tamal/my tasks/cattle identification V2/Cattle-Identification/muzzle detection yolo/YOLO Segmentation (Mask)/runs/segment/train5/weights/best.pt")
    return model

model = load_model()

def process_image(image_path):
    # Read image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO segmentation
    results = model(image_np)
    result = results[0]

    # Run YOLO segmentation
    if result.masks is not None:

        # Get first mask
        mask = result.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)  # Ensure mask is uint8

        # Resize mask to original image size
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

        # Create overlay
        overlay = image_np.copy()
        overlay[mask > 0] = [0, 255, 0]

        # Tight Bounding Box
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            print("Mask detected but no valid region found.")
            return

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

        # Check for blur detection on cropped image
        is_blurry, blur_score = detect_blur(masked_cropped, BLUR_THRESHOLD)
        print(f"Blur Score of Cropped Muzzle: {blur_score:.2f}")
        print(f"Threshold: {BLUR_THRESHOLD}")
        if is_blurry:
            print("Result: The Cropped Muzzle Image is BLURRY ")
        else:
            print("Result: The Cropped Muzzle Image is SHARP ")

        # Check for lighting intensity (too dark?) on cropped image
        is_dark, avg_intensity = detect_lighting(image_bgr, cropped_mask, LIGHT_THRESHOLD)
        print(f"Average Intensity of Cropped Muzzle: {avg_intensity:.2f}")
        if is_dark:
            print("Warning: The cropped muzzle image is too DARK. Consider retaking the picture!")
        else:
            print("Lighting is good in the cropped muzzle.")

        # Check for reflections in the cropped image
        reflection_intensity, reflection_mask = detect_reflection(masked_cropped, REFLECTION_THRESHOLD)
        print(f"Reflection Intensity of Cropped Muzzle: {reflection_intensity:.2f}")
        if reflection_intensity > 0.1:  # If more than 10% of the image has reflections
            print("Warning: Reflections detected in the cropped muzzle! Consider retaking the picture!")
        else:
            print("No significant reflections detected in the cropped muzzle.")

        # Display the images using OpenCV
        cv2.imshow("Original", image_np)
        cv2.imshow("Segmentation Overlay", overlay)
        cv2.imshow("Masked (Tight Crop)", masked_cropped)
        cv2.imshow("Tight Cropped Muzzle", cropped_image)

        # Wait for any key to close the images
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No muzzle detected.")

# Provide image file path here
image_path = "G:/tamal/my tasks/cattle identification V2/Cattle-Identification/muzzle detection yolo/YOLO Segmentation (Mask)/train_dataset/images/val/muzzle_00369.jpg"  
process_image(image_path)
