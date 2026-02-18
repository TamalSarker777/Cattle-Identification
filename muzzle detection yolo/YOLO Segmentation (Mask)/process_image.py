"""
Command-line script for processing individual images for cattle muzzle segmentation and blur detection.
Loads YOLO model, processes specified image, detects blur, runs segmentation,
and displays results using OpenCV windows (original, overlay, masked crop, cropped muzzle).
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

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

# Load YOLO model
def load_model():
    model = YOLO("G:/tamal/my tasks/cattle identification V2/muzzle detection yolo/YOLO Segmentation (Mask)/runs/segment/train5/weights/best.pt")
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

    # Check for blur detection
    is_blurry, blur_score = detect_blur(image_bgr, BLUR_THRESHOLD)

    print(f"Blur Score: {blur_score:.2f}")
    print(f"Threshold: {BLUR_THRESHOLD}")
    if is_blurry:
        print("Result: The Image is BLURRY ")
    else:
        print("Result: The Image is SHARP ")

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
image_path = "G:/tamal/my tasks/cattle identification V2/muzzle detection yolo/YOLO Segmentation (Mask)/train_dataset/images/val/muzzle_00353.jpg"  
process_image(image_path)
