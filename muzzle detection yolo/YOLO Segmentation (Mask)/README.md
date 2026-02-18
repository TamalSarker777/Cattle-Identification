# Cattle Muzzle Detection and Segmentation Project

This project focuses on detecting and segmenting cattle muzzles using YOLO (You Only Look Once) segmentation models. It includes data preprocessing, model training, and inference scripts for both command-line and web-based interfaces.

## Project Structure and File Descriptions

### Core Scripts

#### `app.py`
A Streamlit web application for interactive cattle muzzle segmentation and blur detection.
- **Functionality**: Allows users to upload images, detects blur using Laplacian variance, runs YOLO segmentation to identify muzzle regions, and displays original image, segmentation overlay, masked crop, and tight cropped muzzle.
- **Key Features**:
  - Loads a trained YOLO segmentation model
  - Blur detection with configurable threshold
  - Real-time image processing and visualization
  - Four-panel display: original, overlay, masked crop, cropped muzzle
- **Dependencies**: Streamlit, OpenCV, NumPy, PIL, Ultralytics YOLO

#### `process_image.py`
Command-line script for processing individual images for muzzle segmentation and blur detection.
- **Functionality**: Similar to `app.py` but for single image processing via command line. Loads an image, detects blur, runs segmentation, and displays results using OpenCV windows.
- **Key Features**:
  - Batch or single image processing capability
  - Blur score calculation and thresholding
  - Segmentation mask generation and cropping
  - OpenCV-based visualization (windows for original, overlay, masked, cropped)
- **Usage**: Modify `image_path` variable and run the script
- **Dependencies**: OpenCV, NumPy, PIL, Ultralytics YOLO

#### `masks_to_polygons.py`
Utility script to convert binary mask images to YOLO segmentation polygon annotations.
- **Functionality**: Processes mask images from `train_dataset/masks`, extracts contours, converts to normalized polygon coordinates, and saves as .txt files in YOLO format for segmentation training.
- **Key Features**:
  - Contour detection using OpenCV
  - Polygon simplification and area filtering (>200 pixels)
  - Coordinate normalization (0-1 range)
  - Outputs class ID followed by polygon points
- **Input/Output**: Masks from `./train_dataset/masks` → Labels to `./train_dataset/labels`
- **Dependencies**: OpenCV, OS

### Configuration and Data

#### `config.yaml`
YOLO dataset configuration file for training.
- **Contents**:
  - Dataset path: Points to `train_dataset` folder
  - Train/validation splits: `images/train`, `images/val`
  - Number of classes: 1 (muzzle)
  - Class names: ['muzzle']
- **Purpose**: Defines dataset structure for YOLO training pipeline

### Notebooks

#### `rename_and_resize_data.ipynb`
Jupyter notebook for preprocessing and organizing image data.
- **Functionality**: Renames images from `data` folder to standardized format (`muzzle_XXXXX.jpg`), converts to RGB, compresses to JPEG with quality 80, and saves to `processed_data`.
- **Key Features**:
  - Supports multiple image formats (.jpg, .jpeg, .png, .bmp, .webp)
  - Sequential renaming with zero-padded numbering
  - Quality compression to reduce file size
  - Error handling for corrupted files
- **Note**: Code has commented resize options but currently only compresses without resizing

#### `train.ipynb`
Jupyter notebook for training the YOLO segmentation model.
- **Functionality**: Loads pretrained YOLOv8 nano segmentation model (`yolo26n-seg.pt`), trains on dataset defined in `config.yaml` for 100 epochs with 640x640 images.
- **Key Features**:
  - Uses Ultralytics YOLO library
  - Configurable training parameters (epochs, image size, workers)
  - Includes PyTorch/CUDA version checks
- **Output**: Trained model weights saved in `runs/segment/trainX/weights/`

### Model Files

#### `yolo26n-seg.pt`
Pretrained YOLOv8 nano segmentation model weights.
- **Purpose**: Starting point for training custom segmentation model
- **Source**: Ultralytics YOLOv8 pretrained models

#### `yolo26n.pt`
Pretrained YOLOv8 nano detection model weights.
- **Purpose**: General YOLOv8 nano model (detection, not segmentation)
- **Note**: Not used in current segmentation pipeline

### Data Folders

#### `data/`
Raw input images for preprocessing.

#### `processed_data/`
Processed and renamed images after running `rename_and_resize_data.ipynb`.

#### `train_dataset/`
Structured dataset for YOLO training.
- `images/train/`: Training images
- `images/val/`: Validation images  
- `labels/train/`: Training segmentation labels (polygons)
- `labels/val/`: Validation segmentation labels

#### `runs/segment/`
YOLO training outputs.
- Multiple `trainX/` folders containing:
  - `args.yaml`: Training arguments
  - `weights/`: Model checkpoints (best.pt, last.pt)
  - `results.csv`: Training metrics

#### `task_2025634_annotations_2026_02_17_09_34_56_segmentation mask 1.1/`
External annotation data folder (from annotation tool called CVAT).
- Contains labelmap.txt, ImageSets, SegmentationClass/, SegmentationObject/

## Workflow

1. **Data Preparation**:
   - Place raw images in `data/`
   - Run `rename_and_resize_data.ipynb` to process → `processed_data/`
   - Organize into `train_dataset/images/train` and `val`

2. **Annotation**:
   - Create segmentation masks (if not using external annotations)
   - Run `masks_to_polygons.py` to convert masks to YOLO polygon format

3. **Training**:
   - Configure `config.yaml` with correct paths
   - Run `train.ipynb` to train the model

4. **Inference**:
   - Use `process_image.py` for command-line processing
   - Use `app.py` for web-based interactive processing

## Dependencies

- ultralytics (YOLO)
- opencv-python
- pillow (PIL)
- numpy
- streamlit
- torch (PyTorch)

## Notes

- Blur detection uses Laplacian variance with threshold 120.0
- Segmentation focuses on tight cropping of detected muzzle regions
- Models trained with 640x640 images
- Project uses YOLOv8 segmentation for pixel-level muzzle identification</content>
<parameter name="filePath">g:\tamal\my tasks\cattle identification V2\muzzle detection yolo\YOLO Segmentation (Mask)\README.md