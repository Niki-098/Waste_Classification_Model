# Waste Classification Model ♻
## 1. Approach

The project focuses on automatic waste detection and classification using YOLOv8.

Dataset was organized into six categories: cardboard, glass, metal, paper, plastic, and trash.

A preprocessing pipeline was created to generate YOLO-format bounding boxes automatically using contour detection.

The dataset was split into training and validation sets (80:20).

YOLOv8 was trained on the prepared dataset to detect and classify waste objects.

Finally, the trained model was tested on custom video inputs to evaluate real-world performance.

## 2. Libraries Used

**Ultralytics YOLOv8 –** Object detection and training framework

**OpenCV –** Image preprocessing, contour detection, and visualization

**NumPy –** Numerical operations

**PyTorch –** Deep learning backend for YOLO

**Tkinter –** File dialog for video input selection

**Shutil & OS –** File handling and dataset organization

## 3. How to Run

**Step 1: Preprocess the dataset**<br>
*python preprocess.py*


**Splits data into train/val sets.**<br>

Generates YOLO annotations.<br>
Creates preview images with bounding boxes.

**Step 2: Train the model**<br>
*python train_model.py*


Trains YOLOv8 on the processed dataset.<br>
Saves weights in runs/train/scrap_detector/

**Step 3: Run inference on video**<br>
*python simulation.py*


Select a video file.<br>
Model runs detection using best.pt.<br>
Saves output video in current directory as output_<video_name>.mp4.

## 4. Challenges Faced**

**GPU Limitation:** Training YOLO on CPU was very slow.

**Solution:** Used Google Colab T4 GPU, which significantly accelerated training and testing.

**Bounding Box Generation:** Since the dataset did not have predefined annotations, bounding boxes were auto-generated using thresholding and contours. Some images required fine-tuning of threshold values for better accuracy.

**Video Testing:** Managing different video formats and ensuring the output video was correctly saved and accessible across platforms (Windows/Linux) required additional handling.
