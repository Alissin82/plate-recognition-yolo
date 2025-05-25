# Plate Recognition YOLO Project

This project uses **YOLOv7** for Automatic Number Plate Recognition (ANPR), with support for **Roboflow** datasets and **GPU acceleration** using CUDA.

---

## ✅ Environment Setup

### 1. Install Python 3.10

Make sure Python 3.10 is installed on your system.
You can download it from: [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

> ⚠️ Do not check "Add to PATH" if you don't want it to be your default Python version system-wide.

---

### 2. Create Virtual Environment

In your project root directory, create a virtual environment named `yolov12_p310_env` using Python 3.10:

```powershell
C:\Users\Alissin\AppData\Local\Programs\Python\Python310\python.exe -m venv venv
```

---

### 3. Activate Virtual Environment

Run the appropriate command based on your shell:

#### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### CMD

```cmd
.\venv\Scripts\activate.bat
```

---

### 4. Install PyTorch with CUDA (GPU Support)

Install PyTorch 2.7.0, torchvision, and torchaudio with CUDA 11.8 support:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:

```powershell
nvidia-smi
```

---

### 5. Install YOLOv7 Requirements

```powershell
pip install -r requirements.txt
```

---

## 📥 Download Dataset from Roboflow

go to ./dataset/README.md for futher info

---

## 📦 Download Pretrained YOLOv7 Weights
YOLOv7 provides pretrained weights (yolov7.pt) which are used to initialize the model before fine-tuning on your own dataset. This helps the model converge faster and improves performance, especially when your dataset is small.

These weights were trained on the COCO dataset and provide strong general features for object detection.

### PowerShell Instructions (for Windows)
⬇️ Download into root:

```bash
Invoke-WebRequest -Uri "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt" -OutFile "yolov7.pt"
```

---

## 🎯 Training the YOLOv7 Model

```bash
python train.py --weights yolov7.pt --cfg cfg/training/yolov7.yaml --data dataset/data.yaml --epochs 15 --batch 16 --device 0
```

* `--batch`: Batch size during training
* `--cfg`: Path to YOLOv7 configuration file
* `--epochs`: Number of training epochs
* `--data`: Path to dataset configuration
* `--weights`: Path to pretrained weights
* `--device 0`: Use GPU (GPU #0, e.g., RTX 3050)

---

## 🔍 YOLOv7 Detection - Test Commands & Flag Descriptions

### 📌 Detection Commands

#### 🖼️ 1. Run detection on a batch of images
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.1 --source dataset/test/images --save-txt --save-conf --save-dir output/batch/ --view-img
```
Runs the model on all images in dataset/test/images, saves bounding boxes and confidence scores, and displays the output.

#### 🖼️ 2. Run detection on a single image
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.1 --source dataset/sample/plate_1.jpg --save-txt --save-conf --save-dir output/single/ --view-img
```
Runs the model on a single image (plate_1.jpg), saves the detection results, and opens the output image with annotations.

#### 🎞️ 3. Run detection on a video file
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source dataset/sample/video.mp4 --img-size 640 --save-txt --save-conf --save-dir output/video/ --view-img
```
Runs the model on a video file (frame-by-frame), saves results with confidence scores, and displays annotated video.

#### 🖼🎥 4. Run detection using webcam
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source 0 --img-size 640 --save-txt --save-conf --save-dir output/webcam/ --view-img
```
Activates the webcam for real-time object detection, displays results live, and saves output images with labels.

### 🏷️ YOLOv7 Optional Flags - Explanation Table

| Flag            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `--weights`     | Path to the trained model weights file (e.g., `best.pt`).                   |
| `--conf`        | Confidence threshold to filter weak detections (e.g., 0.1 or 0.25).         |
| `--source`      | Input source: image, folder, video, or webcam (`0`).                        |
| `--img-size`    | Input image resolution (e.g., 640 pixels).                                  |
| `--save-dir`    | Directory where output files (images/videos/labels) will be saved.          |
| `--save-txt`    | Saves bounding box coordinates into `.txt` label files (YOLO format).       |
| `--save-conf`   | Saves confidence scores alongside the bounding box coordinates.             |
| `--view-img`    | Opens and displays the output image or video with annotations.              |
