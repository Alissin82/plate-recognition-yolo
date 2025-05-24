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
C:\Users\Alissin\AppData\Local\Programs\Python\Python310\python.exe -m venv yolov12_p310_env
```

---

### 3. Activate Virtual Environment

Run the appropriate command based on your shell:

#### PowerShell

```powershell
.\yolov12_p310_env\Scripts\Activate.ps1
```

#### CMD

```cmd
yolov12_p310_env\Scripts\activate.bat
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

### 5. Clone YOLOv7 Repository

Clone the official YOLOv7 GitHub repo:

```powershell
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

---

### 6. Install YOLOv7 Requirements

Inside the `yolov7/` directory:

```powershell
pip install -r requirements.txt
```

---

### 7. Install Roboflow Python SDK

```powershell
pip install roboflow
```

---

## 📥 Download Dataset from Roboflow

You can download the dataset either locally or via Google Colab depending on your network accessibility.

---

### Option A: Local Download (If Accessible)

In your Python script:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="your-api-key")
project = rf.workspace("object-recognition-yolo").project("anpr_ir-rsiqu")
version = project.version(1)
dataset = version.download("yolov8", location="C:/Users/Alissin/Desktop/UNI/plate-recognition-yolo/yolov7/roboflow_dataset")
```

> ⚠️ This may not work in some countries (e.g., Iran) due to network restrictions.

---

### Option B: Download via Google Colab (Recommended if A Fails)

1. Open [Google Colab](https://colab.research.google.com/)
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install Roboflow:

```bash
!pip install roboflow
```

4. Run the Roboflow download code:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="your-api-key")
project = rf.workspace("object-recognition-yolo").project("anpr_ir-rsiqu")
version = project.version(1)
dataset = version.download("yolov8", location="/content/drive/MyDrive/roboflow_dataset")
```

5. Go to Google Drive → Locate the `roboflow_dataset` folder → Download it.
6. Place the dataset manually inside:

```
yolov7/roboflow_dataset
```

---

## 📦 Download Pretrained YOLOv7 Weights
YOLOv7 provides pretrained weights (yolov7.pt) which are used to initialize the model before fine-tuning on your own dataset. This helps the model converge faster and improves performance, especially when your dataset is small.

These weights were trained on the COCO dataset and provide strong general features for object detection.

### PowerShell Instructions (for Windows)
⬇️ Download into the current yolov7 folder:

```bash
Invoke-WebRequest -Uri "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt" -OutFile "yolov7.pt"
```

---

## 🎯 Training the YOLOv7 Model

### 1. Update data.yaml

from
```
test: ../test/images
train: ../train/images
val: ../valid/images
```
to
```
test: ./roboflow_dataset/test/images
train: ./roboflow_dataset/train/images
val: ./roboflow_dataset/valid/images
```


### 2. Train the Model

Make sure you're inside the `yolov7/` directory and run the following:

```bash
python train.py \
  --weights yolov7.pt \
  --cfg cfg/training/yolov7.yaml \
  --data roboflow_dataset/data.yaml \
  --epochs 15 \
  --batch 4 \
  --device 0
```

* `--batch`: Batch size during training
* `--cfg`: Path to YOLOv7 configuration file
* `--epochs`: Number of training epochs
* `--data`: Path to dataset configuration
* `--weights`: Path to pretrained weights
* `--device 0`: Use GPU (GPU #0, e.g., RTX 3050)