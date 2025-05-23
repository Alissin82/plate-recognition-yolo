# Plate Recognition YOLO Project

This project uses YOLOv7 for Automatic Number Plate Recognition (ANPR) with support for Roboflow datasets and GPU acceleration.

---

## Environment Setup

### 1. Install Python 3.10

Make sure Python 3.10 is installed on your system.  
You can download it from:  
[https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

❗ Do not check "Add to PATH" if you don't want it to be your default Python version system-wide.

---

### 2. Create Virtual Environment

Run the following command in your project root directory to create a virtual environment named `yolov12_p310_env` using Python 3.10:

example:
```powershell
C:\Users\Alissin\AppData\Local\Programs\Python\Python310\python.exe -m venv yolov12_p310_env
```

### 3. Activate Virtual Environment
Run the appropriate command depending on your shell:

#### PowerShell
```powershell
.\yolov12_p310_env\Scripts\Activate.ps1
```

#### CMD (Command Prompt)
```commandline
yolov12_p310_env\Scripts\activate.bat
```

### 4. Install PyTorch with CUDA (GPU Support)
Install PyTorch 2.7.0, torchvision, and torchaudio with CUDA 11.8 support:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This step will enable hardware acceleration with your NVIDIA GPU (e.g., RTX 3050 Laptop GPU).

You can confirm your GPU support via:

```powershell
nvidia-smi
```

### 5. Clone YOLOv7
Clone the official YOLOv7 repository from GitHub:

```powershell
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

### 6. Install YOLOv7 Requirements
Inside the yolov7 folder, install dependencies:

```powershell
pip install -r requirements.txt
```
