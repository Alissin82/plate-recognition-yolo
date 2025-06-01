import os
from pathlib import Path
import torch
import cv2 as cv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box_PIL
from copy import deepcopy
import platform
import argparse
import subprocess
import torchvision.transforms as transforms
from torch import nn
from PIL import Image

# Device setup
device = select_device("cpu")
half = device.type != 'cpu'
image_size = 640
trace = True

# Model loading
model = attempt_load('./runs/train/exp/weights/best.pt', map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(image_size, s=stride)

if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

# Load custom Persian OCR model
ocr_model_path = Path("./dataset/ocr/model/persian_char_model.pth")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=28):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

o_model = SimpleCNN()
o_model.load_state_dict(torch.load(ocr_model_path, map_location=device))
o_model.eval()
o_model.to(device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Output path
savepath = Path("./output")
savepath.mkdir(parents=True, exist_ok=True)

def detect_plate(source_image):
    img_size = 640
    stride = 32
    img = letterbox(source_image, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=True)[0]

    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    plate_detections = []
    det_confidences = []

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())

    return plate_detections, det_confidences

def crop(image, coord):
    return image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]

def ocr_plate(plate_region):
    h, w, _ = plate_region.shape
    char_width = w // 8  # فرض: 8 کاراکتر
    chars = []
    for i in range(8):
        x_start = i * char_width
        x_end = (i + 1) * char_width
        char_img = plate_region[:, x_start:x_end]
        image_pil = Image.fromarray(cv.cvtColor(char_img, cv.COLOR_BGR2RGB))
        tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = o_model(tensor)
            pred = torch.argmax(output, dim=1)
            chars.append(str(pred.item()))  # یا نگاشت عدد به کاراکتر فارسی
    return ''.join(chars), 1.0

def get_plates_from_image(input_img, filename):
    if input_img is None:
        return None

    plate_detections, det_confidences = detect_plate(input_img)
    plate_texts = []
    ocr_confidences = []
    detected_image = deepcopy(input_img)

    for coords in plate_detections:
        plate_region = crop(input_img, coords)
        plate_text, ocr_confidence = ocr_plate(plate_region)
        plate_texts.append(plate_text)
        ocr_confidences.append(ocr_confidence)
        detected_image = plot_one_box_PIL(coords, detected_image, label=plate_text, color=[0, 150, 255], line_thickness=2)
        print(f"Detected plate text: {plate_text} with OCR confidence: {ocr_confidence}")

    output_filename = savepath / f"{Path(filename).stem}_detected.png"
    cv.imwrite(str(output_filename), detected_image)
    subprocess.run(f'explorer "{savepath}"', shell=True)
    print(f"Saved detected image to {output_filename}")

    return detected_image

def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition CLI")
    parser.add_argument("--image", help="Path to an image file")

    args = parser.parse_args()

    if not args.image:
        print("Please provide --image for input.")
        return

    plate_image = cv.imread(args.image)
    if plate_image is None:
        print(f"Error: Could not read image {args.image}")
        return

    get_plates_from_image(plate_image, args.image)

if __name__ == "__main__":
    main()