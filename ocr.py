import argparse
import cv2 as cv
from pathlib import Path
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import TracedModel, select_device

# Create argument parser to handle command-line inputs
parser = argparse.ArgumentParser(description="License Plate Recognition CLI")
parser.add_argument("--image", help="Path to an image file")  # Argument for input image path

args = parser.parse_args()  # Parse the arguments

# Define the output directory path and create it if it doesn't exist
output_path = Path("./output")
output_path.mkdir(parents=True, exist_ok=True)

# Parameters (extracted from Parameters class)
imgsz = 640  # Input image size for YOLO models
conf_thres = 0.25  # Confidence threshold for detection
cpu_or_cuda = "cpu"  # Device to use (can be "cuda" if GPU is available)

# Paths to the pre-trained YOLO models
modelPlate_path = "./runs/train/exp/weights/best.pt"  # Path to the plate detection model
modelCharX_path = "./runs/characters.pt"  # Path to the character detection model (not used yet)
# Dictionary to map class IDs to characters (based on your previous config)
char_id_dict = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    '10': 'الف', '11': 'ب', '12': 'پ', '13': 'تاکسی', '14': 'ث', '15': 'ج', '16': 'چ', '17': 'ح', '18': 'خ',
    '19': 'د', '20': 'ذ', '21': 'ر', '22': 'ز', '23': 'ژ', '24': 'سین', '25': 'ش', '26': 'ص', '27': 'ض',
    '28': 'ط', '29': 'ظ', '30': 'ع', '31': 'غ', '32': 'ف', '33': 'ق', '34': 'ک', '35': 'گ', '36': 'ل',
    '37': 'م', '38': 'ن', '39': 'ه', '40': 'و', '41': 'ی', '42': 'معلول'
}  # Adjusted based on your config's char_dict

def load_yolo(weights_path: str, device, img_size=imgsz):
    """Load a YOLO model with TracedModel and set it to evaluation mode.

    Args:
        weights_path (str): Path to the model weights file (.pt)
        device: The device (CPU or GPU) to load the model on
        img_size (int): Input image size for the model (default: 640)

    Returns:
        TracedModel: The loaded and traced YOLO model
    """
    model = attempt_load(weights_path, map_location=device)  # Load the model weights
    model = TracedModel(model, device, img_size)  # Trace the model for optimization
    model.to(device).eval()  # Move to the specified device and set to evaluation mode
    return model


def load_image(image_path: str):
    """Load the input image from the given path.

    Args:
        image_path (str): Path to the image file

    Returns:
        numpy.ndarray: The loaded image, or None if failed
    """
    plate_image = cv.imread(image_path)
    if plate_image is None:
        print(f"Error: Could not read image {image_path}")
    return plate_image


def preprocess_image(image: np.ndarray, device):
    """Preprocess the image for YOLO detection.

    Args:
        image (numpy.ndarray): The input image
        device: The device to move the tensor to

    Returns:
        torch.Tensor: Preprocessed image tensor, and original image
    """
    img0 = image.copy()  # Keep original image for drawing
    img = letterbox(img0, new_shape=imgsz, stride=32)[0]  # Resize image to imgsz with padding
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and change dimensions
    img = np.ascontiguousarray(img)  # Ensure contiguous array for torch
    img = torch.from_numpy(img).to(device).float() / 255.0  # Convert to tensor and normalize
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension if needed
    return img, img0


def detect_plates(plate_model, img_tensor, original_shape):
    """Detect plates in the preprocessed image.

    Args:
        plate_model: The loaded YOLO model
        img_tensor (torch.Tensor): Preprocessed image tensor
        original_shape: Shape of the original image for scaling coordinates

    Returns:
        list: List of tuples containing (coordinates, confidence) for detected plates
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        pred = plate_model(img_tensor)[0]  # Get predictions
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, classes=0)  # Filter predictions

    plates = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], original_shape).round()  # Scale coordinates back
            for *xyxy, conf, cls in det:
                plates.append((xyxy, conf.item()))  # Store coordinates and confidence
    return plates


def preprocess_cropped_plate(cropped_plate: np.ndarray):
    """Perform basic preprocessing on the cropped plate image.

    Args:
        cropped_plate (numpy.ndarray): The cropped plate image

    Returns:
        numpy.ndarray: Preprocessed plate image
    """
    # Convert to grayscale for better processing
    gray_plate = cv.cvtColor(cropped_plate, cv.COLOR_BGR2GRAY)

    # Optional: Apply simple contrast adjustment (e.g., CLAHE for better visibility)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_plate = clahe.apply(gray_plate)

    # Convert back to BGR for consistency with original image
    preprocessed_plate = cv.cvtColor(preprocessed_plate, cv.COLOR_GRAY2BGR)
    return preprocessed_plate

def crop_and_save_plate(original_image, plates, save_path=output_path):
    """Crop the first detected plate, preprocess it, detect characters, and save results.

    Args:
        original_image (numpy.ndarray): The original image
        plates (list): List of detected plates with coordinates and confidence
        save_path (Path): Directory to save output files
    """
    if not plates:
        print("No detected plates")
        return

    # Crop the first detected plate
    xyxy, conf = plates[0]
    x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
    cropped_plate = original_image[y1:y2, x1:x2]  # Crop the plate region

    # Preprocess the cropped plate
    preprocessed_plate = preprocess_cropped_plate(cropped_plate)

    # Save the preprocessed cropped plate for debugging
    cropped_path = save_path / f"{Path(args.image).stem}_cropped.jpg"
    cv.imwrite(str(cropped_path), preprocessed_plate)
    print(f"Preprocessed cropped plate saved to {cropped_path} with confidence {conf:.2f}")

    # Draw bounding box and text on original image
    img_with_box = original_image.copy()
    cv.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
    cv.putText(img_with_box, f"Conf: {conf:.2f}%", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Add confidence text
    out_path = save_path / f"{Path(args.image).stem}_detected.png"
    cv.imwrite(str(out_path), img_with_box)
    print(f"Detected plate saved to {out_path}")


def main():

    # Check if an image path is provided
    if not args.image:
        print("Please provide --image for input.")
        return

    # Load and process the image
    image = load_image(args.image)
    if image is None:
        return

    device = select_device(cpu_or_cuda)
    plate_model = load_yolo(modelPlate_path, device)
    img_tensor, img0 = preprocess_image(image, device)
    plates = detect_plates(plate_model, img_tensor, image.shape)
    crop_and_save_plate(img0, plates, output_path)

if __name__ == "__main__":
    main()