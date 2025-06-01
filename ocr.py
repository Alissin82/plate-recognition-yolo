import argparse
import cv2 as cv
from pathlib import Path
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import TracedModel, select_device

# Define the output directory path and create it if it doesn't exist
output_path = Path("./output")
output_path.mkdir(parents=True, exist_ok=True)

# Paths to the pre-trained YOLO models
plate_model_path = "./runs/train/exp/weights/best.pt"  # Path to the plate detection model
char_model_path = "./runs/characters.pt"  # Path to the character detection model (not used yet)


def load_yolo(weights_path: str, device, img_size=640):
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


def main():
    # Create argument parser to handle command-line inputs
    parser = argparse.ArgumentParser(description="License Plate Recognition CLI")
    parser.add_argument("--image", help="Path to an image file")  # Argument for input image path

    args = parser.parse_args()  # Parse the arguments

    # Check if an image path is provided
    if not args.image:
        print("Please provide --image for input.")
        return

    # Read the input image
    plate_image = cv.imread(args.image)
    if plate_image is None:
        print(f"Error: Could not read image {args.image}")  # Error if image fails to load
        return

    # Set up the device (GPU if available, otherwise CPU)
    device = select_device("0" if torch.cuda.is_available() else "cpu")

    # Load the YOLOv7 model for plate detection
    plate_model = load_yolo(plate_model_path, device)

    # Create a copy of the original image and preprocess it for YOLO detection
    img0 = plate_image.copy()  # Keep original image for drawing
    img = letterbox(img0, new_shape=640, stride=32)[0]  # Resize image to 640x640 with padding
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and change dimensions
    img = np.ascontiguousarray(img)  # Ensure contiguous array for torch
    img = torch.from_numpy(img).to(device).float() / 255.0  # Convert to tensor and normalize
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension if needed

    # Perform plate detection using the YOLO model
    with torch.no_grad():  # Disable gradient calculation for inference
        pred = plate_model(img)[0]  # Get predictions
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=0)  # Filter predictions

    # Extract detected plates with their confidence scores
    plates = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  # Scale coordinates back
            for *xyxy, conf, cls in det:
                plates.append((xyxy, conf.item()))  # Store coordinates and confidence

    # Check if any plates were detected
    if not plates:
        print("No detected plates")
        return

    # Crop the first detected plate (for now, only the first one is processed)
    xyxy, conf = plates[0]
    x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
    cropped_plate = img0[y1:y2, x1:x2]  # Crop the plate region from the original image

    # Save the cropped plate for debugging purposes
    cropped_path = output_path / f"{Path(args.image).stem}_cropped.jpg"
    cv.imwrite(str(cropped_path), cropped_plate)
    print(f"Cropped plate saved to {cropped_path} with confidence {conf:.2f}")

    # Optional: Draw bounding box on the original image for verification
    cv.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
    cv.putText(img0, f"Conf: {conf:.2f}%", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
               2)  # Add confidence text
    out_path = output_path / f"{Path(args.image).stem}_detected.png"
    cv.imwrite(str(out_path), img0)
    print(f"Detected plate saved to {out_path}")


if __name__ == "__main__":
    main()