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
import easyocr
import platform
import argparse

device = select_device("cpu")
half = device.type != 'cpu'
image_size=640
trace = True

model = attempt_load('./runs/train/exp/weights/best.pt', map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(image_size, s=stride)

savepath = r"D:\plate-recognition-yolo-master\plate-recognition-yolo-2\output"
os.makedirs(savepath, exist_ok=True)

# OCR
reader = easyocr.Reader(['fa'])

if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

def detect_plate(source_image):
    # Padded resize
    img_size = 640
    stride = 32
    img = letterbox(source_image, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        # Inference
        pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    plate_detections = []
    det_confidences = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()

            # Return results
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())

    return plate_detections, det_confidences


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0):
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def crop(image, coord):
    cropped_image = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    return cropped_image


def ocr_plate(plate_region):
    # Image pre-processing for more accurate OCR
    cv.imwrite(os.path.join(savepath, "plate_img.png"), plate_region)
    rescaled = cv.resize(plate_region, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
    grayscale = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    # OCR the preprocessed image
    grayscale_blur = cv.medianBlur(grayscale, 1)
    ret, thresh1 = cv.threshold(grayscale_blur, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(os.path.join(savepath, "grayscale_blur.png"), grayscale_blur)
    plate_text_easyocr = reader.readtext(grayscale_blur)
    if plate_text_easyocr:
        (bbox, text_easyocr, ocr_confidence) = plate_text_easyocr[0]
        print("plate_text Easyocr ", text_easyocr)
    else:
        text_easyocr = "_"
        ocr_confidence = 0
    # if ocr_confidence == 'nan':

    return text_easyocr, ocr_confidence


def get_plates_from_image(input):
    if input is None:
        return None
    plate_detections, det_confidences = detect_plate(input)
    plate_texts = []
    ocr_confidences = []
    detected_image = deepcopy(input)
    for coords in plate_detections:
        plate_region = crop(input, coords)
        plate_text, ocr_confidence = ocr_plate(plate_region)
        plate_texts.append(plate_text)
        ocr_confidences.append(ocr_confidence)
        detected_image = plot_one_box_PIL(coords, detected_image, label=plate_text, color=[0, 150, 255],
                                          line_thickness=2)
    return detected_image


def pascal_voc_to_coco(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    return [x1, y1, x2 - x1, y2 - y1]


def get_best_ocr(preds, rec_conf, ocr_res, track_id):
    for info in preds:
        # Check if it is current track id
        if info['track_id'] == track_id:
            # Check if the ocr confidenence is maximum or not
            if info['ocr_conf'] < rec_conf:
                info['ocr_conf'] = rec_conf
                info['ocr_txt'] = ocr_res
            else:
                rec_conf = info['ocr_conf']
                ocr_res = info['ocr_txt']
            break
    return preds, rec_conf, ocr_res


def get_plates_from_video(source):
    if source is None:
        return None

    video = cv.VideoCapture(source)

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    temp = f'{Path(source).stem}_temp{Path(source).suffix}'
    export = cv.VideoWriter(temp, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not export.isOpened():
        print("Failed to open video writer.")
        return None

    tracker = DeepSort(embedder_gpu=False)
    preds = []
    total_obj = 0

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            print("Video stream ended or frame not captured.")
            break

        bboxes, scores = detect_plate(frame)
        bboxes = list(map(pascal_voc_to_coco, bboxes))

        if len(bboxes) > 0:
            detections = [(bbox, score, 'number_plate') for bbox, score in zip(bboxes, scores)]
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = [max(0, int(pos)) for pos in track.to_tlbr()]
                plate_region = crop(frame, bbox)
                plate_text, ocr_confidence = ocr_plate(plate_region)
                output_frame = {'track_id': track.track_id, 'ocr_txt': plate_text, 'ocr_conf': ocr_confidence}

                if track.track_id not in [pred['track_id'] for pred in preds]:
                    total_obj += 1
                    preds.append(output_frame)
                else:
                    preds, ocr_confidence, plate_text = get_best_ocr(preds, ocr_confidence, plate_text, track.track_id)

                frame = plot_one_box_PIL(bbox, frame, label=f'{track.track_id}. {plate_text}',
                                         color=[255, 150, 0], line_thickness=3)

        export.write(frame)
        cv.imshow("frame", frame)
        if cv.waitKey(1) == 27:  # ESC to exit early
            break

    cv.destroyAllWindows()
    video.release()
    export.release()

    output = f'{Path(source).stem}_detected{Path(source).suffix}'
    null_output = "NUL" if platform.system() == "Windows" else "/dev/null"

    os.system(
        f'ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k '
        f'-pass 1 -c:a aac -f mp4 {null_output}'
    )
    os.system(
        f'ffmpeg -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k '
        f'-pass 2 -c:a aac -movflags faststart {output}'
    )

    if platform.system() == "Windows":
        os.system(f'del /f /q {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')
    else:
        os.system(f'rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')

    return output


def get_plates_from_webcam():
    # Create a VideoCapture object
    video = cv.VideoCapture(0)

    if not video.isOpened():
        print("Error: Could not open webcam.")
        return

    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    if fps == 0.0:
        fps = 30  # fallback to 30 FPS

    # Define the codec and create VideoWriter object.
    temp = f'cam_temp.mp4'
    export = cv.VideoWriter(temp, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Intializing tracker
    tracker = DeepSort(embedder_gpu=False)

    # Initializing some helper variables.
    preds = []
    total_obj = 0
    fr_count = 0
    while (True):
        ret, frame = video.read()
        if not ret or frame is None:
            print("Video stream ended or frame not captured.")
            break

        fr_count += 1

        if fr_count % 10 != 0:
            export.write(frame)
            continue

        # Run detection
        bboxes, scores = detect_plate(frame)
        bboxes = list(map(lambda bbox: pascal_voc_to_coco(bbox), bboxes))

        if len(bboxes) > 0:
            detections = [(bbox, score, 'number_plate') for bbox, score in zip(bboxes, scores)]
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = [int(position) for position in list(track.to_tlbr())]
                bbox = [max(0, b) for b in bbox]

                plate_region = crop(frame, bbox)
                plate_text, ocr_confidence = ocr_plate(plate_region)

                output_frame = {'track_id': track.track_id, 'ocr_txt': plate_text, 'ocr_conf': ocr_confidence}
                if track.track_id not in [pred['track_id'] for pred in preds]:
                    total_obj += 1
                    preds.append(output_frame)
                else:
                    preds, ocr_confidence, plate_text = get_best_ocr(preds, ocr_confidence, plate_text, track.track_id)

                frame = plot_one_box_PIL(bbox, frame, label=f'{str(track.track_id)}. {plate_text}',
                                         color=[255, 150, 0], line_thickness=3)

                cv.imshow("frame ", frame)
                if cv.waitKey(1) == 27:
                    break

        export.write(frame)

    cv.destroyAllWindows()
    video.release()
    export.release()

    # Compressing the output video for smaller size and web compatibility.
    output = f'cam_detected.mp4'
    # Select system-dependent null device
    null_output = "NUL" if platform.system() == "Windows" else "/dev/null"

    # Perform 2-pass ffmpeg compression
    os.system(
        f'ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k '
        f'-pass 1 -c:a aac -f mp4 {null_output}'
    )
    os.system(
        f'ffmpeg -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k '
        f'-pass 2 -c:a aac -movflags faststart {output}'
    )

    # Delete temporary files
    if platform.system() == "Windows":
        os.system(f'del /f /q {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')
    else:
        os.system(f'rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')

    return output


# ✅ Example usage:
# Detect from image: python ocr.py --image car.jpg
# Detect from video: python ocr.py --video dashcam.mp4
# Detect from webcam: python ocr.py --webcam


def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition CLI")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for live detection")

    args = parser.parse_args()

    if not (args.image or args.video or args.webcam):
        print("Please provide --image, --video, or --webcam as input.")
        return

    if args.image:
        plate_image = cv.imread(args.image)
        if plate_image is None:
            print(f"Error: Could not read image {args.image}")
            return
        detected = get_plates_from_image(plate_image)
        save_path = os.path.join(savepath, "detected_plate.png")
        cv.imwrite(save_path, detected)
        print(f"Saved detected image to {save_path}")
        cv.imshow("Detected Plate (Image)", detected)
        cv.waitKey(3000)  # Display for 3 seconds
        cv.destroyAllWindows()

    elif args.video:
        output_path = get_plates_from_video(args.video)
        print(f"Processed video saved to: {output_path}")

    elif args.webcam:
        output_path = get_plates_from_webcam()
        print(f"Webcam video processed and saved to: {output_path}")

if __name__ == "__main__":
    main()