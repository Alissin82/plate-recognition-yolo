from roboflow import Roboflow

rf = Roboflow(api_key="d6hBMBypniRzqdulp4dp")
project = rf.workspace("object-recognition-yolo").project("anpr_ir-rsiqu")
version = project.version(1)
dataset = version.download("yolov8", location="C:/Users/Alissin/Desktop/UNI/plate-recognition-yolo/yolov7/roboflow_dataset"
)
