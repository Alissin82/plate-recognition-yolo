# How To Download

```python
pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="d6hBMBypniRzqdulp4dp")
project = rf.workspace("object-recognition-yolo").project("anpr_ir-rsiqu")
version = project.version(4)
dataset = version.download("yolov7")
```                