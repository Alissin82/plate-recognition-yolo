# How To Download

1. Open [Google Colab](https://colab.research.google.com/)

2. Mount Google Drive(Authorize Access):

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install and download dataset via roboflow 

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="d6hBMBypniRzqdulp4dp")
project = rf.workspace("object-recognition-yolo").project("anpr_ir-rsiqu")
version = project.version(4)
dataset = version.download("yolov7")
```

4. Go to Google Drive → Locate the `ANPR_ir-4` folder → Download it.

5. Place following folders in ./dataset

* test
* train
* valid