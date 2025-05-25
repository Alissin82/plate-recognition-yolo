
ANPR_ir - v4 2025-05-25 12:01am
==============================

This dataset was exported via roboflow.com on May 24, 2025 at 8:32 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 4217 images.
Number-plate are annotated in YOLO v7 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random Gaussian blur of between 0 and 1 pixels
* Salt and pepper noise was applied to 0.98 percent of pixels


