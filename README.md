Custom Object Detection with YOLO

This project demonstrates custom object detection using YOLO models with a dataset annotated in Label Studio. The workflow covers dataset preparation, training, validation, testing, and deployment of a YOLO model.

---

Table of Contents

1. Project Overview
2. Folder Structure
3. Installation
4. Dataset Preparation
5. Training the Model
6. Testing & Inference
7. Downloading the Trained Model
8. Usage
9. Tips for Improving Model Performance

---

Project Overview
This project trains a YOLO object detection model on a custom dataset. Images were annotated using Label Studio, and the model detects objects of interest based on these annotations.

The training pipeline includes:

* Splitting the dataset into train (80%) and validation (20%) sets
* Configuring YOLO training parameters
* Training the model and monitoring performance (mAP, precision, recall)
* Testing and visualizing detection results
* Exporting and using the trained model for inference

---

Folder Structure
The dataset follows the structure required by Ultralytics YOLO:

custom\_data/
├── images/
├── labels/
├── classes.txt      # list of object classes

data/
├── train/
│   ├── images/
│   └── labels/
├── validation/
│   ├── images/
│   └── labels/

* train/images – Images used for training
* validation/images – Images used for validation
* labels – YOLO-format annotation files
* classes.txt – List of object class names

---

Installation
Install required dependencies:

pip install ultralytics pyyaml

---

Dataset Preparation

1. Unzip images into custom\_data:

unzip -q data.zip -d custom\_data

2. Split dataset into train and validation:

python train\_val\_split.py --datapath="custom\_data" --train\_pct=0.8

3. Create data.yaml configuration file:

from create\_data\_yaml import create\_data\_yaml

create\_data\_yaml("custom\_data/classes.txt", "data.yaml")

This file specifies:

* Dataset paths
* Number of classes (nc)
* Class names

---

Training the Model
Configure training parameters:

* Model: yolo11s.pt (YOLO11 small model)
* Epochs: 60 (adjust depending on dataset size)
* Image size: 640x640

Run training:

yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640

Training results are saved in:
runs/detect/train/

---

Testing & Inference
After training, test the model on validation images:

yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True

Preview results:

import glob
from IPython.display import Image, display

for image\_path in glob.glob('runs/detect/predict/\*.jpg')\[:10]:
display(Image(filename=image\_path, height=400))

---

Downloading the Trained Model
Package the model and training results:

mkdir my\_model
cp runs/detect/train/weights/best.pt my\_model/my\_model.pt
cp -r runs/detect/train my\_model
cd my\_model
zip my\_model.zip my\_model.pt
zip -r my\_model.zip train

---

Usage
Run detection on a live video stream or camera:

python yolo\_detect.py --model my\_model/my\_model.pt --source usb0 --resolution 1280x720

---

Tips for Improving Model Performance

1. Verify dataset annotations for errors
2. Increase the number of training epochs
3. Use a larger model (yolo11l.pt or yolo11xl.pt)
4. Add more annotated images
5. Experiment with image resolution and data augmentation
