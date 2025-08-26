# Custom Object Detection with YOLO

This repository contains a complete workflow for training a **custom YOLO object detection model** on a dataset annotated using Label Studio. The project covers data preparation, model training, evaluation, and deployment.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Training the Model](#training-the-model)
6. [Testing & Inference](#testing--inference)
7. [Downloading the Trained Model](#downloading-the-trained-model)
8. [Usage](#usage)
9. [Tips for Improving Performance](#tips-for-improving-performance)

---

## Project Overview

This project demonstrates how to train a YOLO object detection model on a **custom dataset**. The dataset is annotated with Label Studio, split into training and validation sets, and used to train a YOLO11 model. The trained model can detect objects of interest in images or live video streams.

---

## Folder Structure

```
custom_data/
├── images/
├── labels/
├── classes.txt      # List of object classes

data/
├── train/
│   ├── images/
│   └── labels/
├── validation/
│   ├── images/
│   └── labels/
```

- `train/images` – Images for training
- `validation/images` – Images for validation
- `labels` – YOLO-format annotations
- `classes.txt` – Names of classes

---

## Installation

Install the required Python packages:

```bash
pip install ultralytics pyyaml
```

---

## Dataset Preparation

1. Unzip images into `custom_data`:

```bash
unzip -q data.zip -d custom_data
```

2. Split dataset into training and validation sets:

```bash
python train_val_split.py --datapath="custom_data" --train_pct=0.8
```

3. Generate the `data.yaml` configuration file:

```python
from create_data_yaml import create_data_yaml
create_data_yaml('custom_data/classes.txt', 'data.yaml')
```

---

## Training the Model

**Parameters:**
- Model: `yolo11s.pt` (small YOLO11 model)
- Epochs: 60 (adjust based on dataset size)
- Image size: 640x640

**Training command:**

```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

Trained weights and results are saved in `runs/detect/train/`.

---

## Testing & Inference

Run the trained model on validation images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True
```

Preview results in Python:

```python
import glob
from IPython.display import Image, display

for image_path in glob.glob('runs/detect/predict/*.jpg')[:10]:
    display(Image(filename=image_path, height=400))
```

---

## Downloading the Trained Model

Package your model and training results:

```bash
mkdir my_model
cp runs/detect/train/weights/best.pt my_model/my_model.pt
cp -r runs/detect/train my_model
cd my_model
zip my_model.zip my_model.pt
zip -r my_model.zip train
```

---

## Usage

Run detection on a live camera or video stream:

```bash
python my_model/yolo_detect.py --model my_model/my_model.pt --source usb0 --resolution 1280x720
```

---

## Tips for Improving Performance
1. Check dataset for annotation errors
2. Increase the number of epochs
3. Use a larger model size (`yolo11l.pt` or `yolo11xl.pt`)
4. Add more training images
5. Experiment with image resolution and augmentation

