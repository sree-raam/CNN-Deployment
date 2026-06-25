# CNN-Deployment

## Real-Time Plant Classification Using ResNet50 and Webcam Deployment

This repository contains a real-time image classification system developed during IEEE AI training. The project uses a fine-tuned ResNet50 convolutional neural network to classify plant images through a live webcam feed.

The model performs three-class classification for:

* Banana
* Curry
* Papaya

The system captures frames from a camera, preprocesses each image, performs CNN inference using PyTorch, and displays the predicted class label with real-time FPS on the video output.

---

## Key Features

* Real-time webcam-based image classification
* ResNet50 transfer learning model
* Three-class plant classification
* PyTorch inference pipeline
* OpenCV live video display
* FPS monitoring
* GPU support using CUDA

---

## Main Files

```text
CNN-Deployment/

├── CNN_Webcam_Template.py
├── file.py
├── myCNN.pth
└── README.md
```

---

## Model Architecture

The project uses ResNet50 as the base CNN model.

The final fully connected layer is modified for three output classes:

```python
model.fc = nn.Linear(num_ftrs, 3)
```

---

## Installation

Install the required dependencies:

```bash
pip install torch torchvision opencv-python pillow
```

---

## How to Run

Run the webcam classification script:

```bash
python CNN_Webcam_Template.py
```

Press `q` to exit the video window.

---

## Output

The system displays a live camera window showing:

* Predicted plant class
* Real-time FPS
* Webcam feed with classification overlay

---

## Notes

Make sure the trained model file is available in the repository folder and the model path in the code matches the filename.

Example:

```python
model_load_path = "myCNN.pth"
```

If using CPU instead of GPU, update the device configuration accordingly.

---

## Author

**Sree Raam**

---

## Purpose

This repository was created as part of IEEE AI training to demonstrate CNN model deployment, transfer learning, and real-time computer vision inference using PyTorch and OpenCV.
