# YOLOv8 Tracker

🚀 This is a Python code that uses the YOLOv8 model to detect objects in real-time on a video source, assigning a tracking ID to each detected object.

## Installation

This code requires the following libraries:

- ultralytics
- opencv-python
- torch

They can be installed using pip:


Copy code
```
    pip install -r requirements.txt
```

## Usage

To use this code, follow these steps:

- Clone the repository and open a terminal in the project folder.

- Run the script: python main.py

## Notes

- The code uses the default device camera (0) as the video source. If you want to use another video file, you should change the source parameter in the line for result in model.track(source=0, ...).
- The YOLOv8 model is loaded in the line model = YOLO(model="yolov8n.pt"). If you want to use another model, you should download and specify the corresponding file.
- Object detection is performed in the line detections = sv.Detections.from_yolov8(result). You can adjust the detection filters by modifying the detections object.


> Enjoy real-time object tracking with YOLOv8! 👁️👀