# CodeAlpha_ObjectDetectionAndTracking
# YOLOv8 Webcam Object Detection and Tracking

This project runs real-time object detection and tracking from your webcam using Ultralytics YOLOv8 and OpenCV.

## What It Does

- Opens the default webcam
- Runs YOLOv8 object detection on each frame
- Tracks detected objects across frames with BoT-SORT
- Displays the annotated video feed in a window

## Requirements

- Python 3.9+
- A working webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python det.py
```

On first run, Ultralytics will automatically download the `yolov8n.pt` model weights.

Press `q` to close the video window.

## File Overview

- `det.py`: Main script for webcam capture, detection, tracking, and display

## Notes

- The script uses `video_source = 0`, which targets the default webcam.
- To use a video file instead, replace `0` in `det.py` with a file path such as `cars.mp4`.
- If the webcam does not open, make sure it is connected and not being used by another application.
