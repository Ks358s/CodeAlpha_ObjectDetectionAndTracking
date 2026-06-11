import cv2
from ultralytics import YOLO

def main():
    # 1. Use a pre-trained model
    # YOLOv8 Nano ('yolov8n.pt') is lightweight and ideal for real-time webcam processing.
    # It will automatically download the pre-trained weights on the first run.
    model = YOLO('yolov8n.pt')

    # 2. Set up real-time video input
    # Use 0 for the default webcam, or replace with a video file path (e.g., 'cars.mp4')
    video_source = 0 
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open the video source.")
        return

    print("Starting video stream. Press 'q' to quit.")

    # 3. Process each video frame
    while True:
        success, frame = cap.read()
        if not success:
            print("Video stream ended or cannot be read.")
            break

        # 4. Apply object detection and tracking
        # The .track() method natively handles detection and applies a tracking algorithm.
        # By default, it uses BoT-SORT (an evolution of SORT/Deep SORT).
        # persist=True ensures tracking IDs are maintained across consecutive frames.
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        # 5. Display the output with labels and tracking IDs
        # The .plot() function automatically draws the bounding boxes, class labels, 
        # and unique tracking IDs onto the frame.
        annotated_frame = results[0].plot()

        # Show the annotated frame in a window
        cv2.imshow("YOLOv8 Object Detection and Tracking", annotated_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
