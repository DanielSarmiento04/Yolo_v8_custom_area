from ultralytics import YOLO
import cv2
import torch
import supervision as sv

cap = cv2.VideoCapture(0)

# select a device to run the model on for performance acceleration
device = torch.device("mps")

# Load the YOLOv8 model
model = YOLO(
    model="yolov8n.pt", 
)

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)



for result in model.track(source=0 , show=False, save=False, stream=True, verbose=False, device=device):

    frame = result.orig_img
    
    # Skip frames without detections
    if result.boxes.id is None:
        cv2.imshow("YOLOv8 Inference", frame)
        continue

    # Convert the YOLOv8 results to Supervision Detections
    detections = sv.Detections.from_yolov8(result)

    
    # assign tracker id to each detection
    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
    """
    Detections(xyxy=array([[     262.85,      28.521,      1242.8,      715.25]], dtype=float32), mask=None, class_id=array([0]), confidence=array([    0.83332], dtype=float32), tracker_id=None)
    """
        
    # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

    labels = [
        f"{tracker_id} {model.names[class_id]} {confidence:.2f}"
        for xyxy, mask, confidence, class_id, tracker_id
        in detections
    ]
    # print(detections)

    # Annotate the frame with the detections
    frame = box_annotator.annotate(frame, detections, labels=labels)

    cv2.imshow("YOLOv8 Inference", frame)



cap.release()
cv2.destroyAllWindows()

