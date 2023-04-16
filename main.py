from ultralytics import YOLO
import cv2

import supervision as sv

cap = cv2.VideoCapture(0)

# Load the YOLOv8 model
model = YOLO(
    model="yolov8n.pt", 
)

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)



for result in model.track(source=0 , show=False, save=False, stream=True, verbose=False):

    frame = result.orig_img
    
    # Convert the YOLOv8 results to Supervision Detections
    detections = sv.Detections.from_yolov8(result)
    
    """
    Detections(xyxy=array([[     262.85,      28.521,      1242.8,      715.25]], dtype=float32), mask=None, class_id=array([0]), confidence=array([    0.83332], dtype=float32), tracker_id=None)
    """
        
    # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

    labels = [
        f"{model.names[class_id]} {confidence:.2f}"
        for xyxy, mask, confidence, class_id, tracker_id
        in detections
    ]
    # print(detections)

    # Annotate the frame with the detections
    frame = box_annotator.annotate(frame, detections, labels=labels)

    cv2.imshow("YOLOv8 Inference", frame)


# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if not success:
#         continue

#     # Run YOLOv8 inference on the frame
#     results = model.track(frame, verbose=False, stream=True)
    
#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()
    
#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Inference", annotated_frame)


#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
cap.release()
cv2.destroyAllWindows()

