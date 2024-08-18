import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')



cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
     
        results = model.track(frame, persist=True, classes=[0,1,2,3,7,13])

      
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()