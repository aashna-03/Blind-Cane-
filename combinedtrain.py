import cv2
from ultralytics import YOLO

model1 = YOLO('best_pothole.pt')
model2 = YOLO('best_manhole.pt')


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        
        results1 = model1.track(frame, persist=True)
        results2 = model2.track(frame, persist=True)

       
        merged_results = results1 + results2

       
        annotated_frame = merged_results[0].plot() if merged_results else frame
        cv2.imshow("YOLOv8n and Best1 Tracking", annotated_frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()