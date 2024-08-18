import cv2
from inference_sdk import InferenceHTTPClient


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK")


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()

    result = CLIENT.infer(frame, model_id="tech-tribe/1")


    if 'predictions' in result:
        object_names = [prediction['class'] for prediction in result['predictions']]
        print( object_names)


    for bounding_box in result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()