import cv2
import pygame
import time
from inference_sdk import InferenceHTTPClient

pygame.init()


zebracrossing_detection_audio_files = {
    "english": "zebracrossingdetected.mp3",
    "hindi": "samnezebracrossinghai.mp3",
    "kannada": "mundezebracrossingede.mp3"
}

pothole_detection_audio_files = {
    "english": "potholedetected.mp3",
    "hindi": "samnegaddahai.mp3",
    "kannada": "mundegundiede.mp3"
}

zebracrossing_detection_language_to_audio_path = {
    "english": zebracrossing_detection_audio_files["english"],
    "hindi": zebracrossing_detection_audio_files["hindi"],
    "kannada": zebracrossing_detection_audio_files["kannada"]
}


pothole_detection_language_to_audio_path = {
    "english": pothole_detection_audio_files["english"],
    "hindi": pothole_detection_audio_files["hindi"],
    "kannada": pothole_detection_audio_files["kannada"]
}


def play_notification_sound(language, detection_type):
    if detection_type == "zebracrossing":
        audio_files = zebracrossing_detection_audio_files
        language_to_audio_path = zebracrossing_detection_language_to_audio_path
    elif detection_type == "pothole":
        audio_files = pothole_detection_audio_files
        language_to_audio_path = pothole_detection_language_to_audio_path
    else:
        print("Invalid detection type")
        return

    if language in language_to_audio_path:
        audio_file_path = language_to_audio_path[language]
        notification_sound = pygame.mixer.Sound(audio_file_path)
        notification_sound.play()
    else:
        print("Language not supported")

def calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, object_height):
 
    distance_from_camera = (scale_distance * scale_height_pixels) / object_height
  
    updated_distance = distance_from_camera + scale_distance
    return updated_distance


ZEBRACROSSING_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK"
)

POTHOLE_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK"
)

cap = cv2.VideoCapture(0)  


selected_language = input("Select your preferred language (e.g., 'english', 'hindi', 'kannada'): ").lower()


scale_pixels = 40
scale_height_pixels = 480
scale_height = 15 
scale_distance = 2 

while True:
  
    ret, frame = cap.read()
   
    zebracrossing_result = ZEBRACROSSING_CLIENT.infer(frame, model_id="zebra-crossing-qh5uu/1")


    pothole_result = POTHOLE_CLIENT.infer(frame, model_id="tech-tribe/1")

    zebracrossing_object_names = [prediction['class'] for prediction in zebracrossing_result['predictions']]

  
    pothole_object_names = [prediction['class'] for prediction in pothole_result['predictions']]

    if '0' in zebracrossing_object_names:

        play_notification_sound(selected_language, "zebracrossing")


    if 'pothole' in pothole_object_names:
       
        play_notification_sound(selected_language, "pothole")

    for bounding_box in zebracrossing_result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

  
        distance_from_camera = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, bounding_box['height'])
        cv2.putText(frame, f"Updated Distance: {distance_from_camera:.2f} cm", (box[0][0], box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    for bounding_box in pothole_result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)  

      
        distance_from_camera = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, bounding_box['height'])
        cv2.putText(frame, f"Updated Distance: {distance_from_camera:.2f} cm", (box[0][0], box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



    cv2.imshow('frame', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(2)

cap.release()
cv2.destroyAllWindows()

pygame.quit()
