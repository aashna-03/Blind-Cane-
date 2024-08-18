import cv2
import pygame
import time  
from inference_sdk import InferenceHTTPClient

pygame.init()

audio_files = {
    "english": "cardetected.mp3",
    "hindi": "saamnegaadihai.mp3",
    "kannada":"mundegaadiede.mp3"
   
}


language_to_audio_path = {
    "english": audio_files["english"],
    "hindi": audio_files["hindi"],
    "kannada":audio_files["kannada"],


}


def play_notification_sound(language):
    if language in language_to_audio_path:
        audio_file_path = language_to_audio_path[language]
        notification_sound = pygame.mixer.Sound(audio_file_path)
        notification_sound.play()
    else:
        print("Language not supported")


def calculate_object_height(bbox):
    _, y_min, _, y_max = bbox
    object_height_pixels = abs(y_max - y_min)
    return object_height_pixels 

def calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, scale_height):
 
    scale_factor = scale_height / scale_height_pixels

    height_of_scale = scale_pixels * scale_factor
   
    distance_from_camera = (scale_distance * scale_height_pixels) / scale_height

    updated_distance = distance_from_camera + scale_distance
    return updated_distance  



CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="PIT9ordmgBwkNFwOiLft")


cap = cv2.VideoCapture(0)  


selected_language = input("Select your preferred language (e.g., 'english', 'hindi','kannada'): ").lower()


scale_pixels = 40
scale_height_pixels = 480
scale_height = 15  # in cm
scale_distance = 2 # in cm

while True:
  
    ret, frame = cap.read()

    result = CLIENT.infer(frame, model_id="car_detection-3zagb/1")

    if 'predictions' in result:
        object_names = [prediction['class'] for prediction in result['predictions']]
      
   
    if '1' in object_names:
      
        play_notification_sound(selected_language)


    for bounding_box in result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

          
        object_height_pixels = calculate_object_height((x1, y1, x2, y2))
        updated_distance = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, object_height_pixels)


      
        cv2.putText(frame, f"Updated Distance: {updated_distance:.2f} cm", (box[0][0], box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



  
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


time.sleep(7)

pygame.quit()