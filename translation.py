from gtts import gTTS
from playsound import playsound
from translate import Translator
from ultralytics import YOLO
import cv2
import tempfile
import os


def speak(text, preferred_language_code):
    try:
     
        cleaned_text = text.replace("$", "").replace(",", "")

     
        translator = Translator(to_lang=preferred_language_code)
        translation = translator.translate(cleaned_text)

      
        tts = gTTS(text=translation, lang=preferred_language_code, slow=False)

   
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio.close()

            playsound(temp_audio.name)

        
        os.unlink(temp_audio.name)
    except Exception as e:
        print("Speech Error:", e)


model = YOLO('best.pt')


cap = cv2.VideoCapture(0)


preferred_language_code = input("Enter the language code of your preferred language (e.g., 'hi' for Hindi, 'kn' for Kannada): ")


def translate_text(text, preferred_language_code):
    try:
        
        cleaned_text = text.replace("$", "").replace(",", "")

        translator = Translator(to_lang=preferred_language_code)
        translation = translator.translate(cleaned_text)

        return translation
    except Exception as e:
        print("Translation Error:", e)
        return text  


while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)


    class_labels = [model.names[int(box.cls)] for box in results[0].boxes]

   
    text = ", ".join(class_labels)

   

   
    translated_text = translate_text(text, preferred_language_code)

    speak(translated_text, preferred_language_code)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()