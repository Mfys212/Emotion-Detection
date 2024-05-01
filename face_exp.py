import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ["Marah", "Jijik", "Takut", "Senang", "Netral", "Sedih", "Terkejut"] 


def detect_and_predict_expression(image, model):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_roi = gray_image[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
        predictions = model.predict(reshaped_face)
        max_index = np.argmax(predictions[0])
        emotion_prediction = emotion_labels[max_index]
        confidence_score = predictions[0][max_index]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f"{emotion_prediction}: {confidence_score:.2f}", (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
    return image

video_path = 0
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if video_path == 0:
        frame = cv2.flip(frame, 1)

    frame = detect_and_predict_expression(frame, model)

    cv2.imshow('Expression Detection', cv2.resize(frame, (1100,650),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
