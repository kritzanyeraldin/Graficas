import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp
import cvzone

# referencia: https://github.com/oarriaga/face_classification/tree/master

def emotions_label(num):
    emotions = {0: ['angry', (255, 0, 0)], 1: ['disgust', (255, 64, 0)], 2: ['fear', (255, 165, 0)],
                3: ['happy', (255, 255, 0)], 4: ['sad', (25, 0, 196)], 5: ['surprise', (166, 32, 240)],
                6: ['neutral', (0, 0, 0)]}
    for num_label in emotions:
        if num_label == num:
            return emotions[num_label]



def genders_label(num):
    genders = {0: ['woman', (227, 49, 113)], 1: ['man', (149, 179, 255)]}
    for num_label in genders:
        if num_label == num:
            return genders[num_label]
        else:
            return ['Alien', (10, 124, 13)]


# tf 2.9.1
# keras 2.6.0

face = mp.solutions.face_detection
Face = face.FaceDetection()
mpDwaw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Videos/vd06.mp4')

modelEmotion = load_model("emotion.hdf5", compile=False)
modelGender = load_model("gender.hdf5", compile=False)

while True:
    success, imgOrignal = cap.read()
    # imgOrignal = cv2.resize(imgOrignal,(1200,720))
    imgRGB = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
    results = Face.process(imgRGB)
    facesPoints = results.detections
    hO, wO, _ = imgRGB.shape
    if facesPoints:
        for id, detection in enumerate(facesPoints):
            # mpDwaw.draw_detection(img, detection)
            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * wO), int(bbox.ymin * hO), int(bbox.width * wO), int(bbox.height * hO)
            imgFace = imgGray[y:y + h, x:x + w]
            imgFace1 = imgGray[y:y + h, x:x + w]
            imgFace = cv2.resize(imgFace, (48, 48))
            face = np.asarray(imgFace, dtype=np.float32).reshape(1, 48, 48, 1)
            face = (face / 255.0) - 0.5
            imgFace1 = cv2.resize(imgFace1, (64, 64))
            face1 = np.asarray(imgFace1, dtype=np.float32).reshape(1, 64, 64, 1)
            face1 = (face1 / 255.0) - 0.5

            prediction = modelEmotion.predict(face)
            probability = np.max(prediction) * 100
            probability = round(probability, 2)
            num_emotion = np.argmax(prediction)
            print(num_emotion)

            prediction = modelGender.predict(face1)
            num_gender = np.argmax(prediction)

            emotion_label = emotions_label(num_emotion)
            color_emotion = emotion_label[1]
            emotion_label = emotion_label[0]

            gender_label = genders_label(num_gender)
            color_gender = gender_label[1]
            gender_label = gender_label[0]

            if gender_label != 'None':
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 0, 0), 3)
                cvzone.putTextRect(imgOrignal, f'genero: {gender_label}', (x, y - 15), 2, 2, colorR=color_gender)

            if emotion_label != 'None':
                cvzone.putTextRect(imgOrignal, f'emocion: {emotion_label}', (x, y - 50), 2, 2, colorR=color_emotion)
                cvzone.putTextRect(imgOrignal, f'probabilidad: {probability}%', (x, y - 75), 2, 2, colorR=color_emotion)

    cv2.imshow("Result", imgOrignal)
    cv2.waitKey(15)
