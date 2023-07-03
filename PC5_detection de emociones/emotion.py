import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp
import cvzone


def emotions_label(num):
    emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    for num_label in emotions:
        if num_label == num:
            return emotions[num_label]
        else:
            return 'No se detecto'

def genders_label(num):
    genders = {0:'woman', 1:'man'}
    for num_label in genders:
        if num_label == num:
            return genders[num_label]
        else:
            return 'Alien'

#tf 2.9.1
#keras 2.6.0

face = mp.solutions.face_detection
Face = face.FaceDetection()
mpDwaw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Videos/vd06.mp4')

modelEmotion = load_model("emotion.hdf5", compile=False)
modelGender = load_model("gender.hdf5",compile=False)


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
            #mpDwaw.draw_detection(img, detection)
            bbox = detection.location_data.relative_bounding_box
            x,y,w,h = int(bbox.xmin*wO),int(bbox.ymin*hO),int(bbox.width*wO),int(bbox.height*hO)
            imgFace = imgGray[y:y + h, x:x + w]
            imgFace = cv2.resize(imgFace, (48, 48))
            face = np.asarray(imgFace, dtype=np.float32).reshape(1, 48, 48, 1)
            face = (face / 255.0) - 0.5
            imgFace1 = cv2.resize(imgFace, (64, 64))
            face1 = np.asarray(imgFace1, dtype=np.float32).reshape(1, 64, 64, 1)
            face1 = (face1 / 255.0) - 0.5

            prediction = modelEmotion.predict(face)
            probability = np.max(prediction)*100
            probability = round(probability,2)
            print(probability)
            num_emotion = np.argmax(prediction)

            prediction = modelGender.predict(face1)
            num_gender = np.argmax(prediction)

            emotion_label = emotions_label(num_emotion)
            gender_label = genders_label(num_gender)

            if gender_label != 'None':
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 0, 0), 3)
                cvzone.putTextRect(imgOrignal, f'genero: {gender_label}', (x, y - 15),2, 2,colorR= (0,0,1))

            if emotion_label != 'None':
                cvzone.putTextRect(imgOrignal, f'emocion: {emotion_label}',(x,y-50),2,2, colorR = (0,0,255))
                cvzone.putTextRect(imgOrignal, f'probabilidad: {probability}%', (x, y - 75), 2, 2, colorR=(0,0,255))







    cv2.imshow("Result", imgOrignal)
    cv2.waitKey(15)