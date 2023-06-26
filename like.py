import cv2
import mediapipe as mp
import math

video = cv2.VideoCapture('pulgar2.mp4')
pose = mp.solutions.hands
Pose = pose.Hands(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
contador = 0
check = True

while True:
    success, img = video.read()
    if not success:
        break
    # Convertir la imagen a RGB para procesarla con Mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = Pose.process(img_rgb)
    points= results.multi_hand_landmarks
    if points:

        for hand_landmarks in points:
            # Dibujar los puntos de referencia de las manos en la imagen
            # https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
            draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            p1x = (hand_landmarks.landmark[pose.HandLandmark.THUMB_CMC].x)
            p1y = (hand_landmarks.landmark[pose.HandLandmark.THUMB_CMC].y)
            p2x = (hand_landmarks.landmark[pose.HandLandmark.THUMB_MCP].x)
            p2y = (hand_landmarks.landmark[pose.HandLandmark.THUMB_MCP].y)
            p3x = (hand_landmarks.landmark[pose.HandLandmark.THUMB_IP].x)
            p3y = (hand_landmarks.landmark[pose.HandLandmark.THUMB_IP].y)
            p4x = (hand_landmarks.landmark[pose.HandLandmark.THUMB_TIP].x)
            p4y = (hand_landmarks.landmark[pose.HandLandmark.THUMB_TIP].y)

            p17x = (hand_landmarks.landmark[pose.HandLandmark.PINKY_MCP].x)
            p20x = (hand_landmarks.landmark[pose.HandLandmark.PINKY_TIP].x)
            p13x = (hand_landmarks.landmark[pose.HandLandmark.RING_FINGER_MCP].x)
            p16x = (hand_landmarks.landmark[pose.HandLandmark.RING_FINGER_TIP].x)
            p9x = (hand_landmarks.landmark[pose.HandLandmark.MIDDLE_FINGER_MCP].x)
            p12x = (hand_landmarks.landmark[pose.HandLandmark.MIDDLE_FINGER_TIP].x)
            p5x = (hand_landmarks.landmark[pose.HandLandmark.INDEX_FINGER_MCP].x)
            p8x = (hand_landmarks.landmark[pose.HandLandmark.INDEX_FINGER_TIP].x)

            d1 = p17x - p20x
            d2 = p13x - p16x
            d3 = p9x - p12x
            d4 = p5x - p8x


            angulo_rad = math.atan2(p2y - p3y, p2x - p3x) - math.atan2(p4y - p3y, p4x - p3x)
            angulo1 = math.degrees(angulo_rad)
            angulo1 = (angulo1 + 360) % 360

            angulo = math.atan2(p1y - p2y, p1x - p2x) -math.atan2(p3y -p2y, p3x - p2x)
            angulo2 = math.degrees(angulo)
            angulo2 = (angulo2 + 360) % 360


            # Si los 4 dedos restantes estas cerrados
            if check and d1<0 and d2<0 and d3<0 and d4<0 and 161<angulo1<168.5 and 180<angulo2<184:
                print(f'like\n{angulo1}     {angulo2}')
                contador+=1
                check=False
            #elif d1>0 and d2>0 and d3>0 and d4>0:
            #    print(f'abierto\n{angulo1}      {angulo2}')
            elif angulo1>168.5 and angulo2>184:
                check=True
                print(f'{check}\n{angulo1}      {angulo2}')
            #elif d1<0 and d2<0 and d3<0 and d4<0:
            #    print(f'cerrado\n{angulo1}      {angulo2}')

        texto = f'Likes: {contador}'
        cv2.rectangle(img, (20, 240), (340, 120), (255, 0, 0), -1)
        cv2.putText(img, texto, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)




    cv2.imshow('Resultado', img)
    cv2.waitKey(150)



