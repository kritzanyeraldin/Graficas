import cv2
import numpy as np
import pickle

video = cv2.VideoCapture('ajedrez.mp4')


casillas = []
with open('espacios.pkl', 'rb') as file:
    casillas = pickle.load(file)



while True:
    check, img = video.read()
    img=cv2.resize(img,(1600,900))
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    kernel = np.ones((5,5), np.int8)
    imgDil = cv2.dilate(imgMedian, kernel)

    for x, y, w, h in casillas:
        espacio = imgDil[y:y+h, x:x+w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x,y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        if count < 2300:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('video', img)
    # cv2.imshow('video TH', imgTH)
    # cv2.imshow('video Median', imgMedian)
    # cv2.imshow('video Dilatada', imgDil)
    cv2.waitKey(2)

