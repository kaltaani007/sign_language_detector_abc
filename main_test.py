import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import math

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5" , "model/labels.txt")

labels = ["A" , "B" , "C"]


offset = 20
size = 300
counter = 0

while True:

    flag, img = cap.read()

    imgOutput = img.copy()

    hands, img = detector.findHands(img  )

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((size, size, 3), np.uint8) * 255

        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        shape_crop = imgCrop.shape

        # height and width
        # imgWhite [0:shape_crop[0] , 0 : shape_crop[1]] = imgCrop

        # adjusting the images back on the canvas

        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = size / h
            wcal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wcal, size))
            img_shape_crop = imgResize.shape
            w_gap = math.ceil((size - wcal) / 2)
            imgWhite[:, w_gap:img_shape_crop[1] + w_gap] = imgResize
            prediction , index = classifier.getPrediction(imgWhite)
            #print(prediction , index)

        else:
            k = size / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (size, hcal))
            img_shape_crop = imgResize.shape

            h_gap = math.ceil((size - hcal) / 2)
            imgWhite[h_gap:img_shape_crop[0] + h_gap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50 ), (x - offset + 100 , y - offset - 50 +  50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput , labels[index] , (x , y - offset) , cv2.FONT_HERSHEY_COMPLEX , 2 , ( 255, 255 , 255 ) , 2)
        cv2.rectangle(imgOutput , (x-offset , y -offset) , (x + w +offset , y+h + offset) , (0 ,255, 0 ) , 3)
        cv2.imshow("Image Crop ", imgCrop)
        cv2.imshow("Image Whie ", imgWhite)

    cv2.imshow("Image", imgOutput)



    if cv2.waitKey(1) == ord('q'):
        break
