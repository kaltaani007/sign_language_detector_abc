import cv2
import numpy as np
import cvzone as cz
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import time

import math



cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

folder = "Data/C"

offset = 20
size = 300
counter = 0

while True:

    flag , img = cap.read()

    hands , img = detector.findHands(img)



    if hands:
        hand = hands[0]
        x , y , w , h = hand['bbox']

        imgWhite = np.ones((size, size, 3), np.uint8) * 255


        imgCrop = img[y - offset: y+h + offset ,x - offset : x+w + offset ]
        shape_crop = imgCrop.shape

        # height and width
        #imgWhite [0:shape_crop[0] , 0 : shape_crop[1]] = imgCrop

        # adjusting the images back on the canvas

        aspect_ratio = h/w

        if aspect_ratio > 1 :
            k = size / h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize (imgCrop, (wcal , size))
            img_shape_crop = imgResize.shape
            w_gap = math.ceil((size - wcal)/2)
            imgWhite[: , w_gap:img_shape_crop[1] + w_gap] = imgResize

        else:
            k = size / w
            hcal = math.ceil(k*h)
            imgResize = cv2.resize ( imgCrop , ( size ,  hcal))
            img_shape_crop = imgResize.shape

            h_gap = math.ceil((size - hcal) / 2)
            imgWhite[h_gap:img_shape_crop[0] + h_gap , :] = imgResize




        cv2.imshow("Image Crop ", imgCrop)
        cv2.imshow("Image Whie ", imgWhite)

    cv2.imshow("Image" , img)





    # """ Saving the images into the directory


    #if cv2.waitKey(1) == ord('s') and counter < 50 :
    #   counter +=1
    #   cv2.imwrite(f'{folder}/Image_{time.time()}.jpg' , imgWhite)
    #   print(counter)




    if cv2.waitKey(1) == ord('q'):
        break
