from __future__ import print_function
import cv2
import numpy as np
import argparse
import time

blur = 1
def nothing(x):
  pass

lower_skin = np.array([0,48,80], dtype=np.uint8)
upper_skin = np.array([0,255,255], dtype=np.uint8)

cv2.namedWindow("Bar")
cv2.moveWindow("Bar", 0,0)
cv2.createTrackbar("Blur", "Bar",0,255,nothing)
cv2.createTrackbar("High-hue", "Bar",0,255,nothing)
cv2.createTrackbar("Low-sat", "Bar",0,255,nothing)
cv2.createTrackbar("Low-bri", "Bar",0,255,nothing)

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              Opencv2. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    #backSub = cv2.createBackgroundSubtractorKNN()
    backSub = cv2.BackgroundSubtractorMOG()


cam = cv2.VideoCapture(0)


############## capturing static background

b, frame= cam.read()
if b:
    back_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    back_gray1 = cv2.bilateralFilter(back_gray1, 7, 150, 150) 
    cv2.imshow('back_gray1', back_gray1)
time.sleep(2) 


b, frame= cam.read()
if b:
    back_gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    back_gray2 = cv2.bilateralFilter(back_gray2, 7, 150, 150)
    cv2.imshow('back_gray2', back_gray2)

time.sleep(500) 

############## capturing images

while True:
    
    key = cv2.waitKey(1)&0xFF 
    b, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue saturation brightness

    if False:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
           

# Read track bar
# Good sets : blur = 5, hh = 229, ls = 39, lb =80
        blur = cv2.getTrackbarPos("Blur", "Bar") + 1
        hh = cv2.getTrackbarPos("High-hue", "Bar")
        ls = cv2.getTrackbarPos("Low-sat", "Bar")
        lb = cv2.getTrackbarPos("Low-bri", "Bar")

        lower_skin = np.array([0,60,60], dtype=np.uint8)
        upper_skin = np.array([36,255,255], dtype=np.uint8)
        blur = 10

        # Applies masks
        
        clrMask = cv2.inRange(hsv, lower_skin, upper_skin)
        blrMask = cv2.blur(clrMask, (blur,blur))
        fgMask = backSub.apply(blrMask)
        #fgMask = blrMask
        countours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for countour in countours:
            area = cv2.contourArea(countour)
            if area > 5000:
                cv2.drawContours(frame, countours, -1, (0,255,0), 3)


        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(cam.get(cv2.CAP_PROP_FPS)), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.imshow('Frame', frame)
        cv2.moveWindow("Frame", 100,0)
        cv2.imshow('FG Mask', fgMask)
        cv2.imshow("ColorMask",clrMask)
        
        cv2.waitKey(1)
       
    else:
        print("cam not working")
        break
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()