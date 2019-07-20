from __future__ import print_function
import cv2
import numpy as np
import argparse
import time

blur = 1
def nothing(x):
  pass

# Background substraction def

fgbg = cv2.createBackgroundSubtractorMOG2()


# Color skin definition
#lower_skin = np.array([0,48,80], dtype=np.uint8)
#upper_skin = np.array([0,255,255], dtype=np.uint8)

lower_skin = np.array([0,60,60], dtype=np.uint8)
upper_skin = np.array([36,255,255], dtype=np.uint8)

# setting the control bar window

cv2.namedWindow("Bar")
cv2.moveWindow("Bar", 0,0)
cv2.createTrackbar("Blur", "Bar",0,255,nothing)
cv2.createTrackbar("High-hue", "Bar",0,255,nothing)
cv2.createTrackbar("Low-sat", "Bar",0,255,nothing)
cv2.createTrackbar("Low-bri", "Bar",0,255,nothing)



# Choosing a webcam
cam = cv2.VideoCapture(0)


############## capturing static background

b, frame= cam.read()
 
back_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue saturation brightness
back_hsv = cv2.inRange(back_hsv, lower_skin, upper_skin)
back_hsv = cv2.blur(back_hsv, (blur,blur))
back_hsv = cv2.bilateralFilter(back_hsv, 20, 150, 150) 
    
    
cv2.imshow('back_hsv', back_hsv)
print("read background")

############## capturing images

print("starting capture")
while b:
    
    key = cv2.waitKey(1)&0xFF 
    b, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue saturation brightness
    
           
# Read track bar
# Good sets : blur = 5, hh = 229, ls = 39, lb =80
    blur = cv2.getTrackbarPos("Blur", "Bar") + 1
    hh = cv2.getTrackbarPos("High-hue", "Bar")
    ls = cv2.getTrackbarPos("Low-sat", "Bar")
    lb = cv2.getTrackbarPos("Low-bri", "Bar")

    blur = 10

    # Applies masks
        
    clrMask = cv2.inRange(hsv, lower_skin, upper_skin)
    blrMask = cv2.blur(clrMask, (blur,blur))
    fgMask=fgbg.apply(blrMask)
    #fgMask = blrMask
    
    countours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours = sorted(countours, key=lambda x:cv2.contourArea(x), reverse = True)

    for countour in countours:
        area = cv2.contourArea(countour)
        if area > 5000:
            (x,y,w,h) = cv2.boundingRect(countour)
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        break

    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(cam.get(cv2.CAP_PROP_FPS)), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.imshow('Frame', frame)
    cv2.moveWindow("Frame", 100,0)
       #     cv2.imshow('FG Mask', fgMask)
       #     cv2.imshow("ColorMask",clrMask)
       #     cv2.imshow("back_hsv foreground",back_hsv)
    cv2.waitKey(1)
       
#    else:
#        print("cam not working")
#        break
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()