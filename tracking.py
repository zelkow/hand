# -*- coding: utf-8 -*-
"""
Created on Wed May 14 21:41:48 2014
Module : contour detection ang gestire recognition using open cv for python
Used webcam to capture image, process contors  of hand and perform actions using xdotool package
@author: Prasad
"""

import pickle, time, os, threading
import numpy as np
from numpy import sqrt, arccos, rad2deg

import cv2

# check if latest version of open cv is installed
os.system("xdotool --help")

########################################################################
class HandTracking:

    #----------------------------------------------------------------------
    def __init__(self):
        #self.debugMode = False
        #hide debug data comment below line and un comment above line
        self.debugMode = True



        self.camera = cv2.VideoCapture(0)

        #Resolution settings
        self.camera.set(3,640)
        self.camera.set(4,480)

        self.posPre = 0

        #Dictionary to store our data
        self.Data = {"angles less 90" : 0,
                     "cursor" : (0, 0),
                     "hulls" : 0,
                     "defects" : 0,
                     "fingers": 0,
                     "fingers history": [0],
                     "area": 0,
                     }
        #Last updated
        self.lastData = self.Data

        # Loads the filter variables
        # If they change during execution, the file will be updated
        try:  self.Vars = pickle.load(open(".config", "r"))
        except:
            print "Config file («.config») not found."
            exit()

        # Separate windows for filters
        cv2.namedWindow("Filters")
        cv2.createTrackbar("erode", "Filters", self.Vars["erode"], 255, self.onChange_erode)
        cv2.createTrackbar("dilate", "Filters", self.Vars["dilate"], 255, self.onChange_dilate)
        cv2.createTrackbar("smooth", "Filters", self.Vars["smooth"], 255, self.onChange_smooth)

        cv2.namedWindow("HSV Filters")
        cv2.createTrackbar("upper", "HSV Filters", self.Vars["upper"], 255, self.onChange_upper)
        cv2.createTrackbar("filterUpS", "HSV Filters", self.Vars["filterUpS"], 255, self.onChange_fuS)
        cv2.createTrackbar("filterUpV", "HSV Filters", self.Vars["filterUpV"], 255, self.onChange_fuV)
        cv2.createTrackbar("lower", "HSV Filters", self.Vars["lower"], 255, self.onChange_lower)
        cv2.createTrackbar("filterDownS", "HSV Filters", self.Vars["filterDownS"], 255, self.onChange_fdS)
        cv2.createTrackbar("filterDownV", "HSV Filters", self.Vars["filterDownV"], 255, self.onChange_fdV)

        # Add Text
        self.addText = lambda image, text, point:cv2.putText(image,text, point, cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255))


        while True:
            self.run()  # Process the image
            self.interprete()  #Interprete events (clicks)
            self.updateMousePos()
            if self.debugMode:
                if cv2.waitKey(1) == 27: break


    #Filter up settings
    def onChange_fuS(self, value):
        self.Vars["filterUpS"] = value
        pickle.dump(self.Vars, open(".config", "w"))

    #filter down settings
    def onChange_fdS(self, value):
        self.Vars["filterDownS"] = value
        pickle.dump(self.Vars, open(".config", "w"))

    #filter up hsv
    def onChange_fuV(self, value):
        self.Vars["filterUpV"] = value
        pickle.dump(self.Vars, open(".config", "w"))

    #filter down hsv
    def onChange_fdV(self, value):
        self.Vars["filterDownV"] = value
        pickle.dump(self.Vars, open(".config", "w"))

    #F
    def onChange_upper(self, value):
        self.Vars["upper"] = value
        pickle.dump(self.Vars, open(".config", "w"))

    #----------------------------------------------------------------------
    def onChange_lower(self, value):
        self.Vars["lower"] = value
        pickle.dump(self.Vars, open(".config", "w"))

    #----------------------------------------------------------------------
    def onChange_erode(self, value):
        self.Vars["erode"] = value + 1
        pickle.dump(self.Vars, open(".config", "w"))

    #----------------------------------------------------------------------
    def onChange_dilate(self, value):
        self.Vars["dilate"] = value + 1
        pickle.dump(self.Vars, open(".config", "w"))

    #----------------------------------------------------------------------
    def onChange_smooth(self, value):
        self.Vars["smooth"] = value + 1
        pickle.dump(self.Vars, open(".config", "w"))


    #----------------------------------------------------------------------
    def run(self):
        ret, im = self.camera.read()
        im = cv2.flip(im, 1)
        self.imOrig = im.copy()
        self.imNoFilters = im.copy()

        # Applies smooth
        im = cv2.blur(im, (self.Vars["smooth"], self.Vars["smooth"]))

        #Skin Filter
        filter_ = self.filterSkin(im)

        #erode ffilter
        filter_ = cv2.erode(filter_,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.Vars["erode"], self.Vars["erode"])))

        #Dialate filter
        filter_ = cv2.dilate(filter_,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.Vars["dilate"], self.Vars["dilate"])))


        #Show binary (b/W) image
        if self.debugMode: cv2.imshow("Filter Skin", filter_)

        #Obtain contornos
        contours, hierarchy = cv2.findContours(filter_,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if self.debugMode: cv2.imshow("Filter Skin", filter_)

        # Remove \\ small area
        allIdex = []
        for index in range(len(contours)):
            area = cv2.contourArea(contours[index])
            if area < 5e3: allIdex.append(index)
        allIdex.sort(reverse=True)
        for index in allIdex: contours.pop(index)

        # If no outline, ends here
        if len(contours) == 0: return

        allIdex = []
        index_ = 0
        # We walk each contour
        for cnt in contours:
            self.Data["area"] = cv2.contourArea(cnt)

            tempIm = im.copy()
            tempIm = cv2.subtract(tempIm, im)

            #convex hull
            hull = cv2.convexHull(cnt)
            self.last = None
            self.Data["hulls"] = 0
            for hu in hull:
                if self.last == None: cv2.circle(tempIm, tuple(hu[0]), 10, (0,0,255), 5)
                else:
                    distance = self.distance(self.last, tuple(hu[0]))
                    if distance > 40:  #eliminate false contours
                        self.Data["hulls"] += 1
                        #Red Circles displayed
                        cv2.circle(tempIm, tuple(hu[0]), 10, (0,0,255), 5)
                self.last = tuple(hu[0])


            M = cv2.moments(cnt)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            cv2.circle(tempIm, (centroid_x, centroid_y), 20, (0,255,255), 10)
            self.Data["cursor"] = (centroid_x, centroid_y)


            hull = cv2.convexHull(cnt,returnPoints = False)
            angles = []
            defects = cv2.convexityDefects(cnt,hull)
            if defects == None: return

            self.Data["defects"] = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                if d > 1000 :
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    self.Data["defects"] += 1
                    cv2.circle(tempIm,far,5,[0,255,255],-1)
                    cv2.line(tempIm, start, far, [255, 0, 0], 5)
                    cv2.line(tempIm, far, end, [255, 0, 0], 5)
                    angles.append(self.angle(far, start, end))

            #Filter angles less than 90
            b = filter(lambda a:a<90, angles)

            self.Data["angles less 90"] = len(b)
            self.Data["fingers"] = len(b) + 1

            #history.
            self.Data["fingers history"].append(len(b) + 1)

            if len(self.Data["fingers history"]) > 10: self.Data["fingers history"].pop(0)
            self.imOrig = cv2.add(self.imOrig, tempIm)

            index_ += 1

        #draw contr
        cv2.drawContours(self.imOrig,contours,-1,(64,255,85),-1)

        #Visualize it.
        self.debug()
        if self.debugMode: cv2.imshow("\"Hulk\" Mode :P :P Green monster", self.imOrig)


    #----------------------------------------------------------------------
    def distance(self, cent1, cent2):
        """distance measurement"""
        x = abs(cent1[0] - cent2[0])
        y = abs(cent1[1] - cent2[1])
        d = sqrt(x**2+y**2)
        return d

    #----------------------------------------------------------------------
    def angle(self, cent, rect1, rect2):
        """angle measurement"""
        v1 = (rect1[0] - cent[0], rect1[1] - cent[1])
        v2 = (rect2[0] - cent[0], rect2[1] - cent[1])
        dist = lambda a:sqrt(a[0] ** 2 + a[1] ** 2)
        angle = arccos((sum(map(lambda a, b:a*b, v1, v2))) / (dist(v1) * dist(v2)))
        angle = abs(rad2deg(angle))
        return angle

    #----------------------------------------------------------------------
    def filterSkin(self, im):
        """skin filters."""
        UPPER = np.array([self.Vars["upper"], self.Vars["filterUpS"], self.Vars["filterUpV"]], np.uint8)
        LOWER = np.array([self.Vars["lower"], self.Vars["filterDownS"], self.Vars["filterDownV"]], np.uint8)
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        filter_im = cv2.inRange(hsv_im, LOWER, UPPER)
        return filter_im

    #----------------------------------------------------------------------
    def debug(self):
        """debg messages in video."""
        yPos = 10
        if self.debugMode: self.addText(self.imOrig, "Debug", (yPos, 20))
        pos = 50
        for key in self.Data.keys():
            if self.debugMode: self.addText(self.imOrig, (key+": "+str(self.Data[key])), (yPos, pos))
            pos += 20

    #----------------------------------------------------------------------Added for ubuntu
        #unstable wont work on windows
        # commented by tp on 15 may 2014
    def updateMousePos(self):
        """Actualiza la posisión del cursor."""
        pos = self.Data["cursor"]
        posPre = self.posPre
        npos = np.subtract(pos, posPre)
        self.posPre = pos

        if self.Data["fingers"] in [1]:
            try: elf.t.__stop.set()
            except: pass
            self.t = threading.Thread(target=self.moveMouse, args=(npos))
            self.t.start()

    #----------------------------------------------------------------------
    def interprete(self):
        """Interpret events."""
        cont = 3

        if self.Data["fingers history"][:cont] == [5] * cont:
            os.system("xdotool click 1")
            self.Data["fingers history"] = [0]
        elif self.Data["fingers history"][:cont] == [3] * cont:
            os.system("xdotool click 3")
            self.Data["fingers history"] = [0]

    #----------------------------------------------------------------------
    def moveMouse(self, x, y):

        mini = 10
        mul = 2
        x *= mul
        y *= mul

        posy = lambda n:(y/x) * n
        stepp = 10

        if x > 0:
            for i in range(0, x, stepp): os.system("xdotool mousemove_relative -- %d %d" %(i, posy(i)))
        if x < 0:
            for i in range(x, 0, stepp): os.system("xdotool mousemove_relative -- %d %d" %(i, posy(i)))
        time.sleep(0.2)


if __name__=='__main__':
    HandTracking()
