#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:57:01 2023

@author: indra25
"""
import cv2
import pafy


#computer-webcam
cap = cv2.VideoCapture(0)
print(cap)

#mobile/remote-camera
camera_url = ""
cap.open(camera_url)

#youtube-video
youtube_url = ""
data = pafy.new(youtube_url)
data= data.getbest(preftype = "mp4")
cap.open(data.url)


while cap.Isopened():
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",frame)
    cv2.imshow("gray frame",gray)
    cv2.waitKey(5)


four_cc = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("./output.avi",four_cc,20.0,(640,480),0)

cap.release()
cv2.destroyAllWindows()
