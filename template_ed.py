import numpy as np
import os
import cv2
import sys
import imutils
import pickle
import datetime
from snapshot import snapshot
import math
import time
import argparse
from pyimagesearch.shapedetector import ShapeDetector
from scipy.spatial import distance as dist
from scipy import signal
from imutils import perspective
from imutils import contours
from collections import defaultdict

figPathR = 'cam_data\snapshots\R'
figPathB = 'cam_data\snapshots\B'
figSavePath = 'cam_data\Match'
##
capTemp=r'cam_data\templates\caps'
icTemp=r'cam_data\templates\ic'
ledTempR=r'cam_data\templates\led\R'
ledTempB=r'cam_data\templates\led\B'
resTemp=r'cam_data\templates\resistors'
resTempCnt=r'cam_data\templates\resistors\cnt'
##
my_path = os.path.abspath(os.path.dirname(__file__))
##


for templateFile in os.listdir(os.path.join(my_path,resTempCnt)):
                    tempName= os.path.join(my_path,resTempCnt, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    
                    src_img = np.copy(template)
                    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
                    flag, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                    cv2.imshow("preview", gray)
                    cv2.waitKey(0)
                    cv2.imshow("preview", thresh)
                    cv2.waitKey(0)
                    # Find contours
                    img2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    c=max(contours,key=cv2.contourArea)
                    cnt_img=template.copy()
                    cv2.drawContours(cnt_img, [c], -1, (255, 255, 255), 2)
                    cv2.imshow("preview", cnt_img)
                    cv2.waitKey(0)
                    ## approx the contour
                    approx_img=thresh.copy()
                    epsilon=0.1*cv2.arcLength(c,True)
                    approx=cv2.approxPolyDP(c,epsilon,True)
                    cv2.drawContours(approx_img,[approx],-1,(255,255,255),3)                
                    cv2.imshow("preview",approx_img)
                    cv2.waitKey(0)
                    cr_c=approx
                    ## convex hull the contour
                    hull = cv2.convexHull(cr_c)
                    hull_img=template.copy()
                    cv2.drawContours(hull_img,[hull],-1,(255,255,0),3)
                    cv2.imshow("preview",hull_img)
                    cv2.waitKey(0)
                    ##snapshot(tempName,1,resTempCnt)
