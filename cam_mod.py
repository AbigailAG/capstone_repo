### User places one component, then the component is detected,
### then finds location
### Created 2018-11-18
### Created by: Lindsay Vasilak
### last modified: 2019-Jan-13

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


############## Defining Functions ##############################

def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect
    
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point	
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image,pts):
        
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented
def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections


def calibrate(frame):
        ################ AUTO CALIBRATE #####################
        img_rgb=frame
        img = img_rgb.copy()
        ## make bgr image gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        ## blur the gray image
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        cv2.imshow("preview",blur_gray)
        cv2.waitKey(0)
        ## create binary image
        ret, thresh = cv2.threshold(blur_gray, 80,255, cv2.THRESH_BINARY)
        cv2.imshow("preview",thresh)
        cv2.waitKey(0)
        ## erode and dilate the threshed image to remove noise
        thresh=cv2.erode(thresh, None, iterations=3)
        cv2.imshow("preview",thresh)
        cv2.waitKey(0)
        thresh=cv2.dilate(thresh, None, iterations=1)
        cv2.imshow("preview",thresh)
        cv2.waitKey(0)
        ## create canny edge image
        low_threshold = 90  
        high_threshold =250 
        edges = cv2.Canny(thresh, low_threshold, high_threshold,None,3)
        cv2.imshow("preview",edges)
        edgesCopy=edges.copy()
        cv2.waitKey(0)
        ##h,w=img_rgb.shape[:2]
        ##mask=np.zeros((h+2,w+2),np.uint8)
        ## morph closing of the edges
        kernel=np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE, kernel)
        cv2.imshow("preview",closing)
        cv2.waitKey(0)
        ## get contours of morph closed edges
        h,w=edges.shape[:2]
        mask=np.zeros((h+2,w+2),np.uint8)
        ## sort the contours from left-to-right and initiaize the bounding box
        ## point colors
        cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))            
        ## find the contour with the biggest area
        c=max(cnts,key=cv2.contourArea)
        epsilon=0.1*cv2.arcLength(c,True)
        ## convex hull the contour
        hull = cv2.convexHull(c)
        hull_img=img_rgb.copy()
        cv2.drawContours(hull_img,[hull],-1,(255,255,0),3)
        cv2.imshow("preview",hull_img)
        cv2.waitKey(0)
        ## houghlines on the hull
        hough_img=frame.copy()
        h,w=hough_img.shape[:2]
        mask=np.zeros((h,w),np.uint8)
        cv2.drawContours(mask,[hull],-1,(255,255,255),1)
        cv2.imshow("preview",mask)
        cv2.waitKey(0)        
        threshold=100
        minLinLen=0
        maxLinGap=0
        lines=cv2.HoughLines(mask,1,np.pi/180,threshold)
        for [rho,theta] in lines[:,0,:]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(hough_img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("preview",hough_img)
        cv2.waitKey(0)
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)
        int_img=frame.copy()
        for point in intersections:                
                cv2.circle(int_img, (int(point[0][0]), int(point[0][1])), 5, (255,255,0), -1)
        cv2.imshow("preview",int_img)
        cv2.waitKey(0)
        ## approx the contour
        approx_img=img_rgb.copy()
        epsilon=0.002*cv2.arcLength(hull,True)
        approx=cv2.approxPolyDP(hull,epsilon,True)
        for point in approx:                
                cv2.circle(approx_img, (int(point[0][0]), int(point[0][1])), 5, (255,255,0), -1)
        cv2.drawContours(approx_img,[approx],-1,(255,255,0),3)
        approx=np.array(approx[:,0,:])        
        cv2.imshow("preview",approx_img)
        cv2.waitKey(0)  
        ## get rotated rectangle version of the hull contour                             
        rect=cv2.minAreaRect(approx)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        cv2.drawContours(approx_img, [box], -1, (0, 0, 255), 2)       
        cv2.imshow("preview",approx_img)
        cv2.waitKey(0)
        # compute the rotated bounding box of the contour, then
        # draw the contours
              
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        rect = order_points_old(box)

        ## four point transform warped image
        warped=four_point_transform(frame.copy(),rect)
        bbpoints=rect
        cv2.imshow("preview",warped)
        cv2.waitKey(0)
        h,w=warped.shape[:2]
        rect = np.array([[0,0],[w,0],[w,h],[0,h]],dtype="int")
        rect = order_points_old(rect)
        frame=warped
        order_img=frame.copy()
        # check to see if the new method should be used for
        # ordering the coordinates
        ##if args["new"] > 0:
        ##	rect = perspective.order_points(box)

        # loop over the original points and draw them 
        for ((x, y), color) in zip(rect, colors):
                cv2.circle(order_img, (int(x), int(y)), 5, color, -1)

        topLeft=rect[0]
        topRight=rect[1]
        botRight=rect[2]
        botLeft=rect[3]
        # show the image
        cv2.imshow("preview", order_img)
        cv2.waitKey(0)
        ##  save rectangle points
        filehandler=open("cam_data/bbpoints.obj","wb")
        pickle.dump(bbpoints,filehandler)
        filehandler.close()
        ### Get the regions of the breadboard
        ### region 1
        h = dist.cdist([topLeft],[botLeft], "euclidean")
        htop=dist.cdist([topRight],[topLeft], "euclidean")
        bbot=dist.cdist([botRight],[botLeft], "euclidean")
        side_w=0.037
        r1_w=0.16
        r2_w=0.31
        r3_w=r2_w
        r4_w=r1_w
        r5_w=0.06
        ht=topRight-topLeft
        hbot=botRight-botLeft
        ht_norm=np.linalg.norm(ht)
        bt_norm=np.linalg.norm(hbot)
        uht=np.divide(ht,ht_norm)
        ubt=np.divide(hbot,bt_norm)        
        topLeft=topLeft+side_w*htop*uht
        topRight=topRight-side_w*htop*uht
        botLeft=botLeft+side_w*bbot*ubt
        botRight=botRight-side_w*bbot*ubt        
        order_img2=frame.copy()
        array_new=np.array([np.array(topLeft).flatten(),np.array(topRight).flatten(),np.array(botRight).flatten(),np.array(botLeft).flatten()],dtype="int32")
        cv2.drawContours(order_img2, [array_new], -1, (0, 255, 0), 1)
        cv2.imshow("preview",order_img2)
        cv2.waitKey(0)        
        vtl=topLeft-botLeft
        vtl_norm=np.linalg.norm(vtl)
        utl=np.divide(vtl,vtl_norm)
        vtr=topRight-botRight
        vtr_norm=np.linalg.norm(vtr)
        utr=np.divide(vtr,vtr_norm)
        r1_br=topRight-r1_w*h*utr
        r1_bl=topLeft-r1_w*h*utl  
        r2_tr=r1_br
        r2_tl=r1_bl
        lp_l=r1_bl-r2_w*0.5*h*utl
        lp_r=r1_br-r2_w*0.5*h*utr
        r2_bl=r2_tl-r2_w*h*utl
        r2_br=r2_tr-r2_w*h*utr        
        r5_tr=r2_br
        r5_tl=r2_bl
        r5_bl=r5_tl-r5_w*h*utl
        r5_br=r5_tr-r5_w*h*utr
        r3_tr=r5_br
        r3_tl=r5_bl
        r3_bl=r3_tl-r3_w*h*utl
        r3_br=r3_tr-r3_w*h*utr
        r4_tr=r3_br
        r4_tl=r3_bl
        r4_bl=r4_tl-r4_w*h*utl
        r4_br=r4_tr-r4_w*h*utr 
        region1=np.array([np.array(topLeft).flatten(),np.array(topRight).flatten(),np.array(r1_br).flatten(),np.array(r1_bl).flatten()],dtype="int32")
        region2=np.array([np.array(r2_tl).flatten(),np.array(r2_tr).flatten(),np.array(r2_br).flatten(),np.array(r2_bl).flatten()],dtype="int32")
        region3=np.array([np.array(r3_tl).flatten(),np.array(r3_tr).flatten(),np.array(r3_br).flatten(),np.array(r3_bl).flatten()],dtype="int32")
        region4=np.array([np.array(r4_tl).flatten(),np.array(r4_tr).flatten(),np.array(r4_br).flatten(),np.array(r4_bl).flatten()],dtype="int32")
        region5=np.array([np.array(r5_tl).flatten(),np.array(r5_tr).flatten(),np.array(r5_br).flatten(),np.array(r5_bl).flatten()],dtype="int32")
        img_thing=img_rgb.copy()
        cv2.drawContours(img_thing,[region1],-1,(0,255,0),3)
        cv2.drawContours(img_thing,[region2],-1,(255,255,0),3)
        cv2.drawContours(img_thing,[region5],-1,(0,0,255),3)
        cv2.drawContours(img_thing,[region4],-1,(180,0,90),3)
        cv2.drawContours(img_thing,[region3],-1,(0,255,255),3)
        cv2.imshow("preview",img_thing)
        cv2.waitKey(0)
        #calculate each rail
        numRails=31
        r_xt=np.linspace(topLeft[0][0],topRight[0][0],numRails)
        r_yt=np.linspace(topLeft[0][1],topRight[0][1],numRails)
        r_xb=np.linspace(botLeft[0][0],botRight[0][0],numRails)
        r_yb=np.linspace(botLeft[0][1],botRight[0][1],numRails)
        r_rails=[]        
        r1_xt=np.linspace(region1[0][0],region1[1][0],numRails)
        r1_yt=np.linspace(region1[0][1],region1[1][1],numRails)
        r1_xb=np.linspace(region1[3][0],region1[2][0],numRails)
        r1_yb=np.linspace(region1[3][1],region1[2][1],numRails)
        r1_rails=[]
        r2_xt=np.linspace(region2[0][0],region2[1][0],numRails)
        r2_yt=np.linspace(region2[0][1],region2[1][1],numRails)
        r2_xb=np.linspace(region2[3][0],region2[2][0],numRails)
        r2_yb=np.linspace(region2[3][1],region2[2][1],numRails)
        r2_rails=[]
        r3_xt=np.linspace(region3[0][0],region3[1][0],numRails)
        r3_yt=np.linspace(region3[0][1],region3[1][1],numRails)
        r3_xb=np.linspace(region3[3][0],region3[2][0],numRails)
        r3_yb=np.linspace(region3[3][1],region3[2][1],numRails)
        r3_rails=[]
        r4_xt=np.linspace(region4[0][0],region4[1][0],numRails)
        r4_yt=np.linspace(region4[0][1],region4[1][1],numRails)
        r4_xb=np.linspace(region4[3][0],region4[2][0],numRails)
        r4_yb=np.linspace(region4[3][1],region4[2][1],numRails)
        r4_rails=[]      

        for ind in xrange(r1_xt.size-1):
            r_rectMat=[[r_xt[ind],r_yt[ind]],[r_xt[ind+1],r_yt[ind+1]],[r_xb[ind+1],r_yb[ind+1]],[r_xb[ind],r_yb[ind]]]
            r_rails.append(r_rectMat)
            r1_rectMat=[[r1_xt[ind],r1_yt[ind]],[r1_xt[ind+1],r1_yt[ind+1]],[r1_xb[ind+1],r1_yb[ind+1]],[r1_xb[ind],r1_yb[ind]]]
            r1_rails.append(r1_rectMat)

            r2_rectMat=[[r2_xt[ind],r2_yt[ind]],[r2_xt[ind+1],r2_yt[ind+1]],[r2_xb[ind+1],r2_yb[ind+1]],[r2_xb[ind],r2_yb[ind]]]
            r2_rails.append(r2_rectMat)

            r3_rectMat=[[r3_xt[ind],r3_yt[ind]],[r3_xt[ind+1],r3_yt[ind+1]],[r3_xb[ind+1],r3_yb[ind+1]],[r3_xb[ind],r3_yb[ind]]]
            r3_rails.append(r3_rectMat)

            r4_rectMat=[[r4_xt[ind],r4_yt[ind]],[r4_xt[ind+1],r4_yt[ind+1]],[r4_xb[ind+1],r4_yb[ind+1]],[r4_xb[ind],r4_yb[ind]]]
            r4_rails.append(r4_rectMat)
        #fix the outside rails
        ind=0
        A=r2_tl-0.9*side_w*htop*uht
        B=r2_bl-0.9*side_w*bbot*ubt
        print A
        print B
        print r2_rails[0]
        r2_rails[0]=[[A[0][0],A[0][1]],[r2_xt[ind+1],r2_yt[ind+1]],[r2_xb[ind+1],r2_yb[ind+1]],[B[0][0],B[0][1]]]
        A=r2_tr+0.9*side_w*htop*uht
        B=r2_br+0.9*side_w*bbot*ubt
        ind=r1_xt.size-1
        r2_rails[-1]=[[r2_xt[ind-1],r2_yt[ind-1]],[A[0][0],A[0][1]],[B[0][0],B[0][1]],[r2_xb[ind-1],r2_yb[ind-1]]]

        A=r3_tl-0.9*side_w*htop*uht
        B=r3_bl-0.9*side_w*bbot*ubt
        ind=0
        r3_rails[0]=[[A[0][0],A[0][1]],[r3_xt[ind+1],r3_yt[ind+1]],[r3_xb[ind+1],r3_yb[ind+1]],[B[0][0],B[0][1]]]
        A=r3_tr+0.9*side_w*htop*uht
        B=r3_br+0.9*side_w*bbot*ubt
        ind=r1_xt.size-1
        r3_rails[-1]=[[r3_xt[ind-1],r3_yt[ind-1]],[A[0][0],A[0][1]],[B[0][0],B[0][1]],[r3_xb[ind-1],r3_yb[ind-1]]]
        
        #draw each rail     
        for ind in xrange(len(r_rails)):
              rect_r=np.array([r_rails[ind]],dtype="int32")
              #cv2.drawContours(img_thing,rect_r,-1,(200,25,0),2)

              rect_r2=np.array([r2_rails[ind]],dtype="int32")
              cv2.drawContours(img_thing,rect_r2,-1,(200,25,0),1)

              rect_r3=np.array([r3_rails[ind]],dtype="int32")
              cv2.drawContours(img_thing,rect_r3,-1,(200,25,0),1)

              
        cv2.imshow("preview",img_thing)
        cv2.waitKey(0)   

        ## Save rail coordinates
        filehandler=open("cam_data/r_rails.obj","wb")
        pickle.dump(r_rails,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r2_rails.obj","wb")
        pickle.dump(r2_rails,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r3_rails.obj","wb")
        pickle.dump(r3_rails,filehandler)
        filehandler.close()
        ## Save region coordinates
        filehandler=open("cam_data/topLeft.obj","wb")
        pickle.dump(topLeft,filehandler)
        filehandler.close()
        filehandler=open("cam_data/topRight.obj","wb")
        pickle.dump(topRight,filehandler)
        filehandler.close()
        filehandler=open("cam_data/botLeft.obj","wb")
        pickle.dump(botLeft,filehandler)
        filehandler.close()
        filehandler=open("cam_data/botRight.obj","wb")
        pickle.dump(botRight,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r5_br.obj","wb")
        pickle.dump(r5_br,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r5_bl.obj","wb")
        pickle.dump(r5_bl,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r5_tr.obj","wb")
        pickle.dump(r5_tr,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r5_tl.obj","wb")
        pickle.dump(r5_tl,filehandler)
        filehandler.close()

        filehandler=open("cam_data/r4_br.obj","wb")
        pickle.dump(r4_br,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r4_bl.obj","wb")
        pickle.dump(r4_bl,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r4_tr.obj","wb")
        pickle.dump(r4_tr,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r4_tl.obj","wb")
        pickle.dump(r4_tl,filehandler)
        filehandler.close()

        filehandler=open("cam_data/r3_br.obj","wb")
        pickle.dump(r3_br,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r3_bl.obj","wb")
        pickle.dump(r3_bl,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r3_tr.obj","wb")
        pickle.dump(r3_tr,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r3_tl.obj","wb")
        pickle.dump(r3_tl,filehandler)
        filehandler.close()

        filehandler=open("cam_data/r2_br.obj","wb")
        pickle.dump(r2_br,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r2_bl.obj","wb")
        pickle.dump(r2_bl,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r2_tr.obj","wb")
        pickle.dump(r2_tr,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r2_tl.obj","wb")
        pickle.dump(r2_tl,filehandler)
        filehandler.close()

        filehandler=open("cam_data/r1_br.obj","wb")
        pickle.dump(r1_br,filehandler)
        filehandler.close()
        filehandler=open("cam_data/r1_bl.obj","wb")
        pickle.dump(r1_bl,filehandler)
        filehandler.close()


        cv2.destroyAllWindows()


def componentDetect(frame, resTemp, capTemp,icTemp,ledTemp,my_path):
        ### Detect Components in frame
                cv2.destroyAllWindows()
                resMat=[]
                capMat=[]
                ledMat=[]
                icMat=[]
                resCount=0
                capCount=0
                ledCount=0
                icCount=0
                src_img = np.copy(frame)
                img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
                ## Check if resistors
                for templateFile in os.listdir(os.path.join(my_path,resTemp)):
                    tempName= os.path.join(my_path,resTemp, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    c,w, h  = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)           
                    threshold = 0.63
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (255,0,255), 5)
                        resMat.append([pt, (pt[0] + w, pt[1] + h)])
                        resCount=resCount+1
                print 'resistor matches=', resCount
                for resistor in resMat:
                    cv2.rectangle(src_img,resistor[0],resistor[1],(255,255,0),5)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ## Check if caps
                src_img = np.copy(src_img)
                for templateFile in os.listdir(os.path.join(my_path,capTemp)):        
                    tempName= os.path.join(my_path,capTemp, templateFile)
                    template = cv2.imread(tempName)
                    c,w, h = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)
                    threshold = 0.63
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (250,200,0), 1)
                        capMat.append([pt, (pt[0] + w, pt[1] + h)])
                        capCount=capCount+1
                print 'cap matches=', capCount
                for cap in capMat:
                    cv2.rectangle(src_img,cap[0],cap[1],(0,255,0),5)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()                
                ## Check if LEDs
                for templateFile in os.listdir(os.path.join(my_path,ledTemp)):
                    tempName= os.path.join(my_path,ledTemp, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    c,w, h  = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)           
                    threshold = 0.65
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (200,100,100), 5)
                        ledMat.append([pt, (pt[0] + w, pt[1] + h)])
                        ledCount=ledCount+1
                print 'led matches=', ledCount
                for led in ledMat:
                    cv2.rectangle(src_img,led[0],led[1],(0,255,255),5)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ## Check if ICs
                for templateFile in os.listdir(os.path.join(my_path,icTemp)):
                    tempName= os.path.join(my_path,icTemp, templateFile)
                    tempName=os.path.join(tempName)
                    template = cv2.imread(tempName)
                    c,w, h  = template.shape[::-1]
                    res = cv2.matchTemplate(src_img,template,cv2.TM_CCOEFF_NORMED)           
                    threshold = 0.9
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(src_img, pt, (pt[0] + w, pt[1] + h), (0,100,250), 5)
                        icMat.append([pt, (pt[0] + w, pt[1] + h)])
                        icCount=icCount+1
                print 'IC matches=', icCount
                for ic in icMat:
                    cv2.rectangle(src_img,ic[0],ic[1],(0,0,0),-1)
                cv2.imshow("preview", src_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #return resCount,capCount,ledCount,icCount
                componentName="resistor"
                return componentName

def componentLocate(fgmask,frame, resCount,capCount,ledCount,icCount):
############# LOCATION FIND ######################
    if capCount+resCount+ledCount+icCount>0:
        if icCount>0:
            xrailMat=[]
            yrailMat=[]
            ic=icMat[2]
            xrailMat.append(ic[0][0])
            xrailMat.append(ic[1][0])
            yrailMat.append(ic[0][1])
            yrailMat.append(ic[1][1])
        else:
            xrailMat=[]
            yrailMat=[]
            img=np. copy(fgmask)
##            cv2.imshow("preview",img)
##            cv2.waitKey(0)
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
##            cv2.imshow("preview",blur_gray)
##            cv2.waitKey(0)
            low_threshold = 50  ##50
            high_threshold = 150 ##240
            edges = cv2.Canny(blur_gray, low_threshold, high_threshold,None,3)
##            cv2.imshow("preview",edges)
##            cv2.waitKey(0)
            # Copy edges to the images that will display the results in BGR         
            rho = cv2.HOUGH_PROBABILISTIC  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 25                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 1  # minimum number of pixels making up a line
            max_line_gap = 30  # maximum gap in pixels between connectable line segments
            line_image = np.copy(frame) * 0  # creating a blank to draw lines on
            ##lines = cv2.HoughLines(edges, 1, np.pi / 180, 30, None, 0, 0) ##standard
            img_rgb=frame.copy()
            lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, theta, threshold, np.array([]),min_line_length, max_line_gap)
            if lines is not None:
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                        ##cv2.circle(line_image,(x1,y1),10,(0,255,0),3)
                        ##cv2.circle(line_image,(x2,y2),10,(0,0,255),3)
                        xrailMat.append(x1)
                        xrailMat.append(x2)
                        yrailMat.append(y1)
                        yrailMat.append(y2)
        ##            # Draw the lines on the  image    
                ##img_rgb=cv2.imread(rgbPath)
                lines_edges = cv2.addWeighted(img_rgb, 0.8, line_image, 1, 0)
##                cv2.imshow("preview",lines_edges)
##                cv2.waitKey(0)

        ## Get contours of the image
        #################################
        #################################
        #################################
        ## make bgr image gray
        img=fgmask.copy()
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        ## blur the gray image
        blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
##        cv2.imshow("preview",blur_gray)
##        cv2.waitKey(0)
        ## create binary image
        ret, thresh = cv2.threshold(blur_gray, 80,255, cv2.THRESH_BINARY)
##        cv2.imshow("preview",thresh)
##        cv2.waitKey(0)
        thresh_orig=img.copy()
        ## erode and dilate the threshed image to remove noise
        ##thresh=cv2.erode(thresh, None, iterations=2)
        ##cv2.imshow("preview",thresh)
        ##cv2.waitKey(0)
        thresh=cv2.dilate(thresh, None, iterations=2)
##        cv2.imshow("preview",thresh)
##        cv2.waitKey(0)
        ## create canny edge image
        low_threshold = 90  
        high_threshold =250 
        edges = cv2.Canny(thresh, low_threshold, high_threshold,None,3)
##        cv2.imshow("preview",edges)
        edgesCopy=edges.copy()
##        cv2.waitKey(0)

        ##h,w=img_rgb.shape[:2]
        ##mask=np.zeros((h+2,w+2),np.uint8)

        ## morph closing of the edges
        kernel=np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE, kernel)
##        cv2.imshow("preview",closing)
##        cv2.waitKey(0)

        ## get contours of morph closed edges
        h,w=edges.shape[:2]
        mask=np.zeros((h+2,w+2),np.uint8)
        ##im,contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ## sort the contours from left-to-right and initiaize the bounding box
        ## point colors
        cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE)
        cnts=imutils.grab_contours(cnts)
        ## find the contour with the biggest area
        if len(cnts)>0:
            c=max(cnts,key=cv2.contourArea)
            epsilon=0.1*cv2.arcLength(c,True)
        else:
            print "error no contours found"
            return none
        ##(cnts, _) = contours.sort_contours(cnts)
        ##colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

        ## determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ext_img=frame.copy()
        cv2.drawContours(ext_img,[c],-1,(0,255,255),2)
        cv2.circle(ext_img,extLeft,8, (0, 0, 255), -1)
        cv2.circle(ext_img,extRight,8, (0, 255,0), -1)
        cv2.circle(ext_img,extTop,8, (255, 0, 0), -1)
        cv2.circle(ext_img,extBot,8, (255, 255,0 ), -1)
##        cv2.imshow("preview",ext_img)
##        cv2.waitKey(0)                  

        ## approx the contour
        approx_img=frame.copy()
        approx=cv2.approxPolyDP(c,epsilon,True)
        cv2.drawContours(approx_img,[approx],-1,(255,255,0),3)
##        cv2.imshow("preview",approx_img)
##        cv2.waitKey(0)
        cr_c=approx
        ## convex hull the contour
        hull = cv2.convexHull(c)
        hull_img=frame.copy()
        cv2.drawContours(hull_img,[hull],-1,(255,255,0),3)
##        cv2.imshow("preview",hull_img)
##        cv2.waitKey(0)

         ######################
        #########################
        ######################
        ################

        ## railLocate
        rail1,rail2=railLocate(hull,img_rgb,thresh_orig)
        return rail1, rail2
                
                    
                    
def cross_rails(c_hull,img_rgb,thresh_img,minr3,maxr3,minr2,maxr2):
    thresh=thresh_img.copy()
    img=img_rgb.copy()
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts=imutils.grab_contours(cnts)
    if len(cnts)>0:                
        c=max(cnts,key=cv2.contourArea)
    else:
        print "no contours found, using hull"
        c=c_hull
    if cv2.contourArea(c)<=0.6*cv2.contourArea(c_hull):
        c=c_hull
        print "incorrect contour area, using hull"
##    minInd=np.argmin(c, axis=0)
##    print minInd
##    maxInd=np.argmax(c,axis=0)
##    print maxInd
    
    r2_rails=pickle.load(open("cam_data/r2_rails.obj","rb"))
    r3_rails=pickle.load(open("cam_data/r3_rails.obj","rb"))
    r2_rail_check=np.zeros((len(r2_rails),1))
    r3_rail_check=np.zeros((len(r3_rails),1))
    minInd = np.argmin(c, axis=0)    
    print minInd[0,1]
    minInd=minInd[0,1]
    maxInd = np.argmax(c, axis=0)
    print maxInd[0,1]
    maxInd=maxInd[0,1]
    print minInd
    print maxInd
    
    minmat=c[minInd]    
    maxmat=c[maxInd]
    print minmat[0][1]
    print maxmat[0][1]
    cv2.circle(img, (int(minmat[0][0]), int(minmat[0][1])), 5, (255,255,0), -1)
    cv2.circle(img, (int(maxmat[0][0]), int(maxmat[0][1])), 5, (255,255,0), -1)    
    cv2.imshow("preview",img)
    cv2.waitKey(0)
    for ind1 in xrange(len(r2_rails)): 
        #region2
        check_rail2=check(r2_rails[ind1][0][0],r2_rails[ind1][0][1],r2_rails[ind1][2][0],r2_rails[ind1][2][1],int(minmat[0][0]),int(minmat[0][1]))
        if check_rail2==1:
##            print "match2"
            r2_rail_check[ind1]=r2_rail_check[ind1]+1
                
        #region3
        check_rail3=check(r3_rails[ind1][0][0],r3_rails[ind1][0][1],r3_rails[ind1][2][0],r3_rails[ind1][2][1],int(maxmat[0][0]),int(maxmat[0][1]))
        if check_rail3==1:
##            print "match3"
            r3_rail_check[ind1]=r3_rail_check[ind1]+1
            
    rail1= np.nonzero(r3_rail_check)
    rail2= np.nonzero(r2_rail_check)
    if max(r3_rail_check)==0:
        print "no matches found in region3, using area method"
        rail1=((maxr3+ minr3)/2)-1
    elif max(r2_rail_check)==0:
        print "no matches found in region3, using area method"
        rail2=((maxr2+ minr2)/2)-1
##    print rail1
##    print rail1[0]
##    print len(rail1)
##    print len(rail2)
##    print tuple(rail1)
##    print tuple(rail1[0])
##    print int(rail1[0])
##    print rail2
    if len(rail1)>0:
        #rail1=tuple(np.array(np.nonzero(r3_rail_check),dtype="int").flatten())
        rail1=int(rail1[0])
    else:
        rail1=0
        print "unknown error rail1"
    if len(rail2)>0:
        #rail2=tuple(np.array(np.nonzero(r2_rail_check),dtype="int").flatten())
        rail2=int(rail2[0])
    else:
        rail2=0
        print "unknown error rail2"
    print "rail1:",rail1
    print "rail2:", rail2
    rail1=30-rail1
    rail2=60-rail2
    return rail1,rail2
        
                    
    
##    yMin=np.min(yrailMat)
##    yMax=np.max(yrailMat)
##    xMin=np.min(xrailMat)
##    xMax=np.max(xrailMat)
##    ydist=yMax-yMin
##    xdist=xMax-xMin
##    if xdist > 1.5*ydist:  ## horizontal
##            yavg=0.5*ydist+yMin
##            rail1=railLocate(numRails,xMin,int(np.ceil(yavg)),img_rgb)
##            rail2=railLocate(numRails,xMax,int(np.ceil(yavg)),img_rgb)
##    elif ydist > 1.5*xdist: ## vertical
##            xavg=0.5*xdist+xMin
##            rail1=railLocate(numRails,int(np.ceil(xavg)),yMin,img_rgb)
##            rail2=railLocate(numRails,int(np.ceil(xavg)),yMax,img_rgb)
##    else:  ##angled
##            yavg=0.5*ydist+yMin
##            rail1=railLocate(numRails,xMin,int(np.ceil(yavg)),img_rgb)
##            rail2=railLocate(numRails,xMax,int(np.ceil(yavg)),img_rgb)                    
    #print rail1,rail2
    #return rail1,rail2
                    


def snapshot(str,rval,frame):
    import cv2    
    cv2.imwrite(str,frame)    
    return
        
                
def railLocate(c,img_rgb,thresh_img):               

        #load the region coords
        topLeft=np.array(pickle.load(open("cam_data/topLeft.obj","rb"))).flatten()
        topRight=np.array(pickle.load(open("cam_data/topRight.obj","rb"))).flatten()
        botLeft=np.array(pickle.load(open("cam_data/botLeft.obj","rb"))).flatten()
        botRight=np.array(pickle.load(open("cam_data/botRight.obj","rb"))).flatten()
        r1_bl=np.array(pickle.load(open("cam_data/r1_bl.obj","rb"))).flatten()
        r1_br=np.array(pickle.load(open("cam_data/r1_br.obj","rb"))).flatten()
        r2_bl=np.array(pickle.load(open("cam_data/r2_bl.obj","rb"))).flatten()
        r2_br=np.array(pickle.load(open("cam_data/r2_br.obj","rb"))).flatten()
        r2_tl=np.array(pickle.load(open("cam_data/r2_tl.obj","rb"))).flatten()
        r2_tr=np.array(pickle.load(open("cam_data/r2_tr.obj","rb"))).flatten()
        r3_bl=np.array(pickle.load(open("cam_data/r3_bl.obj","rb"))).flatten()
        r3_br=np.array(pickle.load(open("cam_data/r3_br.obj","rb"))).flatten()
        r3_tl=np.array(pickle.load(open("cam_data/r3_tl.obj","rb"))).flatten()
        r3_tr=np.array(pickle.load(open("cam_data/r3_tr.obj","rb"))).flatten()
        r4_bl=np.array(pickle.load(open("cam_data/r4_bl.obj","rb"))).flatten()
        r4_br=np.array(pickle.load(open("cam_data/r4_br.obj","rb"))).flatten()
        r4_tl=np.array(pickle.load(open("cam_data/r4_tl.obj","rb"))).flatten()
        r4_tr=np.array(pickle.load(open("cam_data/r4_tr.obj","rb"))).flatten()
        r5_bl=np.array(pickle.load(open("cam_data/r5_bl.obj","rb"))).flatten()
        r5_br=np.array(pickle.load(open("cam_data/r5_br.obj","rb"))).flatten()
        r5_tl=np.array(pickle.load(open("cam_data/r5_tl.obj","rb"))).flatten()
        r5_tr=np.array(pickle.load(open("cam_data/r5_tr.obj","rb"))).flatten()

        r2_rails=pickle.load(open("cam_data/r2_rails.obj","rb"))
        r3_rails=pickle.load(open("cam_data/r3_rails.obj","rb"))

        r2_rail_check=np.zeros((len(r2_rails),1))
        r3_rail_check=np.zeros((len(r3_rails),1))
        
        for ind1 in xrange(len(r2_rails)):
            rail_img=img_rgb.copy()
            for ind2 in xrange(len(c)):
                cmat=np.array(c[ind2]).flatten()
                check_rail2=check(r2_rails[ind1][0][0],r2_rails[ind1][0][1],r2_rails[ind1][2][0],r2_rails[ind1][2][1],cmat[0],cmat[1])
                check_rail3=check(r3_rails[ind1][0][0],r3_rails[ind1][0][1],r3_rails[ind1][2][0],r3_rails[ind1][2][1],cmat[0],cmat[1])
                if check_rail2==1:
                    #print "match2"
                    r2_rail_check[ind1]=r2_rail_check[ind1]+1
                if check_rail3==1:
                    #print "match3"
                    r3_rail_check[ind1]=r3_rail_check[ind1]+1
            
                

        ## intersect area method        
        r2_rail_check=np.zeros((len(r2_rails),1))
        r3_rail_check=np.zeros((len(r3_rails),1))
        
        for ind1 in xrange(len(r2_rails)):
            #region2
            cnt_img=thresh_img.copy()
            h,w=cnt_img.shape[:2]
            mask=np.zeros((h,w),np.uint8)
            rect_r2=np.array([r2_rails[ind1]],dtype="int32")
            cv2.drawContours(mask,rect_r2,-1,(200,25,0),1)
            cv2.fillPoly(mask, rect_r2, (200,25,0), 1)
            th,im_th=cv2.threshold(mask,80,255,cv2.THRESH_BINARY)
            im_out=cv2.bitwise_and(cnt_img, im_th)
            cnts = cv2.findContours(im_out, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnts=imutils.grab_contours(cnts)
            if len(cnts)>0:                
                c_rail=max(cnts,key=cv2.contourArea)
                cnt_img=thresh_img.copy()
                h,w=cnt_img.shape[:2]
                mask=np.zeros((h,w),np.uint8)
                cv2.drawContours(mask,[c_rail],-1,(255,255,0),3)
##                cv2.imshow("preview",mask)
##                cv2.waitKey(0)
                area=cv2.contourArea(c_rail)
                r2_rail_check[ind1]=r2_rail_check[ind1]+area
            
            
            #region3
            cnt_img=thresh_img.copy()
            h,w=cnt_img.shape[:2]
            mask=np.zeros((h,w),np.uint8)
            rect_r3=np.array([r3_rails[ind1]],dtype="int32")
            cv2.drawContours(mask,rect_r3,-1,(200,25,0),1)
            cv2.fillPoly(mask, rect_r3, (200,25,0), 1)
            th,im_th=cv2.threshold(mask,80,255,cv2.THRESH_BINARY)
            im_out=cv2.bitwise_and(cnt_img, im_th)
            cnts = cv2.findContours(im_out, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnts=imutils.grab_contours(cnts)
            if len(cnts)>0:                
                c_rail=max(cnts,key=cv2.contourArea)
                cnt_img=thresh_img.copy()
                h,w=cnt_img.shape[:2]
                mask=np.zeros((h,w),np.uint8)
                cv2.drawContours(mask,[c_rail],-1,(255,255,0),3)
##                cv2.imshow("preview",mask)
##                cv2.waitKey(0)
                area=cv2.contourArea(c_rail)
                r3_rail_check[ind1]=r3_rail_check[ind1]+area
            
        #analyze the rail check
        r2_ind=tuple(r2_rail_check.nonzero()[0])
        r3_ind=tuple(r3_rail_check.nonzero()[0])
        numRails=31 # number of rails plus one
              
        if (len(r2_ind)>0) and (len(r3_ind)==0):
            minr2=min(r2_ind)+1
            maxr2=max(r2_ind)+1 
            return (61-maxr2), (61-minr2)            
        elif (len(r3_ind)>0) and (len(r2_ind)==0):
            minr3=min(r3_ind)+1
            maxr3=max(r3_ind)+1
            return (31-maxr3), (31-minr3)            
        elif (len(r3_ind)>0) and (len(r2_ind)>0):
            print "cross rails"
            minr3=min(r3_ind)+1
            maxr3=max(r3_ind)+1
            minr2=min(r2_ind)+1
            maxr2=max(r2_ind)+1
            rail1,rail2=cross_rails(c,img_rgb,thresh_img,minr3,maxr3,minr2,maxr2)
            return rail1,rail2
##            minr3=min(r3_ind)+1
##            minr2=min(r2_ind)+1
##            maxr3=max(r3_ind)+1
##            maxr2=max(r2_ind)+1 
##            if minr2<=minr3:
##                rail1=61-minr2                
##            else:
##                rail1=31-minr3
##            if maxr2>=maxr3:
##                rail2=61-maxr2
##            else:
##                rail2=31-maxr3
##            
##            return rail1,rail2            
        else:
            return 1,1
            
        

def check(x1,y1,x2,y2,x,y):
    if (x1<=x<=x2) and (y1<=y<=y2):
        return 1
    else:
        return 0


##################################################################    
############# End of Function Defining Section ###################
##################################################################
##################################################################


##
##figPathR = 'C:\Python27\shape-detection\shape-detection\snapshots\R'
##figPathB = 'C:\Python27\shape-detection\shape-detection\snapshots\B'
##figSavePath = 'C:\Python27\shape-detection\shape-detection\Match'
####
##capTemp=r'C:\Python27\shape-detection\shape-detection\templates\caps'
##icTemp=r'C:\Python27\shape-detection\shape-detection\templates\ic'
##ledTemp=r'C:\Python27\shape-detection\shape-detection\templates\led'
##resTemp=r'C:\Python27\shape-detection\shape-detection\templates\resistors'
####
##capPath=r'C:\Python27\shape-detection\shape-detection\snapshots\R'
##
##calibrate: camera looks at the blank breadboard 
####
####print getattr(cv2,"CAP_PROP_BRIGHTNESS")
####print getattr(cv2,"CAP_PROP_EXPOSURE")
####print getattr(cv2,"CAP_PROP_GAIN")
##
##cv2.namedWindow("preview")
##vc = cv2.VideoCapture(0)
####vc.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
####vc.set(cv2.CAP_PROP_EXPOSURE,0.25)
####vc.set(cv2.CAP_PROP_BRIGHTNESS,160)
####vc.set(cv2.CAP_PROP_CONTRAST,40)
##calFlag=1
##
##capFlag=1
##
##if calFlag==1:
##    if capFlag==0:
##        picName="fig2019-01-12 120559R.png"
##        tempName= os.path.join(capPath, picName)
##        frame = cv2.imread(tempName)
##    else:
##        if vc.isOpened(): # try to get the first frame
##            rval, frame = vc.read()      
##        else:
##            rval = False
##        while rval:
##                cv2.imshow("preview", frame)
##                rval, frame = vc.read()
##                key = cv2.waitKey(20)
##                if key == 27: # exit on ESC
##                    cv2.destroyAllWindows()
##                    vc.release()
##                    break
##                if key == 99:
##                    rval, frame = vc.read()
##                    break
##        ##        if key == 97:
##        ##            time.sleep(5)
##        ##            vc.set(cv2.CAP_PROP_BRIGHTNESS,11)
##        ##            time.sleep(5)
##        ##        vc.set(cv2.CAP_PROP_EXPOSURE,16)
##        ##        time.sleep(5)
##        ##        vc.set(cv2.CAP_PROP_GAIN,15)
##        ##        time.sleep(5)
##    calibrate(frame)
##    cv2.destroyAllWindows()
##    vc.release()
##
##
###initiate video feed after the calibration is finished
####rect_r=np.array(pickle.load(open("rect_r.obj","rb"))).flatten()
##r2_rails=pickle.load(open("r2_rails.obj","rb"))
##r3_rails=pickle.load(open("r3_rails.obj","rb"))
##cv2.namedWindow("backgroundSubtract")
##cv2.namedWindow("preview")
##vc = cv2.VideoCapture(0)
##fgbg=cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=75, detectShadows=False)
##subFlag=1
##if vc.isOpened(): # try to get the first frame
##    rval, frame = vc.read()   
##    fgmask = fgbg.apply(frame)    
##else:
##    rval = False
##while rval:
##            
##            frame2=frame.copy()
##
##            for ind in xrange(len(r2_rails)):
##                rect_r2=np.array([r2_rails[ind]],dtype="int32")
##                cv2.drawContours(frame2,rect_r2,-1,(200,25,0),1)
##
##                rect_r3=np.array([r3_rails[ind]],dtype="int32")
##                cv2.drawContours(frame2,rect_r3,-1,(200,25,0),1)
##                
##            cv2.imshow("preview", frame2)
##            cv2.imshow("backgroundSubtract",fgmask)
##            rval, frame = vc.read()
##                              
##            if subFlag==1:
##                fgmask = fgbg.apply(frame)
##            key = cv2.waitKey(20)    
##            if key == 27: # exit on ESC
##                cv2.destroyAllWindows()
##                vc.release()
##                break
##            if key == 98: #pause the videoCapture for user to place component
##                subFlag=0
##            if key == 99: #component has been placed. Now time to detect and locate
##                #vc.release()
##                subFlag=1
##                #wait 3 seconds for the camera to focus
##                time.sleep(3)
##                rval, frame = vc.read() #read the frame                
##                fgmask = fgbg.apply(frame)                
##                now=datetime.datetime.now()
##                # save the real frame and background subtracted frame
##                fileName = 'fig' + now.strftime("%Y-%m-%d %H%M%S") 
##                completeNameB = os.path.join(figPathB, fileName + 'B.png')
##                completeNameR = os.path.join(figPathR, fileName + 'R.png')          
##                snapshot(completeNameB,rval,fgmask)
##                snapshot(completeNameR,rval,frame)
##                #component detection
##                #resCount,capCount,ledCount,icCount=componentDetect(frame, resTemp, capTemp,icTemp,ledTemp)
##                #component location
##                rail1,rail2=componentLocate(fgmask,frame, 1,0,0,0)
##                print rail1,rail2
##                # continue while loop for user to add another component.
##                
##cv2.destroyAllWindows()           
