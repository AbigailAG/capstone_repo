# import the necessary packages
# import the necessary packages
##from pyimagesearch.zernikemoments import ZernikeMoments
from imutils.paths import list_images
import numpy as np
import argparse
import pickle
import imutils
import cv2
import os 
import mahotas
# import the necessary packages
from scipy.spatial import distance as dist
 
class Searcher:
	def __init__(self, index):
		# store the index that we will be searching over
		self.index = index
 
	def search(self, queryFeatures):
		# initialize our dictionary of results
		results = {}
 
		# loop over the images in our index
		for (k, features) in self.index.items():
			# compute the distance between the query features
			# and features in our index, then update the results
			d = dist.euclidean(queryFeatures, features)
			results[k] = d
 
		# sort our results, where a smaller distance indicates
		# higher similarity
		results = sorted([(v, k) for (k, v) in results.items()])
 
		# return the results
		return results



	    
class ZernikeMoments:
	def __init__(self, radius):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius
 
	def describe(self, image):
		# return the Zernike moments for the image
		return mahotas.features.zernike_moments(image, self.radius)


###############################
###############################
###############################
figPathR = 'cam_data\snapshots\R'
figPathB = 'cam_data\snapshots\B'
figSavePath = 'cam_data\Match'
##
capTemp=r'cam_data\templates\caps'
icTemp=r'cam_data\templates\ic'
ledTemp=r'cam_data\templates\led'
resTemp=r'cam_data\templates\resistors\C'
resTemp_cnt=r'cam_data\templates\resistors\cnt'

my_path = os.path.abspath(os.path.dirname(__file__))
##
framePath=r'C:\Python27\shape-detection\shape-detection\capstone_repo\capstone_repo\cam_data\snapshots\R'
frameName="fig2019-01-19 140100R.png"
fgmaskName="fig2019-01-17 133929B.png"
fgmaskPath=r'C:\Python27\shape-detection\shape-detection\capstone_repo\capstone_repo\cam_data\snapshots\B'
capFlag=0

frameFilePath= os.path.join(framePath, frameName)
fgmaskFilePath= os.path.join(fgmaskPath, fgmaskName)
frame = cv2.imread(frameFilePath)
warp=frame.copy()
fgmask = cv2.imread(fgmaskFilePath)



sortPath=resTemp
#storePath=os.path.join(my_path, resTemp_cnt)
 
# initialize our descriptor (Zernike Moments with a radius
# of 21 used to characterize the shape of our pokemon) and
# our index dictionary
radius=35
desc = ZernikeMoments(radius)
index = {}

# loop over the sprite images
for spritePath in list_images(sortPath):
	# parse out the pokemon name, then load the image and
	# convert it to grayscale
	pokemon = spritePath[spritePath.rfind("/") + 1:].replace(".png", "")
	image = cv2.imread(spritePath)
	h,w=image.shape[:2]
	print h,w
	image=image[int(h*0.2):int(h*0.8),int(w*0.2):int(w*0.8)]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray=cv2.bitwise_not(gray)
	## blur the gray image
##	kernel_size = 5
##        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
##        cv2.imshow("preview",blur_gray)
##        cv2.waitKey(0)
	## create binary image
        ret, thresh = cv2.threshold(gray, 110,255, cv2.THRESH_BINARY)
        #cv2.imshow("preview",thresh)
       # cv2.waitKey(0)
        ## erode and dilate the threshed image to remove noise
        thresh=cv2.erode(thresh, None, iterations=2)
        thresh=cv2.dilate(thresh, None, iterations=1)
        #cv2.imshow("preview",thresh)
        #cv2.waitKey(0)
        ## create canny edge image
        low_threshold = 90  
        high_threshold =250 
        edges = cv2.Canny(thresh, low_threshold, high_threshold,None,3)
        #cv2.imshow("preview",edges)
        edgesCopy=edges.copy()
        #cv2.waitKey(0)
        image=edges.copy()


	# initialize the outline image, find the outermost
	# contours (the outline) of the pokemon, then draw
	# it
	outline = np.zeros(image.shape, dtype = "uint8")
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c=max(cnts,key=cv2.contourArea)
	#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	cv2.drawContours(outline, [c], -1, 255, -1)
        cv2.imshow("preview",outline)
        cv2.waitKey(0)

	# compute Zernike moments to characterize the shape
	# of pokemon outline, then update the index
	moments = desc.describe(outline)
	index[pokemon] = moments
    # write the index to file
print index
filehandler = open("cam_data/templates/resistors/cnt/index.txt", "wb")
pickle.dump(index,filehandler)
filehandler.close()


#### adjust intensity of pixels....
#####################################
#######################################

# convert the warped image to grayscale and then adjust
# the intensity of the pixels to have minimum and maximum
# values of 0 and 255, respectively
##warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
##warp = exposure.rescale_intensity(warp, out_range = (0, 255))
##cv2.imshow("preview",warp)
##cv2.waitKey(0)
### the pokemon we want to identify will be in the top-right
### corner of the warped image -- let's crop this region out
##(h, w) = warp.shape
##(dX, dY) = (int(w * 0.4), int(h * 0.45))
##crop = warp[10:dY, w - dX:w - 10]
 
# show our images
##cv2.imshow("image", image)
##cv2.imshow("edge", edged)
##cv2.imshow("warp", imutils.resize(warp, height = 300))
##cv2.imshow("crop", imutils.resize(crop, height = 300))
##cv2.waitKey(0)


# load the query image, convert it to grayscale, and
# resize it
image = warp.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("preview",image)
cv2.waitKey(0)
#image = imutils.resize(image, width = 64)
# threshold the image
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
	cv2.THRESH_BINARY_INV, 11, 7)
cv2.imshow("preview",thresh)
cv2.waitKey(0)
# initialize the outline image, find the outermost
# contours (the outline) of the pokemon, then draw
# it
outline = np.zeros(image.shape, dtype = "uint8")
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
cv2.drawContours(outline, [cnts], -1, 255, -1)
cv2.imshow("preview",outline)
cv2.waitKey(0)
# compute Zernike moments to characterize the shape of
# pokemon outline
desc = ZernikeMoments(radius)
queryFeatures = desc.describe(outline)
 
# perform the search to identify the pokemon
searcher = Searcher(index)
results = searcher.search(queryFeatures)
print "That pokemon is: %s" % results[0][1].upper()
 
# show our images
cv2.imshow("image", image)
cv2.imshow("outline", outline)
cv2.waitKey(0)
