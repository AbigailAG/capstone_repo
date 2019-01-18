##import cv2
##import numpy as np
##import os
##import six.moves.urllib as urllib
##import sys
##import tarfile
##import tensorflow as tf
##import zipfile
##from distutils.version import StrictVersion
##from collections import defaultdict
##from io import StringIO
###from matplotlib import pyplot as plt
##from PIL import Image

from cam_mod import *


##
### This is needed since the notebook is stored in the object_detection folder.
##models_path="C:\Python27\shape-detection\shape-detection\models-master\research\object_detection"
###sys.path.append("..")
##from utils import ops as utils_ops
##
##if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
##  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
##
### This is needed to display the images.matplotlib inline
##from utils import label_map_util
##print("hi")
##from utils import visualization_utils as vis_util
### What model to download.
##
##MODEL_NAME = 'component_inference_graph'
##print("hi")
### Path to frozen detection graph. This is the actual model that is used for the object detection.
##PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
##print(PATH_TO_FROZEN_GRAPH)
### List of the strings that is used to add correct label for each box.
##PATH_TO_LABELS = os.path.join('object-detection.pbtxt')
##print(PATH_TO_LABELS)
##NUM_CLASSES = 7
##detection_graph = tf.Graph()
##with detection_graph.as_default():
##  od_graph_def = tf.GraphDef()
##  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
##    serialized_graph = fid.read()
##    od_graph_def.ParseFromString(serialized_graph)
##    tf.import_graph_def(od_graph_def, name='')
##
##    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
##
##def load_image_into_numpy_array(image):
##    (im_width, im_height) = image.size
##    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
##
##
##
##IMAGE_SIZE = (12, 8)
##
##def run_inference_for_single_image(image, graph):
##  with graph.as_default():
##    with tf.Session() as sess:
##      # Get handles to input and output tensors
##      ops = tf.get_default_graph().get_operations()
##      all_tensor_names = {output.name for op in ops for output in op.outputs}
##      tensor_dict = {}
##      for key in [
##          'num_detections', 'detection_boxes', 'detection_scores',
##          'detection_classes', 'detection_masks'
##      ]:
##        tensor_name = key + ':0'
##        if tensor_name in all_tensor_names:
##          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
##              tensor_name)
##      if 'detection_masks' in tensor_dict:
##        # The following processing is only for single image
##        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
##        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
##        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
##        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
##        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
##        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
##        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
##            detection_masks, detection_boxes, image.shape[0], image.shape[1])
##        detection_masks_reframed = tf.cast(
##            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
##        # Follow the convention by adding back the batch dimension
##        tensor_dict['detection_masks'] = tf.expand_dims(
##            detection_masks_reframed, 0)
##      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
##
##      # Run inference
##      output_dict = sess.run(tensor_dict,
##                             feed_dict={image_tensor: np.expand_dims(image, 0)})
##
##      # all outputs are float32 numpy arrays, so convert types as appropriate
##      output_dict['num_detections'] = int(output_dict['num_detections'][0])
##      output_dict['detection_classes'] = output_dict[
##          'detection_classes'][0].astype(np.uint8)
##      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
##      output_dict['detection_scores'] = output_dict['detection_scores'][0]
##      if 'detection_masks' in output_dict:
##        output_dict['detection_masks'] = output_dict['detection_masks'][0]
##  return output_dict
##
##my_path = os.path.abspath(os.path.dirname(__file__))
##testPath=r'cam_data\snapshots\R'
##for templateFile in os.listdir(os.path.join(my_path,testPath)):
##    tempName= os.path.join(my_path,testPath, templateFile)
##    tempName=os.path.join(tempName)
##    image = Image.open(tempName)
##    # result image with boxes and labels on it.
##    image_np = load_image_into_numpy_array(image)
##    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
##    image_np_expanded = np.expand_dims(image_np, axis=0)
##    # Actual detection.
##    output_dict = run_inference_for_single_image(image_np, detection_graph)
##    print(output_dict)
##    # Visualization of the results of a detection.
##    vis_util.visualize_boxes_and_labels_on_image_array(
##      image_np,
##      output_dict['detection_boxes'],
##      output_dict['detection_classes'],
##      output_dict['detection_scores'],
##      category_index,
##      instance_masks=output_dict.get('detection_masks'),
##      use_normalized_coordinates=True,
##      line_thickness=8)
##    cv2.imshow("preview",image_np)
##    cv2.waitKey(0)
##    cv2.imshow("preview",image_np)
##    cv2.waitKey(0)
    
##
##for image_path in TEST_IMAGE_PATHS:
##  image = Image.open(image_path)
##  # the array based representation of the image will be used later in order to prepare the
##  # result image with boxes and labels on it.
##  image_np = load_image_into_numpy_array(image)
##  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
##  image_np_expanded = np.expand_dims(image_np, axis=0)
##  # Actual detection.
##  output_dict = run_inference_for_single_image(image_np, detection_graph)
##  print(output_dict)
##  # Visualization of the results of a detection.
##  vis_util.visualize_boxes_and_labels_on_image_array(
##      image_np,
##      output_dict['detection_boxes'],
##      output_dict['detection_classes'],
##      output_dict['detection_scores'],
##      category_index,
##      instance_masks=output_dict.get('detection_masks'),
##      use_normalized_coordinates=True,
##      line_thickness=8)
##  cv2.imshow("preview",image_np)
##  cv2.waitKey(0)
####  
####  plt.figure(figsize=IMAGE_SIZE)
## ## plt.imshow(image_np)
##
##

figPathR = 'cam_data\snapshots\R'
figPathB = 'cam_data\snapshots\B'
figSavePath = 'cam_data\Match'
##
capTemp=r'cam_data\templates\caps'
icTemp=r'cam_data\templates\ic'
ledTemp=r'cam_data\templates\led'
resTemp=r'cam_data\templates\resistors'
my_path = os.path.abspath(os.path.dirname(__file__))
##
framePath=r'C:\Python27\shape-detection\shape-detection\capstone_repo\capstone_repo\cam_data\snapshots\R'
frameName="fig2019-01-17 133929R.png"
fgmaskName="fig2019-01-17 133929B.png"
fgmaskPath=r'C:\Python27\shape-detection\shape-detection\capstone_repo\capstone_repo\cam_data\snapshots\B'
capFlag=0
################################
    ####################################
    ###########################

LETSBUILD = True
while(LETSBUILD):

    LINDSAYISGO = True
    while(LINDSAYISGO):
        if capFlag==0:          
          frameFilePath= os.path.join(framePath, frameName)
          fgmaskFilePath= os.path.join(fgmaskPath, fgmaskName)
          frame = cv2.imread(frameFilePath)
          fgmask = cv2.imread(fgmaskFilePath)
          compType=componentDetect(frame, fgmask, resTemp, capTemp,icTemp,ledTemp,my_path)
        else:          
          #load bb points
          bbpoints=pickle.load(open("cam_data/bbpoints.obj","rb"))
          #rect_r=np.array(pickle.load(open("cam_data/rect_r.obj","rb"))).flatten()
          r2_rails=pickle.load(open("cam_data/r2_rails.obj","rb"))
          r3_rails=pickle.load(open("cam_data/r3_rails.obj","rb"))

          #snapshot
          ###initiate video feed after the calibration is finished
          cv2.namedWindow("backgroundSubtract")#defining a window
          cv2.namedWindow("preview")
          vc = cv2.VideoCapture(0) #start and define video capture
          vc.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
          vc.set(cv2.CAP_PROP_AUTOFOCUS,1)
          vc.set(cv2.CAP_PROP_BRIGHTNESS,150)
          vc.set(cv2.CAP_PROP_CONTRAST,45)
          fgbg=cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=75, detectShadows=True) #setting function parameters, not actual frames
          subFlag=1 #later
          if vc.isOpened(): # try to get the first frame - check if camera is opened
              rval, frameRaw = vc.read()   #read the frame for fun!
          else:
              rval = False #if it doesn't work, we have problems....deal with those later!
          while rval:

              #perform transformations
              ## four point transform image
              rval, frameRaw= vc.read()
              frame=four_point_transform(frameRaw,bbpoints)
              railOver=frame.copy()
              for ind in xrange(len(r2_rails)):
                  rect_r2=np.array([r2_rails[ind]],dtype="int32")
                  cv2.drawContours(railOver,rect_r2,-1,(200,25,0),1)

                  rect_r3=np.array([r3_rails[ind]],dtype="int32")
                  cv2.drawContours(railOver,rect_r3,-1,(200,25,0),1)
              cv2.imshow("preview", railOver)
              if subFlag==1: #subFlag means we want to get video feed
                  fgmask = fgbg.apply(frame) #background subtract from frame
                  cv2.imshow("backgroundSubtract",fgmask)

              key = cv2.waitKey(20) #check for userkey every whatever # of frames?
              if key == 27: # exit on ESC - get out of jail free
                  cv2.destroyAllWindows()
                  vc.release()
                  break
              if key == 98: #pause the videoCapture for user to place component
                  subFlag=0 #stop taking video feed background subtract - in future may stop video here then background subtract after


              if key == 99: #component has been placed. Now time to detect and locate
                  subFlag=1
                  #wait 3 seconds for the camera to focus
                  time.sleep(3)
                  rval, frameRaw = vc.read() #read the frame
                  for i in xrange(1,15):
                      rval, frameRaw = vc.read() #read the frame
                      key = cv2.waitKey(20)
                      frame=four_point_transform(frameRaw,bbpoints)
                      fgmask = fgbg.apply(frame) #subtract
                      cv2.imshow("backgroundSubtract",fgmask)
                  now=datetime.datetime.now() #getting time for filename
                  # save the real frame and background subtracted frame
                  my_path = os.path.abspath(os.path.dirname(__file__))
                  fileName = 'fig' + now.strftime("%Y-%m-%d %H%M%S")
                  #completeNameB = os.path.join(figPathB, fileName + 'B.png')
                  completeNameB=figPathB+ '\\' + fileName+'B.png'
                  #completeNameR = os.path.join(figPathR, fileName + 'R.png')
                  completeNameR=figPathR+'\\'+fileName+'R.png'
                  snapshot(completeNameB,rval,fgmask) #save snapshot
                  snapshot(completeNameR,rval,frame)
                  #component detection
                  compType=componentDetect(frame, fgmask, resTemp, capTemp,icTemp,ledTemp,my_path)
                  #component location
                  #railOne,railTwo=componentLocate(fgmask,frame, 1,0,0,0)
                  #print railOne,railTwo
                  # continue while loop for user to add another component.
          vc.release()
          cv2.destroyAllWindows()
        







    


