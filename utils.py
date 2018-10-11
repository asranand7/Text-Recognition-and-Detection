import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.models import load_model

import os
from PIL import Image
import networkx as nx

model = load_model('mod1.h5') # This is the model used for differentiating the text and non-text regions


def image_denoising(img):
    '''
    Input : Any image
    Output : Returns the image after denoising
    '''
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return dst



def adaptive_thresholding(img):
    '''
    Input : Any image 
    Output : Returns image after otsu method of image contrast enhancement
    
    '''
    
    ret2,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return th1



def is_text(img):
    '''
    Input: any image
    Output: Gives the probability of presence of text in the image
    
    '''
    img = cv2.resize(img, (32,32))
    
    img= img.reshape(1,32, 32, 1)
    
    return model.predict_proba(img,verbose = 0)[0][1]
    




def extend_boxes(text_boxes,image_shape):
	#convert from the [x,y, width,height]  bounding box format to the [xmin,ymin,xmax,ymax] format for convenience
	#shape of text boxes in N X 4 where N is the no. of rectangular boxes
	xmin = text_boxes[:,0]
	ymin = text_boxes[:,1]
	xmax = xmin +  text_boxes[:,2] -1
	ymax = ymin + text_boxes[:,3] -1 

	#Expand the bounding boxes by small amount
	expansionAmount = 0.02
	xmin = (1-expansionAmount)*xmin
	ymin = (1-expansionAmount)*ymin
	xmax = (1+expansionAmount)*xmax
	ymax = (1+expansionAmount)*ymax

	#clip the bounding boxes to be in the image region
	xmin_idx = (xmin < 1)
	xmin[xmin_idx] = 1
	ymin_idx = (ymin < 1)
	ymin_idx = 1
	xmax_idx = (xmax > image_shape[1])
	xmax[xmax_idx] = image_shape[1]
	ymax_idx = (ymax > image_shape[0])
	ymax[ymax_idx] = (image_shape[0])
	extendedboxes = np.array([xmin,ymin,xmax,ymax]).T

	return extendedboxes



def bool_rect_intersect(A, B):
	#check whether the boxes intersect
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3])


def BBoverlapRatio(A, B):
    in_ = bool_rect_intersect(A, B)
    # if the boxes do not intersect then return zero
    if not in_:
        return 0
    else:
        left = max(A[0], B[0]);
        top = max(A[1], B[1]);
        right = min(A[2], B[2]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        surface_intersection = (intersection[2]-intersection[0])*(intersection[3]-intersection[1]);
        surface_A = (A[2]- A[0])*(A[3]-A[1]) + 0.0;
        return surface_intersection / surface_A # return the overlapratio between two box



def merge_boxes(boxes):
	''' 
	Input : Boxes as an numpy array 
	* Shape of the boxes is N X 4 where N is the no. of boxes

	This function merges the overlapping boxes into one single boxes

	Output : Returns the merged boxes '''
	
	m = len(boxes[:,0])
	overlapRatio = np.ones((m,m)) # calculate the overlap ratio for every pair of boxes
	for i in range(m):
	    for j in range( m ):
	        overlapRatio[i][j] = BBoverlapRatio(boxes[i],boxes[j])

	n = overlapRatio.shape[0]
	#set the overlapratio of the rectangle with itself to be zero
	overlapRatio[:,0] = 0

	g = nx.Graph(overlapRatio) # create a graph of the overlapratio's
	componentIndices = nx.connected_components(g) # find all the connected component i.e all the indices of the boxes which are overlapping with each other in one set
	# componentIndices gives a list of sets. Each set contains the indices of boxes wich are overalapping with each other
	a =sorted(nx.connected_components(g), key = len, reverse=True)

	# Now we join all the overlapping boxes into one single box
	textboxes = []
	for i in range(len(a)):
		component = []
		indices = list(a[i])
		for j in range(len(a[i])):
		    component.append(boxes[indices[j]])

		component = np.array(component)
		xmin = np.min(component[:,0])
		ymin = np.min(component[:,1])
		#xmin_index = np.argmin(component[:,0])
		#ymin = component[:,1][xmin_index]
		xmax = np.max(component[:,2])
		ymax = np.max(component[:,3])
		#xmax_index = np.argmax(component[:,2])
		#ymax = component[:,3][xmax_index]

		textboxes.append([xmin,ymin,xmax,ymax])


	textboxes = np.array(textboxes)
	return textboxes

	    
	    

	







