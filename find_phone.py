import sys
import os
import random
import cv2

##------<SETTINGS>-------
H = 326 #input img height
W = 490 #input img width
area_range = [150, 1500] #range of contour area values likely to contain cell phone
perimeter_range = [50,200] # range of contour perimeter values likely to contain cell phone
random.seed(42) # making results more reproducible/deterministic
##------</SETTINGS>------

def predict(test_img_filename):
    """
    inputs: test_img_filename - string, filename (with path) for the image to be tested
    outputs: (predict_x, predict_y) - tuple, contains predicted normalized (0...1) coordinates for the object of interest (phone in our case)
    """
    image = cv2.imread(test_img_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(gray,(5,5))
    threshold = int((blur.mean() + blur.min())/2) 
    thresholded_img = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)[1] 

    det_x, det_y = [], []
    predict_x, predict_y = 0.0, 0.0
    
    _, contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find all contours in the thresholded image
    if len(contours) == 0: #just in case,  shouldn't happen with this dataset 
        return (0.0, 0.0)
        
    for con in contours:  # pick only contours which satisfy certain constraints (area and perimeter)
        area = cv2.contourArea(con)
        perimeter = cv2.arcLength(con, closed = True)
        (encc_x, encc_y), _ = cv2.minEnclosingCircle(con)
        if (area >= area_range[0]) and (area <= area_range[1]) and (perimeter >= perimeter_range[0]) and (perimeter <= perimeter_range[1]):
            det_x.append(encc_x / float(W))
            det_y.append(encc_y / float(H))
            
    if len(det_x) == 0: #if no suitable contour is detected, randomly pick one of the detected contours
        (encc_x, encc_y), encc_r = cv2.minEnclosingCircle(random.choice(contours))
        predict_x = encc_x / float(W)
        predict_y = encc_y / float(H)
    elif len(det_x) >= 1: #if only one suitable contour is detected, then we're OK. If >1 suitable contour is detected, we're just going to guess and pick the first one
        predict_x = det_x[0]
        predict_y = det_y[0]
    return (predict_x, predict_y)

x, y = predict(sys.argv[1])

print"%f %f" %(x, y)
