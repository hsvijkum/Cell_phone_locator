import sys, os, cv2

def predict(test_img_filename):
    """
    inputs: test_img_filename - string, filename (with path) for the image to be tested
    outputs: (predict_x, predict_y) - tuple, contains predicted normalized (0...1) coordinates for the object of interest (phone in our case)
    """
    image = cv2.imread(test_img_filename)
    H, W = image.shape[0:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(gray,(5,5))
    threshold = int( (blur.mean() + blur.min())/2 )
    thresholded_img = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)[1]
    _, contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find all contours in the thresholded image
    
    if len(contours) == 0: #just in case,  shouldn't happen with this dataset 
        return (0.0, 0.0)

    best_score = -10**8
    target_area, target_perimeter = 555, 75 #contour area & perimeter values likely to contain a cell phone
    for con in contours:  
        
        area, perimeter = cv2.contourArea(con), cv2.arcLength(con, closed = False)
        (encc_x, encc_y), _ = cv2.minEnclosingCircle(con)
        con_score = - abs(area - target_area) - abs(perimeter - target_perimeter)
        if con_score > best_score:
            best_score = con_score
            predict_x, predict_y = encc_x / float(W), encc_y / float(H) 

    return (predict_x, predict_y)

x, y = predict(sys.argv[1])
print("%f %f" %(x, y))
