
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# from rembg import remove, new_session
from shutil import rmtree
import shapely


Threshold = 80

def getSlopes(points):
    

def getLinePoints(x1,y1, slope):
    x = np.linspace(-100,100,1000) + x1
    b = -slope*x1 +  y1
    
    return x, x*slope + b

'''
MAIN
'''

# create list with image names in folder
imagePath="C:/Users/v.jayaweera/Documents/Tim/Average Dimensions"
dirPictures = os.listdir(imagePath)

images = []
for i in dirPictures: 
    if i.split('.')[-1].lower() == "jpg":
        images.append(i)

# create new folder for images without background and new folder for whitened samples
#TODO: make creation directories variable
# try: 
#     os.mkdir("../Without_Background")
# except: 
#     rmtree("../Without_Background")
#     os.mkdir("../Without_Background")

# try: 
#     os.mkdir("../White_Sample")
# except: 
#     rmtree("../White_Sample")
#     os.mkdir("../White_Sample")

# create list to store porosities in
porosities = []

# start of the loop to remove the background
for i in images:
    img = cv2.imread(imagePath + '/' + i) # load image
    h, w = img.shape[:2]
    aspect = w/h
    img = cv2.resize(img, (200, int(200/aspect)), interpolation=cv2.INTER_AREA)
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(imgGray)
    cont, hier = cv2.findContours(inverted, 1,2)
    
        
    mainCont = []
    largestArea = 0  
    for k in cont:
        if(cv2.contourArea(k) > largestArea):
            mainCont = k
    
    rect = cv2.minAreaRect(mainCont)
    M = cv2.moments(mainCont)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    xs,ys = createNormalLine(cx, cy, 3, -34) #createNormalLine(cx,cy, 1, 1.57)
    print(cx,cy)
    contx = mainCont[:,0,0]
    conty = mainCont[:,0,1]
    
    #create shapely Object
    rCont = np.squeeze(k, axis=1)
    polyLine = shapely.geometry.LineString(rCont)
  
    #Reset plots to default figure size
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.gca().invert_yaxis()
    
    plt.plot(*polyLine.xy, 'm.-')
    plt.axline((cx,cy), slope=1/1.5590194017803993,color='g')
    plt.plot(cx,cy,'r.')
    x_left, x_right = plt.gca().get_xlim()
    y_low, y_high = plt.gca().get_ylim()
    plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect)
    plt.show()
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),1)
    cv2.drawContours(img, [mainCont], -1, (0,255,0), 1) 
    cv2.circle(img, (cx,cy), 2, (0,0,255), -1)
    # cv2.rotate(img, rect[3])
    cv2.imshow('g', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()