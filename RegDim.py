
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# from rembg import remove, new_session
from shutil import rmtree
import shapely
import Module.Functions as fb


Threshold = 80

def getSlopes(points):
    s1 = (points[1][1] - points[0][1])/(points[1][0]-points[0][0])
    s2 = ( points[2][1] - points[1][1])/(points[2][0] - points[1][0])
    
    if(abs(s1) >= abs(s2)):
        return s1, s2
    else:
        return s2, s1
    

def getLinePoints(x1, y1, lenP, slope):
    x = np.linspace(-(lenP/2),lenP/2, int(lenP*2)) + x1
    b = -slope*x1 +  y1
    
    return x, x*slope + b

def getPointsofObject(interPoints):
    pointType = shapely.get_type_id(interPoints)
    x,y = [], []
    
    if(pointType == 4):
        interPoints = interPoints.geoms
         
        for p in interPoints:
            x.append(p.x)
            y.append(p.y)
        
    return x,y

def getMaxDist(x,y):
    maxDist = -1
    
    for i in range(len(x)):
        temp = fb.euclidDist(x, y, x[i], y[i])
        
        if(np.max(temp) > maxDist):
            maxDist = np.max(temp)
            
    
    return maxDist

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
    # img = cv2.resize(img, (200, int(200/aspect)), interpolation=cv2.INTER_AREA)
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(imgGray)
    cont, hier = cv2.findContours(inverted, 1,2)
    
        
    mainCont = []
    largestArea = 0  
    for k in cont:
        if(cv2.contourArea(k) > largestArea):
            mainCont = k
    
    #Draw rect
    rect = cv2.minAreaRect(mainCont)
    height = rect[1][0]
    width = rect[1][1]
    
    #get coords of box
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    #get center of mass of contour
    M = cv2.moments(mainCont)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    
    #create shapely Object for regolith
    rCont = np.squeeze(k, axis=1)
    polyLine = shapely.geometry.LineString(rCont)
  
    #generate line going length wise
    vslope, hslope = getSlopes(box)
    vx, vy = getLinePoints(cx, cy, width, vslope)
    
    # calculate all widths
    reg_widths = []
    for p in range(len(vx)):
        #create horizontal line
        nx, ny = getLinePoints(vx[p], vy[p], width, hslope)
        stack = np.stack((nx, ny), axis=-1)
        lineString = shapely.geometry.LineString(stack)
        
        if(lineString.intersects(polyLine)):
            interPoints = lineString.intersection(polyLine)
            a,b = getPointsofObject(interPoints)
           
            reg_widths.append(getMaxDist(a, b))
            # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            # plt.gca().invert_yaxis()
            # plt.plot(vx,vy)
            # plt.plot(nx,ny)
            # plt.plot(*polyLine.xy)
            # x_left, x_right = plt.gca().get_xlim()
            # y_low, y_high = plt.gca().get_ylim()
            # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))/aspect)
            # plt.show()
            
    
    
    #generate line going width wise
    hx, hy = getLinePoints(cx, cy, width, hslope)    
        
    #calculate all heights
    reg_heights = []
    for p in range(len(vx)):
        #create vertical line
        nx, ny = getLinePoints(hx[p], hy[p], width, vslope)
        
        #create Shapely geo object for line
        stack = np.stack((nx, ny), axis=-1)
        lineString = shapely.geometry.LineString(stack)
        
        if(lineString.intersects(polyLine)):
            interPoints = lineString.intersection(polyLine)
            a,b = getPointsofObject(interPoints)
           
            reg_heights.append(getMaxDist(a, b))
            # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            # plt.gca().invert_yaxis()
            # plt.plot(vx,vy)
            # plt.plot(nx,ny)
            # plt.plot(*polyLine.xy)
            # x_left, x_right = plt.gca().get_xlim()
            # y_low, y_high = plt.gca().get_ylim()
            # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))/aspect)
            # plt.show()
    
    
    print("Average height: ", np.average(reg_heights))
    print("Average width: ", np.average(reg_widths))
    print("Area: ", cv2.contourArea(mainCont))
   
    