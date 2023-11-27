import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from shutil import rmtree
import shapely
import Module.Functions as fb
import pandas as pd

#==========================FUNCTIONS=====================================
def drawLabel(image, contour, num):
    '''
    draws label num directly on image at the center of mass of the contour
    '''
    M = cv2.moments(contour)
    cx= int(M['m10']/M['m00'])
    cy= int(M['m01']/M['m00'])
    cv2.putText(image, text= str(num), org=(cx-5,cy),
        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0),
        thickness=3, lineType=cv2.LINE_AA)
    
    
def leftToRight(contours):
    '''
    return the input array of contours sorted from left to right according to their 
    center of mass
    '''
    com_x = []
    
    for c in contours:
        M = cv2.moments(c)
        com_x.append(M['m10']/M['m00'])
        
    return np.argsort(com_x)


def getSlopes(points):
    '''
    points: [x,y,width, height] of rectangle 
    returns: vertical slope, horizontal slope
    '''
    vertical = False
    
    #check if we have vertical line
    if points[1][0]-points[0][0] == 0:
        s1 = None
        vertical = True
    else:
        s1 = (points[1][1] - points[0][1])/(points[1][0]-points[0][0])
        
     #check if we have vertical line    
    if points[2][0] - points[1][0] == 0:
        s2 = None
        vertical = True
    else:      
        s2 = ( points[2][1] - points[1][1])/(points[2][0] - points[1][0])
    
    #always return vertical line first
    if vertical:
        if(s1 is None):
            return s1, s2 
        else:
            return s2, s1 
    else: 
        if abs(s1) >= abs(s2):
            return s1, s2
        else:
            return s2, s1
    

def getLinePoints(x1, y1, lenP, slope):
    '''
    x1,y1: center coordinate of desired line 
    lenP: full length of line 
    slope: slope of line
    
    returns x,y arrays 
    '''
    
    if slope is None:
        x = np.full(shape=(int(lenP*2)), fill_value=x1)
        y = np.linspace(-(lenP/2),lenP/2, int(lenP*2)) + y1
        return x, y 
    else:        
        x = np.linspace(-(lenP/2),lenP/2, int(lenP*2)) + x1
        b = -slope*x1 +  y1
        
        return x, x*slope + b


def getPointsofObject(interPoints):
    '''
    interPoints: intersection geometry returned by shapely
    returns: x,y coordinate of point
    '''
    pointType = shapely.get_type_id(interPoints)
    x,y = [], []
    
    if(pointType == 4):
        interPoints = interPoints.geoms
         
        for p in interPoints:
            x.append(p.x)
            y.append(p.y)
        
    return x,y


def getMaxDist(x,y):
    '''
    x,y : coordinate arrays
    returns: maximum distance between points
    '''
    maxDist = -1
    
    for i in range(len(x)):
        temp = fb.euclidDist(x, y, x[i], y[i])
        
        if(np.max(temp) > maxDist):
            maxDist = np.max(temp)
            
    
    return maxDist


def calcDistance(x_points, y_points, w, s, realContour):
    allWidths = []
    
    for p in range(len(x_points)):
        #create horizontal line
        nx, ny = getLinePoints(x_points[p], y_points[p], w, s)
        stack = np.stack((nx, ny), axis=-1)
        lineString = shapely.geometry.LineString(stack)
        
        if(lineString.intersects(realContour)):
            interPoints = lineString.intersection(realContour)
            a,b = getPointsofObject(interPoints)
           
            allWidths.append(getMaxDist(a, b))
            
            # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            # plt.gca().invert_yaxis()
            # plt.plot(x_points, y_points)
            # plt.plot(nx,ny)
            # plt.plot(*realContour.xy)
            # x_left, x_right = plt.gca().get_xlim()
            # y_low, y_high = plt.gca().get_ylim()
            # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))/aspect)
            # plt.show()
            
            
    return np.array(allWidths)
    

def getProfile(img):
    '''
    img: grayscale image 
    returns: array of vertical intensity profile of image. 
    array length corresponds to image width and each value is the average grayscale value along the vertical in the image
    '''
    intensityProfile = []
    
    for i in range(img.shape[1]):
        col = img[:, i]
        intensityProfile.append(np.average(col))
        
    return intensityProfile


def getMask(img, r, c, width, height):
    '''
    r,c are essentially the coordinate for the top left corner of the rectange
    r: row to start, aka top
    c: column to start, aka left
    returns: rectangular mask
    '''
    blackBoard = np.zeros(shape=img.shape, dtype=np.uint8)    
    blackBoard[r:r+height, c:c+width] = 255
    
    return blackBoard
    
    
def cropImageCoords(img, col, width, padding):
    '''
    return cropped Image,  [left most column, full width]
    '''
    cs, ce = 0,0
    
    if(col < 0):
        cs = 0
    else: 
        cs = col

    if(col + width + 2*padding >= img.shape[1]):
        ce = img.shape[1]-1
    else:
        ce = col + width + 2*padding
         
    return img[:, cs:ce],[cs,ce-cs]


def findRegoliths(img, d, vSearch):
    '''
    img: opencv array
    d: padding for left, right, top, bottom when cropping
    vSearch: boolean representing if the image is has been rotated or not
    returns: cropped image(s) array, cropped coordinates array [[leftmost column, full wdith]..]
    '''
    allCoordinates = []
    croppedRegions = []
    leftPointer = 0
    p = getProfile(img)
    
    mi = np.min(p)
    mx = np.max(p)
    
    if vSearch:
        th = (mi+mx)/2
    else:
        th = mx*0.95

    
    for j in range(len(p)):
        if (leftPointer<1 and p[j]<th):
            leftPointer=j
      
        if (leftPointer>0 and p[j]>th):
            crp, coords = cropImageCoords(img, leftPointer-d, j-leftPointer, d)
            croppedRegions.append(crp)
            allCoordinates.append(coords)
            leftPointer = 0
            
    return croppedRegions, allCoordinates


def findRegolithConoturs(colour_img):
    allContours = []
    imageObj = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)
    imageObj = cv2.bilateralFilter(imageObj, 35, 50, 50)
    
    #get crops of interest
    verticalCrops, column_coords = findRegoliths(imageObj, span, True)
    for i in range(len(verticalCrops)):        
        c_coord = column_coords[i]
        
        piece = verticalCrops[i]
        piece = np.transpose(np.array(piece))
        
        horizontalCrop, row_coords = findRegoliths(piece, span, False)
        
        for m in range(len(horizontalCrop)):            
            r_coord = row_coords[m]
            mask = getMask(imageObj, r_coord[0], c_coord[0], c_coord[1], r_coord[1])
            
            newImg = np.bitwise_and(imageObj,mask)
            r,c = np.where(newImg==0)
            newImg[r,c] = 255
            newImg = cv2.medianBlur(newImg, 9)
            (T, threshInv) = cv2.threshold(newImg, 0, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            
            contours, _ = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            allContours.extend(contours)

    return allContours


def createDir(root, folderName):
    '''
    creates new folder if it doesn't exist
    returns: new folder's path
    '''
    newPath = os.path.join(root, folderName)
    if not os.path.exists(newPath):
        os.makedirs(newPath)
        
    return newPath
#================================MAIN=========================================
imagePath="C:/Users/v.jayaweera/Documents/Tim/Porosity/PNG"
destPath = createDir(imagePath, "Dimension_Routine_Output")
excelPath = destPath

dirPictures = os.listdir(imagePath)
acceptedFileTypes = ["jpg", "png", "tif"]

span = 35
scale = 0.05
df = pd.DataFrame(columns=['Image_Name', 'Position', 'Average_Height (mm)', 'Max_Height (mm)', 'Average_Width (mm)', 'Max_Width  (mm)', 'Area  (mm^2)'])

# create list of image names
images = []
for i in range(len(dirPictures)):
    image_name = dirPictures[i]
    
    if image_name.split('.')[-1].lower() in acceptedFileTypes:
        images.append(image_name)
        
# start of the loop to remove the background
for i in images:
    img = cv2.imread(imagePath + '/' + i) # load image
    h, w = img.shape[:2]
    aspect = w/h
        
    regolith_contours = findRegolithConoturs(img)
    
    #sort contours to appear from left to right
    sortedIndex = leftToRight(regolith_contours)
    regolith_contours = [newK for _, newK in sorted(zip(sortedIndex, regolith_contours))]
    
    contourCounter = 1
    for j in range(len(regolith_contours)):
        cont = regolith_contours[j]
        
        #Draw rect
        rect = cv2.minAreaRect(cont)
        height = rect[1][0]
        width = rect[1][1]
        
        #get coords of box
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    
        #get center of mass of contour
        M = cv2.moments(cont)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

    
        #create shapely Object for regolith
        rCont = np.squeeze(cont, axis=1)
        polyLine = shapely.geometry.LineString(rCont)
  
        #get slopes of line
        vslope, hslope = getSlopes(box)
        
        #generate line going length wise
        vx, vy = getLinePoints(cx, cy, height, vslope)
    
        #generate line going width wise
        hx, hy = getLinePoints(cx, cy, width, hslope)    
        
    
        # calculate all widths
        reg_widths = calcDistance(vx, vy, width, hslope, polyLine)*scale

        # calculate all heights
        reg_heights = calcDistance(hx, hy, height, vslope, polyLine)*scale
      
        
        if(len(reg_heights) > 0 and len(reg_widths) > 0):
            cv2.drawContours(img, [cont], -1, (0,255,0), 3)
            drawLabel(img, cont, contourCounter)
            print("Image: ", i)
            print("Average height: ", np.average(reg_heights))
            print("Max height: ", np.max(reg_heights))
            print("Average width: ", np.average(reg_widths))
            print("Max width: ", np.max(reg_widths))
            print("Area: ", cv2.contourArea(cont)*scale**2)
            print("\n")
            
            
            #create row entry
            df.loc[len(df)] = [i, contourCounter,np.average(reg_heights), np.max(reg_heights),
                               np.average(reg_widths), np.max(reg_widths), cv2.contourArea(cont)*scale**2]
            
            contourCounter = contourCounter + 1
    
    cv2.imwrite(destPath  + '/' + i, img)
    print("Wrote image, ", i) 
    
df.sort_values(["Image_Name", "Position"], inplace=True)
df.to_excel(excelPath+'/'+"Avearge_Dimensions.xlsx", index=False)

    