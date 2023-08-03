'''
Functions
'''
import cv2
import numpy as np
import shapely

def testImported():
    print('Module imported!')
    
def getROI(img, scale):
     '''
     img: image we want take ROI of,
     scale: real world length per pixel,
     returns: [Top_Left_X, Top_Left_Y, Width, Height]
     '''    
     cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
     cv2.resizeWindow('Select ROI', int(img.shape[1]*0.25), int(img.shape[0]*0.25))
     cropCoord = cv2.selectROI('Select ROI', img, showCrosshair=True)
     cropCoord = np.array(cropCoord)
     cv2.destroyWindow('Select ROI')
     #adjust to scale
     cropCoord = np.array(cropCoord)*scale
     
     return cropCoord


def euclidDist(x1, y1, x2, y2):
    '''
    x1: first set of x coordinate(s), 
    y1: first set of y coordinate(s), 
    x2: second set of x coordinate(s), 
    y2: first set of y coordinate(s)
    returns: euclidan distance between points as int or array
    '''
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
 
    
def createNormalLine(x,y, dx, dy):
    xlin = np.linspace(-100,100) 
    ylin = np.linspace(-100,100) 
    
    return xlin*-dy  + x, ylin*dx+ y


def isInImage(r, x, y):
    '''
    r: list or array for ROI like so [Top_Left_X, Top_Left_Y, Width, Height],
    x: column, y: row
    returns: True if point is in ROI, else False
    '''
    xIn, yIn = False, False 
    
    if( x >= r[0] and x <= r[0]+r[2]):
        xIn = True
    
    if y >= r[1] and y <= r[1]+r[3]:
        yIn = True
    
    return xIn and yIn


def gauss1D(size, sigma):
    '''
    size: total length of kernel, must be odd,
    sigma: sigma of gaussian,
    returns: size length array of normalized gaussian kernel
    '''
    size = size+1 if size%2 == 0 else size
    halfLen = (size - 1)/2
    
    filter_range = np.arange(-halfLen, halfLen+1, 1)
    
    gaussFilter  = np.exp(-0.5*(filter_range/sigma)**2)
    gaussFilter = gaussFilter/np.sum(gaussFilter)
    return gaussFilter


def contoursToPoly(cont):
    '''
    cont: list of list of contours from cv2
    returns: list of Shapely Polygons
    '''
    allPolys = []
    for cnt in cont:
        rCont = np.squeeze(cnt, axis=1)                                 
        polyGon = shapely.geometry.Polygon(rCont)
        allPolys.append(polyGon)
        
    
    return allPolys
    

def proccessIntersectionPoint (interPoints, x1, y1):
    '''
    interPoints: shapely intersection geometry, 
    x1: x-coord of point on baseline, 
    y1: y-coord of point on baseline, \n
    returns: x, y of closest inersection point
    '''
    pointType = shapely.get_type_id(interPoints)
    if(pointType == 0):
        return interPoints.x, interPoints.y 
    elif(pointType == 4):
        interPoints = interPoints.geoms
        mx, my = interPoints[0].x, interPoints[0].y
        minDist = euclidDist(x1, y1, mx, my)
        
        for pt in interPoints:
            d = euclidDist(x1, y1, pt.x, pt.y)
            if(d < minDist):
                mx = pt.x
                my = pt.y
                minDist = d
        return mx, my
    elif(pointType == 1):
        all_x, all_y = interPoints.xy
        distances = euclidDist(all_x, all_y, x1, y1)
        
        minIndx = distances.argmin(axis=0) #changed here
        
        return all_x[minIndx], all_y[minIndx]
    elif(pointType == 7):
        allObjects = interPoints.geoms
        mx, my = None, None
        minDist = 100000
        for obj in allObjects:
            nx,ny = proccessIntersectionPoint(obj, x1, y1)
            if(euclidDist(nx,ny,x1,y1) < minDist):
                mx, my = nx, ny
                
        return mx, my
            
def contoursToPolygons(cont):
    '''
    contours returned from cv2
    returns: list of Shapely Polygons
    '''
    allPolys = []
    for cnt in cont:
        rCont = np.squeeze(cnt, axis=1)                                 
        polyGon = shapely.geometry.Polygon(rCont)
        allPolys.append(polyGon)
        
    return allPolys