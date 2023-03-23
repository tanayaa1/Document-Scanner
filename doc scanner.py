import cv2
import numpy as np
##################
widthImg=640
heightImg=480
#####################


frameWidth=640
frameHeight=480
##to use webcam
webcap = cv2.VideoCapture(0)
webcap.set(3, frameWidth) #width has id no.3
webcap.set(4, frameHeight) #height has id no.4
webcap.set(10, 150) #brightness has id 10

def preProcessing(img):
    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1) #size of kernel is 5*5 and sigma x is 1(scale )
    imgCanny= cv2.Canny(imgBlur,200,200)#numbers are threshold intensity of canny imageq
    #dilate to make it thick and reode to make it thin
    kernel=np.ones((5,5))
    imgDial= cv2.dilate(imgCanny,kernel,iterations=1)
    imgThres= cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def getContours(img):
   biggest=np.array([])
   maxArea=0
   contours,hierachy=cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   for cnt in contours:
      area=cv2.contourArea(cnt)
     # print(area)
      if area>5000:
         #cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
         peri= cv2.arcLength(cnt,True)
         print(peri)
         #approimate corner points
         #approx is an array of corners
         approx = cv2.approxPolyDP(cnt,0.02*peri,True)
         if  area>maxArea and len(approx)==4:
             biggest=approx
             maxArea=area
   cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
   return biggest

def reorder(myPoints):
    #by adding points in our array of arrays the smallest will be [0,0] and biggest will be last point
    #by subtracting postitive number is first indes of array and negative is 2nd index of array
    #the points of biggest are(4,1,2) property ie. 4 points in two directioms but 1 is redundant
    myPoints=myPoints.reshape((4,2))
    #the 1 in its shape are not req for addition and subtraction
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1) #add is an array of sum of corner points
    print("add",add)
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    #now take difference to find middle terms
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img,biggest):
    biggest=reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOp= cv2.warpPerspective(img, matrix, (widthImg, heightImg))
#remove 20 pixel from image
    #imgCropped= imgOp[20:imgOp.shape[0]-20,imgOp.shape[1]-20]
   # imgCropped=cv2.resize(imgCropped,(widthImg,heightImg))

    return imgOp


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:

    success, img = webcap.read() #success is a boolean
    cv2.resize(img,(widthImg,heightImg))
    imgContour=img.copy()
    imgThres=preProcessing(img)

    biggest=getContours(imgThres)
    if biggest.size != 0:
       imgWarpped= getWarp(img,biggest)
       print(biggest)

       imgArray=([img,imgContour],
              [imgThres,imgWarpped])
       cv2.imshow("result1", imgWarpped)
    else:
        imgArray = ([img, imgContour],
                    [img, img])
    stackedImages= stackImages(0.6,imgArray)

     #vimg is a series of images in the video
    cv2.imshow("result", stackedImages)

    if cv2.waitKey(1) & 0xFF == ord("q"): #if q is pressed the video stops
        break
