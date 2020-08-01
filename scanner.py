from cv2 import cv2
import numpy as np
import mapper
image=cv2.imread("t4.jpg")  #upload files only in jpg format
image=cv2.resize(image,(1366,768)) #resize to your PC's screen size for better image quality
orig=image.copy()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
cv2.imshow("Title",gray)

blurred=cv2.GaussianBlur(gray,(5,5),0)  #Blurred image
#cv2.imshow("Blur",blurred)

edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
#cv2.imshow("Canny",edged)


contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  
contours=sorted(contours,key=cv2.contourArea,reverse=True)

#the loop extracts the boundary contours of the page
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.08*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapper.mapp(target) #find endpoints of the sheet

pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  #map to 800*800 target window

op=cv2.getPerspectiveTransform(approx,pts)  
dst=cv2.warpPerspective(orig,op,(800,800))


cv2.imshow("Scanned",dst)
cv2.waitKey(0)
