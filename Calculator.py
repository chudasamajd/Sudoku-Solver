import cv2
import numpy as np
import math

vid = cv2.VideoCapture(0)

while vid.isOpened():
    flag,img = vid.read()
    cv2.rectangle(img,(100,100),(300,300),(0,255,255),1)
    crop_image = img[100:300,100:300]

    blur = cv2.GaussianBlur(crop_image,(3,3),0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv,(170,7,79),(172,255,255))


    kernel = np.ones((5,5))


    dilation = cv2.dilate(mask2,kernel,iterations=1)
    erosion = cv2.erode(dilation,kernel,iterations=1)

    filtered = cv2.GaussianBlur(erosion,(3,3),0)
    ret,thresh = cv2.threshold(filtered,127,255,0)

    coun,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    try:
        con = max(coun,key=lambda x:cv2.contourArea(x))

        x,y,w,h = cv2.boundingRect(con)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)

        hull = cv2.convexHull(con)



        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[con],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)


        hull = cv2.convexHull(con,returnPoints=False)
        defects = cv2.convexityDefects(con,hull)

        count_defects = 0

        cv2.imshow('Contours', drawing)
        if cv2.waitKey(1) == 27:
            break
    except:
        pass
        '''
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(con[s][0])
            end = tuple(con[e][0])
            far = tuple(con[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2)/(2*b*c))*180)/3.14

            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)
            cv2.line(crop_image,start,end,[0,255,0],2)


        if count_defects == 0:
            cv2.putText(img,"ONE",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects == 1:
            cv2.putText(img, "TWO", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        elif count_defects == 2:
            cv2.putText(img, "THREE", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        elif count_defects == 3:
            cv2.putText(img, "FOUR", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        elif count_defects == 4:
            cv2.putText(img, "FIVE", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        else:
            pass
    except:
        pass

    cv2.imshow("GESTURE",img)
    all_image = np.hstack((drawing,crop_image))
    cv2.imshow('Contours',all_image)


    if cv2.waitKey(1) == 27:
        break
'''
vid.release()
cv2.destroyAllWindows()