import cv2

vid = cv2.VideoCapture(0)

while vid.isOpened():
    flag,img = vid.read()

    cv2.imshow('Test',img)

    if cv2.waitKey(1) == 27:
        cv2.imwrite('Test.jpg',img)

vid.release()
cv2.destroyAllWindows()