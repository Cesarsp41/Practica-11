import cv2
import imutils

cam = cv2.VideoCapture(1)
##

knn_sub = cv2.createBackgroundSubtractorKNN()
mog2_sub = cv2.createBackgroundSubtractorMOG2()

while(cam.isOpened()):
    ret, frame = cam.read()

    framegauss = cv2.GaussianBlur(frame, (5, 5), 0)
    image_blur = cv2.GaussianBlur(framegauss, (51, 51),cv2.BORDER_DEFAULT)
    image_bw = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    framegauss = cv2.GaussianBlur(image_bw, (5, 5), 0)

    if not ret:
        break
    
    mog_sub_mask = mog2_sub.apply(framegauss)
    knn_sub_mask = knn_sub.apply(framegauss)
    # Suma
    mask2 = cv2.add(mog_sub_mask, knn_sub_mask)
    res = cv2.bitwise_and(frame, frame, mask=mask2)
    frame = imutils.resize(frame, width=500)
    res = imutils.resize(res, width=500)
    cv2.imshow('imgOriginal', frame)
    cv2.imshow('Deteccion de movimiento', res)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        break
