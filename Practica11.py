import cv2
import numpy as np
import imutils

original = cv2.imread("perro1.jpg")
original = imutils.resize(original, width=500)
image_to_compare = cv2.imread("perro2.jpg")
image_to_compare = imutils.resize(image_to_compare, width=500)

#

if original.shape == image_to_compare.shape:
    print('Las imagenes tiene el mismo tamaño y canal')
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)
    print(cv2.countNonZero(b))
    
    if (cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0):
        print('Las imagenes son completamente iguales')
    else:
        print('Las imagenes no son iguales')

#

shift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = shift.detectAndCompute(original, None)
kp_2, desc_2 = shift.detectAndCompute(image_to_compare, None)
print("Keypoints imagen 1", str(len(kp_1)))
print("Keypoints imagen 2", str(len(kp_2)))
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_1, desc_2, k=2)
good_points = []

for m, n in matches:
    
    if m.distance < 0.6*n.distance:
        good_points.append(m)
        
number_keypoints = 0
if (len(kp_1) <= len(kp_2)):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)
    
print("similar matches",len(good_points))
print("Porcentaje de ", len(good_points) / number_keypoints * 100, "%")

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

cv2.imshow("Resultado", cv2.resize(result, None, fx = 1, fy=1))
cv2.imwrite("Feature_matching.jpg", result)
cv2.imshow("Original", original)
cv2.imshow("Duplicate", image_to_compare)
cv2.waitKey(0)

cv2.destroyAllWindows()
