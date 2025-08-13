import cv2 as cv
import numpy as np

img = cv.imread('Week4/Lanes.jpg')
height, width, _ = img.shape

lw_h, lw_s, lw_v = np.array([0, 0, 200])
uw_h, uw_s, uw_v = np.array([179, 50, 255])

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask_img = np.zeros_like(img)

for i in range(height):
    for j in range(width):
        h,s,v = hsv[i][j]
        if lw_h < h < uw_h and lw_s < s < uw_s and lw_v < v < uw_v:
            mask_img[i][j] = (255, 255, 255)
        

cv.imshow('Image', np.hstack([img, mask_img]))
cv.waitKey(0)
cv.destroyAllWindows()