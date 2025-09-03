import cv2 as cv
import numpy as np

img = cv.imread('Week1/img.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
keypoints_with_size = np.copy(img)

cv.drawKeypoints(img, keypoints, keypoints_with_size, color = (255, 0, 0),flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Image', keypoints_with_size)
cv.waitKey(0)