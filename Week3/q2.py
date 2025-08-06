import cv2 as cv
import numpy as np

img = cv.imread('Week1/img.jpg')

grad_x = cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3))
grad_y = cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3))

cv.imshow('Gradient Image', np.hstack([grad_x, grad_y]))

cv.waitKey(0)
cv.destroyAllWindows()