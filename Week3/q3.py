import cv2 as cv
import numpy as np

img = cv.imread('Week1/img.jpg')

box_blur = cv.blur(img, (7,7))
gauss_blur = cv.GaussianBlur(img, (7,7), 0)

cv.imshow('image', np.hstack([box_blur, gauss_blur]))

cv.waitKey(0)
cv.destroyAllWindows()