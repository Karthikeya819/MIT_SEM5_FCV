import cv2 as cv
import numpy as np


img = cv.imread('Week1/images.jpeg')

grad_x = cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3))
grad_y = cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3))

sobel_edges = cv.cvtColor(cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0), cv.COLOR_BGR2GRAY)

_, sobel_edges = cv.threshold(sobel_edges, 100, 255, type=cv.THRESH_BINARY)

cv.imshow("Sobel Edges", sobel_edges)
cv.waitKey(0)
cv.destroyAllWindows()