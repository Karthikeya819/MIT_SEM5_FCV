import cv2 as cv
import numpy as np

img = cv.imread('Week1/img.jpg')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh1 = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
_, thresh2 = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV)
_, thresh3 = cv.threshold(img, 120, 255, cv.THRESH_TRUNC)
_, thresh4 = cv.threshold(img, 120, 255, cv.THRESH_TOZERO)
_, thresh5 = cv.threshold(img, 120, 255, cv.THRESH_TOZERO_INV)

img = cv.putText(img, "Original Image", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
thresh1 = cv.putText(thresh1, "Binary Threshold", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
thresh2 = cv.putText(thresh2, "Binary Threshold Inverted", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
thresh3 = cv.putText(thresh3, "Truncated Threshold", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
thresh4 = cv.putText(thresh4, "To 0", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
thresh5 = cv.putText(thresh5, "To 0 Inverted", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)


cv.imshow('Comaprision', np.vstack([np.hstack([img, thresh1, thresh2]), np.hstack([thresh3, thresh4, thresh5])]))


cv.waitKey(0)
cv.destroyAllWindows()