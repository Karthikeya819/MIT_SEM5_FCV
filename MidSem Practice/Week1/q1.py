import cv2 as cv

img = cv.imread('../../Week1/img.jpg')

img = cv.rectangle(img, (100, 100), (500, 500), (0,0, 255), -1)

cv.imshow('Image', img)
cv.waitKey(0)