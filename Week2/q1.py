import cv2 as cv

img = cv.imread('Week1/img.jpg')

grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist = cv.equalizeHist(grayscale)

cv.imshow('image', img)
cv.imshow('hist', hist)
cv.waitKey(0)