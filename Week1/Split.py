import cv2 as cv

img = cv.imread('Week1/img.jpg')

b, g, r = cv.split(img)

b[100:150, 100:150] = 255

cv.imshow('b_image', b)
cv.imshow('g_image', g)
cv.imshow('r_image', r)

merged = cv.merge([b, g, r])
cv.imshow('merged', merged)

cv.waitKey(0)