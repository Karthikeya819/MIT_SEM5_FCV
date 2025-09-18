import cv2 as cv

img = cv.imread('../../Week1/img.jpg')


img = cv.resize(img, (200, 200), img)
img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imshow('Image', img)

cv.waitKey(0)
cv.destroyAllWindows()