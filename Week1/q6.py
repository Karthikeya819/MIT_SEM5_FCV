import cv2 as cv

img = cv.imread('img.jpg')
img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)


cv.imshow('image', img)

cv.waitKey(0)
cv.destroyAllWindows()