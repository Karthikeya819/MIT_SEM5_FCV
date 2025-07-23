import cv2 as cv

img = cv.imread('img.jpg')

resized_img = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('image', resized_img)

cv.waitKey(0)

cv.destroyAllWindows()