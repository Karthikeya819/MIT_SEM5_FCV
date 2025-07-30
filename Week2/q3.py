import cv2 as cv

img = cv.imread('Week1/img.jpg')

resized_img = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cropped_img = img[100:200, 100:200]
cv.imshow('resized_image', resized_img)
cv.imshow('cropped_image', cropped_img)

cv.waitKey(0)

cv.destroyAllWindows()