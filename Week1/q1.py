import cv2 as cv

img = cv.imread('img.jpg')
cv.imshow('image', img)

cv.waitKey(0)
# Writing same image
cv.imwrite('img.jpg', img)

cv.destroyAllWindows()