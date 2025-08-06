import cv2 as cv
import numpy as np

img = cv.imread('Week1/img.jpg').astype(np.float32)


img_blurred = cv.GaussianBlur(img, (3,3), 0)

unsharp_img = cv.addWeighted(img, 2.3, img_blurred, -1.3, 0)
unsharp_img = np.clip(unsharp_img, 0, 255).astype(np.uint8)

cv.imshow('Image', img.astype(np.uint8))
cv.imshow('Unsharp Image', unsharp_img)

cv.waitKey(0)
cv.destroyAllWindows()