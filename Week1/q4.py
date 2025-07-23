import cv2 as cv
import numpy as np

plain_img = np.zeros((300,300,3))

rect_img = cv.rectangle(plain_img, (50,50), (250,250), (255,255,255), 5)

cv.imshow("Image", rect_img)
cv.waitKey(0)

cv.destroyAllWindows()