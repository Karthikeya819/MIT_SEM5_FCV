import cv2 as cv
import numpy as np

image = cv.imread('Week1/img.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
pixel_vals = image.reshape((-1,3))

pixel_vals = np.float32(pixel_vals)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 3
retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))

cv.imshow('Image', image)
cv.waitKey(0)
cv.destroyAllWindows()