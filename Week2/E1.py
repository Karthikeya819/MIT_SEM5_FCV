# Log Transformations
import cv2 as cv
import numpy as np

img = cv.imread('Week1/img.jpg')

# Log Transformations
log_transformed = np.float32(img)
maxi = np.max(log_transformed)
c = 255/(np.log(1 + maxi))
log_transformed = c * np.log(1 + log_transformed)
log_transformed = np.uint8(log_transformed)

# Gamma Transformations
for gamma in [0.1, 0.5, 1.2, 2.2]:
    gamma_transform = np.array(255 * (img/255)**gamma, dtype='uint8')
    # cv.imshow(f'gamma_transformed_{gamma}', gamma_transform)

# Piecewise linear transformations 

def pixelVal(pix, r1, s1, r2, s2):
    if 0<= pix <=r1:
        return (s1/r1)*pix
    elif r1 < pix < r2:
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
pixelValsec = np.vectorize(pixelVal)
piece_wise_transformed = pixelValsec(img, 70, 0, 140, 255)

cv.imshow('image', img)
cv.imshow('log_transformed', log_transformed)
cv.imshow('piece_wise_transformed', piece_wise_transformed)

cv.waitKey(0)