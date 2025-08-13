import cv2 as cv
import numpy as np

class Thresholding:
    def BINARY_THRESHOLD(img: np.ndarray, thresh: int, maxval: int):
        height, width = img.shape
        out = np.zeros_like(img)
        for i in range(height):
            for j in range(width):
                if img[i, j] > thresh:
                    out[i, j] = maxval
                else:
                    out[i, j] = 0
        return out

    def BINARY_THRESHOLD_INV(img: np.ndarray, thresh: int, maxval: int):
        height, width = img.shape
        out = np.zeros_like(img)
        for i in range(height):
            for j in range(width):
                if img[i, j] < thresh:
                    out[i, j] = maxval
                else:
                    out[i, j] = 0
        return out
    
    def TRUNCATE(img: np.ndarray, thresh: int, maxval: int):
        return np.minimum(img, thresh)
    
    def TOZERO(img: np.ndarray, thresh: int, maxval: int):
        height, width = img.shape
        out = img.copy()
        for i in range(height):
            for j in range(width):
                if img[i, j] < thresh:
                    out[i, j] = 0
        return out
    
    def TOZERO_INV(img: np.ndarray, thresh: int, maxval: int):
        height, width = img.shape
        out = img.copy()
        for i in range(height):
            for j in range(width):
                if img[i, j] > thresh:
                    out[i, j] = 0
        return out


img = cv.imread('Week1/img.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thresh1 = Thresholding.BINARY_THRESHOLD(img_gray, 120, 255)
thresh2 = Thresholding.BINARY_THRESHOLD_INV(img_gray, 120, 255)
thresh3 = Thresholding.TRUNCATE(img_gray, 120, 255)
thresh4 = Thresholding.TOZERO(img_gray, 120, 255)
thresh5 = Thresholding.TOZERO_INV(img_gray, 120, 255)

cv.imshow('Comaprision', np.vstack([np.hstack([img_gray, thresh1, thresh2]), np.hstack([thresh3, thresh4, thresh5])]))
cv.waitKey(0)
cv.destroyAllWindows()