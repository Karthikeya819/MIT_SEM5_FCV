import cv2 as cv
import numpy as np

img = cv.imread('Week5/Chess_Board.jpg')

def Harris_Corners(img:cv.Mat, ksize=3, threshold=0.01):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gx, gy = cv.Sobel(img_gray, cv.CV_64F, 1, 0), cv.Sobel(img_gray, cv.CV_64F, 0, 1)

    Ixx = gx**2
    Iyy = gy**2
    Ixy = gx*gy

    height, width, _ = img.shape
    n = (ksize-1)//2

    Neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    R = np.zeros((height, width), dtype=np.float64)

    for i in range(n, height-n):
        for j in range(n, width-n):
            ixx, iyy, ixy = 0, 0, 0
            for k in range(1, n+1):
                for dx, dy in Neighbours:
                    ixx += Ixx[i + dx*k][j + dy*k]
                    iyy += Iyy[i + dx*k][j + dy*k]
                    ixy += Ixy[i + dx*k][j + dy*k]
            
            R[i][j] = (ixx*iyy - (ixy**2)) - (ixx + iyy)

    R = cv.normalize(R, None, 0, 255, cv.NORM_MINMAX)
    out = img.copy()
    R_thr = threshold * R.max()    
    
    for i in range(height):
        for j in range(width):
            if R[i, j] > R_thr:
                cv.circle(out, (j, i), 2, (0, 0, 255), -1)

    return out


img = Harris_Corners(img)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()