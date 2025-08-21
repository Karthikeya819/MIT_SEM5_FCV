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

def FAST_Corners(img:cv.Mat, n=12, treshold=10):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width, _ = img.shape

    miniCircle = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    bresenhamCircle = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]

    def isActivePixel(I, curI, treshold):
        return I > curI + treshold or I < curI - treshold
    
    output = img.copy()

    for i in range(3, height - 3):
        for j in range(3, width - 3):
            count = 0
            curI = img_gray[i][j]
            CirleI = []

            count = 0
            for dx, dy in miniCircle:
                if isActivePixel(img_gray[i+dx][j+dy], curI, treshold):
                    count += 1
            if count < 2:
                continue

            for dx,dy in bresenhamCircle:
                CirleI.append(img_gray[i+dx][j+dy])

            activePixels = [isActivePixel(a, curI, treshold) for a in CirleI]
            
            activePixels_ext = activePixels + activePixels[:15]
            for num in activePixels_ext:
                if num == 1:
                    count += 1
                    if count == n:
                        output = cv.circle(output, (i, j), 2, (0, 0, 255), -1)
                else:
                    count = 0
    return output

img1 = FAST_Corners(img, n=10, treshold=15)
img2 = Harris_Corners(img)
cv.imshow('Image', np.hstack([img2, img1]))
cv.waitKey(0)
cv.destroyAllWindows()