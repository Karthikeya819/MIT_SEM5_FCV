import cv2 as cv
import numpy as np

img = cv.imread('Week5/Chess_Board.webp')

def Harris_Corners(img:cv.Mat, ksize=3):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gx, gy = cv.Sobel(img_gray, cv.CV_64F, 1, 0), cv.Sobel(img_gray, cv.CV_64F, 0, 1)

    Ixx = gx**2
    Iyy = gy**2
    Ixy = gx*gy

    height, width, _ = img.shape
    n = (ksize-1)//2

    Neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    for i in range(n, height-n):
        for j in range(n, width-n):
            ixx, iyy, ixy = 0, 0, 0
            for k in range(1, n):
                for dx, dy in Neighbours:
                    ixx += Ixx[i]
                




Harris_Corners(img)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()