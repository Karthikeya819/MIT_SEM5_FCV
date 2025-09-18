import cv2 as cv
import numpy as np


def hough_lines(edges: cv.Mat, min_theta: float, max_theta: float, theta: float, rho: int, treshold: float):
    height, width = edges.shape
    diad_len = int(((height**2) + (width**2) )**0.5)

    rho_values = np.arange(-diad_len, diad_len, rho)
    theta_values = np.arange(min_theta, max_theta, theta)

    accummulator = np.zeros((len(rho_values), len(theta_values)))

    sins = np.sin(theta_values)
    coss = np.cos(theta_values)

    xs, ys = np.where(edges > 0)

    for x, y in zip(xs, ys):
        for angle_idx in range(len(theta_values)):
            dist = x*coss[angle_idx] + y*sins[angle_idx]
            rho_idx = np.where(rho_values < dist)[0][-1]

            accummulator[rho_idx][angle_idx] += 1
    accummulator /= np.max(accummulator)

    rho_index, theta_index = np.where(accummulator > treshold)

    return np.vstack([rho_values[rho_index], theta_values[theta_index]]).T

img = cv.imread('../../Week4/Sudoku.jpg')
diag_len = int(np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 50, 150, apertureSize=3)
lines = hough_lines(edges, 0, np.pi, np.pi/180, 1, 0.75)

for r_theta in lines:
    arr = np.array(r_theta, dtype=np.float64)
    r, theta = arr
    a = np.sin(theta)
    b = np.cos(theta)
    x0 = a*r
    y0 = b*r

    x1 = int(x0 + diag_len*(-b)) # sin(ang)*r - cos(ang)*diag_len
    y1 = int(y0 + diag_len*(a))
    x2 = int(x0 - diag_len*(-b))
    y2 = int(y0 - diag_len*(a))

    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()