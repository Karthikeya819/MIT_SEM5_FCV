import cv2 as cv
import numpy as np


def hough_lines(edges: cv.Mat, rho, min_theta, max_theta, theta, threshold):
    diag_len = int(np.ceil(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)))
    
    theta_angles = np.arange(min_theta, max_theta, theta)
    rho_values = np.arange(-diag_len, diag_len +1, rho)

    theta_count = len(theta_angles)
    rho_count = len(rho_values)
    
    accumulator = np.zeros((rho_count, theta_count))

    sins = np.sin(theta_angles)
    coss = np.cos(theta_angles)

    xs, ys = np.where(edges > 0)

    for x, y in zip(xs, ys):
        for angle_idx in range(theta_count):
            cur_rho = x*coss[angle_idx] + y*sins[angle_idx]
            rho_pos = np.where(rho_values < cur_rho)[0][-1]

            accumulator[rho_pos][angle_idx] += 1
    accumulator /= np.max(accumulator)
    rho_index, theta_index = np.where(accumulator > threshold)

    return np.vstack([rho_values[rho_index], theta_angles[theta_index]]).T


img = cv.imread('Week4/Sudoku.jpg')
diag_len = int(np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 50, 150, apertureSize=3)
lines = hough_lines(edges, 1, 0, np.pi, np.pi/180, 0.75)

for r_theta in lines:
    arr = np.array(r_theta, dtype=np.float64)
    r, theta = arr
    a = np.sin(theta)
    b = np.cos(theta)
    x0 = a*r
    y0 = b*r

    x1 = int(x0 + diag_len*(-b))
    y1 = int(y0 + diag_len*(a))
    x2 = int(x0 - diag_len*(-b))
    y2 = int(y0 - diag_len*(a))

    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()