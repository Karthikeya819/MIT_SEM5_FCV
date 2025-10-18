import cv2
import numpy as np
import glob

CHECKERBOARD = (12, 12)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

objpoints = []
imgpoints = []

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

images = glob.glob('./calib_example/*.tif')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH +cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

total_error = 0
print("\n=== Reprojection Verification ===")

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

    img = cv2.imread(images[i])

    for p1, p2 in zip(imgpoints[i], imgpoints2):
        cv2.circle(img, tuple(p1.ravel().astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(img, tuple(p2.ravel().astype(int)), 3, (0, 0, 255), -1)

    cv2.imshow(f'Reprojection Image {i+1}', img)
    cv2.waitKey(500)

cv2.destroyAllWindows()

mean_error = total_error / len(objpoints)
print(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
