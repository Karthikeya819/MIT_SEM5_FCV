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

print("=== Intrinsic Parameters ===")
print("Camera matrix (K):")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)

print("\n=== Extrinsic Parameters (per image) ===")
for i in range(len(rvecs)):
    print(f"\n--- Image {i+1} ---")

    R, _ = cv2.Rodrigues(rvecs[i])
    t = tvecs[i]
    
    print("Rotation vector:")
    print(rvecs[i])
    print("Rotation matrix:")
    print(R)
    print("Translation vector:")
    print(t)

    RT = np.hstack((R, t))
    P = camera_matrix @ RT
    print("Projection matrix (P = K[R|t]):")
    print(P)
