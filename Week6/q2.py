import cv2 as cv
import numpy as np

img1 = cv.imread('Week6/Images/raw_1.png', 0)
img2 = cv.imread('Week6/Images/raw_2.png', 0)

sift = cv.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

def Match_KeyPoints_Knn(descriptors_1, descriptors_2, treshold= 0.8):
    Matches = []
    for i in range(len(descriptors_1)):
        distances = np.sqrt(np.sum((descriptors_2 - descriptors_1[i])**2, axis=1))
        min_dist_j = np.argmin(distances)
        min_d1 = distances[min_dist_j]
        min_d2 = np.min(np.delete(distances, min_dist_j))
        if min_d1/min_d2 < treshold:
            Matches.append(cv.DMatch(i, min_dist_j, distances[min_dist_j]))
    return Matches


matches = Match_KeyPoints_Knn(descriptors_1, descriptors_2, 0.3)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:], img2, matchColor = (0,255,0), singlePointColor=(255,0,0), flags=2)

cv.imshow('Image Matching', img3)
cv.waitKey(0)