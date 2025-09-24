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

def trsfrm_pt_and_msre_dist(H, src_pt, dst_pt):
    src_pt_h = np.array([src_pt[0], src_pt[1], 1])
    transformed_pt_h = np.dot(H, src_pt_h)
    transformed_pt = transformed_pt_h[:2] / transformed_pt_h[2]
    distance = np.linalg.norm(transformed_pt - dst_pt)
    
    return distance

def RANSAC_Homography(keypoints_1, keypoints_2, matches, threshold=5.0, num_iterations=1000):
    best_H, max_inliners = None, 0
    # get exact location of the matches
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches])

    if len(matches) < 4:
        return None
    
    for i in range(num_iterations):
        random_choice = np.random.choice(len(matches), 4, replace=False)
        src_choice = src_pts[random_choice]
        dst_choice = dst_pts[random_choice]
        H_candidate = cv.getPerspectiveTransform(src_choice, dst_choice)
        if H_candidate is None:
            continue
        # count inliers
        inliers_count = 0
        for j in range(len(matches)):
            dist = trsfrm_pt_and_msre_dist(H_candidate, src_pts[j], dst_pts[j])
            if dist < threshold:
                inliers_count += 1
        
        if inliers_count > max_inliners:
            max_inliners = inliers_count
            best_H = H_candidate
    return best_H

print(RANSAC_Homography(keypoints_1, keypoints_2, matches))