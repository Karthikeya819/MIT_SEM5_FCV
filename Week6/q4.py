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

H_mat =  RANSAC_Homography(keypoints_1, keypoints_2, matches)

print(H_mat)

h1, w1 = img1.shape
h2, w2 = img2.shape

corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)

def apply_homography(H, points):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (H @ points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)
    return transformed_points[:, :2]

warped_corners_img2 = apply_homography(H_mat, corners_img2)

all_corners = np.vstack((corners_img1, warped_corners_img2))
xmin, ymin = np.floor(np.min(all_corners, axis=0)).astype(int)
xmax, ymax = np.ceil(np.max(all_corners, axis=0)).astype(int)

translation = [-xmin, -ymin]

panorama = np.zeros((ymax - ymin, xmax - xmin), dtype=img2.dtype)

panorama[translation[1]:translation[1] + h2, translation[0]:translation[0] + w2] = img2

for y in range(h1):
    for x in range(w1):
        pt = np.array([x, y, 1], dtype=float)
        transformed_pt = np.dot(H_mat, pt)
        transformed_pt /= transformed_pt[2]
        tx, ty = int(transformed_pt[0]), int(transformed_pt[1])

        if 0 <= tx + translation[0] < panorama.shape[1] and 0 <= ty + translation[1] < panorama.shape[0]:
            if panorama[ty + translation[1], tx + translation[0]] == 0:  # Unused pixels
                panorama[ty + translation[1], tx + translation[0]] = img1[y, x]
            else:
                pass

cv.imshow('Stitched Image', panorama)
cv.waitKey(0)
cv.destroyAllWindows()