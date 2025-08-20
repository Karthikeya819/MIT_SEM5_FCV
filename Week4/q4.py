import cv2 as cv
import numpy as np

def kmeans(img: cv.Mat, k:int, maxIter:int):
    def dist(pixel1, pixel2):
        return np.sqrt((pixel1[0] - pixel2[0])**2 + (pixel1[1] - pixel2[1])**2 + (pixel1[2] - pixel2[2])**2)

    centroids = np.array([[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(k)])
    prev_centroids = None

    iterCount = 0
    height, width, _ = img.shape
    pixels_reshaped = img.reshape((-1, 3)).astype(np.float32)
    pixels_count = height * width
    assignment = None

    while iterCount < maxIter:
        if prev_centroids is not None and np.allclose(prev_centroids, centroids):
            break
        
        assignment = [0] * pixels_count
        for idx in range(pixels_count):
            distances = []
            for centoid in centroids:
                distances.append(dist(pixels_reshaped[idx], centoid))
            assignment[idx] = np.argmin(distances)
        
        prev_centroids = centroids.copy()
        
        for i in range(k):
            ind = [j for j in range(pixels_count) if assignment[j] == i ]

            if len(ind) > 0:
                centroids[i] = np.mean(pixels_reshaped[ind], axis=0)
            else:
                centroids[i][0] = 0
                centroids[i][1] = 0
                centroids[i][2] = 0
        iterCount += 1
    
    for i in range(height):
        for j in range(width):
            cluster_grp = assignment[i * width + j]
            img[i][j] = centroids[cluster_grp]
    return img
    
img = cv.imread('Week1/img.jpg')

img1 = cv.GaussianBlur(img, ksize=(7, 7), sigmaX=0)
img = kmeans(img, 5, 10)
img1 = kmeans(img1, 5, 10)

cv.imshow('Image', np.hstack([img, img1]))
cv.waitKey(0)
cv.destroyAllWindows()