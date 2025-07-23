import cv2 as cv

# Load Image
img = cv.imread('img.jpg')

print(f'R: {img[0][0][2]} G: {img[0][0][1]} B: {img[0][0][0]}')