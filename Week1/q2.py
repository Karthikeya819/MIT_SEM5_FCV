import cv2 as cv

VideoCapture = cv.VideoCapture('video.mp4')

while True:
    ret, frame = VideoCapture.read()

    if not ret:
        break

    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('k'):
        break

VideoCapture.release()
cv.destroyAllWindows()