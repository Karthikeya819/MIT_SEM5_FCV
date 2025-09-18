import cv2 as cv

video = cv.VideoCapture('../../Week1/video.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()