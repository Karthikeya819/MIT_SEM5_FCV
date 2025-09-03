import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss

dataset_train = 'INRIAPerson/train_64x128_H96/'
dataset_test = 'INRIAPerson/test_64x128_H96/'

neg_list, pos_list =[], []

with open(dataset_train+'pos.lst', 'r') as f:
    pos_list = f.read().strip().split('\n')

with open(dataset_train+'neg.lst', 'r') as f:
    neg_list = f.read().strip().split('\n')

X, Y = [], []

hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
for file_name in pos_list:
    img = cv.imread(dataset_train + file_name, 0)
    X.append(hog.compute(img))
    Y.append(1)

for file_name in neg_list[:300]:
    img = cv.imread(dataset_train + file_name, 0)
    img = cv.resize(img, (96, 160), interpolation=cv.INTER_CUBIC)

    X.append(hog.compute(img))
    Y.append(0)

X , Y = np.array(X), np.array(Y)

model = LinearSVC(max_iter=100)
model.fit(X, Y)

pred = model.predict(X)
print("Log Loss:" + log_loss(Y, pred))