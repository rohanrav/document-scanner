import cv2
import numpy as np
import imutils
from transform import four_point_transform

image = cv2.imread("recipt.jpg")
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(edged, height=650))
cv2.waitKey(0)