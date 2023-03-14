import cv2 as cv
import numpy as np

image = cv.imread("./202-233/10.jpg",0)
k = np.ones((3,3), np.uint8)
open = cv.morphologyEx(image, cv.MORPH_OPEN, k)
close = cv.morphologyEx(image, cv.MORPH_CLOSE, k)
gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, k)
tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, k)
blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, k)
cv.imshow("image", image)
cv.imshow("opening", open)
cv.imshow("closeing", close)
# cv.imshow("tophat", tophat)
# cv.imshow("blackhat", blackhat)
cv.waitKey()
cv.destroyAllWindows()
