
import cv2 as cv
import sys

def waitKey(key, img):
    k = cv.waitKey(0)

    if k == ord(key):
        cv.imwrite("starry_night.png", img)
    else:
        waitKey(key, img)

img = cv.imread("C:/Users/User/Pictures/orth_qual.png")

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
waitKey("s", img)
