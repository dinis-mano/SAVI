# Import OpenCV
import numpy as np
import cv2 as cv

def main():
    image = cv.imread("images/dog_1.jpg")
    cv.imshow("image", image)

    # Get shape and dtype from image
    print("shape=" + str(image.shape))
    print("dtype=" + str(image.dtype))

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray_image", gray_image)
    cv.waitKey(0)




if __name__ == "__main__":
    main()
