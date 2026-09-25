import numpy as np
import cv2 as cv

def showMask(mask, title):
    mask = mask.astype('uint8') * 255
    cv.imshow(title, mask)

def main():
    img = cv.imread("images/dog_1.jpg")
    #cv.imshow("image", img)
    height, width = img.shape[:2]
    img = cv.resize(img, (width//2, height//2))

    # Split image in color channels
    b, g, r = cv.split(img)

    # Criação da mascara para cada color channel
    b_mask = np.logical_and(b>15, b<100)
    showMask(b_mask, "b")
    g_mask = np.logical_and(g>100, g<255)
    showMask(g_mask, "g")
    r_mask = np.logical_and(r>15, r<200)
    showMask(r_mask, "r")

    mask = np.logical_and(b_mask, g_mask)
    mask = np.logical_and(mask, r_mask)
    showMask(mask, "mask")

    # Close small holes using dilate
    kernel = np.ones((5, 5), np.uint8)
    mask = mask.astype('uint8')*255
    mask = cv.dilate(mask, kernel, iterations=2)
    mask = (mask.astype('float')/255).astype('bool')











    cv.waitKey(0)

if __name__ == "__main__":
    main()