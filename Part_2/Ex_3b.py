import cv2
import numpy as np
from numpy.ma.core import masked_array


def showMask(window_name, image):
    image_to_show = image.astype(np.uint8) * 255
    cv2.imshow(window_name, image_to_show)


# Main function
def main():  # this is our main function
    # relative path
    image = cv2.imread("images/person_3.jpg")
    height, width, channels = image.shape
    image = cv2.resize(image, (round(width / 2), round(height / 2)))
    cv2.imshow('Original', image)
    H, W, NC = image.shape

    # --------------------------------
    # Segment green color to isolate the grass
    # --------------------------------

    # conversion to hsv color model
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # split of the color channels
    h, s, v = cv2.split(hsv_image)
    #cv2.imshow('H', h)
    #cv2.imshow('S', s)
    #cv2.imshow('V', v)

    # get masks by imposing limits on the channels
    mask_h = np.logical_and(h > 30, h < 50)  # assumming green color has hue 60
    #showMask('mask_h', mask_h)

    mask_s = np.logical_and(s > 60, s < 255)
    #showMask('mask_s', mask_s)

    mask_v = np.logical_and(v > 60, v < 255)
    #showMask('mask_v', mask_v)

    # Putting the mask alltoghether
    mask = np.logical_and(mask_h, mask_s)
    mask = np.logical_and(mask, mask_v)

    showMask('original mask', mask)

    # Post processing using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_dilated_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)
    mask_dilated = (mask_dilated_uint8.astype(float) / 255).astype(np.bool)

    # cv2.imshow("mask_dilated", mask_dilated_uint8)
    #showMask('dilated mask', mask_dilated)

    # to get the mask of thedog we negate the mask of the grass,
    # assuming the image contains only dog and grass
    mask_dog = np.logical_not(mask_dilated)
    mask_dog = mask_dog.astype(np.uint8) * 255
    cv2.imshow('dog mask', mask_dog)

    #Seperate into multiple components and find the largest one
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dog)
    print('numLabels:', numLabels)

    #Get largest area component
    largest_area = 0
    largest_area_index = None

    for idx in range(1, numLabels): #Start from 1, because 0 will be the background
        area_component = stats[idx, cv2.CC_STAT_AREA]

        if area_component > largest_area:
            largest_area = area_component
            largest_area_index = idx

    print('largest_area_index:', str(largest_area_index))
    print('area', str(largest_area))

    largest_mask = (labels==largest_area_index).astype(np.int8)
    #cv2.imshow('largest mask', largest_mask)

    #Set bonding box for largest
    pts = cv2.findNonZero(largest_mask)
    x, y, width, height = cv2.boundingRect(pts)

    image_annotated = image.copy()
    cv2.rectangle(image_annotated, (x,y), (x+width, y+height), (0,255,0), 2)
    cv2.imshow('annotated', image_annotated)

    # Ex 3b
    #Compute some features
    #Hight to width ratio
    h_w_ratio = height/width
    print('Height to Width Ratio:', str(round(h_w_ratio, 2)))

    #average color of the hue component
    avg_h = np.mean(h)
    print('Average Hue Image:', str(round(avg_h, 2)))

    ma = np.ma.masked_array(h, mask=largest_area, dtype=np.uint8)
    avg_h_object = np.mean(ma)
    print('Average Hue Object:', str(avg_h_object))
    # Atenção que não se pode fazer média de valores circulares. Isto está errado

    cv2.waitKey(0)


if __name__ == "__main__":
    main()