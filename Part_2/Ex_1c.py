import numpy as np
import cv2 as cv

def change_brightness(image, brightness):
    image = image.copy()
    image = image.astype("float") * brightness                    #change brightness
    image = np.clip(image, 0, 255).astype("uint8")    # overflow error protection
    return image

def main():
    time = 3
    img = cv.imread("images/dog_1.jpg")
    cv.imshow("image", img)

    # Create gray image in darker color
    gray_img = img.copy()
    gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)
    gray_img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

    darker_img = img.copy()
    height, width = darker_img.shape[:2]
    cv.waitKey(0)

    time_step = time*20
    for i in range(time_step+1):
        progress = i/time_step          # Value from 0 to 1
        brightness = 1 - progress
        gray_darker_img = change_brightness(gray_img, brightness)
        darker_img[:, :round(width*progress), :] = gray_darker_img[:, :round(width*progress), :]  # curtain for whole image
        cv.imshow("darker_img", darker_img)
        cv.waitKey(50)

    cv.waitKey(0)

if __name__ == "__main__":
    main()