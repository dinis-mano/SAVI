import cv2
import numpy as np

def showMask(window_name, image):
    image_to_show = image.astype(np.uint8) * 255
    cv2.imshow(window_name, image_to_show)

# Main function
def main():  # this is our main function
    # relative path
    image = cv2.imread("images/cenario.jpg")
    template = cv2.imread("images/modelo.png")
    HEIGHT, WIDTH, NC = image.shape
    HEIGHT_TEMP, WIDTH_TEMP, NC_TEMP = template.shape

    # resize to a smaller template, in order to test the algorithm resilience
    #template = cv2.resize(template, (round(WIDTH_TEMP/2), round(HEIGHT_TEMP/2)))
    height, width, channels = template.shape

    cv2.imshow("image", image)
    cv2.imshow("template", template)

    # Metodo para deteção
    method = cv2.TM_CCORR_NORMED

    #Template matching
    res = cv2.matchTemplate(image, template, method) # Matriz com probalilidades de localização

    res = (res*255).astype(np.uint8)
    cv2.imshow("Template Position Probability", res)

    # Encontrar o valor de maior probabilidade
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print('max_loc:', str(max_loc))

    x, y = max_loc
    image_anotated = image.copy()

    cv2.rectangle(image_anotated, (x, y), (x + width, y + height), (255, 0, 0), 2)
    cv2.imshow('image_anotated', image_anotated)


    cv2.waitKey(0)

if __name__ == "__main__":
    main()