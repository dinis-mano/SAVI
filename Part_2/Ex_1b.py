import numpy as np
import cv2 as cv

def main():
    src = cv.imread("images/dog_1.jpg")
    cv.imshow("image", src)

    # recortar a região que contém apenas o cão
    cut_img = src[:, 100:810]
    #cv.imshow("cut", cut_img)

    # redimensionar a imagem para metade do tamanho
    height, width, channels = src.shape
    #resized_img = cv.resize(cut_img, round(height/2), round(width/2))
    #cv.imshow("resized", resized_img)

    # converter para tons de cinzento
    gray_img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray_img)

    # aumentar e diminuir o brilho
    # change to float type and increase all
    bright_img = src.astype('float') + 50

    # To pervent overflow error, clip to 255 and change back to uint8
    bright_img = np.clip(bright_img, 0, 255).astype('uint8')
    cv.imshow("bright", bright_img)

    cv.waitKey(0)

if __name__ == "__main__":
    main()