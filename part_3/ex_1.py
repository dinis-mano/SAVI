import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture('docs/traffic.mp4')
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = round(cap.get(cv2.CAP_PROP_FPS),2)
    print('fps: ', FPS)

    while True:
        #Read frame
        ret, image_rgb = cap.read()
        if not ret:
            print('Finished reading video file')
            break

        cv2.imshow('image', image_rgb)
        key = cv2.waitKey(0)

        if key == 27:
            break


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    cv2.waitKey(0)