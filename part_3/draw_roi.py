import cv2
import numpy as np

# Global variables
image = None
base_image = None
pts = []  # Initialized as an empty list


def mouse_event(event, x, y, flags, param):
    global image, base_image, pts

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

        # Reset image to clean state before redrawing
        image = base_image.copy()

        print(pts)

        # Draw polylines if there are at least 3 points
        if len(pts) >= 3:
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            isClosed = True
            color = (255, 0, 0)
            thickness = 2
            cv2.polylines(image, [pts_arr], isClosed, color, thickness)


def main():
    global image, base_image, pts

    cap = cv2.VideoCapture('docs/traffic.mp4')

    while True:
        _, image = cap.read()

        if image is None:
            print("Error: Image/Video not found or could not be opened.")
            return

        # Keep a clean backup frame for clearing/redrawing
        base_image = image.copy()

        cv2.imshow("image", image)
        cv2.setMouseCallback("image", mouse_event)

        k = cv2.waitKey(0) & 0xFF

        if k == 127:  # DEL key to clear points
            pts = []
            image = base_image.copy()
        elif k == 27:  # ESC key to exit
            break




if __name__ == "__main__":
    main()