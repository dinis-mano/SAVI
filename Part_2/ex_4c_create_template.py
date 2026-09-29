import cv2
import numpy as np

# Initialize global variables for mouse click coordinates
first_pos = None
second_pos = None

def mouse_event(event, x, y, flags, param):
    global first_pos, second_pos, image

    if event == cv2.EVENT_LBUTTONDOWN:
        if first_pos is None:
            first_pos = (x, y)
        elif second_pos is None:
            second_pos = (x, y)
        else:
            first_pos = (x, y)
            second_pos = None

        if first_pos is not None and second_pos is not None:
            cv2.rectangle(image, first_pos, second_pos, (0, 0, 255), 2)

            x_min, x_max = min(first_pos[0], second_pos[0]), max(first_pos[0], second_pos[0])
            y_min, y_max = min(first_pos[1], second_pos[1]), max(first_pos[1], second_pos[1])

            # Corta o ROI garantindo que as coordenadas estão na ordem correta
            roi = image[y_min:y_max, x_min:x_max]
            cv2.imshow("template.jpg", roi)

# Main function
def main():
    global image

    # Relative path
    image = cv2.imread("images/cenario_2.jpg")
    if image is None:
        print("Error: Image not found.")
        return

    HEIGHT, WIDTH, NC = image.shape

    cv2.imshow("image", image)
    cv2.setMouseCallback("image", mouse_event)

    while True:
        cv2.imshow("image", image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()