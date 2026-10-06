import cv2
import numpy as np

def showMask(window_name, image):
    image_to_show = image.astype(np.uint8) * 255
    cv2.imshow(window_name, image_to_show)

def main():
    cap = cv2.VideoCapture('docs/traffic.mp4')
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = round(cap.get(cv2.CAP_PROP_FPS),2)
    print('fps: ', FPS)

    while True:
        #Read frame
        ret, frame = cap.read()
        if not ret:
            print('Finished reading video file')
            break

        # Change into hsv
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        # cv2.imshow('H', h)
        # cv2.imshow('S', s)
        # cv2.imshow('V', v)

        mask_h = np.logical_and(h > 0, h < 60)     # Mask cars and green grass
        mask_s = np.logical_and(s > 0, s < 255)
        mask_v = np.logical_or(np.logical_and(v > 0, v < 180), np.logical_and(v > 210, v < 255))    #Mask out white lines

        # Putting the mask alltoghether
        mask = np.logical_and(mask_h, mask_s)
        mask = np.logical_and(mask, mask_v)

        mask = np.logical_not(mask).astype(np.uint8) * 255
        # showMask('mask', mask)

        # Remove noise with erode
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=4)
        mask_eroded = (mask_eroded.astype(float) / 255).astype(np.bool)
        # showMask('eroded', mask_eroded)

        # Define Road
        pts = [(1, HEIGHT), (1279, HEIGHT), (1277, 659), (930, 233), (776, 84), (660, 1), (562, 1), (553, 19), (444, 155), (218, 412), (2, 680)]

        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        # cv2.polylines(frame, [pts_arr], isClosed, color, thickness)

        mask_road = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_road, [pts_arr], 1)
        mask_road = cv2.erode(mask_road, kernel, iterations=4)
        mask_road = mask_road.astype(bool)
        # showMask('road', mask_road)

        better_mask = np.logical_and(mask_road, mask_eroded).astype(np.uint8)
        showMask('mask', better_mask)

        # Define lane 4
        lane_4_pts = [(979, HEIGHT), (1277, HEIGHT), (944, 270), (723, 41), (651, 2), (638, 4), (687, 39), (757, 160)]
        lane_4_pts = np.array(lane_4_pts, np.int32).reshape((-1, 1, 2))

        lane_3_pts = [(646, HEIGHT), (982, HEIGHT), (840, 344), (730, 97), (679, 31), (640, 6), (620, 7), (642, 25), (667, 77), (673, 210), (666, 408)]
        lane_3_pts = np.array(lane_3_pts, np.int32).reshape((-1, 1, 2))

        lane_2_pts = [(337, HEIGHT), (645, HEIGHT), (672, 316), (670, 121), (651, 38), (620, 6), (601, 7), (617, 28), (613, 80), (584, 171), (493, 372)]
        lane_2_pts = np.array(lane_2_pts, np.int32).reshape((-1, 1, 2))

        lane_1_pts = [(51, HEIGHT), (337, HEIGHT), (482, 367), (573, 156), (598, 75), (603, 24), (588, 2), (577, 24), (531, 104), (329, 356), (171, 566)]
        lane_1_pts = np.array(lane_1_pts, np.int32).reshape((-1, 1, 2))

        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        cv2.polylines(frame, [lane_4_pts], isClosed, color, thickness)
        cv2.polylines(frame, [lane_3_pts], isClosed, color, thickness)
        cv2.polylines(frame, [lane_2_pts], isClosed, color, thickness)
        cv2.polylines(frame, [lane_1_pts], isClosed, color, thickness)

        # Seperate into all compontes
        better_mask = cv2.dilate(better_mask, kernel, iterations=5)
        showMask('mask2', better_mask)
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(better_mask)

        for i in range(1, numLabels): # 0 is the background
            area = stats[i, cv2.CC_STAT_AREA]

            if area > 10:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                # Draw green bounding box on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        frame_time = int(round(1/FPS * 1000))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(frame_time)
        # key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()