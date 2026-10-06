import cv2
import numpy as np

cap = cv2.VideoCapture('docs/traffic.mp4')

# Initialize MOG2
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=40, detectShadows=True
)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# --- STEP 1: WARM-UP PHASE ---
# Run 100 frames with default learning rate so MOG2 learns the road & white arrows as BACKGROUND
print("Warming up background model...")
for _ in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    bg_subtractor.apply(frame, learningRate=-1)

# Reset video back to frame 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- STEP 2: DETECTION PHASE ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Apply background subtraction with a stable learning rate now that background is learned
    fg_mask = bg_subtractor.apply(frame, learningRate=0)

    # Filter shadows (shadows = 127, foreground = 255)
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    # Restrict to driving lanes (ROI)
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    lane_polygon = np.array(
        [
            [
                [int(width * 0.10), height],
                [int(width * 0.42), int(height * 0.22)],
                [int(width * 0.62), int(height * 0.22)],
                [int(width * 0.90), height],
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(roi_mask, lane_polygon, 255)
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

    # Morphological processing to merge car bodies
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # STEP 3: CONTOUR FILTERING
        # Cars have area > 600, width > 30px, and balanced aspect ratio.
        # This discards tall skinny lane dashes (w < 25) and tiny arrow tips.
        aspect_ratio = float(w) / h
        if area > 600 and w > 30 and h > 25 and 0.4 < aspect_ratio < 2.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
