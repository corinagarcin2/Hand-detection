import cv2
import mediapipe as mp
import time
import numpy as np
import math


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def distance(point1, point2):
    """Find distance between two landmark points."""

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# FINGER DETECTION
# ============================================================

def get_fingers(hand_landmarks, hand_label):

    fingers = []

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip = hand_landmarks[4]
    thumb_joint = hand_landmarks[3]

    # Because our camera image is mirrored,
    # thumb direction depends on left/right hand.

    if hand_label == "Right":
        thumb_up = thumb_tip.x > thumb_joint.x
    else:
        thumb_up = thumb_tip.x < thumb_joint.x

    fingers.append(thumb_up)

    # --------------------------------------------------------
    # OTHER FOUR FINGERS
    # --------------------------------------------------------

    # Tip landmark IDs
    tips = [8, 12, 16, 20]

    # Middle joint landmark IDs
    joints = [6, 10, 14, 18]

    for tip, joint in zip(tips, joints):

        # On the screen, smaller Y means higher.
        finger_up = (
            hand_landmarks[tip].y
            <
            hand_landmarks[joint].y
        )

        fingers.append(finger_up)

    return fingers


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def recognize_gesture(fingers, hand_landmarks):

    thumb = fingers[0]
    index = fingers[1]
    middle = fingers[2]
    ring = fingers[3]
    pinky = fingers[4]

    finger_count = sum(fingers)

    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------

    if finger_count == 0:
        return "FIST"

    # --------------------------------------------------------
    # OPEN HAND
    # --------------------------------------------------------

    if finger_count == 5:
        return "OPEN HAND"

    # --------------------------------------------------------
    # PEACE SIGN
    # --------------------------------------------------------

    if (
        index
        and middle
        and not ring
        and not pinky
    ):
        return "PEACE"

    # --------------------------------------------------------
    # POINTING
    # --------------------------------------------------------

    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "POINTING"

    # --------------------------------------------------------
    # THUMBS UP
    # --------------------------------------------------------

    if (
        thumb
        and not index
        and not middle
        and not ring
        and not pinky
    ):

        # Make sure thumb is actually pointing upward.
        if hand_landmarks[4].y < hand_landmarks[2].y:
            return "THUMBS UP"

    return "UNKNOWN"


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("--------------------------------")
print("AI HAND TRACKING")
print("--------------------------------")
print("POINTING = Draw")
print("OPEN HAND = Stop drawing")
print("FIST = Clear drawing")
print("Q = Quit")
print("--------------------------------")


# ============================================================
# DRAWING CANVAS
# ============================================================

canvas = None

previous_point = None


# ============================================================
# TIMING
# ============================================================

start_time = time.monotonic()

previous_time = time.time()


# ============================================================
# MAIN PROGRAM LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # ========================================================
    # CREATE DRAWING CANVAS
    # ========================================================

    if canvas is None:

        canvas = np.zeros_like(frame)


    # ========================================================
    # CONVERT CAMERA IMAGE
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ========================================================
    # MEDIAPIPE TIMESTAMP
    # ========================================================

    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )


    # ========================================================
    # DETECT HANDS
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # Default display information

    detected_hand = "None"
    confidence = 0
    gesture = "None"
    finger_count = 0


    # ========================================================
    # PROCESS DETECTED HANDS
    # ========================================================

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        points = []


        # ====================================================
        # CONVERT LANDMARKS TO SCREEN COORDINATES
        # ====================================================

        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))


        # ====================================================
        # DRAW HAND CONNECTIONS
        # ====================================================

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )


        # ====================================================
        # DRAW 21 LANDMARKS
        # ====================================================

        for x, y in points:

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # ====================================================
        # LEFT / RIGHT HAND
        # ====================================================

        hand_label = "Unknown"
        hand_score = 0

        if (
            len(result.handedness) > hand_index
            and len(result.handedness[hand_index]) > 0
        ):

            category = result.handedness[hand_index][0]

            hand_label = category.category_name

            hand_score = category.score


        detected_hand = hand_label

        confidence = int(hand_score * 100)


        # ====================================================
        # COUNT FINGERS
        # ====================================================

        fingers = get_fingers(
            hand_landmarks,
            hand_label
        )

        finger_count = sum(fingers)


        # ====================================================
        # RECOGNIZE GESTURE
        # ====================================================

        gesture = recognize_gesture(
            fingers,
            hand_landmarks
        )


        # ====================================================
        # INDEX FINGER LOCATION
        # ====================================================

        index_x = points[8][0]
        index_y = points[8][1]


        # ====================================================
        # AIR DRAWING
        # ====================================================

        if gesture == "POINTING":

            # Show drawing cursor

            cv2.circle(
                frame,
                (index_x, index_y),
                12,
                (0, 255, 255),
                -1
            )

            current_point = (
                index_x,
                index_y
            )


            # Draw line from previous position
            # to current finger position.

            if previous_point is not None:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    (255, 0, 255),
                    6
                )

            previous_point = current_point


        # ====================================================
        # OPEN HAND = STOP DRAWING
        # ====================================================

        elif gesture == "OPEN HAND":

            previous_point = None


        # ====================================================
        # FIST = CLEAR SCREEN
        # ====================================================

        elif gesture == "FIST":

            canvas[:] = 0

            previous_point = None


        else:

            previous_point = None


    # ========================================================
    # COMBINE DRAWING WITH CAMERA
    # ========================================================

    frame = cv2.add(
        frame,
        canvas
    )


    # ========================================================
    # FPS COUNTER
    # ========================================================

    current_time = time.time()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (310, 180),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "AI HAND TRACKING",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Hand: {detected_hand}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence}%",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Fingers Up: {finger_count}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (25, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "POINT = Draw | OPEN = Stop | FIST = Clear | Q = Quit",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "AI Hand Tracking",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEAN UP
# ============================================================

camera.release()

landmarker.close()

cv2.destroyAllWindows()