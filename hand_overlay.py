import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# ==============================
# MediaPipe Setup
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

alpha = 0.6
prev_pts = None
trail = [deque(maxlen=15) for _ in range(10)]

mode = "elastic"
active_spell = "None"
combo_count = 0
last_gesture_time = 0
last_switch_time = 0
bg = None

# ==============================
# Smooth
# ==============================
def smooth(points):
    global prev_pts
    if prev_pts is None:
        prev_pts = points
        return points

    sm = []
    for c, p in zip(points, prev_pts):
        sm.append((alpha*c[0] + (1-alpha)*p[0],
                   alpha*c[1] + (1-alpha)*p[1]))
    prev_pts = sm
    return sm

# ==============================
# Gesture Recognition
# ==============================
def detect_gesture(hand):
    fingers = []

    # Tip ids
    tips = [4,8,12,16,20]

    for tip in tips[1:]:
        if hand.landmark[tip].y < hand.landmark[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    count = sum(fingers)

    if count == 0:
        return "Fist"
    elif count == 4:
        return "Open Palm"
    elif count == 2:
        return "Two Fingers"
    else:
        return "None"

# ==============================
# Elastic String
# ==============================
def elastic_string(frame, p1, p2, color):
    midx = (p1[0] + p2[0]) // 2
    midy = (p1[1] + p2[1]) // 2
    tension = int(12 * math.sin(time.time()*4))
    curve = np.array([p1, (midx, midy+tension), p2], np.int32)

    # === OUTER GLOW ===
    overlay1 = frame.copy()
    cv2.polylines(overlay1, [curve], False, color, 10)
    cv2.addWeighted(overlay1, 0.15, frame, 0.85, 0, frame)

    # === MID GLOW ===
    overlay2 = frame.copy()
    cv2.polylines(overlay2, [curve], False, color, 6)
    cv2.addWeighted(overlay2, 0.25, frame, 0.75, 0, frame)

    # === CORE LINE ===
    cv2.polylines(frame, [curve], False, (255,255,255), 2)
    

# ==============================
# HUD Panel
# ==============================
def draw_hud(frame):
    overlay = frame.copy()

    cv2.rectangle(overlay, (10,10), (260,160), (0,255,255), 2)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    cv2.putText(frame, f"MODE: {mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"SPELL: {active_spell}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"COMBO: {combo_count}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

# ==============================
# Crosshair
# ==============================
def draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2
    cv2.circle(frame, (cx,cy), 20, (0,255,0), 2)
    cv2.line(frame, (cx-30,cy), (cx+30,cy), (0,255,0), 1)
    cv2.line(frame, (cx,cy-30), (cx,cy+30), (0,255,0), 1)

# ==============================
# Main Loop
# ==============================
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    if bg is None:
        bg = np.zeros_like(frame)
        h, w, _ = frame.shape
        for _ in range(50):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            cv2.circle(bg, (x,y), 1, (255,255,255), -1)

    frame = cv2.addWeighted(frame, 0.7, bg, 0.3, 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left = None
    right = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            if label == "Left":
                left = hand_landmarks
            elif label == "Right":
                right = hand_landmarks

    if left and right:

        gesture = detect_gesture(left)

        if gesture != "None" and time.time() - last_gesture_time > 0.5:
            active_spell = gesture
            combo_count += 1
            last_gesture_time = time.time()

        h, w, _ = frame.shape
        tips = [4,8,12,16,20]

        left_pts = [(left.landmark[i].x*w,
                     left.landmark[i].y*h) for i in tips]

        right_pts = [(right.landmark[i].x*w,
                      right.landmark[i].y*h) for i in tips]

        pts = smooth(left_pts + right_pts)
        left_sm = pts[:5]
        right_sm = pts[5:]

        for p1, p2 in zip(left_sm, right_sm):
            p1 = (int(p1[0]), int(p1[1]))
            p2 = (int(p2[0]), int(p2[1]))

            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            intensity = min(255, int(dist))
            color = (intensity, 200-intensity//2, 255-intensity//3)

            if mode == "elastic":
                elastic_string(frame, p1, p2, color)
            elif mode == "lightning":
                cv2.line(frame, p1, p2, (255,255,255), 3)
            elif mode == "beam":
                cv2.line(frame, p1, p2, (0,255,255), 3)

        # Pinch mode switch
        thumb = left_sm[0]
        index = left_sm[1]

        if np.hypot(thumb[0]-index[0],
                    thumb[1]-index[1]) < 30:
            if time.time() - last_switch_time > 0.5:
                if mode == "elastic":
                    mode = "lightning"
                elif mode == "lightning":
                    mode = "beam"
                else:
                    mode = "elastic"
                last_switch_time = time.time()

    draw_hud(frame)
    draw_crosshair(frame)

    fps = 1/(time.time()-prev_time)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {int(fps)}", (20,190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("AR HOLOGRAPHIC ENGINE", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()