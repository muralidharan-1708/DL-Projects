import cv2
import mediapipe as mp
import pyautogui
import math

#Mediapipe --> hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#to get a screen size
screen_w, screen_h = pyautogui.size()

#for the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
smoothed_x, smoothed_y = 0, 0
smoothing_factor = 0.5
prev_y = 0  # For scroll detection

# Utility Functions
def fingers_up(landmarks):
    """Return [Index, Middle, Ring, Pinky] finger states"""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = []
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            up.append(True)
        else:
            up.append(False)
    return up

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]

            # to capture the cursor movement 
            cursor_x = int(index_tip.x * screen_w)
            cursor_y = int(index_tip.y * screen_h)

            smoothed_x = int(smoothed_x * (1 - smoothing_factor) + cursor_x * smoothing_factor)
            smoothed_y = int(smoothed_y * (1 - smoothing_factor) + cursor_y * smoothing_factor)

            pyautogui.moveTo(smoothed_x, smoothed_y)

            # hand - gesture detection
            fingers = fingers_up(landmarks)

            # Left - click: Index finger only up
            if fingers[0] and not any(fingers[1:]):
                pyautogui.click()
                cv2.putText(frame, "Left Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Right - click : Index + Middle up
            elif fingers[0] and fingers[1] and not any(fingers[2:]):
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Scroll: Hand moves up/down
            current_y = index_tip.y
            dy = current_y - prev_y

            if abs(dy) > 0.02:
                scroll_amount = -10 if dy > 0 else 10
                pyautogui.scroll(scroll_amount)
                cv2.putText(frame, "Scrolling", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            prev_y = current_y

            # Show cursor circle
            cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 10, (255, 0, 0), 2)

    # UI Text
    cv2.putText(frame, "Index Up = Left Click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "2 Fingers Up = Right Click", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Move Hand Up/Down = Scroll", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'Q' to Quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display Frame
    cv2.imshow("Virtual Mouse Controller", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()