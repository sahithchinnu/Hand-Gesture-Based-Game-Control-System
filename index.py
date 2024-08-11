import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

def count_open_fingers(landmarks):
    # Count how many fingers are open
    open_fingers = 0

    # Thumb: check if the tip is higher (less y-coordinate) than the base
    if landmarks[4].x < landmarks[3].x:
        open_fingers += 1

    # Index finger
    if landmarks[8].y < landmarks[6].y:
        open_fingers += 1

    # Middle finger
    if landmarks[12].y < landmarks[10].y:
        open_fingers += 1

    # Ring finger
    if landmarks[16].y < landmarks[14].y:
        open_fingers += 1

    # Pinky
    if landmarks[20].y < landmarks[18].y:
        open_fingers += 1

    return open_fingers

def thumb_open(landmarks):
    # Check if the thumb is open
    return landmarks[4].x < landmarks[3].x

def all_fingers_open(landmarks):
    # Check if all fingers are open
    return count_open_fingers(landmarks) == 5

def no_fingers_open(landmarks):
    # Check if no fingers are open (all fingers closed)
    return count_open_fingers(landmarks) == 0

def other_fingers_closed(landmarks):
    # Check if all fingers except the thumb are closed
    return count_open_fingers(landmarks) == 1 and thumb_open(landmarks)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to make it more intuitive
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            if thumb_open(landmarks) and other_fingers_closed(landmarks):
                pyautogui.press('w')
                cv2.putText(frame, "Thumb Open & Other Fingers Closed: 'w' Pressed!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            elif no_fingers_open(landmarks):
                pyautogui.press('s')
                cv2.putText(frame, "All Fingers Closed: 's' Pressed!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            elif all_fingers_open(landmarks):
                cv2.putText(frame, "All Fingers Open: No Key Pressed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif count_open_fingers(landmarks) == 2:
                pyautogui.press('a')
                cv2.putText(frame, "Two Fingers Open: 'a' Pressed!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            elif count_open_fingers(landmarks) == 3:
                pyautogui.press('d')
                cv2.putText(frame, "Three Fingers Open: 'd' Pressed!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Other Gesture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
