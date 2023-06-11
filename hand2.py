import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, image = webcam.read()

    results = mp_hands.Hands().process(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break
webcam.release()

