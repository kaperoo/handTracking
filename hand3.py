import cv2
import mediapipe as mp
import time
import pyautogui as pg

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while cap.isOpened():
    success, img = cap.read()
    
    # Flip the image horizontally for a later selfie-view display, and convert
    img = cv2.flip(img, 1)

    # the BGR image to RGB.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Measure and display FPS
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    # get the position of the index finger
    if results.multi_hand_landmarks:
        index = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index.x*1920), int(index.y*1080)
        pg.moveTo(x, y)

    cv2.imshow('MediaPipe Hands', img)
    cv2.waitKey(1)