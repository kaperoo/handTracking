import mediapipe as mp
import cv2
import numpy as np
from draw import draw_landmarks_on_image

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(3)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # convert image to numpy frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # get the timestamp of the current frame
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)
        
cap.release()

