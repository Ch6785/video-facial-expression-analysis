
import cv2
from transformers import pipeline

VIDEO_PATH = "input.mp4"
FRAME_STEP = 5
TEST_FRAMES = 10

print("Loading face detector...", flush=True)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise SystemExit("Cannot open input.mp4")

import os
os.makedirs("test_output", exist_ok=True)

frame_number = 0
saved = 0

while saved < TEST_FRAMES:
    ok, frame = video.read()

    if not ok:
        break

    if frame_number % FRAME_STEP == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

        cv2.imwrite(
            f"test_output/frame_{saved + 1:02d}.jpg",
            frame
        )

        print(
            f"Frame {saved + 1}: {len(faces)} face detections",
            flush=True
        )

        saved += 1

    frame_number += 1

video.release()
print("Done! Check the test_output folder.")
