import cv2
import mediapipe as mp
import os

os.makedirs("test_output_mediapipe", exist_ok=True)

cap = cv2.VideoCapture("input.mp4")
if not cap.isOpened():
    raise SystemExit("Cannot open input.mp4")

detector_module = mp.solutions.face_detection
frame_number = 0
saved = 0

with detector_module.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
) as detector:
    while saved < 10:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_number % 150 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            count = 0

            if result.detections:
                height, width = frame.shape[:2]
                for detection in result.detections:
                    box = detection.location_data.relative_bounding_box
                    x = max(0, int(box.xmin * width))
                    y = max(0, int(box.ymin * height))
                    w = int(box.width * width)
                    h = int(box.height * height)
                    cv2.rectangle(
                        frame, (x, y),
                        (min(x + w, width - 1), min(y + h, height - 1)),
                        (0, 255, 0), 2
                    )
                    count += 1

            saved += 1
            cv2.imwrite(f"test_output_mediapipe/frame_{saved:02d}.jpg", frame)
            print(f"Test image {saved}: {count} faces detected", flush=True)

        frame_number += 1

cap.release()
print("Done! Check test_output_mediapipe.")
