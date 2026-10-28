import cv2
import os

VIDEO = "input.mp4"
PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

detector = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit("Cannot open input.mp4")

os.makedirs("test_output_dnn", exist_ok=True)

frame_number = 0
saved = 0

while saved < 10:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_number % 150 == 0:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        detector.setInput(blob)
        detections = detector.forward()
        count = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * [
                width, height, width, height
            ]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1

        saved += 1
        cv2.imwrite(f"test_output_dnn/frame_{saved:02d}.jpg", frame)
        print(f"Test image {saved}: {count} faces detected", flush=True)

    frame_number += 1

cap.release()
print("Done! Open the test_output_dnn folder.")
import cv2
import os

VIDEO = "input.mp4"
PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

detector = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit("Cannot open input.mp4")

os.makedirs("test_output_dnn", exist_ok=True)

frame_number = 0
saved = 0

while saved < 10:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_number % 150 == 0:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        detector.setInput(blob)
        detections = detector.forward()
        count = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * [
                width, height, width, height
            ]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1

        saved += 1
        cv2.imwrite(f"test_output_dnn/frame_{saved:02d}.jpg", frame)
        print(f"Test image {saved}: {count} faces detected", flush=True)

    frame_number += 1

cap.release()
print("Done! Open the test_output_dnn folder.")
