
import cv2
from transformers import pipeline

MODEL_NAME = "dima806/facial_emotions_image_detection"

print("Loading alternative emotion model...", flush=True)

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
    device=-1
)

detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

cap = cv2.VideoCapture("input.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)

ok, frame = cap.read()
if not ok:
    cap.release()
    raise SystemExit("Could not read input video.")

height, width = frame.shape[:2]

blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)

detector.setInput(blob)
detections = detector.forward()

face_count = 0

for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])

    if confidence < 0.7:
        continue

    box = detections[0, 0, i, 3:7] * [
        width, height, width, height
    ]
    x1, y1, x2, y2 = box.astype(int)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x2 <= x1 or y2 <= y1:
        continue

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        continue

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    predictions = classifier(face_rgb, top_k=3)

    print(f"\nFace {face_count + 1} — top predictions:")
    for prediction in predictions:
        print(
            f"{prediction['label']}: "
            f"{prediction['score']:.1%}"
        )

    face_count += 1
    if face_count >= 5:
        break

cap.release()
print(f"\nTest complete. Faces checked: {face_count}")
