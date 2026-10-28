
import cv2
import time
from transformers import pipeline

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output_annotated.mp4"
FRAME_STEP = 5
FACE_THRESHOLD = 0.70
EMOTION_THRESHOLD = 0.60

print("Loading facial-expression model...", flush=True)

classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=-1
)

print("Loading face detector...", flush=True)

detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit("Cannot open input.mp4")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    max(fps / FRAME_STEP, 1),
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise SystemExit("Cannot create output video.")

emotion_counts = {}
frame_number = 0
processed_frames = 0
start_time = time.time()

print(f"Video frames: {total_frames}", flush=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    current_frame = frame_number
    frame_number += 1

    if current_frame % FRAME_STEP != 0:
        continue

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        face_confidence = float(detections[0, 0, i, 2])

        if face_confidence < FACE_THRESHOLD:
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
        prediction = classifier(face_rgb, top_k=1)[0]

        score = float(prediction["score"])
        label = prediction["label"].upper()

        if score < EMOTION_THRESHOLD:
            label = "UNCERTAIN"

        emotion_counts[label] = emotion_counts.get(label, 0) + 1

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"{label} ({score:.0%})",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)
    processed_frames += 1

    if processed_frames % 30 == 0:
        print(
            f"Processed {frame_number}/{total_frames}",
            flush=True
        )

cap.release()
out.release()

print("\n===== EXPRESSION ANALYSIS SUMMARY =====")
print(f"Frames analyzed: {processed_frames}")
print(f"Time taken: {time.time() - start_time:.1f} seconds")

for label, count in sorted(emotion_counts.items()):
    print(f"{label}: {count}")

print(f"\nAnnotated video saved as: {OUTPUT_PATH}")
