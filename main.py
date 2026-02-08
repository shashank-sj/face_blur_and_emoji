import cv2
from ultralytics import YOLO

from expressions.fer_model import FERModel
from expressions.smoother import ExpressionSmoother
from overlays.emoji import overlay_emoji


video_path = "input.mp4"
out_path = "output.mp4"

cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# YOLO face model
model = YOLO("weights/yolov8n-face.pt")

# FER
fer = FERModel("weights/fer_simple.pt")
smoother = ExpressionSmoother(window=7)

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5,
        verbose=False
    )

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy()

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            # Run FER every N frames (speed!)
            if frame_id % 5 == 0:
                expr, conf = fer.predict(face)
            else:
                expr, conf = "neutral", 0.0

            final_expr = smoother.update(int(tid), expr, conf)

            overlay_emoji(frame, (x1, y1, x2, y2), final_expr)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
print("✅ Done →", out_path)
