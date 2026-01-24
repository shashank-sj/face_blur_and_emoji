from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path="weights/yolov8n-face.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Args:
            frame (np.ndarray): BGR image

        Returns:
            List of [x1, y1, x2, y2, confidence]
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=0.5,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

        return detections
