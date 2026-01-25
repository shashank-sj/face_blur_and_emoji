class Tracker:
    def __init__(self, yolo_model):
        """
        Args:
            yolo_model: ultralytics.YOLO instance
        """
        self.model = yolo_model

    def update(self, frame):
        """
        Args:
            frame (np.ndarray)

        Returns:
            List of [x1, y1, x2, y2, track_id, conf]
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False
        )

        tracks = []

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                tracks.append([x1, y1, x2, y2, int(tid), float(conf)])

        return tracks
