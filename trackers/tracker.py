from ultralytics.trackers.byte_tracker import BYTETracker
import numpy as np


class Tracker:
    def __init__(self, fps=30):
        self.tracker = BYTETracker(
            track_thresh=0.4,
            match_thresh=0.8,
            frame_rate=fps
        )

    def update(self, detections):
        """
        Args:
            detections: List of [x1, y1, x2, y2, conf]

        Returns:
            List of [x1, y1, x2, y2, track_id]
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets, None)

        results = []
        for t in tracks:
            x1, y1, x2, y2, track_id = t[:5]
            results.append([x1, y1, x2, y2, int(track_id)])

        return results
