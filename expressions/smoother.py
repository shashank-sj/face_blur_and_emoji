from collections import deque, Counter


class ExpressionSmoother:
    def __init__(self, window=7, min_conf=0.5):
        self.window = window
        self.min_conf = min_conf
        self.history = {}  # track_id -> deque

    def update(self, track_id, expr, conf):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window)

        # only trust confident predictions
        if conf >= self.min_conf:
            self.history[track_id].append(expr)

        if len(self.history[track_id]) == 0:
            return "neutral"

        # majority vote
        return Counter(self.history[track_id]).most_common(1)[0][0]
