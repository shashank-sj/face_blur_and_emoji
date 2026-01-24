from collections import deque, Counter


class ExpressionSmoother:
    def __init__(self, window=7, min_conf=0.5):
        self.window = window
        self.min_conf = min_conf
        self.history = {}

    def update(self, track_id, expr, conf):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window)

        if conf >= self.min_conf:
            self.history[track_id].append(expr)

        if len(self.history[track_id]) == 0:
            return "neutral"

        return Counter(self.history[track_id]).most_common(1)[0][0]
