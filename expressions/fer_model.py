import cv2
import torch
import numpy as np


class FERModel:
    def __init__(self, model):
        self.model = model.eval()
        self.labels = [
            "angry", "disgust", "fear",
            "happy", "sad", "surprise", "neutral"
        ]

    def predict(self, face_bgr):
        if face_bgr.size == 0:
            return "neutral", 0.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))

        tensor = torch.tensor(resized, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        return self.labels[idx], probs[idx].item()
