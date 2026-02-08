import torch
import torch.nn as nn
import cv2
import numpy as np


# -----------------------------
# SAME model architecture used in training
# -----------------------------
class SimpleFER(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, 7)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------
# Inference wrapper
# -----------------------------
class FERModel:
    def __init__(self, weights_path="weights/fer_simple.pt", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.model = SimpleFER().to(device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        self.model.eval()

        self.labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral"
        ]

    def preprocess(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        img = gray.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

        return img.to(self.device)

    def predict(self, face_bgr):
        tensor = self.preprocess(face_bgr)
        if tensor is None:
            return "neutral", 0.0

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)

        expr = self.labels[idx.item()]
        confidence = conf.item()

        # Confidence gating (VERY important)
        if confidence < 0.5:
            return "neutral", confidence

        return expr, confidence
