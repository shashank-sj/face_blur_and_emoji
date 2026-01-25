import torch
import torch.nn as nn
import cv2
import numpy as np


class SimpleFER(nn.Module):
    """
    Lightweight FER CNN (FER2013 style)
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
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

        self.classifier = nn.Linear(128, 7)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class FERModel:
    def __init__(self, weights_path, device="cpu"):
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

    def predict(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return "neutral", 0.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))

        tensor = torch.tensor(
            resized, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0) / 255.0

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        confidence = probs[idx].item()

        return self.labels[idx], confidence
