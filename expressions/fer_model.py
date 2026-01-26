import torch
import cv2
import numpy as np


class FERModel:
    def __init__(self, device=None):
        """
        Loads pretrained FER ResNet18 via torch.hub
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Load pretrained FER model
        self.model = torch.hub.load(
            "nateraw/fer",
            "resnet18",
            pretrained=True
        ).to(device)

        self.model.eval()

        # Label order used by this model
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
        """
        Input: BGR face crop
        Output: normalized tensor [1, 3, 224, 224]
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))

        tensor = torch.from_numpy(face_rgb).float()
        tensor = tensor.permute(2, 0, 1) / 255.0

        # ImageNet normalization (important!)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tensor = (tensor - mean) / std

        return tensor.unsqueeze(0).to(self.device)

    def predict(self, face_bgr):
        """
        Returns:
            expression (str)
            confidence (float)
        """
        tensor = self.preprocess(face_bgr)
        if tensor is None:
            return "neutral", 0.0

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        confidence = probs[idx].item()

        return self.labels[idx], confidence
