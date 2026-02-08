import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


# -----------------------------
# Dataset
# -----------------------------
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split="Training"):
        df = pd.read_csv(csv_path)
        self.df = df[df["Usage"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pixels = np.array(row["pixels"].split(), dtype=np.float32)
        img = pixels.reshape(48, 48) / 255.0

        # augment (light, safe)
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()  # <-- IMPORTANT

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = int(row["emotion"])

        return img, label


# -----------------------------
# Model (simple & stable)
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
# Train
# -----------------------------
def train(csv_path, device="cpu"):
    train_ds = FER2013Dataset(csv_path, "Training")
    val_ds   = FER2013Dataset(csv_path, "PublicTest")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    model = SimpleFER().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 26):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch}: loss={running_loss:.3f}, val_acc={acc:.3f}")

    torch.save(model.state_dict(), "weights/fer_simple.pt")
    print("✅ Saved weights → weights/fer_simple.pt")


# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(r"C:\Users\dell\Desktop\face_blur\face_blur_and_emoji\data\fer2013.csv\fer2013.csv", device)
