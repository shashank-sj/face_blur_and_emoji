import cv2

EMOJI_MAP = {
    "happy": "assets/emojis/happy.png",
    "sad": "assets/emojis/sad.png",
    "angry": "assets/emojis/angry.png",
    "surprise": "assets/emojis/surprise.png",
    "neutral": "assets/emojis/neutral.png",
}


def overlay_emoji(frame, bbox, emotion):
    if emotion not in EMOJI_MAP:
        emotion = "neutral"

    emoji = cv2.imread(EMOJI_MAP[emotion], cv2.IMREAD_UNCHANGED)
    if emoji is None:
        return

    x1, y1, x2, y2 = map(int, bbox)
    size = max(1, x2 - x1)

    emoji = cv2.resize(emoji, (size, size))

    y = max(0, y1 - size)
    x = max(0, x1)

    h, w = emoji.shape[:2]

    alpha = emoji[:, :, 3] / 255.0
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (
            alpha * emoji[:, :, c]
            + (1 - alpha) * frame[y:y+h, x:x+w, c]
        )
