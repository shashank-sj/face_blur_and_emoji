import cv2

EMOJI_PATHS = {
    "happy": "assets/emojis/happy.png",
    "sad": "assets/emojis/sad.png",
    "angry": "assets/emojis/angry.png",
    "neutral": "assets/emojis/neutral.png",
    "surprise": "assets/emojis/surprise.png"
}


def overlay_emoji(frame, bbox, emotion):
    if emotion not in EMOJI_PATHS:
        emotion = "neutral"

    emoji = cv2.imread(EMOJI_PATHS[emotion], cv2.IMREAD_UNCHANGED)
    if emoji is None:
        return

    x1, y1, x2, y2 = map(int, bbox)
    size = max(1, x2 - x1)

    emoji = cv2.resize(emoji, (size, size))

    y = max(0, y1 - size)
    x = max(0, x1)

    frame[y:y+size, x:x+size] = emoji[:, :, :3]
