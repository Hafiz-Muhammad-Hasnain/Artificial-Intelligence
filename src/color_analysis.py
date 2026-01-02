from typing import Tuple
import numpy as np
import cv2

# Simple color buckets in HSV for demo
COLOR_BUCKETS = {
    "Red": ((0, 70, 50), (10, 255, 255)),
    "Red2": ((170, 70, 50), (180, 255, 255)),
    "Yellow": ((20, 70, 50), (35, 255, 255)),
    "Green": ((35, 70, 50), (85, 255, 255)),
    "Orange": ((10, 70, 50), (20, 255, 255)),
}


def dominant_color_hsv(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    scores = {}
    for name, (low, high) in COLOR_BUCKETS.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        scores[name] = int(mask.sum())
    # Merge Red and Red2
    total_red = scores.get("Red", 0) + scores.get("Red2", 0)
    scores["Red"] = total_red
    if "Red2" in scores:
        del scores["Red2"]
    # Choose max
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best
