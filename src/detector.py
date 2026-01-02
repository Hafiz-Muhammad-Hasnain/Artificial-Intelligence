from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
from .utils import load_class_names

# Project root for finding weights
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FruitDetector:
    def __init__(self, weights: str | Path | None = None, conf: float = 0.5):
        self.conf = conf
        if weights is None:
            # Try using trained weights if available, else fallback to base model
            trained = PROJECT_ROOT / "runs/detect/train/weights/best.pt"
            if trained.exists():
                weights = trained
            else:
                weights = PROJECT_ROOT / "archive/Fruits-detection/yolov8s.pt"
        self.model = YOLO(str(weights))
        self.class_names = load_class_names()

    def detect(self, image_path: str | Path) -> List[Dict[str, Any]]:
        results = self.model.predict(source=str(image_path), conf=self.conf, verbose=False)
        detections: List[Dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int)[0]  # [x1, y1, x2, y2]
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                score = float(b.conf.item()) if b.conf is not None else 0.0
                name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                detections.append({
                    "bbox": xyxy.tolist(),
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": score,
                })
        return detections

    def annotate(self, image_path: str | Path, detections: List[Dict[str, Any]]) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img
