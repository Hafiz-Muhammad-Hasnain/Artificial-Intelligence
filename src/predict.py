import argparse
from pathlib import Path
import cv2
from .detector import FruitDetector
from .color_analysis import dominant_color_hsv
from .price_config import PRICES


def main():
    parser = argparse.ArgumentParser(description="Fruit detection and pricing")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    det = FruitDetector(conf=args.conf)
    detections = det.detect(args.image)
    img = cv2.imread(args.image)

    # Quantity per class and pricing
    counts = {}
    total = 0.0
    for d in detections:
        cls = d["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
        price = PRICES.get(cls, 0.0)
        total += price
        color = dominant_color_hsv(img, tuple(d["bbox"]))
        print(f"Detected {cls} (color: {color}) conf={d['confidence']:.2f}")

    print("\nSummary:")
    for cls, c in counts.items():
        print(f"  {cls}: {c} x {PRICES.get(cls, 0.0):.2f}")
    print(f"Total price: {total:.2f}")

    # Save annotated image
    annotated = det.annotate(args.image, detections)
    out_path = Path("outputs") / (Path(args.image).stem + "_det.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image: {out_path}")


if __name__ == "__main__":
    main()
