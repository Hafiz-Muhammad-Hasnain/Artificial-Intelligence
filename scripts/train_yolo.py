from ultralytics import YOLO
import torch

# Train YOLOv8 on local fruit dataset
# Requires GPU for speed, but will run on CPU slowly

def main():
    # Use pretrained YOLOv8s from archive
    weights = 'archive/Fruits-detection/yolov8s.pt'
    model = YOLO(weights)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model.train(
        data='data/data.yaml',
        epochs=1,
        imgsz=416,
        batch=8,
        device=device,
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=True,
        workers=0,
        fraction=0.1,  # Use only 10% data for quick demo
    )
    print("Training complete. Best weights saved to runs/detect/train/weights/best.pt")


if __name__ == '__main__':
    main()
