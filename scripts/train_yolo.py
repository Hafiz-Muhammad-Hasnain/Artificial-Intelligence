from ultralytics import YOLO
import torch
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Train YOLOv8 on local fruit dataset
# Requires GPU for speed, but will run on CPU slowly

def main():
    # Initialize Weights & Biases
    wandb.init(
        project="fruit-detection",
        name="yolov8-fruits",
        config={
            "model": "yolov8s",
            "epochs": 50,
            "imgsz": 416,
            "batch": 8,
            "confidence_threshold": 0.5,
        }
    )

    # Use pretrained YOLOv8s from archive
    weights = 'archive/Fruits-detection/yolov8s.pt'
    model = YOLO(weights)
    
    # Add W&B callback for logging
    add_wandb_callback(model, enable_model_checkpointing=True)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    model.train(
        data='data/data.yaml',
        epochs=50,
        imgsz=416,
        batch=8,
        device=device,
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=True,
        workers=0,
    )
    
    # Log final metrics to W&B
    wandb.log({
        "final_model": "runs/detect/train/weights/best.pt"
    })
    
    wandb.finish()
    print("Training complete. Best weights saved to runs/detect/train/weights/best.pt")


if __name__ == '__main__':
    main()
