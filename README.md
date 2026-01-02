# Fruit Detection and Pricing System

This project implements the proposal for an AI-based fruit detection and pricing system using YOLOv8. It detects multiple fruits in an image, identifies types, estimates quantity and color, and calculates the total price using predefined rates. A simple Streamlit web app is included for easy interaction.
// python -m src.predict archive/Fruits-detection/test/images/<image>.jpg --conf 0.25
// streamlit run app.py
## Features
- YOLOv8-based fruit detection (Apple, Banana, Grape, Orange, Pineapple, Watermelon)
- Quantity counting per class
- Simple color estimation (HSV-based buckets)
- Pricing integration via configurable dictionary
- Training script to fine-tune YOLOv8 on provided dataset
- CLI prediction and Streamlit UI

## Project Structure
- `archive/Fruits-detection/`: YOLO-format dataset (images + labels)
- `data/data.yaml`: Local dataset config for Ultralytics
- `src/`: Core modules
  - `detector.py`: YOLO wrapper (loads `runs/detect/train/weights/best.pt` if present, else falls back to `archive/yolov8s.pt`)
  - `color_analysis.py`: Simple HSV color bucketing
  - `price_config.py`: Price dictionary
  - `predict.py`: CLI for detection + pricing
  - `app_streamlit.py`: Web UI
- `scripts/train_yolo.py`: YOLOv8 training
- `requirements.txt`: Dependencies
- `docs/Project-Proposal.txt`: Extracted proposal text

## Setup
### 1) Create and activate virtual environment
On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

## Training (per proposal)
Fine-tune YOLOv8 on the local dataset:
```powershell
python scripts/train_yolo.py
```
This writes weights to `runs/detect/train/weights/best.pt`. The app/CLI will use this automatically.

Notes:
- Adjust epochs/batch/imgsz in `scripts/train_yolo.py` based on your hardware.
- GPU is recommended for reasonable training times.

## Inference via CLI
Run detection on an image and compute pricing:
```powershell
python -m src.predict archive/Fruits-detection/test/images/<some_image>.jpg --conf 0.5
```
Outputs an annotated image under `outputs/` and prints a summary.

## Streamlit Web App
Launch the app to upload images:
```powershell
streamlit run src/app_streamlit.py
```
Features:
- Confidence slider
- Editable prices in sidebar
- Annotated image, detections table, color estimates, and total price

## Color Estimation
Color is estimated using simple HSV buckets (Red, Yellow, Green, Orange). This is a heuristic and can be improved with more advanced color analysis.

## Performance Evaluation
- After training, measure mAP@0.5 and precision/recall via Ultralytics training logs.
- Validate quantity counts against labels on the validation set.
- Observe response time in Streamlit.

## Future Work
- Improve color estimation robustness
- Add Weights & Biases logging
- Support camera feed in the app
- Expand classes/prices
