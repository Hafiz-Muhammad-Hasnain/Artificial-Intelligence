import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from .detector import FruitDetector
from .color_analysis import dominant_color_hsv
from .price_config import PRICES

st.set_page_config(page_title="Fruit Detection & Pricing", layout="wide")

st.title("Fruit Detection and Pricing System")
st.write("Upload an image with fruits. The app will detect, count, estimate color, and compute total price.")

conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

# Allow adjusting prices
st.sidebar.subheader("Prices (per fruit)")
prices = {k: st.sidebar.number_input(k, min_value=0.0, value=v, step=0.1) for k, v in PRICES.items()}

uploaded = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Save to temp
    tmp_dir = Path(".streamlit-cache")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getvalue())

    det = FruitDetector(conf=conf)
    detections = det.detect(tmp_path)
    img = cv2.imread(str(tmp_path))
    annotated = det.annotate(tmp_path, detections)

    # Compute counts and total
    counts = {}
    total = 0.0
    colors = []
    for d in detections:
        cls = d["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
        total += prices.get(cls, 0.0)
        color = dominant_color_hsv(img, tuple(d["bbox"]))
        colors.append((cls, color, d['confidence']))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annotated Image")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Detections")
        if detections:
            st.table({
                "Class": [d["class_name"] for d in detections],
                "Confidence": [round(d["confidence"], 3) for d in detections]
            })
        else:
            st.info("No detections above threshold. Try lowering confidence or train the model.")

    st.subheader("Summary")
    st.write({cls: counts.get(cls, 0) for cls in prices.keys()})
    st.write(f"Total Price: {total:.2f}")

    st.subheader("Colors")
    if colors:
        st.table({
            "Class": [c[0] for c in colors],
            "Color": [c[1] for c in colors],
            "Confidence": [round(c[2], 3) for c in colors]
        })
