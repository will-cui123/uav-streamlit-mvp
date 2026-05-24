"""
Streamlit MVP for ENGG2112 UAV landing-zone segmentation.

This app loads the trained residual U-Net model from the notebook
`cnn_improved2_GPU.ipynb` and lets a user upload an image or choose a sample
image. The model outputs a pixel-level map with three classes:
safe, caution and unsafe.

Expected local files:
- app.py
- best_uav_unsafe_f1_unet.pth
- best_thresholds_unsafe_f1.json  optional, but recommended
- sample_images/                 optional folder for demo images
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix


# ============================================================
# App configuration and constants
# ============================================================

st.set_page_config(
    page_title="UAV Landing Zone Safety Segmentation",
    page_icon="🛰️",
    layout="wide",
)

IMG_SIZE = 256
NUM_CLASSES = 3
CLASS_NAMES = ["safe", "caution", "unsafe"]

# CSV mode is optional. It is included only for batch-style demos where a CSV
# points to local image files. The actual ML model is image-based, not tabular.
REQUIRED_CSV_COLUMNS = ["image_path"]

MODEL_PATH = Path("best_uav_unsafe_f1_unet.pth")
THRESHOLD_PATH = Path("best_thresholds_unsafe_f1.json")
SAMPLE_IMAGE_DIR = Path("sample_images")

DEFAULT_THRESHOLDS = {
    "unsafe_threshold": 0.37,
    "caution_threshold": 0.35,
    "postprocess_min_area": 25,
    "postprocess_close_kernel": 3,
    "postprocess_dilate_kernel": 3,
}

# RGB colours used in the displayed segmentation mask.
CLASS_COLOURS = {
    0: (36, 176, 82),     # safe = green
    1: (245, 196, 66),    # caution = yellow
    2: (224, 64, 64),     # unsafe = red
}

# ============================================================
# Model architecture copied from the training notebook
# ============================================================

class ResidualDoubleConv(nn.Module):
    """Two convolution layers with a residual skip connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),

            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class UNetBetter(nn.Module):
    """Small residual U-Net used for pixel-level UAV landing-zone segmentation."""

    def __init__(self, in_channels: int = 14, num_classes: int = 3):
        super().__init__()

        self.enc1 = ResidualDoubleConv(in_channels, 32, dropout=0.02)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualDoubleConv(32, 64, dropout=0.03)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualDoubleConv(64, 128, dropout=0.05)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResidualDoubleConv(128, 256, dropout=0.10)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualDoubleConv(256, 128, dropout=0.05)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualDoubleConv(128, 64, dropout=0.03)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualDoubleConv(64, 32, dropout=0.02)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.final(d1)


# ============================================================
# Feature generation copied from the training notebook
# ============================================================

def make_post_features(post_bgr: np.ndarray) -> np.ndarray:
    """
    Build the same 14-channel feature tensor used during model training.

    Input:
        post_bgr: resized OpenCV BGR image with shape H x W x 3.

    Output:
        H x W x 14 float32 array.
    """
    post_float = post_bgr.astype(np.float32) / 255.0

    post_gray = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    post_gray_ch = post_gray[..., None]

    hsv = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

    edges = cv2.Canny(post_bgr, 80, 160).astype(np.float32) / 255.0
    edges_ch = edges[..., None]

    edge_density = cv2.blur(edges.astype(np.float32), (5, 5))
    edge_density_ch = edge_density[..., None]

    gray_mean = cv2.blur(post_gray, (5, 5))
    gray_sq_mean = cv2.blur(post_gray ** 2, (5, 5))
    texture_var = gray_sq_mean - (gray_mean ** 2)
    texture_var = texture_var / (texture_var.max() + 1e-8)
    texture_var_ch = texture_var[..., None]

    sobel_x = cv2.Sobel(post_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(post_gray, cv2.CV_32F, 0, 1, ksize=3)

    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)
    gradient_mag_ch = gradient_mag[..., None]

    features = np.concatenate(
        [
            post_float,          # 3 channels
            post_gray_ch,        # 1 channel
            hsv,                 # 3 channels
            lab,                 # 3 channels
            edges_ch,            # 1 channel
            edge_density_ch,     # 1 channel
            texture_var_ch,      # 1 channel
            gradient_mag_ch,     # 1 channel
        ],
        axis=-1,
    )

    return features.astype(np.float32)


# ============================================================
# Prediction and post-processing helpers
# ============================================================

def predict_with_thresholds(
    probs: np.ndarray,
    unsafe_threshold: float = 0.37,
    caution_threshold: float = 0.35,
) -> np.ndarray:
    """
    Convert class probabilities into labels using tuned thresholds.

    probs shape:
        N x C x H x W

    The model first chooses between safe/caution, then upgrades pixels to
    caution or unsafe when probabilities cross the selected thresholds.
    """
    caution_p = probs[:, 1]
    unsafe_p = probs[:, 2]

    pred = np.argmax(probs[:, 0:2], axis=1).astype(np.uint8)
    pred[caution_p >= caution_threshold] = 1
    pred[unsafe_p >= unsafe_threshold] = 2

    return pred


def postprocess_unsafe_predictions(
    preds: np.ndarray,
    min_area: int = 25,
    close_kernel: int = 3,
    dilate_kernel: int = 3,
) -> np.ndarray:
    """
    Clean unsafe predictions by closing small gaps, removing tiny unsafe regions,
    and slightly expanding unsafe areas into neighbouring caution pixels.
    """
    processed = preds.copy()

    for i in range(processed.shape[0]):
        mask = processed[i].copy()
        unsafe = (mask == 2).astype(np.uint8)

        unsafe = cv2.morphologyEx(
            unsafe,
            cv2.MORPH_CLOSE,
            np.ones((close_kernel, close_kernel), np.uint8),
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            unsafe,
            connectivity=8,
        )

        cleaned = np.zeros_like(unsafe)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == label_id] = 1

        dilated = cv2.dilate(
            cleaned,
            np.ones((dilate_kernel, dilate_kernel), np.uint8),
            iterations=1,
        )

        mask[cleaned == 1] = 2
        mask[(dilated == 1) & (mask == 1)] = 2

        processed[i] = mask

    return processed


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device_name: str) -> UNetBetter:
    """Load the trained PyTorch model once and reuse it across Streamlit reruns."""
    device = torch.device(device_name)
    model = UNetBetter(in_channels=14, num_classes=NUM_CLASSES).to(device)

    try:
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Place best_uav_unsafe_f1_unet.pth in the same folder as app.py."
        ) from exc

    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint was found, but it does not match the U-Net architecture "
            "defined in this app. Check that you are using best_uav_unsafe_f1_unet.pth "
            "from the same training notebook."
        ) from exc


def load_thresholds(threshold_path: Path) -> Dict[str, float]:
    """Load threshold settings from JSON, falling back to notebook defaults."""
    if not threshold_path.exists():
        return DEFAULT_THRESHOLDS.copy()

    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        thresholds = DEFAULT_THRESHOLDS.copy()
        thresholds.update(loaded)
        return thresholds

    except (json.JSONDecodeError, OSError):
        st.warning(
            "Could not read best_thresholds_unsafe_f1.json. "
            "Using default threshold values from the notebook output."
        )
        return DEFAULT_THRESHOLDS.copy()


def run_segmentation(
    model: UNetBetter,
    image: Image.Image,
    device_name: str,
    thresholds: Dict[str, float],
    use_thresholds: bool,
    use_postprocessing: bool,
) -> Dict[str, np.ndarray | float | str | Dict[str, float]]:
    """
    Run the full inference pipeline:
    1. Resize uploaded image to 256x256.
    2. Build 14-channel features.
    3. Predict per-pixel class probabilities.
    4. Convert probabilities into labels.
    5. Resize output mask back to the original image size.
    """
    device = torch.device(device_name)

    original_rgb = np.array(image.convert("RGB"))
    original_h, original_w = original_rgb.shape[:2]

    resized_rgb = cv2.resize(
        original_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

    features = make_post_features(resized_bgr)
    tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    if use_thresholds:
        pred_small = predict_with_thresholds(
            probs,
            unsafe_threshold=float(thresholds["unsafe_threshold"]),
            caution_threshold=float(thresholds["caution_threshold"]),
        )
    else:
        pred_small = np.argmax(probs, axis=1).astype(np.uint8)

    if use_postprocessing:
        pred_small = postprocess_unsafe_predictions(
            pred_small,
            min_area=int(thresholds["postprocess_min_area"]),
            close_kernel=int(thresholds["postprocess_close_kernel"]),
            dilate_kernel=int(thresholds["postprocess_dilate_kernel"]),
        )

    pred_small = pred_small[0]

    pred_full = cv2.resize(
        pred_small,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    pixel_confidence_small = probs.max(axis=1)[0]
    mean_confidence = float(pixel_confidence_small.mean())

    counts = {
        CLASS_NAMES[class_id]: int((pred_full == class_id).sum())
        for class_id in range(NUM_CLASSES)
    }
    total_pixels = max(1, int(pred_full.size))
    percentages = {
        class_name: count / total_pixels * 100.0
        for class_name, count in counts.items()
    }
    dominant_class = max(percentages, key=percentages.get)

    return {
        "original_rgb": original_rgb,
        "resized_rgb": resized_rgb,
        "probabilities": probs[0],
        "confidence_map": pixel_confidence_small,
        "mean_confidence": mean_confidence,
        "pred_mask_small": pred_small,
        "pred_mask_full": pred_full,
        "counts": counts,
        "percentages": percentages,
        "dominant_class": dominant_class,
    }


# ============================================================
# Input validation helpers
# ============================================================

def open_uploaded_image(uploaded_file) -> Image.Image:
    """Validate and open an uploaded image file."""
    if uploaded_file is None:
        raise ValueError("No image was uploaded.")

    try:
        image = Image.open(uploaded_file)
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "The uploaded file is not a valid image. Please upload a PNG, JPG or JPEG file."
        ) from exc

    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")

    width, height = image.size
    if width < 32 or height < 32:
        raise ValueError(
            "The uploaded image is too small for a useful segmentation demo. "
            "Please use an image at least 32 x 32 pixels."
        )

    return image


def open_image_from_path(image_path: str) -> Image.Image:
    """Validate and open an image from a local file path."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image path does not exist: {image_path}. "
            "Check the path in your CSV or place the file in the project folder."
        )

    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Could not open image: {image_path}. Use PNG, JPG or JPEG files."
        ) from exc


def validate_csv(uploaded_csv) -> pd.DataFrame:
    """Validate the optional batch CSV and check required columns."""
    if uploaded_csv is None:
        raise ValueError("No CSV was uploaded.")

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as exc:
        raise ValueError(
            "Could not read the CSV. Please upload a valid comma-separated file."
        ) from exc

    missing_columns = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Invalid CSV format. Missing required column(s): "
            + ", ".join(missing_columns)
            + f". Required columns: {REQUIRED_CSV_COLUMNS}"
        )

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    return df


# ============================================================
# Display and visualisation helpers
# ============================================================

def make_colour_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class IDs into an RGB colour mask."""
    colour_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, colour in CLASS_COLOURS.items():
        colour_mask[mask == class_id] = colour
    return colour_mask


def make_overlay(rgb_image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the original image with the predicted segmentation colours."""
    colour_mask = make_colour_mask(mask)
    overlay = (rgb_image * (1.0 - alpha) + colour_mask * alpha).clip(0, 255)
    return overlay.astype(np.uint8)


def image_array_to_png_bytes(image_array: np.ndarray) -> bytes:
    """Convert a numpy RGB image array into downloadable PNG bytes."""
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def plot_distribution(percentages: Dict[str, float]):
    """Create a simple bar chart showing predicted pixel share by class."""
    labels = CLASS_NAMES
    values = [percentages[label] for label in labels]
    colours = [np.array(CLASS_COLOURS[i]) / 255.0 for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values, color=colours)
    ax.set_ylabel("Predicted pixel share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Predicted landing-zone safety distribution")

    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    return fig


def plot_probability_heatmap(probability_map: np.ndarray, title: str):
    """Create a heatmap for one class probability map."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(probability_map, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Create an optional confusion matrix if a ground-truth mask is provided."""
    cm = confusion_matrix(
        y_true.flatten(),
        y_pred.flatten(),
        labels=[0, 1, 2],
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalised = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_normalised, vmin=0, vmax=1)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground-truth label")
    ax.set_title("Normalised confusion matrix")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                f"{cm_normalised[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def standardise_ground_truth_mask(mask_image: Image.Image, expected_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert an optional ground-truth mask image into class IDs.

    Supported formats:
    - Grayscale mask with values 0, 1, 2.
    - Grayscale mask with rough levels: dark=safe, middle=caution, bright=unsafe.
    - RGB mask using colours close to the displayed app colours.
    """
    expected_h, expected_w = expected_shape
    arr = np.array(mask_image)

    if arr.ndim == 2:
        if arr.max() <= 2:
            labels = arr.astype(np.uint8)
        else:
            labels = np.zeros_like(arr, dtype=np.uint8)
            labels[(arr > 85) & (arr <= 170)] = 1
            labels[arr > 170] = 2

    else:
        rgb = arr[:, :, :3].astype(np.float32)
        palette = np.array([CLASS_COLOURS[i] for i in range(NUM_CLASSES)], dtype=np.float32)
        distances = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
        labels = np.argmin(distances, axis=-1).astype(np.uint8)

    if labels.shape != (expected_h, expected_w):
        labels = cv2.resize(
            labels,
            (expected_w, expected_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return labels.astype(np.uint8)


def show_legend():
    """Display the colour legend used in the mask and overlay."""
    st.markdown(
        """
        **Legend:** 🟩 Safe &nbsp;&nbsp; 🟨 Caution &nbsp;&nbsp; 🟥 Unsafe
        """
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("How to use this demo")

    st.markdown(
        """
        1. Place `best_uav_unsafe_f1_unet.pth` beside `app.py`.
        2. Upload a UAV/post-disaster image, choose a sample image, or load an image path from CSV.
        3. Review the labelled image, overlay, confidence score and class distribution.
        4. Use the interpretation section to explain the result to a non-technical marker.
        """
    )

    st.divider()

    st.subheader("Model settings")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Detected device: `{device_name}`")

    thresholds = load_thresholds(THRESHOLD_PATH)

    # THRESHOLD SLIDER
    st.caption("Adjust these if the model is over-predicting caution or unsafe.")

    thresholds["caution_threshold"] = st.slider(
        "Caution threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(thresholds.get("caution_threshold", 0.60)),
        step=0.05,
    )
    
    thresholds["unsafe_threshold"] = st.slider(
        "Unsafe threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(thresholds.get("unsafe_threshold", 0.65)),
        step=0.05,
    )

    use_thresholds = st.checkbox(
        "Use tuned safety thresholds",
        value=True,
        help="Uses the threshold settings saved by the training notebook.",
    )

    use_postprocessing = st.checkbox(
        "Apply unsafe-region post-processing",
        value=True,
        help="Removes tiny unsafe noise and smooths unsafe regions.",
    )

    overlay_alpha = st.slider(
        "Overlay strength",
        min_value=0.10,
        max_value=0.80,
        value=0.45,
        step=0.05,
    )

    st.divider()

    st.subheader("Limitations")
    st.markdown(
        """
        - This is a demo tool, not a real flight-safety system.
        - Predictions are resized to 256 x 256 before inference.
        - The output depends on the training data and pseudo-labelling rules.
        - Shadows, unusual camera angles and unseen disaster types may reduce reliability.
        - A human reviewer should always verify unsafe/caution areas.
        """
    )


# ============================================================
# Main page: problem explanation
# ============================================================

st.title("🛰️ UAV Landing Zone Safety Segmentation MVP")

st.markdown(
    """
    This app demonstrates a machine learning model that labels each pixel in a UAV-style
    post-disaster image as **safe**, **caution** or **unsafe** for potential landing-zone
    assessment. Instead of only showing code, the app presents the model output visually
    so a tutor or industry stakeholder can understand what the model is predicting.
    """
)

show_legend()


# ============================================================
# Model readiness check
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Model file is missing. Expected `best_uav_unsafe_f1_unet.pth` in the same "
        "folder as `app.py`. Train the notebook first, then copy the saved checkpoint "
        "into this app folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH), device_name)
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# Input selection: upload image, sample image, or optional CSV
# ============================================================

st.header("1. Choose input data")

input_mode = st.radio(
    "Input source",
    ["Upload image", "Use sample image", "Upload CSV with image paths"],
    horizontal=True,
)

selected_image: Optional[Image.Image] = None
selected_image_name = "uploaded_image"

if input_mode == "Upload image":
    uploaded_image = st.file_uploader(
        "Upload a UAV or aerial image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_image is not None:
        try:
            selected_image = open_uploaded_image(uploaded_image)
            selected_image_name = uploaded_image.name
            st.success(f"Loaded image: {uploaded_image.name}")
        except ValueError as exc:
            st.error(str(exc))

elif input_mode == "Use sample image":
    sample_files = []
    if SAMPLE_IMAGE_DIR.exists():
        sample_files = sorted(
            [
                p for p in SAMPLE_IMAGE_DIR.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )

    if not sample_files:
        st.warning(
            "No sample images found. Create a folder called `sample_images` and place "
            "PNG/JPG demo images inside it, or use the upload option."
        )
    else:
        sample_choice = st.selectbox(
            "Choose a sample image",
            sample_files,
            format_func=lambda p: p.name,
        )

        try:
            selected_image = Image.open(sample_choice).convert("RGB")
            selected_image_name = sample_choice.name
            st.success(f"Loaded sample image: {sample_choice.name}")
        except Exception as exc:
            st.error(f"Could not open sample image: {sample_choice}. {exc}")

else:
    uploaded_csv = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=f"Required column: {REQUIRED_CSV_COLUMNS}. Example: image_path",
    )

    if uploaded_csv is not None:
        try:
            df = validate_csv(uploaded_csv)
            st.success("CSV loaded and validated.")
            st.write("Sample rows:")
            st.dataframe(df.head(10), width="stretch")

            row_index = st.number_input(
                "Select CSV row to predict",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1,
            )

            image_path = str(df.loc[row_index, "image_path"])
            selected_image = open_image_from_path(image_path)
            selected_image_name = Path(image_path).name

        except Exception as exc:
            st.error(str(exc))


if selected_image is None:
    st.info("Choose or upload an image to run the segmentation model.")
    st.stop()


# ============================================================
# Preview selected input image
# ============================================================

with st.expander("Preview selected input", expanded=True):
    st.image(
        selected_image,
        caption=f"Input image: {selected_image_name}",
        width="stretch",
    )
    st.caption(f"Original size: {selected_image.size[0]} x {selected_image.size[1]} pixels")


# ============================================================
# Run prediction
# ============================================================

st.header("2. Model prediction")

with st.spinner("Running pixel-level segmentation..."):
    result = run_segmentation(
        model=model,
        image=selected_image,
        device_name=device_name,
        thresholds=thresholds,
        use_thresholds=use_thresholds,
        use_postprocessing=use_postprocessing,
    )

original_rgb = result["original_rgb"]
pred_mask_full = result["pred_mask_full"]
pred_mask_small = result["pred_mask_small"]
probabilities = result["probabilities"]
percentages = result["percentages"]
counts = result["counts"]
dominant_class = result["dominant_class"]
mean_confidence = result["mean_confidence"]

labelled_mask = make_colour_mask(pred_mask_full)
overlay = make_overlay(original_rgb, pred_mask_full, alpha=overlay_alpha)

safe_pct = percentages["safe"]
caution_pct = percentages["caution"]
unsafe_pct = percentages["unsafe"]

metric_cols = st.columns(4)
metric_cols[0].metric("Dominant prediction", dominant_class.title())
metric_cols[1].metric("Mean confidence", f"{mean_confidence * 100:.1f}%")
metric_cols[2].metric("Unsafe pixels", f"{unsafe_pct:.1f}%")
metric_cols[3].metric("Caution pixels", f"{caution_pct:.1f}%")

view_cols = st.columns(3)
with view_cols[0]:
    st.image(original_rgb, caption="Original image", width="stretch")
with view_cols[1]:
    st.image(labelled_mask, caption="Full pixel-labelled mask", width="stretch")
with view_cols[2]:
    st.image(overlay, caption="Overlay on original image", width="stretch")

download_cols = st.columns(2)
with download_cols[0]:
    st.download_button(
        "Download labelled mask",
        data=image_array_to_png_bytes(labelled_mask),
        file_name=f"{Path(selected_image_name).stem}_labelled_mask.png",
        mime="image/png",
    )
with download_cols[1]:
    st.download_button(
        "Download overlay",
        data=image_array_to_png_bytes(overlay),
        file_name=f"{Path(selected_image_name).stem}_overlay.png",
        mime="image/png",
    )


# ============================================================
# Visualisations
# ============================================================

st.header("3. Visualisations")

summary_df = pd.DataFrame(
    {
        "Class": [name.title() for name in CLASS_NAMES],
        "Pixel count": [counts[name] for name in CLASS_NAMES],
        "Share of image (%)": [round(percentages[name], 2) for name in CLASS_NAMES],
        "Interpretation": [
            "Likely usable area",
            "Needs human checking before use",
            "High-risk area to avoid",
        ],
    }
)

table_col, chart_col = st.columns([1.1, 1])
with table_col:
    st.subheader("Prediction summary")
    st.dataframe(summary_df, width="stretch", hide_index=True)

with chart_col:
    st.subheader("Class distribution")
    st.pyplot(plot_distribution(percentages), width="stretch")

heatmap_cols = st.columns(3)
with heatmap_cols[0]:
    st.pyplot(
        plot_probability_heatmap(probabilities[0], "Safe probability"),
        width="stretch",
    )
with heatmap_cols[1]:
    st.pyplot(
        plot_probability_heatmap(probabilities[1], "Caution probability"),
        width="stretch",
    )
with heatmap_cols[2]:
    st.pyplot(
        plot_probability_heatmap(probabilities[2], "Unsafe probability"),
        width="stretch",
    )


# ============================================================
# Optional confusion matrix using uploaded ground-truth mask
# ============================================================

st.header("4. Optional validation against a ground-truth mask")

st.markdown(
    """
    Upload a ground-truth mask only if you have one. This enables a confusion matrix
    for the selected image. The mask should use class IDs `0=safe`, `1=caution`,
    `2=unsafe`, or colours close to the app legend.
    """
)

ground_truth_upload = st.file_uploader(
    "Optional: upload ground-truth mask",
    type=["png", "jpg", "jpeg"],
)

if ground_truth_upload is not None:
    try:
        gt_image = open_uploaded_image(ground_truth_upload)
        gt_mask = standardise_ground_truth_mask(gt_image, pred_mask_full.shape)

        cm_col, gt_col = st.columns([1, 1])
        with cm_col:
            st.pyplot(
                plot_confusion_matrix(gt_mask, pred_mask_full),
                width="stretch",
            )
        with gt_col:
            st.image(
                make_colour_mask(gt_mask),
                caption="Ground-truth mask interpreted by app",
                width="stretch",
            )

    except Exception as exc:
        st.error(f"Could not process ground-truth mask: {exc}")


# ============================================================
# Non-technical interpretation
# ============================================================

st.header("5. Stakeholder interpretation")

if unsafe_pct >= 25:
    recommendation = (
        "The image contains a large unsafe region. For a UAV landing decision, this "
        "area should be treated as unsuitable unless confirmed otherwise by a human reviewer."
    )
elif caution_pct + unsafe_pct >= 40:
    recommendation = (
        "The image contains a substantial amount of caution or unsafe area. The site may "
        "still contain usable zones, but it needs careful review before any landing decision."
    )
else:
    recommendation = (
        "The model predicts that most of the image is safe. However, any caution or unsafe "
        "patches should still be checked because the model is a decision-support tool, not a final authority."
    )

st.markdown(
    f"""
    **Plain-English result:** The model predicts that **{safe_pct:.1f}%** of the image is
    safe, **{caution_pct:.1f}%** is caution, and **{unsafe_pct:.1f}%** is unsafe.

    **What this means:** Green areas are the most suitable candidate regions. Yellow areas
    may contain obstacles, rough texture, damage, edges or uncertain visual features. Red
    areas are predicted as unsafe and should be avoided or manually inspected.

    **Recommended stakeholder message:** {recommendation}

    **Confidence note:** The mean pixel confidence is **{mean_confidence * 100:.1f}%**.
    This is useful for presentation, but it should not be interpreted as a guarantee of
    real-world safety.
    """
)


# ============================================================
# Technical notes for marker/demo
# ============================================================

with st.expander("Technical notes"):
    st.markdown(
        f"""
        - Model type: residual U-Net-style CNN.
        - Input: image resized to `{IMG_SIZE} x {IMG_SIZE}`.
        - Feature tensor: 14 channels created from RGB/BGR colour, grayscale, HSV, LAB,
          Canny edges, edge density, texture variance and gradient magnitude.
        - Classes: `{CLASS_NAMES}`.
        - Threshold mode: `{use_thresholds}`.
        - Post-processing mode: `{use_postprocessing}`.
        - Unsafe threshold: `{thresholds["unsafe_threshold"]}`.
        - Caution threshold: `{thresholds["caution_threshold"]}`.
        - Checkpoint path: `{MODEL_PATH}`.
        """
    )
