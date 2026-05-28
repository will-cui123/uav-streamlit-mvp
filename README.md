# UAV Landing Zone Safety Segmentation MVP

This is a Streamlit frontend MVP for an ENGG2112 machine learning project. It demonstrates a trained residual U-Net-style CNN that predicts a pixel-level safety label for a UAV/post-disaster image.

The app is designed for industry stakeholders, so it focuses on visual output and interpretation rather than only showing model code.

## What the app does

- Loads a trained PyTorch segmentation model.
- Lets the user upload an image, choose a sample image, or select an image path from a CSV.
- Validates input files and gives clear error messages.
- Predicts every pixel as one of three classes:
  - `safe`
  - `caution`
  - `unsafe`
- Displays:
  - original image
  - full labelled mask
  - overlay on the original image
  - class distribution chart
  - class probability heatmaps
  - optional confusion matrix if a ground-truth mask is uploaded
- Provides a non-technical interpretation section for stakeholder presentation.

## Required files

Place these files in the same project folder:

```text
uav_streamlit_mvp/
├── app.py
├── requirements.txt
├── README.md
├── best_uav_unsafe_f1_unet.pth
├── best_thresholds_unsafe_f1.json      # optional but recommended
└── sample_images/                      # optional
    ├── sample1.png
    └── sample2.jpg
```

The model checkpoint should come from the training notebook. The expected checkpoint name is:

```text
best_uav_unsafe_f1_unet.pth
```

The threshold file is optional. If it is missing, the app uses these default values from the notebook output:

```json
{
  "unsafe_threshold": 0.37,
  "caution_threshold": 0.35,
  "postprocess_min_area": 25,
  "postprocess_close_kernel": 3,
  "postprocess_dilate_kernel": 3
}
```

## CSV input mode

The main demo mode is image upload. CSV mode is included only for batch-style demonstrations where the CSV points to local image files.

Required CSV column:

```text
image_path
```

Example:

```csv
image_path
sample_images/sample1.png
sample_images/sample2.jpg
```

The app validates that the required column exists and that the selected image path can be opened.

## How to run locally

### macOS / Linux

```bash
cd path/to/uav_streamlit_mvp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
cd path\to\uav_streamlit_mvp
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Notes about PyTorch installation

The `requirements.txt` installs the standard PyTorch package. This is enough for a CPU demo.

If you want GPU acceleration, install the PyTorch version that matches your CUDA setup from the official PyTorch installation selector, then run:

```bash
streamlit run app.py
```

## Model limitations

This app is a demonstration tool, not a real flight-safety system.

Important limitations:

- The image is resized to 256 x 256 before inference.
- The model depends on the training data and pseudo-labelling process.
- Predictions may be less reliable on unseen disaster types, unusual camera angles, shadows or image quality issues.
- The output should support human decision-making, not replace human review.
- `safe`, `caution` and `unsafe` describe model predictions, not certified landing-zone safety.

## Suggested presentation script

A concise way to explain the demo:

> This frontend turns the trained CNN into an interactive decision-support prototype. A user can upload a UAV image, and the model predicts a safety class for every pixel. The output is shown as a coloured segmentation mask and overlay, so stakeholders can quickly see which areas are likely safe, uncertain or unsafe. The charts summarise the pixel distribution, while the interpretation section translates the prediction into plain English for non-technical users.
