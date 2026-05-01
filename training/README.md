# ID Card Detection — Training Guide

## Overview

This trains a YOLOv8-nano object detection model to recognize ID cards in webcam frames.  
The trained model gets exported to TensorFlow.js format and loaded by the React app.

## Prerequisites

- Python 3.9+ (`python3 --version`)
- pip (`pip3 --version`)
- ~2GB disk space for the dataset + model

## Quick Start

```bash
cd training

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (needs free Roboflow API key)
#    Get your key at: https://app.roboflow.com/settings/api
python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

# 4. Train the model (~10-30 min depending on GPU)
python train.py

# 5. Export to TensorFlow.js (copies to public/model/)
python export_tfjs.py
```

Then go back to the project root and start the dev server:

```bash
cd ..
npm run dev
```

## Step-by-step Details

### 1. Get the Dataset

The default dataset is **"ID Card"** from Roboflow Universe (augmented-startups/id-card-new).

**Option A: Roboflow (recommended)**

1. Go to https://app.roboflow.com/settings/api and create a free account
2. Copy your API key
3. Run:

```bash
python download_dataset.py --api-key YOUR_KEY
```

**Option B: Use a different Roboflow dataset**

```bash
python download_dataset.py --api-key YOUR_KEY \
  --workspace WORKSPACE_NAME \
  --project PROJECT_NAME \
  --version VERSION_NUMBER
```

**Option C: Use your own images**

```bash
# Create the folder structure
python download_dataset.py --local-only
```

Then add images to `dataset/train/images/` and YOLO-format labels to `dataset/train/labels/`.

#### YOLO Label Format

Each image needs a `.txt` file with the same name. Each line represents one bounding box:

```
class_id center_x center_y width height
```

All values are normalized to [0, 1] relative to image dimensions. For a single-class detector (just "id-card"), `class_id` is always `0`:

```
0 0.5 0.5 0.8 0.6
```

Tools for labeling:

- [Roboflow Annotate](https://roboflow.com/annotate) (web-based, free)
- [Label Studio](https://labelstud.io/) (self-hosted)
- [CVAT](https://cvat.ai/) (open source)

### 2. Train

```bash
python train.py
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--imgsz` | 640 | Input image size |
| `--batch` | 16 | Batch size (reduce if OOM) |
| `--model` | yolov8n.pt | Base model (n=nano, s=small, m=medium) |
| `--resume` | — | Resume from last checkpoint |

**On Apple Silicon (M1/M2/M3):** Training uses MPS acceleration automatically.  
**Without GPU:** Training still works on CPU, just slower (~30-60 min for 50 epochs).

Training output goes to `training/runs/id-card-detect/`.

### 3. Export

```bash
python export_tfjs.py
```

This:

1. Loads the best checkpoint from `runs/id-card-detect/weights/best.pt`
2. Exports to TensorFlow.js format
3. Copies the model files to `public/model/`

The React app loads the model from `/model/model.json` at runtime.

## Adding Custom Images Later

To improve the model with your own ID card images:

1. Take photos of ID cards in various conditions (angles, lighting, backgrounds)
2. Annotate them with bounding boxes (use Roboflow Annotate for easiest workflow)
3. Add to `dataset/train/images/` and `dataset/train/labels/`
4. Optionally add ~20% to `dataset/valid/images/` + `labels/` for validation
5. Re-run training:

```bash
python train.py --epochs 30
python export_tfjs.py
```

**Tips for better detection:**

- Include diverse backgrounds (desk, hand-held, wallet, etc.)
- Vary lighting conditions
- Include partial occlusions
- Mix orientations (landscape + portrait)
- Aim for at least 100-200 training images for good results
