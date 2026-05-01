"""
Download an ID card detection dataset.

Usage:
  # Download from Roboflow Universe (public datasets, needs free API key):
  python download_dataset.py --api-key YOUR_KEY

  # Or set the env var:
  ROBOFLOW_API_KEY=YOUR_KEY python download_dataset.py

  # Just create the local folder structure (to add your own images):
  python download_dataset.py --local-only
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"

# Roboflow Universe public download URL for an ID card detection dataset
# Format: yolov8, hosted on universe.roboflow.com
UNIVERSE_URL = "https://universe.roboflow.com/ds/FBYBpkDv5C" "?key={api_key}"


def download_universe(api_key):
    """Download directly from a Roboflow Universe public dataset link."""
    import requests

    url = UNIVERSE_URL.format(api_key=api_key)
    print("Downloading ID card dataset from Roboflow Universe...")
    print("(This may take a minute)")

    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        print(f"Download failed (HTTP {resp.status_code})")
        print("Falling back to Roboflow SDK method...")
        return download_roboflow_sdk(api_key)

    # Extract zip to dataset dir
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    content = io.BytesIO(resp.content)
    with zipfile.ZipFile(content) as zf:
        zf.extractall(DATASET_DIR)

    _print_dataset_info()
    _ensure_data_yaml()
    return True


def download_roboflow_sdk(api_key):
    """Fallback: use the Roboflow SDK."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow first: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)

    # Try multiple known public ID card datasets
    datasets = [
        ("augmented-startups", "id-card-new", 2),
        ("objectdetection-bdhv0", "id-card-detection-zrqjn", 1),
        ("i-card", "id-card-detection-rkjqs", 1),
    ]

    for workspace, project, version in datasets:
        try:
            print(f"Trying {workspace}/{project} v{version}...")
            proj = rf.workspace(workspace).project(project)
            proj.version(version).download("yolov8", location=str(DATASET_DIR))
            _print_dataset_info()
            return True
        except Exception as e:
            print(f"  Skipped: {e}")
            continue

    print("\nCould not access any public dataset.")
    print("Creating local folder structure instead...")
    setup_local_structure()
    return False


def _print_dataset_info():
    """Print info about the downloaded dataset."""
    print(f"\nDataset downloaded to: {DATASET_DIR}")
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*")))
            print(f"  {split}: {count} images")


def _ensure_data_yaml():
    """Create data.yaml if the download didn't include one."""
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(
            f"train: {DATASET_DIR / 'train' / 'images'}\n"
            f"val: {DATASET_DIR / 'valid' / 'images'}\n"
            f"test: {DATASET_DIR / 'test' / 'images'}\n"
            f"\nnc: 1\n"
            f"names: ['id-card']\n"
        )


def setup_local_structure():
    """Create the expected folder structure for a custom dataset."""
    for split in ["train", "valid", "test"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(
            f"train: {DATASET_DIR / 'train' / 'images'}\n"
            f"val: {DATASET_DIR / 'valid' / 'images'}\n"
            f"test: {DATASET_DIR / 'test' / 'images'}\n"
            f"\nnc: 1\n"
            f"names: ['id-card']\n"
        )
    print(f"Local dataset structure created at: {DATASET_DIR}")
    print("Add your images to train/images and labels (YOLO format) to train/labels")
    print(f"Config: {data_yaml}")


def main():
    parser = argparse.ArgumentParser(description="Download ID card detection dataset")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Just create the local folder structure without downloading",
    )
    args = parser.parse_args()

    if args.local_only:
        setup_local_structure()
        return

    if not args.api_key:
        print("No API key provided.")
        print("Get a free key at: https://app.roboflow.com/settings/api")
        print("Then run: python download_dataset.py --api-key YOUR_KEY")
        print("\nCreating local folder structure instead...")
        setup_local_structure()
        return

    download_universe(args.api_key)


if __name__ == "__main__":
    main()
