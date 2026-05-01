"""
Train a YOLOv8-nano model for ID card detection.

Usage:
  python train.py                          # Use default dataset path
  python train.py --data path/to/data.yaml # Custom dataset
  python train.py --epochs 100             # More epochs
  python train.py --resume                 # Resume interrupted training
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent / "dataset"
RUNS_DIR = Path(__file__).parent / "runs"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 ID card detector")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (default: yolov8n.pt = nano)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    args = parser.parse_args()

    # Find data.yaml
    data_path = args.data
    if data_path is None:
        data_path = DATASET_DIR / "data.yaml"
        if not data_path.exists():
            print(f"No data.yaml found at {data_path}")
            print("Run download_dataset.py first, or pass --data path/to/data.yaml")
            return

    # When resuming, use last.pt checkpoint
    model_path = args.model
    if args.resume:
        last_pt = RUNS_DIR / "id-card-detect" / "weights" / "last.pt"
        if last_pt.exists():
            model_path = str(last_pt)
        else:
            print(f"No last.pt found at {last_pt}, starting fresh")
            args.resume = False

    print(f"Training config:")
    print(f"  Dataset:  {data_path}")
    print(f"  Model:    {model_path}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print(f"  Resume:   {args.resume}")
    print()

    model = YOLO(model_path)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name="id-card-detect",
        exist_ok=True,
        patience=10,  # early stopping
        save=True,
        plots=True,
        resume=args.resume,
    )

    # Print results
    best_model = RUNS_DIR / "id-card-detect" / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best model saved to: {best_model}")
    print(f"\nNext step: python export_tfjs.py")

    return results


if __name__ == "__main__":
    main()
