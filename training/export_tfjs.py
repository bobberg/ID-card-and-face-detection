"""
Export the trained YOLOv8 model to TensorFlow.js format for browser use.

Step-by-step: .pt -> ONNX -> TF SavedModel -> TF.js GraphModel

Usage:
  python export_tfjs.py                         # Use default best.pt path
  python export_tfjs.py --model path/to/best.pt # Custom model path
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

RUNS_DIR = Path(__file__).parent / "runs"
PUBLIC_MODEL_DIR = Path(__file__).parent.parent / "public" / "model"


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TF.js")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to trained .pt model"
    )
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = RUNS_DIR / "id-card-detect" / "weights" / "best.pt"

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("Train a model first with: python train.py")
        return

    weights_dir = model_path.parent
    onnx_path = weights_dir / model_path.stem
    onnx_file = onnx_path.with_suffix(".onnx")
    saved_model_dir = weights_dir / f"{model_path.stem}_saved_model"
    tfjs_dir = weights_dir / f"{model_path.stem}_web_model"

    # Step 1: Export to ONNX
    if not onnx_file.exists():
        print(f"Loading model from: {model_path}")
        model = YOLO(str(model_path))
        print("Step 1: Exporting to ONNX...")
        model.export(format="onnx")
        print(f"  ONNX saved: {onnx_file}")
    else:
        print(f"Step 1: ONNX already exists: {onnx_file}")

    # Step 2: ONNX -> TF SavedModel using onnx2tf CLI
    if not saved_model_dir.exists():
        print("Step 2: Converting ONNX to TF SavedModel...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "onnx2tf",
                "-i",
                str(onnx_file),
                "-o",
                str(saved_model_dir),
                "-osd",
                "-coion",
            ],
            check=True,
        )
        print(f"  SavedModel saved: {saved_model_dir}")
    else:
        print(f"Step 2: SavedModel already exists: {saved_model_dir}")

    # Step 3: TF SavedModel -> TF.js using tensorflowjs_converter CLI
    print("Step 3: Converting SavedModel to TF.js...")
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            str(saved_model_dir),
            str(tfjs_dir),
        ],
        check=True,
    )
    print(f"  TF.js model saved: {tfjs_dir}")

    # Step 4: Copy to public/model/
    print("Step 4: Copying to public/model/...")
    PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in PUBLIC_MODEL_DIR.glob("*"):
        f.unlink()
    for f in tfjs_dir.iterdir():
        dest = PUBLIC_MODEL_DIR / f.name
        shutil.copy2(f, dest)
        print(f"  Copied: {f.name}")

    # Get class names from model
    model = YOLO(str(model_path))
    class_names = list(model.names.values())
    print(f"  Classes: {class_names}")

    metadata = PUBLIC_MODEL_DIR / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "inputSize": 640,
                "classes": class_names,
                "source": model_path.name,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nModel ready at: {PUBLIC_MODEL_DIR}")
    print("The React app will load it from /model/model.json")


if __name__ == "__main__":
    main()
