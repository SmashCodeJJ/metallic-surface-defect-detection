#!/usr/bin/env bash
# Setup TensorFlow Object Detection API for washer defect detection training/inference.
# Run from repo root: bash scripts/setup_tfod.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/Tensorflow"
VENV="$ROOT/.venv-tfod"

echo "=== TFOD Setup for Metallic Surface Defect Detection ==="

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "tensorflow==2.15.0" opencv-python matplotlib pandas protobuf==3.20.3

mkdir -p "$TF_DIR"/{workspace/{annotations,images/{train,test},models,pre-trained-models},scripts,models}

if [ ! -d "$TF_DIR/models/research/object_detection" ]; then
  echo "Cloning TensorFlow models (object_detection only)..."
  git clone --depth 1 --filter=blob:none --sparse https://github.com/tensorflow/models "$TF_DIR/models"
  cd "$TF_DIR/models"
  git sparse-checkout set research/object_detection research/slim
fi

cd "$TF_DIR/models/research"
pip install .

# Link local annotated dataset
ln -sfn "$ROOT/data/annotated/train" "$TF_DIR/workspace/images/train" 2>/dev/null || true
ln -sfn "$ROOT/data/annotated/test" "$TF_DIR/workspace/images/test" 2>/dev/null || true
cp "$ROOT/data/annotated/label_map.pbtxt" "$TF_DIR/workspace/annotations/" 2>/dev/null || true

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Generate TFRecords: see notebooks/02_training_and_detection.ipynb"
echo "  2. Train: python Tensorflow/models/research/object_detection/model_main_tf2.py ..."
echo "  3. Run detection: python scripts/run_detection.py"
