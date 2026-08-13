#!/usr/bin/env python3
"""
Run SSD MobileNet V2 defect detection on test images.
Requires TFOD setup: bash scripts/setup_tfod.sh

Set MODEL_PATH to your trained checkpoint directory, e.g.:
  export MODEL_PATH="/path/to/TFODCourse/Tensorflow/workspace/models/my_ssd_mobnet"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(
    "/Users/apple/Documents/Programming Language/Python/Tensorflow Object Detection"
    "/TFODCourse/Tensorflow/workspace/models/my_ssd_mobnet"
)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL))
LABEL_MAP = ROOT / "data" / "annotated" / "label_map.pbtxt"
TEST_DIR = ROOT / "data" / "annotated" / "test"
OUTPUT_DIR = ROOT / "results" / "detection"


def main() -> int:
    tfod_root = ROOT / "Tensorflow" / "models" / "research"
    legacy_tfod = Path(
        "/Users/apple/Documents/Programming Language/Python/Tensorflow Object Detection"
        "/TFODCourse/Tensorflow/models/research"
    )
    research_path = tfod_root if tfod_root.exists() else legacy_tfod

    if not research_path.exists():
        print("TFOD not installed. Run: bash scripts/setup_tfod.sh")
        return 1

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Set MODEL_PATH to your trained checkpoint directory.")
        return 1

    sys.path.insert(0, str(research_path))
    sys.path.insert(0, str(research_path / "slim"))

    import tensorflow as tf
    from object_detection.builders import model_builder
    from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils

    configs = config_util.get_configs_from_pipeline_file(str(MODEL_PATH / "pipeline.config"))
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.train.Checkpoint(model=detection_model)
    latest = tf.train.latest_checkpoint(str(MODEL_PATH))
    if not latest:
        print("No checkpoint found in model directory")
        return 1
    ckpt.restore(latest).expect_partial()
    print(f"Loaded checkpoint: {latest}")

    @tf.function
    def detect(image_np):
        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
        return detection_model(input_tensor)

    category_index = label_map_util.create_category_index_from_labelmap(str(LABEL_MAP))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    import cv2
    import numpy as np

    for image_path in sorted(TEST_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detect(image_rgb)

        num_detections = int(detections.pop("num_detections"))
        detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_rgb,
            detections["detection_boxes"],
            detections["detection_classes"] + 1,
            detections["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=0.5,
            line_thickness=2,
        )

        out_path = OUTPUT_DIR / f"detected_{image_path.name}"
        cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        top_score = float(detections["detection_scores"].max()) if num_detections else 0.0
        top_class_id = int(detections["detection_classes"][0]) + 1 if num_detections else -1
        results.append({
            "image": image_path.name,
            "detections": num_detections,
            "top_score": round(top_score, 3),
            "top_class_id": top_class_id,
            "output": str(out_path.relative_to(ROOT)),
        })
        print(f"  {image_path.name}: {num_detections} detection(s), top score={top_score:.2f}")

    summary_path = OUTPUT_DIR / "detection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"model": str(MODEL_PATH), "results": results}, f, indent=2)

    print(f"Saved detection results to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
