#!/usr/bin/env python3
"""
Analyze metallic washer sample images with OpenCV preprocessing.
Demonstrates the image processing stage of the defect detection pipeline.
Full SSD MobileNet V2 training requires TensorFlow Object Detection API setup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "docs" / "images"
OUTPUT_DIR = ROOT / "results" / "image_analysis"


def analyze_image(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"file": path.name, "error": "could not read image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Highlight potential surface irregularities via edge density
    edge_density = float(np.count_nonzero(edges) / edges.size)
    mean_intensity = float(np.mean(gray))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_gray.jpg"), gray)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_edges.jpg"), edges)

    return {
        "file": path.name,
        "shape": list(img.shape),
        "edge_density": round(edge_density, 4),
        "mean_intensity": round(mean_intensity, 2),
        "likely_defect": edge_density > 0.08,
    }


def main() -> int:
    images = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return 1

    results = [analyze_image(path) for path in images[:8]]
    summary = {
        "images_analyzed": len(results),
        "defect_candidates": sum(1 for r in results if r.get("likely_defect")),
        "details": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "analysis_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Analyzed {len(results)} sample images")
    for item in results:
        flag = "DEFECT?" if item.get("likely_defect") else "ok"
        print(f"  {item['file']}: edge_density={item.get('edge_density')} [{flag}]")
    print(f"Saved edge maps and summary to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
