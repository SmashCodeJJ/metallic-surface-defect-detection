#!/usr/bin/env python3
"""Visualize Pascal VOC annotations for the washer defect dataset."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "annotated"
OUTPUT_DIR = ROOT / "results" / "annotations"

CLASS_COLORS = {
    "DefectiveWashers": (0, 0, 255),
    "NonDefectiveWashers": (0, 200, 0),
}


def parse_voc_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        objects.append({
            "class": name,
            "xmin": int(float(bbox.find("xmin").text)),
            "ymin": int(float(bbox.find("ymin").text)),
            "xmax": int(float(bbox.find("xmax").text)),
            "ymax": int(float(bbox.find("ymax").text)),
        })
    return objects


def visualize_split(split: str) -> list[dict]:
    split_dir = DATA_DIR / split
    results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for xml_path in sorted(split_dir.glob("*.xml")):
        image_path = xml_path.with_suffix(".jpg")
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        objects = parse_voc_xml(xml_path)
        for obj in objects:
            color = CLASS_COLORS.get(obj["class"], (255, 255, 0))
            cv2.rectangle(
                image,
                (obj["xmin"], obj["ymin"]),
                (obj["xmax"], obj["ymax"]),
                color,
                2,
            )
            cv2.putText(
                image,
                obj["class"],
                (obj["xmin"], max(obj["ymin"] - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        out_path = OUTPUT_DIR / f"{split}_{image_path.stem}_labeled.jpg"
        cv2.imwrite(str(out_path), image)
        results.append({
            "split": split,
            "image": image_path.name,
            "objects": len(objects),
            "classes": [o["class"] for o in objects],
            "output": str(out_path.relative_to(ROOT)),
        })
        print(f"  {split}/{image_path.name}: {len(objects)} box(es) -> {out_path.name}")

    return results


def main() -> int:
    if not DATA_DIR.exists():
        print(f"Missing dataset at {DATA_DIR}")
        return 1

    print("Visualizing washer defect annotations...")
    all_results = visualize_split("test") + visualize_split("train")

    summary = {
        "total_images": len(all_results),
        "defective": sum(1 for r in all_results if "DefectiveWashers" in r["classes"]),
        "non_defective": sum(1 for r in all_results if "NonDefectiveWashers" in r["classes"]),
        "details": all_results,
    }

    summary_path = OUTPUT_DIR / "annotation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLabeled {summary['total_images']} images")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
