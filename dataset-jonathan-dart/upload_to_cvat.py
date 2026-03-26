"""
Upload YOLO pose annotations to CVAT task via API.

Usage:
    python upload_to_cvat.py --model runs/pose/dart_pose_overfit/weights/best.pt
"""

import argparse
import getpass
import shutil
import zipfile
from pathlib import Path

from cvat_sdk import make_client
from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL = DATASET_DIR / "labels" / "val"
EXPORT_DIR = DATASET_DIR / "cvat_export"

CVAT_HOST = "https://app.cvat.ai"
TASK_ID = 2094676


def yolo_label_from_result(result) -> str:
    """Convert a YOLO prediction result to YOLO keypoint label format."""
    lines = []
    if result.boxes is None or len(result.boxes) == 0:
        return ""

    boxes = result.boxes.xywhn.cpu()
    classes = result.boxes.cls.cpu().int()
    keypoints = result.keypoints.xyn.cpu() if result.keypoints is not None else None
    kp_conf = result.keypoints.conf.cpu() if result.keypoints is not None else None

    for i in range(len(boxes)):
        cls = int(classes[i])
        cx, cy, w, h = boxes[i].tolist()
        parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

        if keypoints is not None:
            kps = keypoints[i]
            confs = kp_conf[i] if kp_conf is not None else None
            for j in range(len(kps)):
                kx, ky = kps[j].tolist()
                v = 2 if (confs is not None and confs[j] > 0.5) else 0
                parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(v)])

        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def build_zip(model, conf):
    """Build the annotations zip and return its path."""
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    labels_out = EXPORT_DIR / "labels" / "Train"
    labels_out.mkdir(parents=True)

    image_files = sorted(IMAGES_TRAIN.glob("*.jpg"))
    image_files = [f for f in image_files if "_aug_" not in f.stem]

    label_dirs = [LABELS_TRAIN, LABELS_VAL]
    labeled = 0
    predicted = 0
    train_txt_lines = []

    for img_path in image_files:
        label_path = None
        for ld in label_dirs:
            candidate = ld / (img_path.stem + ".txt")
            if candidate.exists():
                label_path = candidate
                break

        out_lbl = labels_out / (img_path.stem + ".txt")
        train_txt_lines.append(f"data/images/Train/{img_path.name}")

        if label_path:
            shutil.copy2(label_path, out_lbl)
            labeled += 1
        else:
            results = model(str(img_path), conf=conf, verbose=False)
            label_str = yolo_label_from_result(results[0])
            if label_str:
                out_lbl.write_text(label_str)
                predicted += 1

    (EXPORT_DIR / "Train.txt").write_text("\n".join(train_txt_lines) + "\n")
    (EXPORT_DIR / "data.yaml").write_text(
        "Train: Train.txt\n"
        "kpt_shape:\n"
        "- 2\n"
        "- 3\n"
        "names:\n"
        "  0: Dart\n"
        "path: .\n"
    )

    zip_path = DATASET_DIR / "cvat_annotations.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(EXPORT_DIR.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(EXPORT_DIR))

    print(f"Built zip: {labeled} existing + {predicted} predicted annotations")
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--username", type=str, required=True)
    args = parser.parse_args()

    password = getpass.getpass("CVAT password: ")

    # Build annotations zip
    model = YOLO(args.model)
    zip_path = build_zip(model, args.conf)

    # Upload to CVAT
    print(f"\nConnecting to {CVAT_HOST}...")
    with make_client(CVAT_HOST, credentials=(args.username, password)) as client:
        task = client.tasks.retrieve(TASK_ID)
        print(f"Task: {task.name} (id={task.id})")
        print(f"Task size: {task.size} images")
        # Get available formats
        formats = client.api_client.server_api.retrieve_annotation_formats()
        if isinstance(formats, tuple):
            formats = formats[0]
        print("\nAvailable YOLO/pose formats:")
        for f in formats.importers:
            if "yolo" in f.name.lower() or "pose" in f.name.lower():
                print(f"  - '{f.name}'")

        # Get task labels
        labels = client.api_client.labels_api.list(task_id=TASK_ID)
        if isinstance(labels, tuple):
            labels = labels[0]
        for l in labels.results:
            print(f"Label: name='{l.name}', type='{l.type}'")
            if hasattr(l, 'sublabels') and l.sublabels:
                for sl in l.sublabels:
                    print(f"  Sublabel: name='{sl.name}', type='{sl.type}'")
            # Print full label dict for debugging
            print(f"  Full: {l.to_dict()}")

        print("\nUploading annotations...")
        task.import_annotations(
            format_name="Ultralytics YOLO Pose 1.0",
            filename=str(zip_path),
        )
        print("Done! Annotations uploaded successfully.")


if __name__ == "__main__":
    main()
