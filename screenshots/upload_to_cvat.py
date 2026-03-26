"""
Create a CVAT task with screenshot images and upload model predictions as annotations.

Usage:
    python upload_to_cvat.py --model ../runs/pose/dart_pose_orientation_colours/weights/best.pt --username your_user
"""

import argparse
import getpass
import shutil
import zipfile
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent
EXPORT_DIR = DATASET_DIR / "cvat_export"

CVAT_HOST = "https://app.cvat.ai"

DEFAULT_MODEL = (
    DATASET_DIR.parent / "runs" / "pose" / "dart_pose_orientation_colours" / "weights" / "best.pt"
)


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
    """Run model on all images and build annotations zip."""
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    labels_out = EXPORT_DIR / "labels" / "Train"
    labels_out.mkdir(parents=True)

    image_files = sorted(
        f for f in DATASET_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    predicted = 0
    skipped = 0
    train_txt_lines = []

    for img_path in image_files:
        train_txt_lines.append(f"data/images/Train/{img_path.name}")

        results = model(str(img_path), conf=conf, verbose=False)
        label_str = yolo_label_from_result(results[0])
        if label_str:
            out_lbl = labels_out / (img_path.stem + ".txt")
            out_lbl.write_text(label_str)
            predicted += 1
        else:
            skipped += 1

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

    print(f"Built zip: {predicted} predicted, {skipped} with no detections")
    return zip_path, image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--username", type=str, default='johnzakaria98')
    parser.add_argument("--task-name", type=str, default="screenshots-dart")
    args = parser.parse_args()

    password = 'tyx7TEG5uba5cjb!yqj'

    # Build annotations
    model = YOLO(args.model)
    zip_path, image_files = build_zip(model, args.conf)

    # Connect to CVAT
    print(f"\nConnecting to {CVAT_HOST}...")
    with make_client(CVAT_HOST, credentials=(args.username, password)) as client:
        # Create task with images
        print(f"Creating task '{args.task_name}' with {len(image_files)} images...")
        task = client.tasks.create_from_data(
            spec={
                "name": args.task_name,
                "labels": [
                    {
                        "name": "Dart",
                        "type": "skeleton",
                        "sublabels": [
                            {"name": "tip", "type": "points"},
                            {"name": "base", "type": "points"},
                        ],
                        "svg": '<line x1="25.00" y1="15.00" x2="25.00" y2="35.00" '
                               'stroke="black" data-type="edge" '
                               'data-node-from="1" data-node-to="2" />'
                               '<circle r="1.5" stroke="black" fill="#b3b3b3" '
                               'cx="25.00" cy="15.00" data-type="element" '
                               'data-element-id="1" data-node-id="1" data-label-name="tip" />'
                               '<circle r="1.5" stroke="black" fill="#b3b3b3" '
                               'cx="25.00" cy="35.00" data-type="element" '
                               'data-element-id="2" data-node-id="2" data-label-name="base" />',
                    }
                ],
            },
            resource_type=ResourceType.LOCAL,
            resources=[str(f) for f in image_files],
        )
        print(f"Task created: id={task.id}, size={task.size} images")

        # Upload annotations
        print("Uploading annotations...")
        task.import_annotations(
            format_name="Ultralytics YOLO Pose 1.0",
            filename=str(zip_path),
        )
        print(f"Done! Annotations uploaded to task {task.id}")
        print(f"Open: {CVAT_HOST}/tasks/{task.id}")


if __name__ == "__main__":
    main()
