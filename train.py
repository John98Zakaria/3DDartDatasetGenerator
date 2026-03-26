"""
Train a YOLO pose model on the dart keypoint dataset-solo-dart.

Usage:
    python train.py
    python train.py --epochs 200 --batch 32 --model yolo11n-pose.pt
    python train.py --epochs 150 --batch 32 --name full-dataset-with-augmentation --dataset dataset-combined
    python train.py --epochs 150 --name full-dataset-with-augmentation --register-model screenshot-optimzed-hyper-params
    python train.py --epochs 150 --batch 32 --model runs/pose/full-dataset-with-copy-paste-cont10/weights/best.pt --name biased-set--register-model full-dataset-with-copy-paste-cont --dataset screenshots-label-conf
    python train.py --resume runs/pose/dart_pose/weights/last.pt
"""

import argparse
import os
from pathlib import Path

import mlflow
import yaml

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
from ultralytics import YOLO, settings

settings.update({"mlflow": True})

import numpy as np

DATASET_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-multi-dart", "dataset.yaml")
DATASET_DIR = Path(DATASET_YAML).parent

def _count_images(split_dir):
    """Count images in a dataset split directory."""
    d = Path(split_dir)
    return sum(1 for _ in d.glob("*.jpg")) + sum(1 for _ in d.glob("*.png")) if d.exists() else 0


def _log_dataset(data_yaml):
    """Log the dataset to the active MLflow run."""
    with open(data_yaml) as f:
        ds_cfg = yaml.safe_load(f)
    ds_path = Path(ds_cfg.get("path", Path(data_yaml).parent))
    train_count = _count_images(ds_path / ds_cfg.get("train", "images/train"))
    val_count = _count_images(ds_path / ds_cfg.get("val", "images/val"))

    dataset = mlflow.data.from_pandas(
        __import__("pandas").DataFrame({
            "split": ["train", "val"],
            "image_count": [train_count, val_count],
        }),
        source=str(ds_path),
        name=ds_path.name,
    )
    mlflow.log_input(dataset, context="training")
    mlflow.log_artifact(str(data_yaml), artifact_path="dataset")
    mlflow.set_tag("dataset.classes", ", ".join(ds_cfg.get("names", {}).values()))
    mlflow.set_tag("dataset.kpt_shape", str(ds_cfg.get("kpt_shape", [])))
    print(f"  Dataset logged: {ds_path.name} (train={train_count}, val={val_count})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO pose model for dart detection")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt",
                        help="Pretrained model to start from")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset directory name (e.g. dataset-combined), resolved to <dir>/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default='0',
                        help="Device: 0 for GPU, cpu for CPU, None for auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", type=str, default="dart_pose")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint (path to last.pt)")
    parser.add_argument("--register-model", type=str, default=None,
                        help="Registered model name in MLflow (e.g. screenshot-optimzed-hyper-params)")
    args = parser.parse_args()
    # Resolve --dataset dir name to --data yaml path; default to DATASET_YAML
    if args.dataset and not args.data:
        project_root = os.path.dirname(os.path.abspath(__file__))
        args.data = os.path.join(project_root, args.dataset, "dataset.yaml")
    if not args.data:
        args.data = DATASET_YAML
    return args


def _ensure_experiment(name):
    """Restore a deleted experiment so the name can be reused."""
    exp = mlflow.get_experiment_by_name(name)
    if exp and exp.lifecycle_stage == "deleted":
        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
        print(f"Restored deleted experiment '{name}'")


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    _ensure_experiment(args.name)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.name

    try:
        if args.resume:
            # Resume the most recent MLflow run for this experiment so metrics continue there
            mlflow.set_experiment(args.name)
            experiment = mlflow.get_experiment_by_name(args.name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if not runs.empty:
                    run_id = runs.iloc[0].run_id
                    mlflow.start_run(run_id=run_id)
                    print(f"Resuming MLflow run {run_id}")

            print(f"Model: {args.resume}")
            model = YOLO(args.resume)
            model.train(resume=True, workers=args.workers)
        else:
            print(f"Model: {args.model}")
            model = YOLO(args.model)
            model.train(
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                name=args.name,
                box=3.0,
                cls=12.0,
                pose=16.0,
                kobj=2.0,
                degrees=20.0,
                scale=0.7,
                close_mosaic=20,
                patience=50,
                pretrained=True,
                # Noise/color robustness augmentations
                hsv_h=0.02,    # hue jitter
                hsv_s=0.75,    # saturation jitter
                hsv_v=0.5,     # brightness jitter
                erasing=0.3,   # random erasing (simulates occlusion/noise patches)
                translate=0.15, # random translate (sub-pixel shift robustness)
                flipud=0.1,    # vertical flip
                fliplr=0.5,    # horizontal flip
            )
    except Exception as e:
        # Mark the MLflow run as failed
        run = mlflow.active_run()
        if run:
            mlflow.set_tag("error", str(e)[:250])
            mlflow.end_run(status="FAILED")
            print(f"MLflow run {run.info.run_id} marked as FAILED: {e}")
        else:
            # Ultralytics may have already ended the run; try to find and update it
            mlflow.set_experiment(args.name)
            experiment = mlflow.get_experiment_by_name(args.name)
            if experiment:
                recent = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if not recent.empty:
                    client = mlflow.tracking.MlflowClient()
                    rid = recent.iloc[0].run_id
                    client.set_tag(rid, "error", str(e)[:250])
                    client.set_terminated(rid, status="FAILED")
                    print(f"MLflow run {rid} marked as FAILED: {e}")
        raise

    # Find best weights — try local path, then look for ultralytics' numbered dirs
    best = os.path.join("runs", "pose", args.name, "weights", "best.pt")
    if not os.path.exists(best):
        # Ultralytics may append a number (e.g. dart_pose2, dart_pose3)
        pose_dir = os.path.join("runs", "pose")
        if os.path.isdir(pose_dir):
            candidates = sorted(
                [d for d in os.listdir(pose_dir) if d.startswith(args.name)],
                key=lambda d: os.path.getmtime(os.path.join(pose_dir, d)),
                reverse=True,
            )
            for c in candidates:
                p = os.path.join(pose_dir, c, "weights", "best.pt")
                if os.path.exists(p):
                    best = p
                    break

    if os.path.exists(best):
        print(f"\nRunning validation with best weights from {best}...")
        val_model = YOLO(best)
        val_model.val(data=args.data)

        # Find the MLflow run to attach registration to
        run = mlflow.active_run()
        if not run:
            # Ultralytics ended the run; find the most recent one for this experiment
            mlflow.set_experiment(args.name)
            experiment = mlflow.get_experiment_by_name(args.name)
            if experiment:
                recent = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if not recent.empty:
                    run = mlflow.start_run(run_id=recent.iloc[0].run_id)
            if not run:
                run = mlflow.start_run(run_name=args.name)

        with run:
            # Log model lineage: which base model this was trained from
            mlflow.set_tag("mlflow.source.name", args.model)
            mlflow.set_tag("base_model", args.model)
            if args.resume:
                mlflow.set_tag("resumed_from", args.resume)

            # Log dataset info
            _log_dataset(args.data)

            # Upload weights if not already present in artifacts
            client = mlflow.tracking.MlflowClient()
            existing = [f.path for f in client.list_artifacts(run.info.run_id, "weights")]
            if "weights/best.pt" not in existing:
                mlflow.log_artifact(best, artifact_path="weights")

            # Register model
            register_name = args.register_model or args.name
            model_uri = f"runs:/{run.info.run_id}/weights"
            mv = mlflow.register_model(model_uri, name=register_name)
            print(f"  Registered model '{mv.name}' version {mv.version}")


if __name__ == "__main__":
    main()
