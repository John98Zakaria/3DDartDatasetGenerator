"""
Parallel dataset generation launcher.

Spawns multiple Blender instances, each generating a slice of the dataset.

Usage:
    python launch_dataset_gen.py --num_samples 50000 --workers 4
    python launch_dataset_gen.py --num_samples 1000 --workers 8 --blender "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"
"""

import argparse
import subprocess
import math
import os
import sys
import time


def find_blender():
    """Try to find Blender executable."""
    common_paths = [
        "blender",  # On PATH
        "C:/Program Files/Blender Foundation/Blender 4.3/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.1/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
    ]
    for path in common_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, timeout=10)
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Parallel YOLO dataset generation")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\John\PycharmProjects\PythonProject\dataset-multi-dart")
    parser.add_argument("--blend_file", type=str, default="blender/dart-view.blend")
    parser.add_argument("--blender", type=str, default=None, help="Path to Blender executable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--render_width", type=int, default=640)
    parser.add_argument("--render_height", type=int, default=640)
    args = parser.parse_args()

    blender = args.blender or find_blender()
    if blender is None:
        print("ERROR: Could not find Blender. Specify with --blender <path>")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_script = os.path.join(script_dir, "generate_yolo_dataset.py")
    blend_file = os.path.join(script_dir, args.blend_file)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(blend_file):
        print(f"ERROR: Blend file not found: {blend_file}")
        sys.exit(1)

    # Split samples across workers
    samples_per_worker = math.ceil(args.num_samples / args.workers)
    processes = []

    print(f"Launching {args.workers} Blender workers")
    print(f"  Total samples: {args.num_samples}")
    print(f"  Samples per worker: {samples_per_worker}")
    print(f"  Output: {output_dir}")
    print(f"  Blender: {blender}")
    print()

    start_time = time.time()

    for w in range(args.workers):
        start_index = w * samples_per_worker
        # Last worker may have fewer samples
        worker_samples = min(samples_per_worker, args.num_samples - start_index)
        if worker_samples <= 0:
            break

        cmd = [
            blender,
            blend_file,
            "--background",
            "--python", gen_script,
            "--",
            "--num_samples", str(worker_samples),
            "--start_index", str(start_index),
            "--output_dir", output_dir,
            "--seed", str(args.seed),
            "--val_ratio", str(args.val_ratio),
            "--render_width", str(args.render_width),
            "--render_height", str(args.render_height),
        ]

        log_path = os.path.join(output_dir, f"worker_{w}.log")
        os.makedirs(output_dir, exist_ok=True)
        log_file = open(log_path, "w")

        print(f"  Worker {w}: samples {start_index}-{start_index + worker_samples - 1} -> {log_path}")

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((proc, log_file, w))

    print(f"\nAll {len(processes)} workers launched. Waiting for completion...")
    print("(Check worker_*.log files in output dir for progress)\n")

    # Wait for all workers to finish
    for proc, log_file, w in processes:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
        print(f"  Worker {w}: {status}")

    elapsed = time.time() - start_time
    print(f"\nAll workers done in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
