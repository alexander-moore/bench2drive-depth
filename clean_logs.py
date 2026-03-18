#!/usr/bin/env python3
"""
clean_logs.py — Delete abandoned TensorBoard / Lightning log trials.

Trials with <= --threshold validation iterations (default 1) that have not
been modified within --min-age-hours (default 2) are considered abandoned.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOG_ROOT


def get_max_mtime(trial_root: Path) -> float:
    """Return the most recent mtime of any file under trial_root."""
    mtimes = [f.stat().st_mtime for f in trial_root.rglob("*") if f.is_file()]
    return max(mtimes) if mtimes else 0.0


def count_val_steps(trial_root: Path) -> int:
    """
    Find the event file under trial_root and count val/* scalar entries.
    Returns 0 if no event file is found or loading fails.
    """
    event_files = list(trial_root.rglob("events.out.tfevents.*"))
    if not event_files:
        return 0

    # Use the directory containing the most recently modified event file
    event_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    version_dir = event_files[0].parent

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        ea = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
        ea.Reload()
        val_steps = sum(
            len(ea.Scalars(tag))
            for tag in ea.Tags().get("scalars", [])
            if tag.startswith("val/")
        )
        return val_steps
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Clean up abandoned TensorBoard/Lightning log trials."
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_ROOT),
        help=f"Root of all experiment logs (default: {LOG_ROOT})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Max val iterations to still count as abandoned (default: 1)",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=2.0,
        help="Only consider trials last modified more than N hours ago (default: 2)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete abandoned trials (omit for dry run)",
    )
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    if not log_root.exists():
        print(f"Log directory not found: {log_root}")
        return

    min_age_seconds = args.min_age_hours * 3600
    now = time.time()

    # Collect all trial roots: log_root / {experiment} / {trial}
    trial_roots = []
    for experiment_dir in sorted(log_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for trial_dir in sorted(experiment_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial_roots.append(trial_dir)

    if not trial_roots:
        print("No trials found.")
        return

    # Header
    col_w = 55
    print(
        f"\n{'Trial Path':<{col_w}}  {'Val Iters':>10}  {'Last Modified':<22}  Action"
    )
    print("-" * (col_w + 50))

    to_delete = []

    for trial_root in trial_roots:
        max_mtime = get_max_mtime(trial_root)
        age_seconds = now - max_mtime
        last_modified = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(max_mtime))
            if max_mtime
            else "N/A"
        )

        if age_seconds < min_age_seconds:
            val_steps = "?"
            action = "SKIP (too recent)"
        else:
            val_steps = count_val_steps(trial_root)
            if val_steps <= args.threshold:
                action = "DELETE" if args.delete else "would delete"
                to_delete.append(trial_root)
            else:
                action = "keep"

        path_str = str(trial_root)
        val_str = str(val_steps) if val_steps != "?" else "?"
        print(f"{path_str:<{col_w}}  {val_str:>10}  {last_modified:<22}  {action}")

    print("-" * (col_w + 50))
    print(f"\n{len(to_delete)} trial(s) marked as abandoned (val iters <= {args.threshold}).")

    if args.delete:
        for trial_root in to_delete:
            shutil.rmtree(trial_root)
            print(f"  Deleted: {trial_root}")
        print("Done.")
    else:
        print("Dry run — pass --delete to remove them.")


if __name__ == "__main__":
    main()
