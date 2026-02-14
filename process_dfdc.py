"""
Process DFDC part zips ONE AT A TIME — fits in 80 GB.
=====================================================

Download each part zip from Kaggle website, then run:

    python process_dfdc.py --zip D:\\Downloads\\dfdc_train_part_00.zip
    python process_dfdc.py --zip D:\\Downloads\\dfdc_train_part_01.zip
    ...etc for all 10 parts

OR if you extracted a part to a folder:

    python process_dfdc.py --folder D:\\Downloads\\dfdc_train_part_00

What it does per part:
  1. Opens the zip (or folder) — reads metadata.json inside
  2. Extracts ALL real videos  → dataset/dfdc_videos/real/
  3. Extracts SAME COUNT of fake (deterministic seed) → dataset/dfdc_videos/fake/
  4. DELETES the zip to free space (--keep-zip to skip deletion)
  5. Shows progress + running totals

Space budget (80 GB available):
  - One zip at a time: ~10 GB temp
  - Accumulated output: ~4 GB per part × 10 = ~40 GB
  - Peak usage: ~50 GB  ✓ fits easily

After all 10 parts:
    python train.py --dataset dataset/dfdc_videos
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path("dataset/dfdc_videos")
PROGRESS_FILE = Path("dataset/dfdc_progress.json")


# =========================================================================
# Progress tracking — remembers which parts are done across runs
# =========================================================================
def load_progress():
    """Load progress from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"parts_done": [], "real_count": 0, "fake_count": 0}


def save_progress(progress):
    """Save progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# =========================================================================
# Detect part index from zip/folder name
# =========================================================================
def detect_part_index(path_str):
    """Extract part number from a path like dfdc_train_part_03.zip or folder name."""
    import re
    # Match patterns: part_0, part_00, part_03, part_9, etc.
    m = re.search(r'part[_\-]?(\d+)', str(path_str))
    if m:
        return int(m.group(1))
    return None


# =========================================================================
# Process from a ZIP file (without full extraction)
# =========================================================================
def process_zip(zip_path, output_dir, keep_zip=False):
    """
    Open a DFDC part zip, selectively extract only needed videos.
    Reads metadata.json from inside the zip to determine labels.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"  ERROR: File not found: {zip_path}")
        sys.exit(1)

    part_idx = detect_part_index(zip_path.name)
    if part_idx is None:
        print(f"  WARNING: Can't detect part number from '{zip_path.name}'")
        print(f"  Assuming part 0. Use --part N to override.")
        part_idx = 0

    progress = load_progress()
    if part_idx in progress["parts_done"]:
        print(f"  Part {part_idx} already processed! Skipping.")
        print(f"  (Delete {PROGRESS_FILE} to reprocess)")
        return

    zip_size_gb = zip_path.stat().st_size / (1024**3)
    print(f"\n  Processing: {zip_path.name} ({zip_size_gb:.1f} GB)")
    print(f"  Part index: {part_idx}")

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        all_names = zf.namelist()
        print(f"  Files in zip: {len(all_names)}")

        # Find metadata.json inside the zip
        meta_entries = [n for n in all_names if n.endswith('metadata.json')]
        if not meta_entries:
            print("  ERROR: No metadata.json found in zip!")
            sys.exit(1)

        meta_entry = meta_entries[0]
        print(f"  Metadata: {meta_entry}")

        # Read metadata
        with zf.open(meta_entry) as mf:
            metadata = json.load(mf)

        # Classify videos
        real_vids = []
        fake_vids = []
        for filename, info in metadata.items():
            label = info.get("label", "").upper()
            # Find this file in the zip (could be nested in subfolder)
            matching = [n for n in all_names if n.endswith('/' + filename) or n == filename]
            if not matching:
                continue
            zip_name = matching[0]
            if label == "REAL":
                real_vids.append((filename, zip_name))
            elif label == "FAKE":
                fake_vids.append((filename, zip_name))

        n_real = len(real_vids)
        n_fake_total = len(fake_vids)
        print(f"  Real: {n_real}, Fake: {n_fake_total}")

        # Balance: keep ALL real + same count of fake (deterministic per-part)
        random.seed(SEED + part_idx)  # Deterministic per part
        random.shuffle(fake_vids)
        fake_selected = fake_vids[:n_real]
        print(f"  Keeping: {n_real} real + {len(fake_selected)} fake")

        # Extract real videos
        print(f"\n  Extracting real videos...")
        real_ok = 0
        for i, (filename, zip_name) in enumerate(real_vids):
            dst = real_dir / f"part{part_idx}_{filename}"
            if dst.exists() and dst.stat().st_size > 1000:
                real_ok += 1
                continue
            try:
                data = zf.read(zip_name)
                with open(dst, 'wb') as f:
                    f.write(data)
                if dst.stat().st_size > 1000:
                    real_ok += 1
                else:
                    dst.unlink()
            except Exception as e:
                pass

            if (i + 1) % 20 == 0 or i == n_real - 1:
                print(f"\r    [{i+1}/{n_real}] {real_ok} extracted    ",
                      end="", flush=True)

        print(f"\n    Real: {real_ok}/{n_real} extracted")

        # Extract fake videos
        print(f"  Extracting fake videos...")
        fake_ok = 0
        n_fake_sel = len(fake_selected)
        for i, (filename, zip_name) in enumerate(fake_selected):
            dst = fake_dir / f"part{part_idx}_{filename}"
            if dst.exists() and dst.stat().st_size > 1000:
                fake_ok += 1
                continue
            try:
                data = zf.read(zip_name)
                with open(dst, 'wb') as f:
                    f.write(data)
                if dst.stat().st_size > 1000:
                    fake_ok += 1
                else:
                    dst.unlink()
            except Exception as e:
                pass

            if (i + 1) % 20 == 0 or i == n_fake_sel - 1:
                print(f"\r    [{i+1}/{n_fake_sel}] {fake_ok} extracted    ",
                      end="", flush=True)

        print(f"\n    Fake: {fake_ok}/{n_fake_sel} extracted")

    # Update progress
    progress["parts_done"].append(part_idx)
    progress["real_count"] += real_ok
    progress["fake_count"] += fake_ok
    save_progress(progress)

    # Delete zip to free space
    if not keep_zip:
        print(f"\n  Deleting zip to free {zip_size_gb:.1f} GB...")
        zip_path.unlink()
        print(f"  Deleted: {zip_path.name}")
    else:
        print(f"\n  Keeping zip (--keep-zip)")

    # Show running totals
    total_real = len(list(real_dir.glob("*.mp4")))
    total_fake = len(list(fake_dir.glob("*.mp4")))
    gb = sum(f.stat().st_size for f in real_dir.glob("*.mp4")) / (1024**3)
    gb += sum(f.stat().st_size for f in fake_dir.glob("*.mp4")) / (1024**3)

    parts_remaining = 10 - len(progress["parts_done"])
    print(f"\n  {'='*50}")
    print(f"  Part {part_idx} done!")
    print(f"  Running total: {total_real} real + {total_fake} fake ({gb:.1f} GB)")
    print(f"  Parts processed: {sorted(progress['parts_done'])}")
    print(f"  Parts remaining: {parts_remaining}")
    if parts_remaining > 0:
        print(f"\n  Next: download the next part zip from Kaggle and run:")
        print(f"    python process_dfdc.py --zip <path_to_next_zip>")
    else:
        print(f"\n  ALL PARTS DONE! Ready to train:")
        print(f"    python train.py --dataset {output_dir}")
    print(f"  {'='*50}")


# =========================================================================
# Process from an extracted folder
# =========================================================================
def process_folder(folder_path, output_dir, keep_folder=False):
    """Process an already-extracted DFDC part folder."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"  ERROR: Folder not found: {folder_path}")
        sys.exit(1)

    part_idx = detect_part_index(folder_path.name)
    if part_idx is None:
        # Check parent
        part_idx = detect_part_index(str(folder_path))
    if part_idx is None:
        part_idx = 0

    progress = load_progress()
    if part_idx in progress["parts_done"]:
        print(f"  Part {part_idx} already processed! Skipping.")
        return

    print(f"\n  Processing folder: {folder_path}")
    print(f"  Part index: {part_idx}")

    # Find metadata.json (could be in subfolder)
    meta_candidates = list(folder_path.rglob("metadata.json"))
    if not meta_candidates:
        print("  ERROR: No metadata.json found!")
        sys.exit(1)

    meta_path = meta_candidates[0]
    meta_dir = meta_path.parent  # Videos are in same dir as metadata

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_vids = []
    fake_vids = []
    for filename, info in metadata.items():
        label = info.get("label", "").upper()
        src = meta_dir / filename
        if not src.exists():
            # Search deeper
            matches = list(folder_path.rglob(filename))
            src = matches[0] if matches else None
        if src and src.exists():
            if label == "REAL":
                real_vids.append((filename, src))
            elif label == "FAKE":
                fake_vids.append((filename, src))

    n_real = len(real_vids)
    print(f"  Real: {n_real}, Fake: {len(fake_vids)}")

    # Balance
    random.seed(SEED + part_idx)
    random.shuffle(fake_vids)
    fake_selected = fake_vids[:n_real]
    print(f"  Keeping: {n_real} real + {len(fake_selected)} fake")

    # Copy real
    print(f"  Copying real videos...")
    real_ok = 0
    for i, (filename, src) in enumerate(real_vids):
        dst = real_dir / f"part{part_idx}_{filename}"
        if dst.exists() and dst.stat().st_size > 1000:
            real_ok += 1
            continue
        try:
            shutil.copy2(str(src), str(dst))
            real_ok += 1
        except Exception:
            pass
        if (i + 1) % 50 == 0 or i == n_real - 1:
            print(f"\r    [{i+1}/{n_real}] {real_ok} copied    ",
                  end="", flush=True)
    print(f"\n    Real: {real_ok}/{n_real}")

    # Copy fake
    print(f"  Copying fake videos...")
    fake_ok = 0
    n_fake_sel = len(fake_selected)
    for i, (filename, src) in enumerate(fake_selected):
        dst = fake_dir / f"part{part_idx}_{filename}"
        if dst.exists() and dst.stat().st_size > 1000:
            fake_ok += 1
            continue
        try:
            shutil.copy2(str(src), str(dst))
            fake_ok += 1
        except Exception:
            pass
        if (i + 1) % 50 == 0 or i == n_fake_sel - 1:
            print(f"\r    [{i+1}/{n_fake_sel}] {fake_ok} copied    ",
                  end="", flush=True)
    print(f"\n    Fake: {fake_ok}/{n_fake_sel}")

    # Update progress
    progress["parts_done"].append(part_idx)
    progress["real_count"] += real_ok
    progress["fake_count"] += fake_ok
    save_progress(progress)

    # Optionally delete folder
    if not keep_folder:
        folder_size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
        gb = folder_size / (1024**3)
        print(f"\n  Deleting extracted folder to free {gb:.1f} GB...")
        shutil.rmtree(str(folder_path))
        print(f"  Deleted: {folder_path}")

    # Running totals
    total_real = len(list(real_dir.glob("*.mp4")))
    total_fake = len(list(fake_dir.glob("*.mp4")))
    gb = sum(f.stat().st_size for f in real_dir.glob("*.mp4")) / (1024**3)
    gb += sum(f.stat().st_size for f in fake_dir.glob("*.mp4")) / (1024**3)

    parts_remaining = 10 - len(progress["parts_done"])
    print(f"\n  {'='*50}")
    print(f"  Part {part_idx} done!")
    print(f"  Running total: {total_real} real + {total_fake} fake ({gb:.1f} GB)")
    print(f"  Parts processed: {sorted(progress['parts_done'])}")
    print(f"  Parts remaining: {parts_remaining}")
    if parts_remaining > 0:
        print(f"\n  Next: download the next part and run:")
        print(f"    python process_dfdc.py --zip <next_zip>")
        print(f"    python process_dfdc.py --folder <next_folder>")
    else:
        print(f"\n  ALL PARTS DONE! Ready to train:")
        print(f"    python train.py --dataset {output_dir}")
    print(f"  {'='*50}")


# =========================================================================
# Status command
# =========================================================================
def show_status(output_dir):
    """Show current progress."""
    progress = load_progress()
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"

    n_real = len(list(real_dir.glob("*.mp4"))) if real_dir.exists() else 0
    n_fake = len(list(fake_dir.glob("*.mp4"))) if fake_dir.exists() else 0

    gb = 0
    if real_dir.exists():
        gb += sum(f.stat().st_size for f in real_dir.glob("*.mp4")) / (1024**3)
    if fake_dir.exists():
        gb += sum(f.stat().st_size for f in fake_dir.glob("*.mp4")) / (1024**3)

    print(f"\n  {'='*50}")
    print(f"  DFDC Dataset Status")
    print(f"  {'='*50}")
    print(f"  Parts done: {sorted(progress.get('parts_done', []))}")
    print(f"  Parts remaining: {sorted(set(range(10)) - set(progress.get('parts_done', [])))}")
    print(f"  Real videos: {n_real}")
    print(f"  Fake videos: {n_fake}")
    print(f"  Total size:  {gb:.1f} GB")
    print(f"  Output dir:  {output_dir}")
    print(f"  {'='*50}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Process DFDC part zips one at a time (fits in 80 GB)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", type=str,
                       help="Path to a downloaded DFDC part .zip file")
    group.add_argument("--folder", type=str,
                       help="Path to an already-extracted DFDC part folder")
    group.add_argument("--status", action="store_true",
                       help="Show current progress")

    parser.add_argument("--output", type=str, default="dataset/dfdc_videos",
                        help="Output directory (default: dataset/dfdc_videos)")
    parser.add_argument("--part", type=int, default=None,
                        help="Override part index (auto-detected from filename)")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Don't delete the zip after processing")
    parser.add_argument("--keep-folder", action="store_true",
                        help="Don't delete the extracted folder after processing")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("  DFDC One-Part-At-A-Time Processor")
    print("=" * 60)

    if args.status:
        show_status(output_dir)
        return

    if args.zip:
        process_zip(Path(args.zip), output_dir, keep_zip=args.keep_zip)
    elif args.folder:
        process_folder(Path(args.folder), output_dir, keep_folder=args.keep_folder)


if __name__ == "__main__":
    main()
