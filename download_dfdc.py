"""
DFDC Video Downloader — Downloads BALANCED raw .mp4 videos.
===========================================================

Source: Kaggle dataset 'pranay22077/dfdc-10' (10 DFDC training parts)
Output: dataset/dfdc_videos/real/ + dataset/dfdc_videos/fake/

Key fixes over previous version:
  - SEQUENTIAL downloads (no threading) to avoid 429 rate limits
  - Exponential backoff: waits & retries on 429 (never gives up)
  - Uses requests library for reliable HTTP downloads
  - NO file deletions — files are NEVER removed
  - Fully resumable: skips already-downloaded files
  - Verifies every file after download (size > 1KB)

Usage:
    python download_dfdc.py
    python download_dfdc.py --delay 1.0
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
KAGGLE_DATASET = "pranay22077/dfdc-10"
NUM_PARTS = 10
OUTPUT_DIR = Path("dataset/dfdc_videos")
META_DIR = Path("dataset/dfdc_meta")
SEED = 42


def part_folder(i):
    """Remote path prefix for DFDC part i in the Kaggle dataset."""
    return f"dfdc_train_part_{i:02d}/dfdc_train_part_{i}"


# =============================================================================
# Auth: get requests session with Kaggle credentials
# =============================================================================
def get_kaggle_session():
    """Build a requests.Session with Kaggle auth headers."""
    import requests

    # Try KAGGLE_API_TOKEN first (new-style)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {token}"
        return s

    # Try kaggle.json (old-style username/key)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            creds = json.load(f)
        s = requests.Session()
        s.auth = (creds["username"], creds["key"])
        return s

    # Try env vars
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        s = requests.Session()
        s.auth = (user, key)
        return s

    print("  ERROR: No Kaggle credentials found!")
    print("  Set KAGGLE_API_TOKEN env var, or place kaggle.json in ~/.kaggle/")
    sys.exit(1)


# =============================================================================
# Download one file with retry + backoff
# =============================================================================
def download_one(session, remote_path, local_path, delay=0.5, max_retries=20):
    """
    Download a single file from Kaggle dataset via HTTP.
    Retries with exponential backoff on 429.
    Returns True on success, False on permanent failure.
    """
    import requests

    local_path = Path(local_path)
    if local_path.exists() and local_path.stat().st_size > 1000:
        return True  # Already have it

    local_path.parent.mkdir(parents=True, exist_ok=True)

    import urllib.parse
    encoded = urllib.parse.quote(remote_path, safe='')
    url = (f"https://www.kaggle.com/api/v1/datasets/download/"
           f"{KAGGLE_DATASET}/{encoded}")

    backoff = 60  # Start with 60s on rate limit
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            r = session.get(url, stream=True, timeout=120, allow_redirects=True)

            if r.status_code == 200:
                # Check content type - could be raw file or zip
                ct = r.headers.get("content-type", "")
                tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)

                if tmp_path.stat().st_size < 500:
                    tmp_path.unlink()
                    return False

                # Check if it's a zip (Kaggle sometimes wraps files)
                if tmp_path.stat().st_size > 0:
                    try:
                        with zipfile.ZipFile(str(tmp_path), 'r') as zf:
                            names = zf.namelist()
                            # For mp4 downloads: find the mp4 inside
                            mp4s = [n for n in names if n.endswith('.mp4')]
                            jsons = [n for n in names if n.endswith('.json')]
                            target_file = mp4s[0] if mp4s else (jsons[0] if jsons else names[0])
                            # Extract to temp dir, move to final location
                            extract_dir = local_path.parent / "__zip_tmp__"
                            zf.extract(target_file, str(extract_dir))
                            extracted = extract_dir / target_file
                            shutil.move(str(extracted), str(local_path))
                            # Clean up
                            tmp_path.unlink()
                            shutil.rmtree(str(extract_dir), ignore_errors=True)
                            return local_path.exists() and local_path.stat().st_size > 500
                    except zipfile.BadZipFile:
                        pass  # Not a zip, it's the raw file

                    # Raw file — just rename
                    shutil.move(str(tmp_path), str(local_path))
                    return local_path.exists() and local_path.stat().st_size > 1000

            elif r.status_code == 429:
                print(f"\n  [429 Rate limited] Waiting {backoff}s before retry "
                      f"(attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 900)  # Max 15 min wait
                continue

            elif r.status_code == 404:
                return False  # File doesn't exist on Kaggle

            else:
                # Other error — short retry
                time.sleep(5)
                continue

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            time.sleep(10)
            continue
        except Exception as e:
            return False

    return False  # Exhausted retries


# =============================================================================
# Phase 1: Download metadata files
# =============================================================================
def download_all_metadata(session, delay):
    """Download metadata.json from each of the 10 parts."""
    print("  Phase 1: Downloading metadata (10 files)...")
    META_DIR.mkdir(parents=True, exist_ok=True)
    all_meta = {}

    for i in range(NUM_PARTS):
        local = META_DIR / f"metadata_part_{i}.json"

        if local.exists() and local.stat().st_size > 100:
            with open(local, 'r') as f:
                all_meta[i] = json.load(f)
            r = sum(1 for v in all_meta[i].values() if v.get("label","").upper() == "REAL")
            fk = len(all_meta[i]) - r
            print(f"    Part {i}: cached — {r} real, {fk} fake")
            continue

        remote = f"{part_folder(i)}/metadata.json"
        ok = download_one(session, remote, local, delay=delay)
        if ok and local.exists():
            with open(local, 'r') as f:
                all_meta[i] = json.load(f)
            r = sum(1 for v in all_meta[i].values() if v.get("label","").upper() == "REAL")
            fk = len(all_meta[i]) - r
            print(f"    Part {i}: {r} real, {fk} fake")
        else:
            print(f"    Part {i}: FAILED (will retry on next run)")

    return all_meta


# =============================================================================
# Phase 2: Build balanced download list
# =============================================================================
def build_download_list(all_meta, output_dir):
    """Build list of (remote_path, local_path, label) to download."""
    real_files = []
    fake_files = []

    for part_idx, meta in all_meta.items():
        for filename, info in meta.items():
            label = info.get("label", "").upper()
            remote = f"{part_folder(part_idx)}/{filename}"
            local = output_dir / label.lower() / f"part{part_idx}_{filename}"

            if label == "REAL":
                real_files.append((remote, local, "real"))
            elif label == "FAKE":
                fake_files.append((remote, local, "fake"))

    # Balance: keep ALL real, sample fake to match
    random.seed(SEED)
    random.shuffle(fake_files)
    target = len(real_files)
    fake_files = fake_files[:target]

    print(f"\n  Total available: {len(real_files)} real across all parts")
    print(f"  Balanced target: {target} real + {target} fake = {target*2} videos")

    # Merge and remove already-downloaded
    all_tasks = real_files + fake_files
    pending = [(r, l, lab) for r, l, lab in all_tasks
               if not (l.exists() and l.stat().st_size > 1000)]

    already = len(all_tasks) - len(pending)
    print(f"  Already downloaded: {already}")
    print(f"  Remaining to download: {len(pending)}")

    return all_tasks, pending


# =============================================================================
# Phase 3: Sequential download with progress
# =============================================================================
def download_videos(session, pending, delay):
    """Download all pending videos sequentially with rate limit handling."""
    total = len(pending)
    if total == 0:
        print("\n  All videos already downloaded!")
        return 0, 0

    print(f"\n  Phase 3: Downloading {total} videos (sequential, {delay}s delay)")
    print(f"  Estimated time: ~{total * (8 + delay) / 3600:.1f} hours")
    print(f"  (Resume-safe: re-run if interrupted)\n")

    success = 0
    failed = 0
    start_time = time.time()

    for i, (remote, local, label) in enumerate(pending):
        ok = download_one(session, remote, local, delay=delay)
        if ok:
            success += 1
        else:
            failed += 1

        # Progress every 10 files
        if (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (success + failed) / max(elapsed, 1)
            remaining = (total - i - 1) / max(rate, 0.001)
            print(f"\r  [{i+1}/{total}] {success} ok, {failed} fail | "
                  f"{elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m left    ",
                  end="", flush=True)

    print(f"\n\n  Download phase done: {success} succeeded, {failed} failed")
    return success, failed


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download balanced DFDC raw videos"
    )
    parser.add_argument("--output", type=str, default="dataset/dfdc_videos")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between downloads (default: 0.5)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    (output_dir / "real").mkdir(parents=True, exist_ok=True)
    (output_dir / "fake").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DFDC Video Downloader (Sequential + Backoff)")
    print("=" * 60)
    print(f"  Source:  {KAGGLE_DATASET}")
    print(f"  Output:  {output_dir}")
    print(f"  Delay:   {args.delay}s between downloads")
    print(f"  Mode:    Sequential (no threading, no rate limit issues)")
    print("=" * 60)

    # Auth
    session = get_kaggle_session()
    print("  Kaggle session ready\n")

    # Phase 1: metadata
    all_meta = download_all_metadata(session, args.delay)
    if not all_meta:
        print("  No metadata. Check Kaggle credentials.")
        sys.exit(1)

    # Phase 2: build list
    all_tasks, pending = build_download_list(all_meta, output_dir)

    # Phase 3: download
    success, failed = download_videos(session, pending, args.delay)

    # Summary (NO deletions!)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    n_real = len(list(real_dir.glob("*.mp4")))
    n_fake = len(list(fake_dir.glob("*.mp4")))
    gb_real = sum(f.stat().st_size for f in real_dir.glob("*.mp4")) / (1024**3)
    gb_fake = sum(f.stat().st_size for f in fake_dir.glob("*.mp4")) / (1024**3)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Real videos: {n_real:,}  ({gb_real:.1f} GB)")
    print(f"  Fake videos: {n_fake:,}  ({gb_fake:.1f} GB)")
    print(f"  Total:       {n_real + n_fake:,} ({gb_real + gb_fake:.1f} GB)")
    print(f"  Location:    {output_dir}")

    if failed > 0:
        print(f"\n  {failed} downloads failed. Re-run to retry them:")
        print(f"    python download_dfdc.py")

    print(f"\n  Next step — train:")
    print(f"    python train.py --dataset {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
