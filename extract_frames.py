import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_DIR = Path("dataset")
OUTPUT_DIR = DATASET_DIR / "frames"
REAL_VIDEO_DIR = DATASET_DIR / "DFD_original sequences"
FAKE_VIDEO_DIR = DATASET_DIR / "DFD_manipulated_sequences" / "DFD_manipulated_sequences"

# Extraction settings
FRAMES_PER_VIDEO = 15       # Extract 15 frames per video (evenly spaced)
VAL_SPLIT = 0.15            # 15% of videos go to validation
IMAGE_SIZE = (300, 300)      # Resize frames to 300x300 (crop/resize happens in training)
JPEG_QUALITY = 95            # Save quality


def extract_frames_from_video(video_path, output_dir, prefix, max_frames=15):
    """Extract evenly-spaced frames from a single video.
    
    Args:
        video_path: Path to .mp4 file
        output_dir: Directory to save extracted frames
        prefix: Filename prefix for saved frames
        max_frames: Maximum number of frames to extract
        
    Returns:
        Number of frames extracted
    """
    # Skip if frames already exist for this video
    existing = list(output_dir.glob(f"{prefix}_f*.jpg"))
    if len(existing) >= max_frames:
        return len(existing)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Could not open {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0
    
    # Calculate frame indices (evenly spaced)
    num_to_extract = min(max_frames, total_frames)
    if num_to_extract <= 0:
        cap.release()
        return 0
    
    # Skip first and last 5% of video (often black/transition frames)
    start_frame = int(total_frames * 0.05)
    end_frame = int(total_frames * 0.95)
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames - 1
    
    indices = [int(start_frame + i * (end_frame - start_frame) / num_to_extract) 
               for i in range(num_to_extract)]
    
    count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # Resize
        frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        # Save
        filename = f"{prefix}_f{idx:06d}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        count += 1
    
    cap.release()
    return count


def main():
    print("=" * 60)
    print("  DFD Dataset Frame Extraction")
    print("=" * 60)
    
    # Check directories exist
    if not REAL_VIDEO_DIR.exists():
        print(f"ERROR: Real video directory not found: {REAL_VIDEO_DIR}")
        return
    if not FAKE_VIDEO_DIR.exists():
        print(f"ERROR: Fake video directory not found: {FAKE_VIDEO_DIR}")
        return
    
    # Collect video files
    real_videos = sorted([f for f in REAL_VIDEO_DIR.iterdir() if f.suffix.lower() == '.mp4'])
    fake_videos = sorted([f for f in FAKE_VIDEO_DIR.iterdir() if f.suffix.lower() == '.mp4'])
    
    print(f"  Real videos: {len(real_videos)}")
    print(f"  Fake videos: {len(fake_videos)}")
    print(f"  Frames per video: {FRAMES_PER_VIDEO}")
    print(f"  Val split: {VAL_SPLIT*100:.0f}%")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("ERROR: No videos found!")
        return
    
    # Split videos into train/val
    random.seed(42)  # Reproducible splits
    
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    n_real_val = max(1, int(len(real_videos) * VAL_SPLIT))
    n_fake_val = max(1, int(len(fake_videos) * VAL_SPLIT))
    
    real_val = real_videos[:n_real_val]
    real_train = real_videos[n_real_val:]
    fake_val = fake_videos[:n_fake_val]
    fake_train = fake_videos[n_fake_val:]
    
    print(f"\n  Train: {len(real_train)} real + {len(fake_train)} fake videos")
    print(f"  Val:   {len(real_val)} real + {len(fake_val)} fake videos")
    
    # Create output directories
    dirs = {
        'train_real': OUTPUT_DIR / 'train' / 'real',
        'train_fake': OUTPUT_DIR / 'train' / 'fake',
        'val_real': OUTPUT_DIR / 'val' / 'real',
        'val_fake': OUTPUT_DIR / 'val' / 'fake',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    total_frames = 0
    
    # Extract from training real videos
    print(f"\n--- Extracting TRAIN REAL frames ---")
    for video in tqdm(real_train, desc="Train Real"):
        prefix = video.stem
        n = extract_frames_from_video(video, dirs['train_real'], prefix, FRAMES_PER_VIDEO)
        total_frames += n
    
    # Extract from training fake videos
    print(f"\n--- Extracting TRAIN FAKE frames ---")
    for video in tqdm(fake_train, desc="Train Fake"):
        prefix = video.stem
        n = extract_frames_from_video(video, dirs['train_fake'], prefix, FRAMES_PER_VIDEO)
        total_frames += n
    
    # Extract from validation real videos
    print(f"\n--- Extracting VAL REAL frames ---")
    for video in tqdm(real_val, desc="Val Real"):
        prefix = video.stem
        n = extract_frames_from_video(video, dirs['val_real'], prefix, FRAMES_PER_VIDEO)
        total_frames += n
    
    # Extract from validation fake videos
    print(f"\n--- Extracting VAL FAKE frames ---")
    for video in tqdm(fake_val, desc="Val Fake"):
        prefix = video.stem
        n = extract_frames_from_video(video, dirs['val_fake'], prefix, FRAMES_PER_VIDEO)
        total_frames += n
    
    # Print summary
    train_real_count = len(list(dirs['train_real'].glob('*.jpg')))
    train_fake_count = len(list(dirs['train_fake'].glob('*.jpg')))
    val_real_count = len(list(dirs['val_real'].glob('*.jpg')))
    val_fake_count = len(list(dirs['val_fake'].glob('*.jpg')))
    
    print(f"\n{'='*60}")
    print(f"  Frame Extraction Complete!")
    print(f"{'='*60}")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Train: {train_real_count} real + {train_fake_count} fake = {train_real_count + train_fake_count}")
    print(f"  Val:   {val_real_count} real + {val_fake_count} fake = {val_real_count + val_fake_count}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\n  To start training, run:")
    print(f"    python train.py --data_dir dataset/frames --epochs 30 --batch_size 16 --resume weights/best_model.pth")


if __name__ == '__main__':
    main()
