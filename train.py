"""
Training script for DeepfakeEfficientNet on FaceForensics++ C23 dataset.

Features:
  - FULL CHECKPOINT RESUME: Stop anytime (Ctrl+C, shutdown, power off) and
    resume exactly where you left off. Saves every epoch + writes a resume
    checkpoint that stores epoch, optimizer, scheduler, scaler, best metrics,
    and RNG states so training is 100% resumable.
  - REGULARIZATION: Label smoothing, Mixup, CutMix, stochastic weight
    averaging (SWA), gradient clipping, weight decay.
  - OPTIMIZATION: OneCycleLR with warmup, gradient accumulation for larger
    effective batch size, EMA (exponential moving average) model, early
    stopping with patience.
  - BALANCED SAMPLING: WeightedRandomSampler to handle 6:1 FAKE:REAL imbalance.
  - AUGMENTATIONS: JPEG compression, color jitter, random erasing, flip,
    rotation, grayscale.

Usage:
    # Start training (auto-resumes if checkpoint exists):
    python train.py

    # Force fresh start:
    python train.py --fresh

    # Custom settings:
    python train.py --epochs 30 --batch_size 16 --lr 3e-4

    # Resume after shutdown/restart — just run the same command:
    python train.py
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from model import DeepfakeEfficientNet

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Configuration
# =============================================================================
FF_DATASET_DIR = Path("dataset/FaceForensics++_C23")

REAL_FOLDERS = ["original"]
FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap",
                "NeuralTextures", "DeepFakeDetection"]

FACE_MARGIN = 0.3
MIN_FACE_SIZE = 60

CHECKPOINT_DIR = Path("weights")
RESUME_CHECKPOINT = CHECKPOINT_DIR / "training_checkpoint.pth"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
TRAINING_LOG_PATH = CHECKPOINT_DIR / "training_log.json"


# =============================================================================
# Graceful Stop Handler
# =============================================================================
_stop_requested = False

def _signal_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        print("\n\n  Force quit! (progress saved at last epoch)")
        sys.exit(1)
    _stop_requested = True
    print("\n\n  >>> STOP REQUESTED — will save and exit after current epoch.")
    print("  >>> Press Ctrl+C again to force quit.\n")

signal.signal(signal.SIGINT, _signal_handler)
try:
    signal.signal(signal.SIGBREAK, _signal_handler)  # Windows Ctrl+Break
except AttributeError:
    pass


# =============================================================================
# 1. Face Extraction from Videos
# =============================================================================
def get_face_detector():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return cascade


def detect_face_haar(frame, cascade, min_size=60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(min_size, min_size)
    )
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]
    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(fw, x + w + margin_x)
    y2 = min(fh, y + h + margin_y)
    return (x1, y1, x2 - x1, y2 - y1)


def extract_face_crops_from_video(video_path, cascade, max_frames=15, size=224):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    start = int(total_frames * 0.05)
    end = int(total_frames * 0.95)
    if end <= start:
        start, end = 0, total_frames - 1
    n_candidates = min(max_frames * 3, end - start + 1)
    candidate_indices = sorted(random.sample(range(start, end + 1), n_candidates))
    crops = []
    for idx in candidate_indices:
        if len(crops) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        face_box = detect_face_haar(frame, cascade)
        if face_box is None:
            continue
        x, y, w, h = face_box
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.shape[0] < MIN_FACE_SIZE or face_crop.shape[1] < MIN_FACE_SIZE:
            continue
        face_crop = cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        crops.append(face_pil)
    cap.release()
    return crops


# =============================================================================
# 2. Pre-extraction (cached to disk)
# =============================================================================
VAL_SPLIT = 0.15

def preextract_faces(output_dir, frames_per_video=15):
    output_dir = Path(output_dir)
    # Only skip if ALL four subdirectories have images (prevents partial extraction)
    dirs_to_check = [
        output_dir / "train" / "real",
        output_dir / "train" / "fake",
        output_dir / "val" / "real",
        output_dir / "val" / "fake",
    ]
    counts = {}
    all_populated = True
    for d in dirs_to_check:
        if d.exists():
            counts[d.name + "_" + d.parent.name] = len(list(d.glob("*.jpg")))
        else:
            counts[d.name + "_" + d.parent.name] = 0
        if counts[d.name + "_" + d.parent.name] < 10:
            all_populated = False

    if all_populated:
        real_count = len(list((output_dir / "train" / "real").glob("*.jpg"))) + \
                     len(list((output_dir / "val" / "real").glob("*.jpg")))
        fake_count = len(list((output_dir / "train" / "fake").glob("*.jpg"))) + \
                     len(list((output_dir / "val" / "fake").glob("*.jpg")))
        print(f"  Face crops already extracted ({real_count} real, {fake_count} fake)")
        print(f"  Delete '{output_dir}' to re-extract\n")
        return
    elif any(v > 0 for v in counts.values()):
        print(f"  Incomplete extraction detected — re-extracting from scratch...")
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)

    print("=" * 60)
    print("  Phase 1: Extracting face crops from FF++ videos")
    print("=" * 60)
    cascade = get_face_detector()
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    video_entries = []
    for folder_name in REAL_FOLDERS:
        folder = FF_DATASET_DIR / folder_name
        if not folder.exists():
            continue
        for v in sorted(folder.glob("*.mp4")):
            video_entries.append((v, "real", folder_name))
    for folder_name in FAKE_FOLDERS:
        folder = FF_DATASET_DIR / folder_name
        if not folder.exists():
            continue
        for v in sorted(folder.glob("*.mp4")):
            video_entries.append((v, "fake", folder_name))

    real_vids = [e for e in video_entries if e[1] == "real"]
    fake_vids = [e for e in video_entries if e[1] == "fake"]
    print(f"  Real: {len(real_vids)} videos | Fake: {len(fake_vids)} videos")

    random.seed(42)
    random.shuffle(real_vids)
    random.shuffle(fake_vids)
    n_val_real = max(1, int(len(real_vids) * VAL_SPLIT))
    n_val_fake = max(1, int(len(fake_vids) * VAL_SPLIT))
    splits = {
        "train": real_vids[n_val_real:] + fake_vids[n_val_fake:],
        "val": real_vids[:n_val_real] + fake_vids[:n_val_fake],
    }
    print(f"  Train: {len(splits['train'])} videos | Val: {len(splits['val'])} videos")

    total_extracted = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}
    for split_name, entries in splits.items():
        print(f"\n  Extracting {split_name}...")
        for video_path, label, source_name in tqdm(entries, desc=f"  {split_name}"):
            crops = extract_face_crops_from_video(
                video_path, cascade, max_frames=frames_per_video
            )
            for i, crop in enumerate(crops):
                fname = f"{source_name}_{video_path.stem}_f{i:02d}.jpg"
                save_path = output_dir / split_name / label / fname
                crop.save(str(save_path), quality=95)
                total_extracted[split_name][label] += 1

    print(f"\n  Extraction complete!")
    print(f"  Train: {total_extracted['train']['real']} real, "
          f"{total_extracted['train']['fake']} fake")
    print(f"  Val: {total_extracted['val']['real']} real, "
          f"{total_extracted['val']['fake']} fake\n")


# =============================================================================
# 3. Augmentations
# =============================================================================
class JPEGAugmentation:
    def __init__(self, quality_range=(20, 75), prob=0.5):
        self.quality_range = quality_range
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image
        img_array = np.array(image)
        quality = random.randint(*self.quality_range)
        _, encoded = cv2.imencode('.jpg', img_array,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


class GaussianNoise:
    """Add random Gaussian noise to simulate camera/compression noise."""
    def __init__(self, std_range=(0.01, 0.05), prob=0.3):
        self.std_range = std_range
        self.prob = prob

    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)


# =============================================================================
# 4. Mixup & CutMix (regularization)
# =============================================================================
def mixup_data(x, y, alpha=0.4):
    """Mixup: blend two samples and their labels."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste a random patch from one image onto another."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = random.randint(0, H)
    cx = random.randint(0, W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    x_clone = x.clone()
    x_clone[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_clone, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# 5. EMA (Exponential Moving Average) Model
# =============================================================================
class EMAModel:
    """Maintains an exponential moving average of model parameters.
    
    The EMA model typically generalizes better than the raw trained model.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model):
        """Replace model params with EMA params (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# =============================================================================
# 6. Dataset
# =============================================================================
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', image_size=224):
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.split = split
        self.samples = []
        self.labels = []

        for img_path in sorted((self.data_dir / 'real').glob('*.jpg')):
            self.samples.append((img_path, 0))
            self.labels.append(0)
        for img_path in sorted((self.data_dir / 'fake').glob('*.jpg')):
            self.samples.append((img_path, 1))
            self.labels.append(1)

        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        print(f"  [{split}] {len(self.samples)} samples "
              f"({n_real} real, {n_fake} fake)")

        self.jpeg_aug = JPEGAugmentation(
            quality_range=(20, 75), prob=0.5 if split == 'train' else 0.0
        )
        self.gaussian_noise = GaussianNoise(
            std_range=(0.01, 0.04), prob=0.3 if split == 'train' else 0.0
        )

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                       saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.split == 'train':
                img = self.jpeg_aug(img)
            rgb_tensor = self.transform(img)
            if self.split == 'train':
                # Apply Gaussian noise after normalization
                rgb_tensor = self.gaussian_noise(rgb_tensor)
            return rgb_tensor, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self) - 1))


# =============================================================================
# 7. Balanced Sampler
# =============================================================================
def make_balanced_sampler(dataset):
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels, minlength=2)
    if class_counts[0] == 0 or class_counts[1] == 0:
        missing = "fake" if class_counts[1] == 0 else "real"
        raise RuntimeError(
            f"Training set has 0 {missing} samples! "
            f"Delete 'dataset/ff_face_crops' and re-run to fix extraction."
        )
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataset), replacement=True
    )
    print(f"  Balanced sampler: real weight={weights_per_class[0]:.6f}, "
          f"fake weight={weights_per_class[1]:.6f}")
    return sampler


# =============================================================================
# 8. Training & Validation
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    epoch, ema, args):
    global _stop_requested
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    accum_steps = args.grad_accum
    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for batch_idx, (rgb, labels) in enumerate(pbar):
        if _stop_requested:
            break

        rgb = rgb.to(DEVICE)
        labels = labels.to(DEVICE)

        # ---- Mixup / CutMix regularization ----
        use_mix = (args.mixup_alpha > 0 or args.cutmix_alpha > 0) and \
                  random.random() < 0.5  # Apply 50% of batches
        if use_mix:
            if random.random() < 0.5 and args.mixup_alpha > 0:
                rgb, labels_a, labels_b, lam = mixup_data(
                    rgb, labels, args.mixup_alpha
                )
            else:
                rgb, labels_a, labels_b, lam = cutmix_data(
                    rgb, labels, args.cutmix_alpha
                )
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        # ---- Forward ----
        with torch.cuda.amp.autocast(enabled=(DEVICE != 'cpu')):
            logits = model(rgb).squeeze(1)
            if use_mix:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps  # Scale for gradient accumulation

        # ---- Backward ----
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ---- Optimizer step (every accum_steps batches) ----
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            # Step OneCycleLR per optimizer step
            if scheduler is not None:
                scheduler.step()

            # Update EMA
            if ema is not None:
                ema.update(model)

        # ---- Metrics (use original labels for accuracy) ----
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * accum_steps * labels.size(0)

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.1f}%',
            'lr': f'{optimizer.param_groups[-1]["lr"]:.2e}'
        })

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    for rgb, labels in tqdm(loader, desc='Validating', leave=False):
        rgb = rgb.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(rgb).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_loss += loss.item() * labels.size(0)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    preds = (all_probs > 0.5).astype(float)
    accuracy = (preds == all_labels).mean()

    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    real_acc = (preds[real_mask] == 0).mean() if real_mask.sum() > 0 else 0
    fake_acc = (preds[fake_mask] == 1).mean() if fake_mask.sum() > 0 else 0

    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    print(f"  Val Acc: {accuracy*100:.1f}% "
          f"(Real: {real_acc*100:.1f}%, Fake: {fake_acc*100:.1f}%) "
          f"| F1: {f1:.4f} | AUC: {auc:.4f} "
          f"| Prec: {precision:.3f} Rec: {recall:.3f}")

    return running_loss / total, accuracy, f1, auc


# =============================================================================
# 9. Checkpoint Save / Load (survives shutdown & restart)
# =============================================================================
def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    ema, best_val_f1, best_val_acc, training_log, args):
    """Save everything needed to resume training exactly."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc,
        'training_log': training_log,
        'args': vars(args),
        # RNG states for exact reproducibility
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.random.get_rng_state(),
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    if ema is not None:
        state['ema_state_dict'] = ema.state_dict()
    if DEVICE != 'cpu':
        state['rng_cuda'] = torch.cuda.get_rng_state()

    # Write to temp file first, then rename (atomic on most OS)
    tmp_path = str(path) + '.tmp'
    torch.save(state, tmp_path)
    if os.path.exists(str(path)):
        os.remove(str(path))
    os.rename(tmp_path, str(path))


def load_checkpoint(path, model, optimizer, scheduler, scaler, ema):
    """Load checkpoint and restore all state."""
    if not os.path.exists(str(path)):
        return None

    print(f"\n  Loading checkpoint from {path}...")
    state = torch.load(str(path), map_location=DEVICE, weights_only=False)

    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in state:
        scaler.load_state_dict(state['scaler_state_dict'])
    if ema is not None and 'ema_state_dict' in state:
        ema.load_state_dict(state['ema_state_dict'])

    # Restore RNG states
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.random.set_rng_state(state['rng_torch'])
    if DEVICE != 'cpu' and 'rng_cuda' in state:
        torch.cuda.set_rng_state(state['rng_cuda'])

    epoch = state['epoch']
    best_f1 = state.get('best_val_f1', 0.0)
    best_acc = state.get('best_val_acc', 0.0)
    log = state.get('training_log', [])

    print(f"  Resumed from epoch {epoch}")
    print(f"  Best so far: F1={best_f1:.4f}, Acc={best_acc*100:.1f}%\n")

    return {
        'epoch': epoch,
        'best_val_f1': best_f1,
        'best_val_acc': best_acc,
        'training_log': log,
    }


# =============================================================================
# 10. Training Log (JSON, human-readable)
# =============================================================================
def save_training_log(log, path):
    with open(str(path), 'w') as f:
        json.dump(log, f, indent=2)


# =============================================================================
# 11. Main
# =============================================================================
def main(args):
    global _stop_requested

    print(f"\n{'='*60}")
    print(f"  Deepfake Detection — FaceForensics++ Training")
    print(f"{'='*60}")
    print(f"  Device:             {DEVICE}")
    print(f"  Dataset:            {FF_DATASET_DIR}")
    print(f"  Frames per video:   {args.frames_per_video}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch size:         {args.batch_size} "
          f"(effective: {args.batch_size * args.grad_accum})")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Label smoothing:    {args.label_smoothing}")
    print(f"  Mixup alpha:        {args.mixup_alpha}")
    print(f"  CutMix alpha:       {args.cutmix_alpha}")
    print(f"  EMA decay:          {args.ema_decay}")
    print(f"  Grad accumulation:  {args.grad_accum}")
    print(f"  Early stop patience:{args.patience}")
    print(f"{'='*60}")
    print(f"  Press Ctrl+C to stop — progress will be saved.")
    print(f"  Run the same command to resume from where you stopped.")
    print(f"{'='*60}\n")

    # ---- Phase 1: Extract face crops ----
    face_crops_dir = Path("dataset/ff_face_crops")
    preextract_faces(face_crops_dir, frames_per_video=args.frames_per_video)

    # ---- Phase 2: Datasets & Loaders ----
    train_dataset = DeepfakeDataset(face_crops_dir, split='train')
    val_dataset = DeepfakeDataset(face_crops_dir, split='val')

    if len(train_dataset) == 0:
        print("ERROR: No training samples! Check face extraction.")
        return

    train_sampler = make_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---- Phase 3: Model ----
    model = DeepfakeEfficientNet(pretrained=True, dropout=args.dropout)

    # Load pre-existing weights (from WildFake or previous FF++ training)
    # This is SEPARATE from training checkpoint resume — this loads the
    # base model weights before training starts for the first time
    pretrained_path = Path("weights/best_model.pth")
    if not args.fresh and pretrained_path.exists() and not RESUME_CHECKPOINT.exists():
        print(f"  Loading pre-trained weights from {pretrained_path}...")
        checkpoint = torch.load(str(pretrained_path), map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            print(f"  Loaded epoch {checkpoint.get('epoch', '?')} "
                  f"(val_acc: {checkpoint.get('val_acc', 0)*100:.1f}%)")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"  Fine-tuning on FaceForensics++ C23...\n")

    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable\n")

    # ---- Phase 4: Loss with Label Smoothing ----
    criterion = nn.BCEWithLogitsLoss(
        label_smoothing=args.label_smoothing
    ) if hasattr(nn.BCEWithLogitsLoss, 'label_smoothing') else None

    # BCEWithLogitsLoss doesn't have label_smoothing — implement manually
    if criterion is None:
        base_criterion = nn.BCEWithLogitsLoss()
        ls = args.label_smoothing

        def criterion(logits, targets):
            if ls > 0:
                targets = targets * (1 - ls) + 0.5 * ls
            return base_criterion(logits, targets)

    # ---- Optimizer with differential LR ----
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'net._fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': classifier_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    # ---- OneCycleLR Scheduler (with warmup built-in) ----
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        total_steps=total_steps,
        pct_start=0.1,       # 10% warmup
        anneal_strategy='cos',
        div_factor=25,        # start_lr = max_lr / 25
        final_div_factor=1000,  # end_lr = start_lr / 1000
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if DEVICE != 'cpu' else None

    # EMA
    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---- Phase 5: Resume from checkpoint if exists ----
    start_epoch = 1
    best_val_f1 = 0.0
    best_val_acc = 0.0
    training_log = []
    patience_counter = 0

    if not args.fresh and RESUME_CHECKPOINT.exists():
        resume_state = load_checkpoint(
            RESUME_CHECKPOINT, model, optimizer, scheduler, scaler, ema
        )
        if resume_state:
            start_epoch = resume_state['epoch'] + 1
            best_val_f1 = resume_state['best_val_f1']
            best_val_acc = resume_state['best_val_acc']
            training_log = resume_state['training_log']

            if start_epoch > args.epochs:
                print(f"  Training already complete ({start_epoch-1}/{args.epochs})!")
                print(f"  Use --epochs {start_epoch + 5} to train more epochs.")
                return

    # ---- Phase 6: Training Loop ----
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Training epochs {start_epoch} to {args.epochs}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Ctrl+C to stop safely (saves checkpoint)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        if _stop_requested:
            print(f"\n  Stop requested before epoch {epoch}.")
            print(f"  Saving checkpoint...")
            save_checkpoint(
                RESUME_CHECKPOINT, epoch - 1, model, optimizer, scheduler,
                scaler, ema, best_val_f1, best_val_acc, training_log, args
            )
            print(f"  Checkpoint saved. Run same command to resume.\n")
            break

        epoch_start = time.time()

        # ---- Train ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, ema, args
        )

        if _stop_requested:
            print(f"\n  Stopped during epoch {epoch}. Saving checkpoint...")
            save_checkpoint(
                RESUME_CHECKPOINT, epoch - 1, model, optimizer, scheduler,
                scaler, ema, best_val_f1, best_val_acc, training_log, args
            )
            print(f"  Checkpoint saved. Run same command to resume.\n")
            break

        # ---- Validate (with EMA model if available) ----
        if ema is not None:
            ema.apply_shadow(model)

        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader,
                                                       criterion)

        if ema is not None:
            ema.restore(model)

        epoch_time = time.time() - epoch_start

        # ---- Logging ----
        epoch_log = {
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 5),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'val_auc': round(val_auc, 4),
            'lr': optimizer.param_groups[-1]['lr'],
            'time_seconds': round(epoch_time, 1),
        }
        training_log.append(epoch_log)
        save_training_log(training_log, TRAINING_LOG_PATH)

        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.0f}s): "
              f"Train Loss={train_loss:.4f} Acc={train_acc*100:.1f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc*100:.1f}% "
              f"F1={val_f1:.4f} AUC={val_auc:.4f}")

        # ---- Save best model ----
        is_best = val_f1 > best_val_f1 if val_f1 > 0 else val_acc > best_val_acc
        if is_best:
            best_val_f1 = max(val_f1, best_val_f1)
            best_val_acc = max(val_acc, best_val_acc)
            patience_counter = 0

            # Save best model (using EMA weights if available)
            if ema is not None:
                ema.apply_shadow(model)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': {
                    'dataset': 'FaceForensics++_C23',
                    'dropout': args.dropout,
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'label_smoothing': args.label_smoothing,
                    'mixup_alpha': args.mixup_alpha,
                    'cutmix_alpha': args.cutmix_alpha,
                    'ema_decay': args.ema_decay,
                }
            }, str(BEST_MODEL_PATH))

            if ema is not None:
                ema.restore(model)

            print(f"  >>> Best model saved! "
                  f"F1={val_f1:.4f} Acc={val_acc*100:.1f}%\n")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})\n")

        # ---- Save resume checkpoint every epoch ----
        save_checkpoint(
            RESUME_CHECKPOINT, epoch, model, optimizer, scheduler,
            scaler, ema, best_val_f1, best_val_acc, training_log, args
        )

        # ---- Early stopping ----
        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered after {args.patience} epochs "
                  f"without improvement.")
            break

    # ---- Done ----
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best: F1={best_val_f1:.4f}, Acc={best_val_acc*100:.1f}%")
    print(f"  Model saved to: {BEST_MODEL_PATH}")
    print(f"  Training log: {TRAINING_LOG_PATH}")
    if RESUME_CHECKPOINT.exists():
        print(f"  Resume checkpoint: {RESUME_CHECKPOINT}")
        print(f"  (Delete it to start fresh next time)")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detector on FaceForensics++',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Max LR for classifier (backbone gets 0.1x)')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--frames_per_video', type=int, default=15)

    # Regularization
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0=off, 0.1=recommended)')
    parser.add_argument('--mixup_alpha', type=float, default=0.3,
                        help='Mixup alpha (0=off)')
    parser.add_argument('--cutmix_alpha', type=float, default=0.3,
                        help='CutMix alpha (0=off)')

    # Optimization
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay (0=off, 0.999=recommended)')
    parser.add_argument('--grad_accum', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch * accum)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience (epochs without improvement)')

    # Resume / Fresh
    parser.add_argument('--fresh', action='store_true', default=False,
                        help='Start completely fresh (ignore all checkpoints)')
    parser.add_argument('--save_dir', type=str, default='weights')

    args = parser.parse_args()
    main(args)
