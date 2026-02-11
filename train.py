import argparse
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import DeepfakeMobileNetV3, compute_frequency_features

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 1. JPEG Augmentation
# =============================================================================
class JPEGAugmentation:
    """Apply random JPEG compression to simulate real-world artifacts.
    
    This is critical because:
    - Real deepfake videos are always re-compressed (YouTube, social media)
    - Training without JPEG augmentation causes model to learn compression artifacts
      (not manipulation artifacts), leading to false negatives on re-compressed fakes
    """
    
    def __init__(self, quality_range=(30, 70), prob=0.5):
        """
        Args:
            quality_range: (min, max) JPEG quality. 30-70 covers social media range.
            prob: probability of applying augmentation per sample
        """
        self.quality_range = quality_range
        self.prob = prob
    
    def __call__(self, image):
        """
        Args:
            image: PIL Image or numpy array (uint8 BGR)
        Returns:
            Augmented image (same type as input)
        """
        if random.random() > self.prob:
            return image
        
        # Convert PIL to numpy if needed
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image
        
        # Apply JPEG compression
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img_bgr, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        if is_pil:
            return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        return decoded


class DoubleJPEGAugmentation:
    """Apply JPEG compression twice — simulates upload+re-download cycle."""
    
    def __init__(self, quality_range_1=(50, 85), quality_range_2=(30, 70), prob=0.3):
        self.aug1 = JPEGAugmentation(quality_range_1, prob=1.0)
        self.aug2 = JPEGAugmentation(quality_range_2, prob=1.0)
        self.prob = prob
    
    def __call__(self, image):
        if random.random() > self.prob:
            return image
        return self.aug2(self.aug1(image))


# =============================================================================
# 2. Dataset with Frequency Features
# =============================================================================
class DeepfakeDataset(Dataset):
    """Dataset that returns RGB images + precomputed frequency features.
    
    Each sample returns:
        rgb_tensor: (3, 224, 224) - normalized RGB for MobileNetV3
        freq_tensor: (2, 224, 224) - FFT magnitude + DCT coefficients
        label: 0 (real) or 1 (fake)
    """
    
    def __init__(self, data_dir, split='train', image_size=224):
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.is_train = (split == 'train')
        
        # Collect image paths
        self.samples = []  # list of (path, label)
        
        real_dir = self.data_dir / 'real'
        fake_dir = self.data_dir / 'fake'
        
        if real_dir.exists():
            for img_path in self._get_images(real_dir):
                self.samples.append((str(img_path), 0))
        
        if fake_dir.exists():
            for img_path in self._get_images(fake_dir):
                self.samples.append((str(img_path), 1))
        
        # Shuffle
        random.shuffle(self.samples)
        
        print(f"[{split}] Loaded {len(self.samples)} images "
              f"(real: {sum(1 for _, l in self.samples if l == 0)}, "
              f"fake: {sum(1 for _, l in self.samples if l == 1)})")
        
        # JPEG augmentation (training only)
        self.jpeg_aug = JPEGAugmentation(quality_range=(30, 70), prob=0.5) if self.is_train else None
        self.double_jpeg_aug = DoubleJPEGAugmentation(prob=0.2) if self.is_train else None
        
        # Standard RGB augmentations (training only)
        if self.is_train:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    
    def _get_images(self, directory):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = []
        for ext in exts:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        return sorted(images)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback: random other sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        # Apply JPEG augmentation BEFORE extracting frequency features
        # (so the model learns to detect fakes even through compression)
        if self.is_train:
            image = self.jpeg_aug(image)
            image = self.double_jpeg_aug(image)
        
        # Compute frequency features from the (potentially JPEG-augmented) image
        freq_features = compute_frequency_features(image, size=self.image_size)
        freq_tensor = torch.from_numpy(freq_features)  # (2, 224, 224)
        
        # Convert to PIL for torchvision transforms
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Apply RGB transforms
        rgb_tensor = self.rgb_transform(pil_image)  # (3, 224, 224)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return rgb_tensor, freq_tensor, label_tensor


# =============================================================================
# 3. Contrastive Loss
# =============================================================================
class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon).
    
    Pulls together embeddings of the same class (real-real, fake-fake)
    and pushes apart embeddings of different classes (real-fake).
    
    This creates a much better feature space than BCE alone:
    - Real samples cluster tightly
    - Fake samples cluster tightly  
    - Large margin between clusters
    - Model becomes more confident and calibrated
    
    Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, projections, labels):
        """
        Args:
            projections: (B, D) L2-normalized embeddings from projection head
            labels: (B,) binary labels (0=real, 1=fake)
            
        Returns:
            scalar loss
        """
        device = projections.device
        batch_size = projections.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        labels = labels.contiguous().view(-1, 1)
        
        # Mask: 1 where labels match (same class), 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Similarity matrix
        similarity = torch.matmul(projections, projections.T) / self.temperature
        
        # Remove self-similarity (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Compute log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # Mean of log-prob over positive pairs
        mask_pos_pairs = mask.sum(dim=1)
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_pos_pairs
        
        loss = -mean_log_prob.mean()
        return loss


# =============================================================================
# 4. Combined Loss
# =============================================================================
class CombinedLoss(nn.Module):
    """BCE + Supervised Contrastive Loss.
    
    The contrastive loss improves feature quality while BCE provides
    direct classification signal. Together they outperform either alone.
    """
    
    def __init__(self, bce_weight=0.6, contrastive_weight=0.4, temperature=0.07):
        super().__init__()
        self.bce_weight = bce_weight
        self.contrastive_weight = contrastive_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
    
    def forward(self, logits, projections, labels):
        """
        Args:
            logits: (B, 1) raw classifier output
            projections: (B, 128) L2-normalized projection embeddings
            labels: (B,) binary labels
        """
        bce = self.bce_loss(logits.squeeze(), labels)
        contrastive = self.contrastive_loss(projections, labels)
        
        total = self.bce_weight * bce + self.contrastive_weight * contrastive
        return total, bce, contrastive


# =============================================================================
# 5. Training Loop
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_con = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for rgb, freq, labels in pbar:
        rgb = rgb.to(DEVICE)
        freq = freq.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(DEVICE != 'cpu')):
            logits, projections = model.forward_with_projection(rgb, freq)
            loss, bce, con = criterion(logits, projections, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accuracy
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        running_loss += loss.item()
        running_bce += bce.item()
        running_con += con.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bce': f'{bce.item():.4f}',
            'con': f'{con.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })
    
    n = len(loader)
    return running_loss / n, running_bce / n, running_con / n, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    for rgb, freq, labels in tqdm(loader, desc='Validating'):
        rgb = rgb.to(DEVICE)
        freq = freq.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits, projections = model.forward_with_projection(rgb, freq)
        loss, _, _ = criterion(logits, projections, labels)
        
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    n = len(loader)
    acc = correct / total
    avg_loss = running_loss / n
    
    # Compute AUC if sklearn available
    auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except:
        pass
    
    return avg_loss, acc, auc


def main(args):
    print("=" * 60)
    print("  Enhanced Deepfake Detection Training")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  BCE weight: {args.bce_weight}")
    print(f"  Contrastive weight: {args.contrastive_weight}")
    print(f"  JPEG augmentation: ON (quality {30}-{70})")
    print(f"  Frequency input: FFT + DCT")
    print("=" * 60)
    
    # Datasets
    train_dataset = DeepfakeDataset(args.data_dir, split='train', image_size=224)
    val_dataset = DeepfakeDataset(args.data_dir, split='val', image_size=224)
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        print(f"Expected structure: {args.data_dir}/train/real/ and {args.data_dir}/train/fake/")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for contrastive loss (need pairs)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Model
    model = DeepfakeMobileNetV3(pretrained=True, dropout=args.dropout)
    
    # Optionally load existing weights to fine-tune
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(DEVICE)
    
    # Loss
    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=0.07,
    )
    
    # Optimizer — differential learning rates
    # (Lower LR for pretrained backbone, higher for new freq branch + classifier)
    backbone_params = list(model.backbone.parameters())
    new_params = list(model.freq_branch.parameters()) + \
                 list(model.classifier.parameters()) + \
                 list(model.projection_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},   # Pretrained: lower LR
        {'params': new_params, 'lr': args.lr},                # New layers: full LR
    ], weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs // 3, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if DEVICE != 'cpu' else None
    
    # Training
    best_val_auc = 0.0
    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_bce, train_con, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        
        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        
        scheduler.step()
        
        print(f"\nTrain — Loss: {train_loss:.4f} | BCE: {train_bce:.4f} | "
              f"Contrastive: {train_con:.4f} | Acc: {train_acc*100:.1f}%")
        print(f"Val   — Loss: {val_loss:.4f} | Acc: {val_acc*100:.1f}% | AUC: {val_auc:.4f}")
        
        # Save best model
        is_best = val_auc > best_val_auc if val_auc > 0 else val_acc > best_val_acc
        if is_best:
            best_val_auc = max(val_auc, best_val_auc)
            best_val_acc = max(val_acc, best_val_acc)
            
            save_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
                'train_loss': train_loss,
                'architecture': 'DeepfakeMobileNetV3',
            }, save_path)
            print(f"✓ Best model saved! AUC: {val_auc:.4f}, Acc: {val_acc*100:.1f}%")
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_auc': val_auc,
        }, save_dir / 'latest_checkpoint.pth')
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best validation Acc: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Enhanced Deepfake Detector')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset (must contain train/real, train/fake, val/real, val/fake)')
    parser.add_argument('--save_dir', type=str, default='weights',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for new layers (backbone gets 0.1x)')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--bce_weight', type=float, default=0.6,
                        help='Weight for BCE loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.4,
                        help='Weight for contrastive loss')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    main(args)
