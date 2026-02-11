"""
MobileNetV3-Large + Frequency-Domain Fusion for Deepfake Detection.

Why MobileNetV3-Large over EfficientNet-B0:
  - ~2x faster inference on CPU (optimized for mobile/edge with h-swish + SE blocks)
  - Built into torchvision — no extra pip dependency
  - Same param count (~5.4M) but much faster forward pass
  - Better suited for real-time browser extension use

Architecture:
  RGB frame  → MobileNetV3-Large backbone → 960-dim features
  FFT/DCT    → FrequencyBranch (3-layer CNN) → 128-dim features
  Concatenate → [1088-dim] → Classifier → binary output (real/fake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models


class FrequencyBranch(nn.Module):
    """Lightweight CNN that processes frequency-domain representations (FFT + DCT).
    
    Extracts spectral features from:
    - FFT magnitude spectrum (captures GAN periodic artifacts)
    - DCT coefficients (captures compression/manipulation artifacts)
    
    Output: 128-dim feature vector (~0.2M params, <1ms inference)
    """
    
    def __init__(self):
        super().__init__()
        # Input: 2 channels (FFT magnitude + DCT coefficients), 224x224
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 2, 224, 224) - FFT magnitude + DCT channels
        Returns:
            (B, 128) feature vector
        """
        return self.features(x).flatten(1)


class DeepfakeMobileNetV3(nn.Module):
    """MobileNetV3-Large + Frequency-Domain Fusion for deepfake detection.
    
    Architecture:
        RGB frame → MobileNetV3-Large backbone → 960-dim features
        FFT/DCT   → FrequencyBranch            → 128-dim features
        Concatenate → Classifier head → binary output
    
    Speed: ~15-25ms per frame on CPU (vs ~40-60ms for EfficientNet-B0)
    """
    
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        
        # RGB backbone — MobileNetV3-Large (pretrained on ImageNet)
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.backbone = models.mobilenet_v3_large(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_large(weights=None)
        
        # Backbone feature dim = 960 for MobileNetV3-Large
        self.backbone_features = self.backbone.classifier[0].in_features  # 960
        
        # Remove original classifier — we replace it with our fused classifier
        self.backbone.classifier = nn.Identity()
        
        # Frequency-domain branch
        self.freq_branch = FrequencyBranch()  # outputs 128-dim
        
        # Fused classifier: backbone (960) + frequency (128) = 1088
        fused_dim = self.backbone_features + 128
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
        
        # Projection head for contrastive learning (only used during training)
        self.projection_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def extract_features(self, rgb_input, freq_input):
        """Extract fused features from both branches.
        
        Args:
            rgb_input: (B, 3, 224, 224) normalized RGB
            freq_input: (B, 2, 224, 224) FFT magnitude + DCT
            
        Returns:
            (B, 1088) fused feature vector
        """
        rgb_features = self.backbone(rgb_input)       # (B, 960)
        freq_features = self.freq_branch(freq_input)  # (B, 128)
        fused = torch.cat([rgb_features, freq_features], dim=1)  # (B, 1088)
        return fused
    
    def forward(self, rgb_input, freq_input=None):
        """
        Args:
            rgb_input: (B, 3, 224, 224) normalized RGB image
            freq_input: (B, 2, 224, 224) frequency-domain input (FFT + DCT).
                       If None, generates it from rgb_input (slower but works for inference).
        
        Returns:
            logits: (B, 1) raw logits for binary classification
        """
        if freq_input is None:
            freq_input = self._rgb_to_freq(rgb_input)
        
        fused = self.extract_features(rgb_input, freq_input)
        logits = self.classifier(fused)
        return logits
    
    def forward_with_projection(self, rgb_input, freq_input):
        """Forward pass that also returns projection for contrastive loss.
        Used only during training.
        
        Returns:
            logits: (B, 1)
            projections: (B, 128) L2-normalized embeddings
        """
        fused = self.extract_features(rgb_input, freq_input)
        logits = self.classifier(fused)
        proj = self.projection_head(fused)
        proj = F.normalize(proj, p=2, dim=1)
        return logits, proj
    
    def get_feature_extractor(self):
        """Get the last conv layer for GradCAM (on the RGB backbone)."""
        # MobileNetV3's last conv block is backbone.features[-1]
        return self.backbone.features[-1]
    
    @torch.no_grad()
    def _rgb_to_freq(self, rgb_tensor):
        """Convert normalized RGB tensor to frequency-domain input on-the-fly.
        
        This is used during inference when freq_input is not pre-computed.
        """
        device = rgb_tensor.device
        B = rgb_tensor.shape[0]
        freq_batch = torch.zeros(B, 2, 224, 224, device=device)
        
        for i in range(B):
            # Denormalize from ImageNet normalization
            img = rgb_tensor[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            freq = compute_frequency_features(img)  # (2, H, W)
            freq_batch[i] = torch.from_numpy(freq).to(device)
        
        return freq_batch


def compute_frequency_features(image_bgr_or_rgb, size=224):
    """Compute FFT magnitude + DCT features from an image.
    
    This is the canonical function used by both training and inference.
    
    Args:
        image_bgr_or_rgb: uint8 image, shape (H, W, 3)
        size: output spatial size
    
    Returns:
        numpy array of shape (2, size, size) — channel 0 = FFT magnitude, channel 1 = DCT
    """
    # Convert to grayscale
    if len(image_bgr_or_rgb.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_rgb
    
    gray = cv2.resize(gray, (size, size)).astype(np.float32)
    
    # --- Channel 0: FFT magnitude spectrum ---
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    # Normalize to [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 1e-6:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)
    
    # --- Channel 1: DCT coefficients ---
    dct = cv2.dct(gray / 255.0)
    dct_abs = np.abs(dct)
    # Log scale for better dynamic range
    dct_log = np.log1p(dct_abs)
    dct_min, dct_max = dct_log.min(), dct_log.max()
    if dct_max - dct_min > 1e-6:
        dct_log = (dct_log - dct_min) / (dct_max - dct_min)
    else:
        dct_log = np.zeros_like(dct_log)
    
    # Stack: (2, H, W)
    features = np.stack([magnitude, dct_log], axis=0).astype(np.float32)
    return features
