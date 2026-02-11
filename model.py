"""
EfficientNet-B0 for Deepfake Detection.

This architecture EXACTLY matches the trained weights from best_model.pth.
The model uses:
  - self.net = EfficientNet (backbone)
  - net._fc = Sequential classifier (1280 → 512 → 256 → 1)

Architecture:
  RGB frame → EfficientNet-B0 backbone → 1280-dim features → Classifier → binary output (real/fake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from efficientnet_pytorch import EfficientNet


class DeepfakeEfficientNet(nn.Module):
    """EfficientNet-B0 for deepfake detection.
    
    Architecture matches trained weights exactly:
        RGB frame → EfficientNet-B0 (self.net) → 1280-dim → Classifier (net._fc) → binary output
    
    The classifier structure:
        - Dropout(0.5)
        - Linear(1280, 512) + BatchNorm + ReLU
        - Dropout(0.35)
        - Linear(512, 256) + BatchNorm + ReLU
        - Dropout(0.25)
        - Linear(256, 1)
    """
    
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        
        # RGB backbone — EfficientNet-B0 (stored as self.net to match trained weights)
        if pretrained:
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.net = EfficientNet.from_name('efficientnet-b0')
        
        # Backbone feature dim = 1280 for EfficientNet-B0
        in_features = self.net._fc.in_features  # 1280
        
        # Replace classifier with our custom one (matches trained weights layer indices)
        # Indices: 0=Dropout, 1=Linear, 2=BN, 3=ReLU, 4=Dropout, 5=Linear, 6=BN, 7=ReLU, 8=Dropout, 9=Linear
        self.net._fc = nn.Sequential(
            nn.Dropout(dropout),           # index 0
            nn.Linear(in_features, 512),   # index 1: net._fc.1.weight
            nn.BatchNorm1d(512),           # index 2: net._fc.2.weight
            nn.ReLU(),                     # index 3
            nn.Dropout(dropout * 0.7),     # index 4
            nn.Linear(512, 256),           # index 5: net._fc.5.weight
            nn.BatchNorm1d(256),           # index 6: net._fc.6.weight
            nn.ReLU(),                     # index 7
            nn.Dropout(dropout * 0.5),     # index 8
            nn.Linear(256, 1)              # index 9: net._fc.9.weight
        )
    
    def forward(self, rgb_input, freq_input=None):
        """
        Args:
            rgb_input: (B, 3, 224, 224) normalized RGB image
            freq_input: Ignored (kept for API compatibility)
        
        Returns:
            logits: (B, 1) raw logits for binary classification
        """
        return self.net(rgb_input)
    
    def extract_features(self, rgb_input):
        """Extract features from backbone (before classifier).
        
        Args:
            rgb_input: (B, 3, 224, 224) normalized RGB
            
        Returns:
            (B, 1280) feature vector
        """
        # Use EfficientNet's extract_features method
        x = self.net.extract_features(rgb_input)
        # Global average pooling
        x = self.net._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
    def forward_with_projection(self, rgb_input, freq_input=None):
        """Forward pass for training compatibility.
        
        Returns:
            logits: (B, 1)
            projections: None (no projection head in original architecture)
        """
        logits = self.net(rgb_input)
        return logits, None
    
    def get_feature_extractor(self):
        """Get the last conv layer for GradCAM."""
        return self.net._conv_head


def compute_frequency_features(image_bgr_or_rgb, size=224):
    """Compute FFT magnitude + DCT features from an image.
    
    This is kept for compatibility and potential future use.
    
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
