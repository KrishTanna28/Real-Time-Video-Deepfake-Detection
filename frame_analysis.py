"""
Frame-Level Forensic Analysis Module
Detects deepfake artifacts in video frames WITHOUT requiring face detection.
Uses multiple lightweight forensic signals for comprehensive analysis.

Techniques used:
1. Frequency Domain Analysis (FFT) - detects GAN spectral artifacts
2. Noise Pattern Consistency - detects inconsistent sensor noise
3. Error Level Analysis (ELA) - detects compression manipulation
4. Edge Coherence Analysis - detects compositing artifacts
5. Color Space Analysis - detects unnatural color distributions
6. Temporal Consistency - detects frame-to-frame anomalies

All methods are optimized for real-time performance (~10-15ms total per frame).
"""

import cv2
import numpy as np
from collections import deque


class FrameForensicAnalyzer:
    """
    Analyzes video frames for deepfake artifacts using forensic techniques.
    Works on ANY video content - with or without visible faces.
    """

    def __init__(self, analysis_size=(256, 256)):
        """
        Args:
            analysis_size: Resize frames to this size for analysis (speed vs accuracy).
                          (256, 256) is a good balance for real-time.
        """
        self.analysis_size = analysis_size
        self.prev_frame_gray = None
        self.temporal_diffs = deque(maxlen=30)
        self.frame_count = 0

        # Pre-compute frequency masks for reuse
        h, w = analysis_size
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        self._dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        self._inner_r = min(h, w) // 8
        self._mid_r = min(h, w) // 4
        self._outer_r = min(h, w) // 2

        # Weights for combining signals (tuned for general deepfake detection)
        self.weights = {
            'frequency': 0.25,
            'noise': 0.20,
            'ela': 0.20,
            'edge': 0.15,
            'color': 0.10,
            'temporal': 0.10,
        }

    def analyze(self, frame):
        """
        Run all forensic analyses on a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict with individual scores, overall fake_probability, and analysis metadata
        """
        self.frame_count += 1

        # Resize for consistent and fast analysis
        resized = cv2.resize(frame, self.analysis_size, interpolation=cv2.INTER_LINEAR)

        scores = {}

        # 1. Frequency domain analysis (GAN spectral fingerprints)
        scores['frequency'] = self._analyze_frequency(resized)

        # 2. Noise pattern consistency
        scores['noise'] = self._analyze_noise(resized)

        # 3. Error Level Analysis (compression artifact detection)
        scores['ela'] = self._analyze_ela(resized)

        # 4. Edge coherence (compositing artifacts)
        scores['edge'] = self._analyze_edges(resized)

        # 5. Color space analysis
        scores['color'] = self._analyze_color(resized)

        # 6. Temporal consistency (frame-to-frame)
        scores['temporal'] = self._analyze_temporal(resized)

        # Combined weighted score
        combined = sum(scores[k] * self.weights[k] for k in self.weights)

        return {
            'scores': scores,
            'fake_probability': float(np.clip(combined, 0.0, 1.0)),
            'analysis_type': 'frame_forensic',
            'frame_number': self.frame_count,
        }

    def analyze_fast(self, frame):
        """
        Lightweight analysis using only the fastest signals.
        Use this for every frame and full analyze() every Nth frame.

        Takes ~3-5ms vs ~10-15ms for full analyze().
        """
        self.frame_count += 1
        resized = cv2.resize(frame, self.analysis_size, interpolation=cv2.INTER_LINEAR)

        scores = {}
        scores['frequency'] = self._analyze_frequency(resized)
        scores['temporal'] = self._analyze_temporal(resized)
        scores['edge'] = self._analyze_edges(resized)

        fast_weights = {'frequency': 0.45, 'temporal': 0.25, 'edge': 0.30}
        combined = sum(scores[k] * fast_weights[k] for k in fast_weights)

        return {
            'scores': scores,
            'fake_probability': float(np.clip(combined, 0.0, 1.0)),
            'analysis_type': 'frame_forensic_fast',
            'frame_number': self.frame_count,
        }

    def _analyze_frequency(self, frame):
        """
        Detect GAN artifacts via FFT analysis.
        GAN-generated content often has:
        - Abnormally low high-frequency content (over-smoothing)
        - Periodic checkerboard artifacts in mid frequencies
        - Unusual spectral energy distribution
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        dist = self._dist_from_center

        # Extract frequency band energies (using precomputed masks)
        low_mask = dist <= self._inner_r
        mid_mask = (dist > self._inner_r) & (dist <= self._mid_r)
        high_mask = (dist > self._mid_r) & (dist <= self._outer_r)

        low_freq = magnitude[low_mask].mean() if np.any(low_mask) else 0
        mid_freq = magnitude[mid_mask].mean() if np.any(mid_mask) else 0
        high_freq = magnitude[high_mask].mean() if np.any(high_mask) else 0

        total = low_freq + mid_freq + high_freq + 1e-10

        high_freq_ratio = high_freq / total
        mid_freq_ratio = mid_freq / total

        score = 0.0

        # Low high-frequency ratio suggests generated/over-smoothed content
        if high_freq_ratio < 0.18:
            score += 0.4
        elif high_freq_ratio < 0.22:
            score += 0.2

        # Check for periodic artifacts in mid frequencies (GAN checkerboard)
        mid_values = magnitude[mid_mask]
        if len(mid_values) > 10:
            mid_cv = np.std(mid_values) / (np.mean(mid_values) + 1e-10)
            if mid_cv > 0.6:
                score += 0.25
            elif mid_cv > 0.45:
                score += 0.1

        # Abnormal mid-to-high frequency ratio
        if mid_freq_ratio > 0.45 and high_freq_ratio < 0.2:
            score += 0.15

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_noise(self, frame):
        """
        Check noise pattern consistency across image blocks.
        Real camera sensors produce spatially uniform noise.
        Manipulated/generated content has inconsistent noise patterns.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract noise residual using high-pass filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        # Divide into blocks and measure noise variance per block
        block_size = 32
        h, w = noise.shape
        block_stds = []

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = noise[i:i + block_size, j:j + block_size]
                block_stds.append(np.std(block))

        if len(block_stds) < 4:
            return 0.0

        block_stds = np.array(block_stds)
        mean_noise = np.mean(block_stds)
        noise_cv = np.std(block_stds) / (mean_noise + 1e-10)

        score = 0.0

        # Very inconsistent noise = likely manipulated
        if noise_cv > 0.7:
            score += 0.5
        elif noise_cv > 0.5:
            score += 0.25

        # Suspiciously smooth (denoised or generated)
        if mean_noise < 1.0:
            score += 0.3
        elif mean_noise < 2.0:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_ela(self, frame):
        """
        Error Level Analysis - detect compression inconsistencies.
        Re-compresses the frame and checks if different regions show
        different error levels (indicates editing/splicing).
        """
        # Re-compress at known quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if recompressed is None:
            return 0.0

        # Compute absolute difference
        diff = cv2.absdiff(frame, recompressed)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Analyze difference distribution across blocks
        block_size = 32
        h, w = diff_gray.shape
        block_means = []

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = diff_gray[i:i + block_size, j:j + block_size]
                block_means.append(np.mean(block))

        if len(block_means) < 4:
            return 0.0

        block_means = np.array(block_means)
        ela_mean = np.mean(block_means)
        ela_cv = np.std(block_means) / (ela_mean + 1e-10)

        score = 0.0

        # Very inconsistent compression = likely manipulated
        if ela_cv > 0.9:
            score += 0.5
        elif ela_cv > 0.6:
            score += 0.2

        # Very high mean error = possibly generated content
        if ela_mean > 15:
            score += 0.2
        elif ela_mean > 10:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_edges(self, frame):
        """
        Analyze edge coherence for compositing/generation artifacts.
        Generated content often has:
        - Unusually low edge density (over-smoothing)
        - Low Laplacian variance (lack of fine detail)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Laplacian variance (sharpness/detail measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)

        score = 0.0

        # Very low edge density = over-smoothed/generated
        if edge_density < 0.02:
            score += 0.35
        elif edge_density < 0.04:
            score += 0.15

        # Very low Laplacian variance = blurry/smooth
        if lap_var < 50:
            score += 0.3
        elif lap_var < 100:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_color(self, frame):
        """
        Analyze color distribution for anomalies.
        GAN-generated content may have:
        - Unusually uniform saturation/brightness
        - Color banding from limited color palette
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1].astype(np.float32)
        sat_std = np.std(saturation)

        value = hsv[:, :, 2].astype(np.float32)
        val_std = np.std(value)

        score = 0.0

        # Unusually uniform saturation
        if sat_std < 15:
            score += 0.3
        elif sat_std < 25:
            score += 0.1

        # Unusually uniform brightness
        if val_std < 15:
            score += 0.25
        elif val_std < 25:
            score += 0.1

        # Check for color banding / limited palette
        unique_hues = len(np.unique(hsv[:, :, 0]))
        if unique_hues < 30:
            score += 0.25
        elif unique_hues < 50:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_temporal(self, frame):
        """
        Analyze temporal consistency between consecutive frames.
        Deepfakes may exhibit:
        - Erratic frame-to-frame differences (unstable generation)
        - Near-zero differences (frozen/looped content)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0

        # Frame difference
        diff = cv2.absdiff(gray, self.prev_frame_gray)
        mean_diff = np.mean(diff)
        self.temporal_diffs.append(mean_diff)
        self.prev_frame_gray = gray

        if len(self.temporal_diffs) < 5:
            return 0.0

        diffs = np.array(self.temporal_diffs)
        mean_diffs = np.mean(diffs)
        temporal_cv = np.std(diffs) / (mean_diffs + 1e-10)

        score = 0.0

        # Very erratic frame differences = unstable generation
        if temporal_cv > 1.5:
            score += 0.4
        elif temporal_cv > 1.0:
            score += 0.2

        # Near-zero differences (looping/frozen) after initial frames
        if mean_diff < 0.3 and self.frame_count > 10:
            score += 0.3
        elif mean_diff < 0.8 and self.frame_count > 10:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def reset(self):
        """Reset all analyzer state for a new session."""
        self.prev_frame_gray = None
        self.temporal_diffs.clear()
        self.frame_count = 0
