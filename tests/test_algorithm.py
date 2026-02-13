"""
╔══════════════════════════════════════════════════════════════════╗
║   ALGORITHM TESTS — Real-Time Video Deepfake Detection         ║
║   Validates detection logic: threshold behaviour, voting       ║
║   accuracy, forensic signal correctness, frequency features.   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import pytest
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────── Helpers ───────────────────────

def make_smooth_image(size=(256, 256)):
    """Create an artificially smooth (GAN-like) image."""
    img = np.full((*size, 3), 128, dtype=np.uint8)
    img = cv2.GaussianBlur(img, (31, 31), 10)
    return img


def make_noisy_image(size=(256, 256)):
    """Create a noisy (camera-like) image."""
    img = np.random.randint(60, 200, (*size, 3), dtype=np.uint8)
    return img


def make_gradient_image(size=(256, 256)):
    """Create a smooth gradient image with clear edges."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img[i, :, :] = int(255 * i / h)
    # Add strong edges
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), 3)
    cv2.circle(img, (128, 128), 60, (0, 255, 0), 3)
    return img


# ════════════════════════════════════════════════════════
#  7. DETECTION THRESHOLD CORRECTNESS
# ════════════════════════════════════════════════════════

class TestThresholdBehaviour:
    """Verify that the configurable threshold correctly classifies frames."""

    @pytest.fixture
    def tracker_low(self):
        from deepfake_detection import TemporalTracker
        return TemporalTracker(window_size=60, voting_window=5, detection_threshold=0.5)

    @pytest.fixture
    def tracker_high(self):
        from deepfake_detection import TemporalTracker
        return TemporalTracker(window_size=60, voting_window=5, detection_threshold=0.75)

    def test_prob_above_threshold_classified_fake(self, tracker_low):
        """Prob=0.6 with threshold=0.5 should classify as FAKE."""
        for _ in range(5):
            tracker_low.update(0.6)
        assert tracker_low.get_confidence_level() == "FAKE"

    def test_prob_below_threshold_classified_real(self, tracker_high):
        """Prob=0.6 with threshold=0.75 should classify as REAL."""
        for _ in range(5):
            tracker_high.update(0.6)
        assert tracker_high.get_confidence_level() == "REAL"

    def test_borderline_exact_threshold(self):
        """Prob exactly at threshold should classify as REAL (not >)."""
        from deepfake_detection import TemporalTracker
        t = TemporalTracker(voting_window=5, detection_threshold=0.75)
        for _ in range(5):
            t.update(0.75)  # exactly at threshold — NOT > 0.75
        assert t.get_confidence_level() == "REAL"

    def test_different_thresholds_opposite_results(self, tracker_low, tracker_high):
        """Same probability should give opposite verdicts with different thresholds."""
        for _ in range(5):
            tracker_low.update(0.65)
            tracker_high.update(0.65)
        assert tracker_low.get_confidence_level() == "FAKE"
        assert tracker_high.get_confidence_level() == "REAL"


# ════════════════════════════════════════════════════════
#  8. VOTING ACCURACY
# ════════════════════════════════════════════════════════

class TestVotingAccuracy:
    """Validate majority-vote logic with various vote distributions."""

    @pytest.fixture
    def tracker(self):
        from deepfake_detection import TemporalTracker
        return TemporalTracker(voting_window=10, detection_threshold=0.5)

    def test_unanimous_fake(self, tracker):
        """10/10 FAKE → FAKE."""
        for _ in range(10):
            tracker.update(0.9)
        assert tracker.get_confidence_level() == "FAKE"
        stats = tracker.get_voting_stats()
        assert stats["fake_count"] == 10

    def test_unanimous_real(self, tracker):
        """10/10 REAL → REAL."""
        for _ in range(10):
            tracker.update(0.1)
        assert tracker.get_confidence_level() == "REAL"
        stats = tracker.get_voting_stats()
        assert stats["real_count"] == 10

    def test_slim_majority_fake(self, tracker):
        """6 FAKE + 4 REAL → FAKE."""
        for _ in range(6):
            tracker.update(0.9)
        for _ in range(4):
            tracker.update(0.1)
        assert tracker.get_confidence_level() == "FAKE"

    def test_slim_majority_real(self, tracker):
        """4 FAKE + 6 REAL → REAL."""
        for _ in range(4):
            tracker.update(0.9)
        for _ in range(6):
            tracker.update(0.1)
        assert tracker.get_confidence_level() == "REAL"

    def test_tie_goes_to_real(self, tracker):
        """5 FAKE + 5 REAL → REAL (not greater than)."""
        for _ in range(5):
            tracker.update(0.9)
        for _ in range(5):
            tracker.update(0.1)
        # fake_count == real_count → not (fake > real) → REAL
        assert tracker.get_confidence_level() == "REAL"

    def test_sliding_window_verdict_update(self, tracker):
        """Pushing new votes should slide out old ones and update verdict."""
        # Start with 10 FAKE
        for _ in range(10):
            tracker.update(0.9)
        assert tracker.get_confidence_level() == "FAKE"
        # Push 8 REAL → window now has 2 FAKE + 8 REAL
        for _ in range(8):
            tracker.update(0.1)
        assert tracker.get_confidence_level() == "REAL"


# ════════════════════════════════════════════════════════
#  9. FORENSIC SIGNAL CORRECTNESS
# ════════════════════════════════════════════════════════

class TestForensicSignalCorrectness:
    """Validate that forensic signals behave as expected for different inputs."""

    @pytest.fixture
    def analyzer(self):
        from frame_analysis import FrameForensicAnalyzer
        return FrameForensicAnalyzer(analysis_size=(256, 256))

    def test_smooth_image_higher_frequency_score(self, analyzer):
        """Smooth (GAN-like) image should score higher on frequency analysis."""
        smooth = make_smooth_image()
        noisy = make_noisy_image()
        r_smooth = analyzer.analyze(smooth)
        analyzer.reset()
        r_noisy = analyzer.analyze(noisy)
        # Smooth should score >= noisy on frequency (low high-freq = GAN-like)
        assert r_smooth["scores"]["frequency"] >= r_noisy["scores"]["frequency"], \
            f"Smooth freq={r_smooth['scores']['frequency']:.3f} should >= noisy freq={r_noisy['scores']['frequency']:.3f}"

    def test_uniform_color_scores_higher(self, analyzer):
        """An image with very uniform color should score higher on color analysis."""
        uniform = np.full((256, 256, 3), 100, dtype=np.uint8)
        varied = make_noisy_image()
        r_uni = analyzer.analyze(uniform)
        analyzer.reset()
        r_var = analyzer.analyze(varied)
        assert r_uni["scores"]["color"] >= r_var["scores"]["color"]

    def test_edge_rich_image_lower_edge_score(self, analyzer):
        """An image with many edges should score lower (more natural)."""
        edgy = make_gradient_image()
        smooth = make_smooth_image()
        r_edgy = analyzer.analyze(edgy)
        analyzer.reset()
        r_smooth = analyzer.analyze(smooth)
        # Smooth image has fewer edges → higher edge score (suspicious)
        assert r_smooth["scores"]["edge"] >= r_edgy["scores"]["edge"]

    def test_weighted_combination(self, analyzer):
        """Combined probability should be weighted sum of individual scores."""
        frame = make_noisy_image()
        result = analyzer.analyze(frame)
        manual_sum = sum(result["scores"][k] * analyzer.weights[k] for k in analyzer.weights)
        manual_sum = float(np.clip(manual_sum, 0.0, 1.0))
        assert abs(result["fake_probability"] - manual_sum) < 1e-6


# ════════════════════════════════════════════════════════
#  10. FREQUENCY FEATURES
# ════════════════════════════════════════════════════════

class TestFrequencyFeatures:
    """Validate FFT + DCT feature computation for the model."""

    def test_output_shape(self):
        """compute_frequency_features should return (2, 224, 224)."""
        from model import compute_frequency_features
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        features = compute_frequency_features(img, size=224)
        assert features.shape == (2, 224, 224)

    def test_output_dtype(self):
        """Features should be float32."""
        from model import compute_frequency_features
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        features = compute_frequency_features(img, size=224)
        assert features.dtype == np.float32

    def test_values_normalized(self):
        """FFT and DCT channels should be in [0, 1] range."""
        from model import compute_frequency_features
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        features = compute_frequency_features(img, size=224)
        assert features.min() >= -0.01, f"Min={features.min()}"
        assert features.max() <= 1.01, f"Max={features.max()}"

    def test_different_images_different_features(self):
        """Two different images should produce different frequency features."""
        from model import compute_frequency_features
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        f1 = compute_frequency_features(img1, size=224)
        f2 = compute_frequency_features(img2, size=224)
        assert not np.allclose(f1, f2), "Different images should have different frequency features"


# ════════════════════════════════════════════════════════
#  11. STABILITY SCORE CALCULATION
# ════════════════════════════════════════════════════════

class TestStabilityScore:
    """Validate the stability/consistency metric."""

    def test_stable_predictions_high_score(self):
        """Consistent predictions should give a high stability score."""
        from deepfake_detection import TemporalTracker
        t = TemporalTracker(voting_window=10, detection_threshold=0.5)
        for _ in range(30):
            t.update(0.85)  # very consistent
        stability = t.get_stability_score()
        assert stability > 0.9, f"Stable predictions should give >0.9, got {stability}"

    def test_unstable_predictions_low_score(self):
        """Wildly varying predictions should give a lower stability score."""
        from deepfake_detection import TemporalTracker
        t = TemporalTracker(voting_window=10, detection_threshold=0.5)
        for i in range(30):
            t.update(0.1 if i % 2 == 0 else 0.9)  # oscillating
        stability = t.get_stability_score()
        assert stability < 0.5, f"Unstable predictions should give <0.5, got {stability}"

    def test_stability_zero_with_few_frames(self):
        """Stability should be 0 with fewer than 10 frames."""
        from deepfake_detection import TemporalTracker
        t = TemporalTracker(voting_window=10, detection_threshold=0.5)
        for _ in range(5):
            t.update(0.5)
        assert t.get_stability_score() == 0.0
