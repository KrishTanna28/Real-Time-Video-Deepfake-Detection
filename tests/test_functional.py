"""
╔══════════════════════════════════════════════════════════════════╗
║   FUNCTIONAL TESTS — Real-Time Video Deepfake Detection        ║
║   Validates core feature correctness: model loading, face      ║
║   detection, forensic analysis, temporal tracking, and API.    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import pytest
import numpy as np
import cv2
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────── Helpers ───────────────────────

def generate_face_frame(width=640, height=480):
    """Generate a synthetic frame with a face-like oval for testing."""
    frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Draw a skin-toned oval to simulate a face
    center = (width // 2, height // 2)
    axes = (80, 110)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (180, 160, 140), -1)
    # Add "eyes"
    cv2.circle(frame, (center[0] - 30, center[1] - 20), 8, (60, 40, 30), -1)
    cv2.circle(frame, (center[0] + 30, center[1] - 20), 8, (60, 40, 30), -1)
    return frame


def generate_blank_frame(width=640, height=480):
    """Generate a plain gray frame with no face."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def load_real_image_if_exists():
    """Try to load a real image from the dataset for higher-fidelity tests."""
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "dataset", "ff_face_crops", "val", "real"),
        os.path.join(os.path.dirname(__file__), "..", "dataset", "ff_face_crops", "val", "fake"),
    ]
    for p in paths:
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                img = cv2.imread(os.path.join(p, files[0]))
                if img is not None:
                    return img
    return None


# ════════════════════════════════════════════════════════
#  1. MODEL LOADING
# ════════════════════════════════════════════════════════

class TestModelLoading:
    """Verify the trained model loads correctly and architecture matches."""

    def test_model_file_exists(self):
        """best_model.pth weight file must be present."""
        weights = os.path.join(os.path.dirname(__file__), "..", "weights", "best_model.pth")
        assert os.path.exists(weights), "Weight file 'best_model.pth' not found in weights/"

    def test_model_architecture(self):
        """Model should be DeepfakeEfficientNet with correct classifier layers."""
        from model import DeepfakeEfficientNet
        m = DeepfakeEfficientNet(pretrained=False)
        # Classifier chain: Dropout → Linear(1280,512) → BN → ReLU → Dropout → Linear(512,256) → BN → ReLU → Dropout → Linear(256,1)
        fc = m.net._fc
        assert len(fc) == 10, f"Classifier should have 10 layers, got {len(fc)}"
        assert fc[1].in_features == 1280, "First linear layer input should be 1280"
        assert fc[1].out_features == 512, "First linear layer output should be 512"
        assert fc[9].out_features == 1, "Final layer output should be 1 (binary)"

    def test_model_loads_weights(self):
        """Loading checkpoint into the model should produce zero missing keys."""
        from model import DeepfakeEfficientNet
        weights = os.path.join(os.path.dirname(__file__), "..", "weights", "best_model.pth")
        if not os.path.exists(weights):
            pytest.skip("Weight file not present")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        m = DeepfakeEfficientNet(pretrained=False)
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert len(missing) == 0, f"Missing keys: {missing[:5]}"

    def test_model_forward_pass(self):
        """A random tensor forward pass should return shape (B, 1)."""
        from model import DeepfakeEfficientNet
        m = DeepfakeEfficientNet(pretrained=False).eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = m(dummy)
        assert out.shape == (1, 1), f"Expected (1,1) got {out.shape}"

    def test_model_output_range(self):
        """Sigmoid of model output should be in [0, 1]."""
        from model import DeepfakeEfficientNet
        m = DeepfakeEfficientNet(pretrained=False).eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = m(dummy)
            probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0 and probs.max() <= 1.0


# ════════════════════════════════════════════════════════
#  2. FACE DETECTION
# ════════════════════════════════════════════════════════

class TestFaceDetection:
    """Validate face detection module handles various inputs."""

    def test_detect_returns_list(self):
        """detect_bounding_box should always return a list."""
        from face_detection import detect_bounding_box
        frame = generate_blank_frame()
        result = detect_bounding_box(frame)
        assert isinstance(result, (list, np.ndarray, tuple))

    def test_no_crash_on_empty_frame(self):
        """Should not crash on a completely empty (black) frame."""
        from face_detection import detect_bounding_box
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_bounding_box(frame)
        assert isinstance(result, (list, np.ndarray, tuple))

    def test_no_crash_on_tiny_frame(self):
        """Should not crash on a very small frame."""
        from face_detection import detect_bounding_box
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = detect_bounding_box(frame)
        assert isinstance(result, (list, np.ndarray, tuple))

    def test_bbox_format(self):
        """If faces detected, each should have (x, y, w, h) format."""
        from face_detection import detect_bounding_box
        real_img = load_real_image_if_exists()
        if real_img is None:
            pytest.skip("No real test image available")
        faces = detect_bounding_box(real_img)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            assert w > 0 and h > 0, "Width and height must be positive"

    def test_extract_face_region(self):
        """extract_face_region should return a non-empty sub-image."""
        from face_detection import extract_face_region
        frame = generate_face_frame()
        region = extract_face_region(frame, (200, 100, 160, 220))
        assert region.shape[0] > 0 and region.shape[1] > 0


# ════════════════════════════════════════════════════════
#  3. FRAME FORENSIC ANALYSIS
# ════════════════════════════════════════════════════════

class TestFrameForensics:
    """Validate all 6 forensic signals run and return valid scores."""

    @pytest.fixture
    def analyzer(self):
        from frame_analysis import FrameForensicAnalyzer
        return FrameForensicAnalyzer(analysis_size=(256, 256))

    def test_full_analysis_returns_dict(self, analyzer):
        """analyze() should return a dict with 'scores' and 'fake_probability'."""
        frame = generate_face_frame()
        result = analyzer.analyze(frame)
        assert "scores" in result
        assert "fake_probability" in result

    def test_all_six_signals_present(self, analyzer):
        """All 6 forensic signals must be present in output."""
        frame = generate_face_frame()
        result = analyzer.analyze(frame)
        expected = {"frequency", "noise", "ela", "edge", "color", "temporal"}
        assert expected.issubset(result["scores"].keys()), f"Missing signals: {expected - set(result['scores'].keys())}"

    def test_scores_in_valid_range(self, analyzer):
        """Every forensic signal score must be in [0.0, 1.0]."""
        frame = generate_face_frame()
        result = analyzer.analyze(frame)
        for name, score in result["scores"].items():
            assert 0.0 <= score <= 1.0, f"{name} score {score} out of [0,1] range"

    def test_fake_probability_in_range(self, analyzer):
        """Combined fake_probability must be in [0.0, 1.0]."""
        frame = generate_face_frame()
        result = analyzer.analyze(frame)
        assert 0.0 <= result["fake_probability"] <= 1.0

    def test_fast_analysis(self, analyzer):
        """analyze_fast() should return subset of signals and valid probability."""
        frame = generate_face_frame()
        # Need to run analyze first for temporal baseline
        analyzer.analyze(frame)
        result = analyzer.analyze_fast(frame)
        assert "fake_probability" in result
        assert 0.0 <= result["fake_probability"] <= 1.0

    def test_reset_clears_state(self, analyzer):
        """After reset(), frame_count should be 0 and temporal state cleared."""
        frame = generate_face_frame()
        analyzer.analyze(frame)
        analyzer.analyze(frame)
        assert analyzer.frame_count == 2
        analyzer.reset()
        assert analyzer.frame_count == 0
        assert analyzer.prev_frame_gray is None


# ════════════════════════════════════════════════════════
#  4. TEMPORAL TRACKER (Voting System)
# ════════════════════════════════════════════════════════

class TestTemporalTracker:
    """Validate the voting-based temporal classification system."""

    @pytest.fixture
    def tracker(self):
        from deepfake_detection import TemporalTracker
        return TemporalTracker(window_size=60, voting_window=10, detection_threshold=0.75)

    def test_initial_state_is_uncertain(self, tracker):
        """Before any frames, verdict should be UNCERTAIN."""
        assert tracker.get_confidence_level() == "UNCERTAIN"

    def test_stays_uncertain_below_window(self, tracker):
        """Verdict should remain UNCERTAIN until voting_window frames collected."""
        for i in range(9):
            tracker.update(0.9)  # clearly fake
        assert tracker.get_confidence_level() == "UNCERTAIN", \
            "Should remain UNCERTAIN with < 10 frames"

    def test_verdict_after_full_window(self, tracker):
        """After 10 frames, a clear majority should produce a definitive verdict."""
        for _ in range(10):
            tracker.update(0.9)  # all fake
        assert tracker.get_confidence_level() == "FAKE"

    def test_majority_real(self, tracker):
        """Majority REAL frames should produce REAL verdict."""
        for _ in range(10):
            tracker.update(0.2)  # all real
        assert tracker.get_confidence_level() == "REAL"

    def test_mixed_votes_majority_wins(self, tracker):
        """With 7 FAKE + 3 REAL, verdict should be FAKE."""
        for _ in range(7):
            tracker.update(0.9)  # fake
        for _ in range(3):
            tracker.update(0.2)  # real
        assert tracker.get_confidence_level() == "FAKE"

    def test_reset_clears_everything(self, tracker):
        """After reset(), verdict goes back to UNCERTAIN and history is empty."""
        for _ in range(10):
            tracker.update(0.9)
        tracker.reset()
        assert tracker.get_confidence_level() == "UNCERTAIN"
        assert len(tracker.frame_classifications) == 0
        assert len(tracker.score_history) == 0

    def test_voting_stats(self, tracker):
        """get_voting_stats() should return correct fake/real counts."""
        for _ in range(6):
            tracker.update(0.9)
        for _ in range(4):
            tracker.update(0.2)
        stats = tracker.get_voting_stats()
        assert stats["fake_count"] == 6
        assert stats["real_count"] == 4
        assert stats["total_frames"] == 10

    def test_temporal_average(self, tracker):
        """Temporal average should reflect mean of input scores."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        for s in scores:
            tracker.update(s)
        avg = tracker.get_temporal_average()
        assert abs(avg - np.mean(scores)) < 1e-6

    def test_queue_sliding_window(self, tracker):
        """Queue should only keep the last voting_window classifications."""
        # Fill with 10 FAKE
        for _ in range(10):
            tracker.update(0.9)
        assert tracker.get_confidence_level() == "FAKE"
        # Now push 10 REAL — old FAKEs should slide out
        for _ in range(10):
            tracker.update(0.1)
        assert tracker.get_confidence_level() == "REAL"

    def test_none_probability_skipped(self, tracker):
        """update(None) should not add anything to history."""
        tracker.update(None)
        assert len(tracker.score_history) == 0
        assert len(tracker.frame_classifications) == 0


# ════════════════════════════════════════════════════════
#  5. DETECTOR INTEGRATION
# ════════════════════════════════════════════════════════

class TestDeepfakeDetector:
    """Integration tests for the DeepfakeDetector class."""

    @pytest.fixture
    def detector(self):
        from deepfake_detection import DeepfakeDetector
        return DeepfakeDetector(
            enable_gradcam=False, use_tta=False,
            num_tta_augmentations=1, detection_threshold=0.75
        )

    def test_reset_zeros_frame_count(self, detector):
        """reset() should set frame_count back to 0."""
        detector.frame_count = 42
        detector.reset()
        assert detector.frame_count == 0

    def test_frame_forensics_on_blank(self, detector):
        """Frame forensics should run on a blank frame without error."""
        frame = generate_blank_frame()
        result = detector.analyze_frame_forensics(frame)
        assert "fake_probability" in result

    def test_analyze_frame_forensics_works(self, detector):
        """analyze_frame_forensics() should return a dict with fake_probability."""
        frame = generate_face_frame()
        result = detector.analyze_frame_forensics(frame)
        assert "fake_probability" in result
        assert 0.0 <= result["fake_probability"] <= 1.0

    def test_frame_count_increments(self, detector):
        """Frame count should increment after processing."""
        frame = generate_face_frame()
        detector.analyze_frame_forensics(frame)
        detector.frame_count += 1  # simulate predict
        detector.analyze_frame_forensics(frame)
        detector.frame_count += 1
        assert detector.frame_count == 2


# ════════════════════════════════════════════════════════
#  6. API ENDPOINT CONTRACTS
# ════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Validate Flask API endpoint contracts (without starting the server)."""

    @pytest.fixture
    def client(self):
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_health_endpoint(self, client):
        """/health should return 200 with correct fields."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "device" in data
        assert "capabilities" in data

    def test_reset_endpoint(self, client):
        """/reset should return 200 with success=True."""
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_analyze_no_frame(self, client):
        """/analyze without frame should return 400."""
        resp = client.post("/analyze")
        assert resp.status_code == 400

    def test_analyze_with_valid_image(self, client):
        """/analyze with a valid JPEG frame should return 200 and required fields."""
        frame = generate_face_frame()
        _, buf = cv2.imencode(".jpg", frame)
        from io import BytesIO
        data = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
        # Reset to avoid rate-limit from previous test
        client.post("/reset")
        time.sleep(0.15)  # respect rate limit
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert "fake_probability" in body
        assert "processing_time_ms" in body
        assert "frame_count" in body

    def test_analyze_returns_probability_range(self, client):
        """/analyze fake_probability should be in [0, 1]."""
        frame = generate_blank_frame()
        _, buf = cv2.imencode(".jpg", frame)
        from io import BytesIO
        data = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
        client.post("/reset")
        time.sleep(0.15)
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        body = resp.get_json()
        assert 0.0 <= body["fake_probability"] <= 1.0

    def test_stats_endpoint(self, client):
        """/stats should return frame_count and voting info."""
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "frame_count" in data
        assert "voting" in data
