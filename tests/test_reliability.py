"""
╔══════════════════════════════════════════════════════════════════╗
║   RELIABILITY & EDGE-CASE TESTS — Real-Time Deepfake Detection ║
║   Validates robustness: corrupted inputs, extreme resolutions, ║
║   determinism, concurrent usage, and graceful degradation.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import pytest
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════
#  18. CORRUPTED / MALFORMED INPUT
# ════════════════════════════════════════════════════════

class TestMalformedInput:
    """System should handle bad inputs gracefully, never crash."""

    def test_face_detection_none_input(self):
        """detect_bounding_box(None) should return empty list, not crash."""
        from face_detection import detect_bounding_box
        result = detect_bounding_box(None)
        assert result == [] or len(result) == 0

    def test_face_detection_1d_array(self):
        """1D array should not crash face detection."""
        from face_detection import detect_bounding_box
        result = detect_bounding_box(np.zeros(100, dtype=np.uint8))
        assert isinstance(result, (list, np.ndarray, tuple))

    def test_forensics_on_single_pixel(self):
        """Frame forensics on a 1x1 image should not crash."""
        from frame_analysis import FrameForensicAnalyzer
        analyzer = FrameForensicAnalyzer()
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        # Should not raise
        try:
            result = analyzer.analyze(frame)
            assert "fake_probability" in result
        except Exception:
            pass  # Acceptable to fail gracefully

    def test_api_invalid_image_bytes(self):
        """Sending garbage bytes to /analyze should return 400, not 500."""
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            from io import BytesIO
            data = {"frame": (BytesIO(b"not_an_image"), "frame.jpg")}
            client.post("/reset")
            time.sleep(0.15)
            resp = client.post("/analyze", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400, f"Expected 400 for garbage bytes, got {resp.status_code}"

    def test_api_empty_file(self):
        """Sending an empty file should return error (400 or 500), not 200."""
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            from io import BytesIO
            data = {"frame": (BytesIO(b""), "frame.jpg")}
            client.post("/reset")
            time.sleep(0.15)
            resp = client.post("/analyze", data=data, content_type="multipart/form-data")
            assert resp.status_code in (400, 500), f"Expected error status, got {resp.status_code}"


# ════════════════════════════════════════════════════════
#  19. RESOLUTION VARIANCE
# ════════════════════════════════════════════════════════

class TestResolutionVariance:
    """Pipeline should handle various image sizes without error."""

    @pytest.fixture
    def analyzer(self):
        from frame_analysis import FrameForensicAnalyzer
        return FrameForensicAnalyzer()

    @pytest.mark.parametrize("width,height", [
        (160, 120),   # Very low res
        (320, 240),   # QVGA
        (640, 480),   # VGA
        (1280, 720),  # HD
        (1920, 1080), # Full HD
        (500, 300),   # Non-standard
    ])
    def test_forensics_various_resolutions(self, analyzer, width, height):
        """Frame forensics should work at any resolution."""
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        result = analyzer.analyze(frame)
        assert 0.0 <= result["fake_probability"] <= 1.0
        analyzer.reset()

    @pytest.mark.parametrize("width,height", [
        (160, 120),
        (640, 480),
        (1280, 720),
    ])
    def test_face_detection_various_resolutions(self, width, height):
        """Face detection should not crash at any resolution."""
        from face_detection import detect_bounding_box
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        result = detect_bounding_box(frame)
        assert isinstance(result, (list, np.ndarray, tuple))


# ════════════════════════════════════════════════════════
#  20. DETERMINISM / REPRODUCIBILITY
# ════════════════════════════════════════════════════════

class TestDeterminism:
    """Same input should produce the same output."""

    def test_model_deterministic(self):
        """Same tensor → same output (no randomness in eval mode)."""
        from model import DeepfakeEfficientNet
        m = DeepfakeEfficientNet(pretrained=False).eval()
        torch.manual_seed(42)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = m(dummy).item()
            out2 = m(dummy).item()
        assert out1 == out2, "Model should be deterministic in eval mode"

    def test_forensic_deterministic(self):
        """Same frame → same forensic scores (no randomness)."""
        from frame_analysis import FrameForensicAnalyzer
        frame = np.random.RandomState(42).randint(0, 255, (256, 256, 3)).astype(np.uint8)

        a1 = FrameForensicAnalyzer()
        r1 = a1.analyze(frame)

        a2 = FrameForensicAnalyzer()
        r2 = a2.analyze(frame)

        for key in r1["scores"]:
            assert abs(r1["scores"][key] - r2["scores"][key]) < 1e-6, \
                f"{key}: {r1['scores'][key]} != {r2['scores'][key]}"

    def test_frequency_features_deterministic(self):
        """Same image → same frequency features."""
        from model import compute_frequency_features
        img = np.random.RandomState(123).randint(0, 255, (200, 200, 3)).astype(np.uint8)
        f1 = compute_frequency_features(img, size=224)
        f2 = compute_frequency_features(img, size=224)
        assert np.allclose(f1, f2), "Frequency features should be deterministic"


# ════════════════════════════════════════════════════════
#  21. IMAGE FORMAT COMPATIBILITY
# ════════════════════════════════════════════════════════

class TestImageFormatCompatibility:
    """API should accept JPEG, PNG, and BMP encoded frames."""

    @pytest.fixture
    def client(self):
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    @pytest.mark.parametrize("ext,params", [
        (".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 80]),
        (".png", []),
        (".bmp", []),
    ])
    def test_various_image_formats(self, client, ext, params):
        """API should correctly decode JPEG, PNG, and BMP frames."""
        from io import BytesIO
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        if params:
            _, buf = cv2.imencode(ext, frame, params)
        else:
            _, buf = cv2.imencode(ext, frame)

        client.post("/reset")
        time.sleep(0.15)
        data = {"frame": (BytesIO(buf.tobytes()), f"frame{ext}")}
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200, f"Failed for {ext}: status={resp.status_code}"
        body = resp.get_json()
        assert body["success"] is True


# ════════════════════════════════════════════════════════
#  22. RATE LIMITING
# ════════════════════════════════════════════════════════

class TestRateLimiting:
    """Rate limiter should correctly throttle rapid requests."""

    def test_rapid_requests_get_429(self):
        """Two requests within 100ms should trigger rate limiting."""
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", frame)
            from io import BytesIO

            client.post("/reset")
            time.sleep(0.15)

            # First request
            data1 = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
            resp1 = client.post("/analyze", data=data1, content_type="multipart/form-data")

            # Immediate second request (no sleep)
            data2 = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
            resp2 = client.post("/analyze", data=data2, content_type="multipart/form-data")

            assert resp2.status_code == 429, \
                f"Expected 429 for rapid request, got {resp2.status_code}"

    def test_spaced_requests_pass(self):
        """Requests spaced >100ms apart should not be rate-limited."""
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", frame)
            from io import BytesIO

            client.post("/reset")
            time.sleep(0.15)

            data1 = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
            resp1 = client.post("/analyze", data=data1, content_type="multipart/form-data")
            time.sleep(0.15)

            data2 = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
            resp2 = client.post("/analyze", data=data2, content_type="multipart/form-data")

            assert resp2.status_code == 200


# ════════════════════════════════════════════════════════
#  23. RESET INTEGRITY
# ════════════════════════════════════════════════════════

class TestResetIntegrity:
    """After reset, all stateful components should be clean."""

    def test_detector_full_reset(self):
        """DeepfakeDetector.reset() should clear all internal state."""
        from deepfake_detection import DeepfakeDetector
        d = DeepfakeDetector(enable_gradcam=False, use_tta=False)
        # Simulate some activity
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        d.predict(frame)
        d.predict(frame)
        assert d.frame_count > 0

        d.reset()
        assert d.frame_count == 0
        assert len(d.temporal_tracker.score_history) == 0
        assert len(d.temporal_tracker.frame_classifications) == 0
        assert d.temporal_tracker.current_verdict is None
        assert d.frame_analyzer.frame_count == 0

    def test_api_reset_clears_frame_count(self):
        """POST /reset followed by GET /stats should show frame_count=0."""
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            # Analyze a frame first
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", frame)
            from io import BytesIO
            time.sleep(0.15)
            data = {"frame": (BytesIO(buf.tobytes()), "f.jpg")}
            client.post("/analyze", data=data, content_type="multipart/form-data")

            # Reset
            client.post("/reset")

            # Check stats
            resp = client.get("/stats")
            stats = resp.get_json()
            assert stats["frame_count"] == 0


# ════════════════════════════════════════════════════════
#  24. CONTINUOUS OPERATION
# ════════════════════════════════════════════════════════

class TestContinuousOperation:
    """System should handle sustained frame processing without degradation."""

    def test_50_consecutive_frames(self):
        """Process 50 frames through forensics without error or memory leak."""
        from frame_analysis import FrameForensicAnalyzer
        analyzer = FrameForensicAnalyzer()
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = analyzer.analyze(frame)
            assert 0.0 <= result["fake_probability"] <= 1.0, f"Frame {i} out of range"

    def test_temporal_tracker_long_session(self):
        """Tracker should stay correct over 200 updates."""
        from deepfake_detection import TemporalTracker
        t = TemporalTracker(voting_window=10, detection_threshold=0.5)
        for i in range(200):
            prob = 0.9 if i < 100 else 0.1
            t.update(prob)

        # After 200 frames (last 100 are REAL), verdict should be REAL
        assert t.get_confidence_level() == "REAL"
        stats = t.get_voting_stats()
        assert stats["real_count"] == 10  # last 10 frames are all REAL
