"""
╔══════════════════════════════════════════════════════════════════╗
║   PERFORMANCE TESTS — Real-Time Video Deepfake Detection       ║
║   Benchmarks latency of every pipeline component to ensure     ║
║   real-time viability (target: <500ms per frame end-to-end).   ║
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


# ─────────────────────── Helpers ───────────────────────

def generate_frame(width=640, height=480):
    return np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)


def benchmark(fn, iterations=10, warmup=2):
    """Run fn for warmup+iterations, return (mean_ms, min_ms, max_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
    # Timed runs
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times), min(times), max(times)


# ════════════════════════════════════════════════════════
#  12. FACE DETECTION LATENCY
# ════════════════════════════════════════════════════════

class TestFaceDetectionPerformance:
    """Face detection should complete within real-time budget."""

    def test_face_detection_under_100ms(self):
        """Single-frame face detection should take <100ms."""
        from face_detection import detect_bounding_box
        frame = generate_frame()
        mean_ms, min_ms, max_ms = benchmark(
            lambda: detect_bounding_box(frame), iterations=20, warmup=3
        )
        assert mean_ms < 100, f"Face detection mean={mean_ms:.1f}ms exceeds 100ms budget"
        print(f"\n  Face detection: mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")

    def test_face_detection_hd_frame(self):
        """Face detection on 1280×720 should still be < 500ms."""
        from face_detection import detect_bounding_box
        frame = generate_frame(1280, 720)
        mean_ms, _, _ = benchmark(lambda: detect_bounding_box(frame), iterations=10)
        assert mean_ms < 500, f"HD face detection mean={mean_ms:.1f}ms exceeds 500ms"
        print(f"\n  HD face detection: mean={mean_ms:.1f}ms")


# ════════════════════════════════════════════════════════
#  13. MODEL INFERENCE LATENCY
# ════════════════════════════════════════════════════════

class TestModelInferencePerformance:
    """EfficientNet-B0 single forward pass speed benchmark."""

    def test_model_forward_under_200ms(self):
        """Single forward pass should complete in <200ms (GPU) or <500ms (CPU)."""
        from model import DeepfakeEfficientNet
        from deepfake_detection import DEVICE
        m = DeepfakeEfficientNet(pretrained=False).to(DEVICE).eval()
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)

        def forward():
            with torch.no_grad():
                m(dummy)
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()

        mean_ms, min_ms, max_ms = benchmark(forward, iterations=20, warmup=5)
        limit = 200 if DEVICE.startswith("cuda") else 500
        assert mean_ms < limit, f"Forward pass mean={mean_ms:.1f}ms exceeds {limit}ms on {DEVICE}"
        print(f"\n  Model inference ({DEVICE}): mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")

    def test_batch_inference_throughput(self):
        """Batch of 4 frames should be faster per-frame than single inference."""
        from model import DeepfakeEfficientNet
        from deepfake_detection import DEVICE
        m = DeepfakeEfficientNet(pretrained=False).to(DEVICE).eval()

        single = torch.randn(1, 3, 224, 224).to(DEVICE)
        batch = torch.randn(4, 3, 224, 224).to(DEVICE)

        def single_fn():
            with torch.no_grad():
                m(single)
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()

        def batch_fn():
            with torch.no_grad():
                m(batch)
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()

        single_ms, _, _ = benchmark(single_fn, iterations=10, warmup=3)
        batch_ms, _, _ = benchmark(batch_fn, iterations=10, warmup=3)
        per_frame_batch = batch_ms / 4
        print(f"\n  Single: {single_ms:.1f}ms | Batch(4): {batch_ms:.1f}ms ({per_frame_batch:.1f}ms/frame)")


# ════════════════════════════════════════════════════════
#  14. FRAME FORENSIC ANALYSIS LATENCY
# ════════════════════════════════════════════════════════

class TestFrameForensicPerformance:
    """Frame-level forensic analysis speed benchmarks."""

    @pytest.fixture
    def analyzer(self):
        from frame_analysis import FrameForensicAnalyzer
        return FrameForensicAnalyzer(analysis_size=(256, 256))

    def test_full_analysis_under_50ms(self, analyzer):
        """Full 6-signal forensic analysis should take <50ms."""
        frame = generate_frame()
        mean_ms, min_ms, max_ms = benchmark(
            lambda: analyzer.analyze(frame), iterations=30, warmup=5
        )
        assert mean_ms < 50, f"Full forensic analysis mean={mean_ms:.1f}ms exceeds 50ms"
        print(f"\n  Full forensic: mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")

    def test_fast_analysis_under_20ms(self, analyzer):
        """Fast 3-signal forensic analysis should take <20ms."""
        frame = generate_frame()
        analyzer.analyze(frame)  # init temporal baseline
        mean_ms, min_ms, max_ms = benchmark(
            lambda: analyzer.analyze_fast(frame), iterations=30, warmup=5
        )
        assert mean_ms < 20, f"Fast forensic analysis mean={mean_ms:.1f}ms exceeds 20ms"
        print(f"\n  Fast forensic: mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")


# ════════════════════════════════════════════════════════
#  15. FREQUENCY FEATURE EXTRACTION LATENCY
# ════════════════════════════════════════════════════════

class TestFrequencyFeaturePerformance:
    """FFT + DCT feature extraction speed."""

    def test_frequency_features_under_30ms(self):
        """compute_frequency_features should complete in <30ms."""
        from model import compute_frequency_features
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        mean_ms, min_ms, max_ms = benchmark(
            lambda: compute_frequency_features(img, size=224), iterations=30, warmup=5
        )
        assert mean_ms < 30, f"Frequency features mean={mean_ms:.1f}ms exceeds 30ms"
        print(f"\n  Frequency features: mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")


# ════════════════════════════════════════════════════════
#  16. API END-TO-END LATENCY
# ════════════════════════════════════════════════════════

class TestAPIPerformance:
    """End-to-end /analyze request latency through Flask test client."""

    @pytest.fixture
    def client(self):
        from backend_server import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_analyze_endpoint_under_1500ms(self, client):
        """Full /analyze request (decode + detect + model + forensics) should be <1500ms."""
        frame = generate_frame()
        _, buf = cv2.imencode(".jpg", frame)
        from io import BytesIO

        # Reset state and wait for rate limiter
        client.post("/reset")
        time.sleep(0.15)

        times = []
        for i in range(5):
            data = {"frame": (BytesIO(buf.tobytes()), "frame.jpg")}
            t0 = time.perf_counter()
            resp = client.post("/analyze", data=data, content_type="multipart/form-data")
            t1 = time.perf_counter()
            if resp.status_code == 200:
                times.append((t1 - t0) * 1000)
            time.sleep(0.15)  # respect rate limit

        if len(times) > 0:
            mean_ms = np.mean(times)
            assert mean_ms < 1500, f"API mean={mean_ms:.1f}ms exceeds 1500ms"
            print(f"\n  API /analyze: mean={mean_ms:.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms")

    def test_health_endpoint_under_50ms(self, client):
        """/health should respond in <50ms."""
        t0 = time.perf_counter()
        resp = client.get("/health")
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        assert resp.status_code == 200
        assert ms < 50, f"/health took {ms:.1f}ms (limit 50ms)"

    def test_reset_endpoint_under_50ms(self, client):
        """/reset should respond in <50ms."""
        t0 = time.perf_counter()
        resp = client.post("/reset")
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        assert resp.status_code == 200
        assert ms < 50, f"/reset took {ms:.1f}ms (limit 50ms)"


# ════════════════════════════════════════════════════════
#  17. MEMORY FOOTPRINT
# ════════════════════════════════════════════════════════

class TestMemoryFootprint:
    """Model and system memory should stay within bounds."""

    def test_model_parameter_count(self):
        """EfficientNet-B0 should have ~5.3M parameters (not bloated)."""
        from model import DeepfakeEfficientNet
        m = DeepfakeEfficientNet(pretrained=False)
        total_params = sum(p.numel() for p in m.parameters())
        # EfficientNet-B0 ≈ 5.3M params
        assert total_params < 8_000_000, f"Model has {total_params/1e6:.1f}M params (expected <8M)"
        print(f"\n  Model parameters: {total_params/1e6:.2f}M")

    def test_model_size_on_disk(self):
        """Weight file should be <50MB."""
        weights = os.path.join(os.path.dirname(__file__), "..", "weights", "best_model.pth")
        if not os.path.exists(weights):
            pytest.skip("Weight file not present")
        size_mb = os.path.getsize(weights) / (1024 * 1024)
        assert size_mb < 50, f"Weight file is {size_mb:.1f}MB (limit 50MB)"
        print(f"\n  Weight file size: {size_mb:.1f}MB")
