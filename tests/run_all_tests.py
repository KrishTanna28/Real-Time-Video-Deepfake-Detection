"""
╔══════════════════════════════════════════════════════════════════╗
║   UNIFIED TEST RUNNER — Generates a Summary Table              ║
║   Run: pytest tests/ -v --tb=short -q                          ║
║   Or:  python tests/run_all_tests.py   (for the full report)   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import subprocess
import json
import re

# Add project root
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


# ──────────── Test Plan Definition ────────────
# Mirrors the "Unified Testing & Validation Summary" table format

TEST_PLAN = [
    # (Sr, Testing Type, What is Tested, Scenario/Input, Metric, Expected Result)
    (1,  "Functional",     "Model Loading",         "Load best_model.pth",        "Missing keys",         "0 missing"),
    (2,  "Functional",     "Model Architecture",    "EfficientNet-B0 classifier", "Layer count/dims",     "10 layers, 1280→1"),
    (3,  "Functional",     "Model Forward Pass",    "Random tensor (1,3,224,224)","Output shape",         "(1, 1)"),
    (4,  "Functional",     "Face Detection",        "Various frames",             "Returns list",         "Always list"),
    (5,  "Functional",     "Frame Forensics",       "Random frame",               "All 6 signals present","6/6 signals"),
    (6,  "Functional",     "Temporal Tracker",      "10 FAKE frames",             "Verdict",              "FAKE"),
    (7,  "Functional",     "API /health",           "GET request",                "HTTP status",          "200 + healthy"),
    (8,  "Functional",     "API /reset",            "POST request",               "success field",        "True"),
    (9,  "Functional",     "API /analyze",          "Valid JPEG frame",           "Response fields",      "All present"),
    (10, "Algorithm",      "Threshold Accuracy",    "Prob vs threshold",          "Classification",       "Correct FAKE/REAL"),
    (11, "Algorithm",      "Voting Majority",       "6F+4R / 4F+6R",             "Verdict",              "Majority wins"),
    (12, "Algorithm",      "Tie-Break Rule",        "5 FAKE + 5 REAL",           "Verdict",              "REAL (safe default)"),
    (13, "Algorithm",      "Sliding Window",        "10F then 8R",               "Verdict update",       "FAKE → REAL"),
    (14, "Algorithm",      "Forensic Signals",      "Smooth vs noisy images",    "Score ordering",       "Correct direction"),
    (15, "Algorithm",      "Weighted Combination",  "Manual weight calc",         "Match combined score", "Exact match"),
    (16, "Algorithm",      "Frequency Features",    "FFT + DCT extraction",      "Shape & range",        "(2,224,224) in [0,1]"),
    (17, "Algorithm",      "Stability Score",       "Consistent vs oscillating",  "Score value",          ">0.9 vs <0.5"),
    (18, "Performance",    "Face Detection Speed",  "640×480 frame",             "Latency",              "<100ms"),
    (19, "Performance",    "Model Inference",       "Single forward pass",        "Latency",              "<200ms GPU/<500ms CPU"),
    (20, "Performance",    "Full Forensics",        "6-signal analysis",          "Latency",              "<50ms"),
    (21, "Performance",    "Fast Forensics",        "3-signal analysis",          "Latency",              "<20ms"),
    (22, "Performance",    "API End-to-End",        "/analyze full pipeline",     "Latency",              "<1500ms"),
    (23, "Performance",    "Memory Footprint",      "Model parameters",           "Param count",          "<8M params"),
    (24, "Performance",    "Weight File Size",      "best_model.pth",            "File size",            "<50MB"),
    (25, "Reliability",    "Corrupted Input",       "None / garbage bytes",       "Error handling",       "No crash (400)"),
    (26, "Reliability",    "Resolution Variance",   "120p to 1080p",             "All work",             "Valid probability"),
    (27, "Reliability",    "Determinism",           "Same input twice",           "Output match",         "Identical"),
    (28, "Reliability",    "Image Formats",         "JPEG, PNG, BMP",            "API acceptance",       "200 OK"),
    (29, "Reliability",    "Rate Limiting",         "Rapid requests",             "HTTP 429",             "Throttled correctly"),
    (30, "Reliability",    "Reset Integrity",       "Reset + check state",        "All counters",         "Zero / None"),
    (31, "Reliability",    "Continuous Operation",   "50 consecutive frames",     "No degradation",       "All valid"),
]


def print_summary_table():
    """Print a formatted summary table of the test plan."""
    print()
    print("=" * 110)
    print("  UNIFIED TESTING & VALIDATION SUMMARY — Real-Time Video Deepfake Detection System")
    print("=" * 110)
    print(f"{'Sr':>3} | {'Testing Type':<15} | {'What is Tested':<22} | {'Scenario / Input':<24} | {'Metric':<20} | {'Expected':<18}")
    print("-" * 110)
    for row in TEST_PLAN:
        sr, ttype, what, scenario, metric, expected = row
        print(f"{sr:>3} | {ttype:<15} | {what:<22} | {scenario:<24} | {metric:<20} | {expected:<18}")
    print("=" * 110)
    print()


def run_tests():
    """Run all tests via pytest and display results."""
    print_summary_table()

    print("\nRunning all tests...\n")
    test_dir = os.path.dirname(__file__)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    print(result.stdout)
    if result.stderr:
        # Only print relevant stderr (filter out warnings)
        lines = [l for l in result.stderr.splitlines() if "warning" not in l.lower()]
        if lines:
            print("\n".join(lines))

    # Parse results
    passed = len(re.findall(r"PASSED", result.stdout))
    failed = len(re.findall(r"FAILED", result.stdout))
    skipped = len(re.findall(r"SKIPPED", result.stdout))
    total = passed + failed + skipped

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"  Pass Rate: {passed/max(total,1)*100:.1f}%")
    print("=" * 60)

    return result.returncode


if __name__ == "__main__":
    exit(run_tests())
