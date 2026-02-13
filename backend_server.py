"""
Flask Backend Server for Browser Extension
Handles frame analysis requests from the browser extension.

Production-ready with:
- Multi-signal analysis (face + frame forensics)
- Works on any video (with or without faces)
- Proper logging and error handling
- Rate limiting
- Health monitoring
"""

import logging
import time
import traceback
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import base64
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("Starting Backend Server...")
print("=" * 60)
print("Loading models (this may take 10-30 seconds)...")

from deepfake_detection import DeepfakeDetector, mtcnn, model, DEVICE
from face_detection import detect_bounding_box

print("Models loaded successfully!")
print("=" * 60)

app = Flask(__name__)
# Enable CORS for browser extension
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize detector
print("Initializing detector...")
detector = DeepfakeDetector(enable_gradcam=False, use_tta=False, num_tta_augmentations=1, detection_threshold=0.55)
print("Detector initialized!")
print("=" * 60)

# --- Rate limiting ---
_last_request_time = 0
_min_request_interval = 0.1  # 100ms minimum between requests


def rate_limit(f):
    """Simple rate limiter to prevent overload."""
    @wraps(f)
    def decorated(*args, **kwargs):
        global _last_request_time
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _min_request_interval:
            return jsonify({
                'error': 'Rate limited',
                'retry_after_ms': int((_min_request_interval - elapsed) * 1000)
            }), 429
        _last_request_time = now
        return f(*args, **kwargs)
    return decorated

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system info."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': DEVICE,
        'gpu_name': gpu_name,
        'frame_count': detector.frame_count,
        'capabilities': {
            'face_detection': True,
            'frame_forensics': True,
            'temporal_tracking': True,
        }
    }), 200

@app.route('/reset', methods=['POST'])
def reset_detector():
    """Reset detector state (frame count, temporal tracker, forensics)."""
    try:
        detector.reset()
        return jsonify({
            'success': True,
            'message': 'Detector reset successfully'
        }), 200
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
@rate_limit
def analyze_frame():
    """
    Analyze a single frame for deepfake detection.
    
    Works on ANY frame - with or without faces:
    - If faces found: returns face model + frame forensic combined score
    - If no faces: returns frame forensic analysis only
    
    Expects: multipart/form-data with 'frame' field containing image
    Returns: JSON with comprehensive detection results
    """
    start_time = time.time()

    try:
        # Check if frame is in request
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        file = request.files['frame']

        # Read and decode image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # --- Run frame-level forensic analysis (always, regardless of faces) ---
        frame_forensic = detector.analyze_frame_forensics(frame)
        frame_forensic_prob = frame_forensic['fake_probability']
        frame_forensic_scores = frame_forensic.get('scores', {})

        # --- Detect faces ---
        faces = detect_bounding_box(frame)
        
        # Increment frame count
        detector.frame_count += 1

        if len(faces) > 0:
            # ---- FACE(S) DETECTED: use face model as PRIMARY signal ----
            x, y, w, h = faces[0]  # Primary face
            face_region = frame[y:y + h, x:x + w]

            # Get face model prediction
            fake_prob, real_score, gradcam = detector.analyze_face(face_region)

            if fake_prob is not None:
                # Use FACE probability directly for voting (it's the trained signal!)
                # Frame forensics is supplementary info, not used in voting

                # Update temporal tracker with FACE score directly
                detector.temporal_tracker.update(fake_prob)
                confidence_level = detector.temporal_tracker.get_confidence_level()
                temporal_avg = detector.temporal_tracker.get_temporal_average()
                stability = detector.temporal_tracker.get_stability_score()

                processing_time = (time.time() - start_time) * 1000

                response = {
                    'success': True,
                    'analysis_mode': 'face+frame',
                    'faces_detected': len(faces),
                    'fake_probability': float(fake_prob),  # Use face prob directly
                    'face_probability': float(fake_prob),
                    'frame_forensic_probability': float(frame_forensic_prob),
                    'real_probability': float(1 - fake_prob),
                    'confidence_level': confidence_level,
                    'temporal_average': float(temporal_avg),
                    'stability_score': float(stability),
                    'frame_count': detector.frame_count,
                    'processing_time_ms': round(processing_time, 1),
                    'face_bbox': {
                        'x': int(x), 'y': int(y),
                        'width': int(w), 'height': int(h)
                    },
                }

                logger.info(
                    f"Frame {detector.frame_count} | Face: {fake_prob*100:.0f}% | "
                    f"Forensic: {frame_forensic_prob*100:.0f}% | "
                    f"Verdict: {confidence_level} | {processing_time:.0f}ms"
                )

                return jsonify(response), 200

        # ---- NO FACES (or face analysis failed): frame forensics only ----
        detector.temporal_tracker.update(frame_forensic_prob)
        confidence_level = detector.temporal_tracker.get_confidence_level()
        temporal_avg = detector.temporal_tracker.get_temporal_average()
        stability = detector.temporal_tracker.get_stability_score()

        processing_time = (time.time() - start_time) * 1000

        response = {
            'success': True,
            'analysis_mode': 'frame_only',
            'faces_detected': len(faces),
            'fake_probability': float(frame_forensic_prob),
            'frame_forensic_probability': float(frame_forensic_prob),
            'real_probability': float(1 - frame_forensic_prob),
            'confidence_level': confidence_level,
            'temporal_average': float(temporal_avg),
            'stability_score': float(stability),
            'frame_count': detector.frame_count,
            'processing_time_ms': round(processing_time, 1),
        }

        logger.info(
            f"Frame {detector.frame_count} [NO FACE] | "
            f"Forensic: {frame_forensic_prob*100:.0f}% | "
            f"Verdict: {confidence_level} | {processing_time:.0f}ms"
        )

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get current detection statistics."""
    try:
        voting_stats = detector.temporal_tracker.get_voting_stats()
        return jsonify({
            'frame_count': detector.frame_count,
            'temporal_average': float(detector.temporal_tracker.get_temporal_average()),
            'stability_score': float(detector.temporal_tracker.get_stability_score()),
            'confidence_level': detector.temporal_tracker.get_confidence_level(),
            'history_length': len(detector.temporal_tracker.score_history),
            'voting': voting_stats,
            'device': DEVICE,
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Deepfake Detection Backend Server")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Model loaded: {model is not None}")
    print(f"  Capabilities:")
    print(f"    - Face detection (EfficientNet-B0)")
    print(f"    - Frequency-domain analysis (FFT magnitude + DCT)")
    print(f"    - Frame-level forensics (Noise, ELA, Edge, Color)")
    print(f"    - Temporal consistency tracking")
    print(f"    - Works on ANY video (face or no face)")
    print(f"\n  Server running on http://localhost:5000")
    print("=" * 60)
    print("  READY! You can now use the extension.")
    print("=" * 60 + "\n")

    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
