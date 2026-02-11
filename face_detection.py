"""
Face Detection Module
Uses OpenCV DNN (fast) as primary with Haar Cascade fallback.
Optimized for real-time performance.
"""

import cv2
import numpy as np
import os

# Load the pre-trained Haar Cascade classifier (fast fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to load OpenCV DNN face detector (more accurate than Haar)
_dnn_net = None
_dnn_available = False

# OpenCV DNN model paths (ships with opencv-contrib or can be downloaded)
_prototxt_path = os.path.join(os.path.dirname(__file__), "weights", "deploy.prototxt")
_model_path = os.path.join(os.path.dirname(__file__), "weights", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

if os.path.exists(_prototxt_path) and os.path.exists(_model_path):
    try:
        _dnn_net = cv2.dnn.readNetFromCaffe(_prototxt_path, _model_path)
        _dnn_available = True
        print("✓ OpenCV DNN face detector loaded (high accuracy)")
    except Exception as e:
        print(f"⚠️  DNN face detector failed to load: {e}")
        print("  Falling back to Haar Cascade (lower accuracy)")
else:
    print("ℹ️  DNN face detector model not found, using Haar Cascade")
    print(f"  To improve face detection, download:")
    print(f"    deploy.prototxt and res10_300x300_ssd_iter_140000_fp16.caffemodel")
    print(f"    into the weights/ directory")


def detect_bounding_box(frame, confidence_threshold=0.5):
    """
    Detect faces in a frame using the best available detector.
    
    Uses DNN detector if available (better accuracy, handles side profiles),
    falls back to Haar Cascade otherwise.
    
    Args:
        frame: Input image/frame (BGR format)
        confidence_threshold: Minimum confidence for DNN detections (default: 0.5)
        
    Returns:
        List of tuples: [(x, y, w, h), ...] representing face bounding boxes
    """
    try:
        if frame is None or frame.size == 0:
            return []
        
        if len(frame.shape) < 2 or frame.shape[0] < 30 or frame.shape[1] < 30:
            return []
        
        if _dnn_available:
            return _detect_dnn(frame, confidence_threshold)
        else:
            return _detect_haar(frame)
    
    except Exception as e:
        # If primary detector fails, try fallback
        try:
            return _detect_haar(frame)
        except Exception:
            return []


def _detect_dnn(frame, confidence_threshold=0.5):
    """Detect faces using OpenCV DNN SSD model (more accurate)."""
    h, w = frame.shape[:2]
    
    # Prepare input blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    
    _dnn_net.setInput(blob)
    detections = _dnn_net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            bw = x2 - x1
            bh = y2 - y1
            
            # Filter too small or invalid boxes
            if bw > 20 and bh > 20:
                face_boxes.append((x1, y1, bw, bh))
    
    return face_boxes


def _detect_haar(frame):
    """Detect faces using Haar Cascade (fast fallback)."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def draw_bounding_boxes(frame, faces, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes around detected faces
    
    Args:
        frame: Input image/frame
        faces: List of face bounding boxes [(x, y, w, h), ...]
        color: Box color (BGR format)
        thickness: Line thickness
        
    Returns:
        Frame with bounding boxes drawn
    """
    frame_copy = frame.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
    
    return frame_copy

def extract_face_region(frame, face_box, padding=0):
    """
    Extract face region from frame with optional padding
    
    Args:
        frame: Input image/frame
        face_box: Tuple (x, y, w, h) representing face bounding box
        padding: Extra pixels to add around the face (default: 0)
        
    Returns:
        Extracted face region as numpy array
    """
    x, y, w, h = face_box
    
    # Add padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(frame.shape[1], x + w + padding)
    y_end = min(frame.shape[0], y + h + padding)
    
    # Extract region
    face_region = frame[y_start:y_end, x_start:x_end]
    
    return face_region

def detect_and_extract_faces(frame, padding=0):
    """
    Detect faces and extract face regions in one step
    
    Args:
        frame: Input image/frame
        padding: Extra pixels to add around each face
        
    Returns:
        List of tuples: [(face_region, (x, y, w, h)), ...]
    """
    faces = detect_bounding_box(frame)
    
    face_data = []
    for face_box in faces:
        face_region = extract_face_region(frame, face_box, padding)
        face_data.append((face_region, face_box))
    
    return face_data

# Test function
if __name__ == "__main__":
    print("Face Detection Module")
    print("=" * 50)
    print("Available functions:")
    print("- detect_bounding_box(frame)")
    print("- draw_bounding_boxes(frame, faces)")
    print("- extract_face_region(frame, face_box)")
    print("- detect_and_extract_faces(frame)")
    print("=" * 50)
    print("✓ Module loaded successfully")
