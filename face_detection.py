"""
Face Detection Module
Uses OpenCV Haar Cascades for fast face detection
"""

import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_bounding_box(frame):
    """
    Detect faces in a frame using Haar Cascade classifier
    
    Args:
        frame: Input image/frame (BGR format)
        
    Returns:
        List of tuples: [(x, y, w, h), ...] representing face bounding boxes
    """
    try:
        # Validate input
        if frame is None or frame.size == 0:
            return []
        
        # Ensure frame has valid dimensions
        if len(frame.shape) < 2 or frame.shape[0] < 30 or frame.shape[1] < 30:
            return []
        
        # Convert to grayscale for face detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect faces with error handling
        # Parameters:
        # - scaleFactor: How much the image size is reduced at each image scale
        # - minNeighbors: How many neighbors each candidate rectangle should have to retain it
        # - minSize: Minimum possible object size
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append((x, y, w, h))
        
        return face_boxes
    
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

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
