import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np 
import cv2
from collections import deque
import time
import random
import pickle
import os

from face_detection import detect_bounding_box

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize models
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# Create EfficientNet-B0 model with custom classifier for binary deepfake detection
class DeepfakeEfficientNet(nn.Module):
    """EfficientNet-B0 backbone with binary classification head"""
    def __init__(self, pretrained=True, dropout=0.5):
        super(DeepfakeEfficientNet, self).__init__()
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        
        # Get the number of features from the last layer
        num_features = self.efficientnet._fc.in_features
        
        # Replace the classifier with GENERALIZED architecture
        # This matches the train_generalized_colab.py model
        # More layers with BatchNorm for better generalization
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
    
    def get_feature_extractor(self):
        """Get the last convolutional layer for GradCAM"""
        return self.efficientnet._conv_head

# Initialize model
print("Initializing EfficientNet-B0 for deepfake detection...")
model = DeepfakeEfficientNet(pretrained=True)

# Load trained deepfake detection weights if available
import os

weights_paths = [
    os.path.join(os.path.dirname(__file__), "weights", "best_model.pth")
]

model_loaded = False
for weights_path in weights_paths:
    if os.path.exists(weights_path):
        print(f"Loading trained model from {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            # Handle potential state dict key mismatches
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Fix key mismatch (net. -> efficientnet.)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('net.'):
                    new_key = key.replace('net.', 'efficientnet.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            print("✓ Trained model loaded successfully")
            model_loaded = True
            break
        except Exception as e:
            print(f"⚠️  Warning: Could not load {weights_path}: {e}")
            continue

if not model_loaded:
    print(f"⚠️  Warning: No trained model found")
    print("Using pretrained ImageNet weights from EfficientNet-B0")
    print("NOTE: Model needs to be retrained for optimal deepfake detection")

model.to(DEVICE)
model.eval()


class TemporalTracker:
    """
    Layer 2: Enhanced Temporal Consistency Analysis
    Tracks predictions across frames with voting-based classification
    """
    
    def __init__(self, window_size=60, high_confidence_threshold=0.75, voting_window=10):
        """
        Args:
            window_size: Number of frames to track (60 frames ~ 2 seconds at 30fps)
            high_confidence_threshold: Threshold for high confidence detection
            voting_window: Number of frames to collect before updating verdict (default: 10)
        """
        self.window_size = window_size
        self.high_confidence_threshold = high_confidence_threshold
        self.voting_window = voting_window
        self.score_history = deque(maxlen=window_size)
        self.variance_history = deque(maxlen=30)  # Track prediction variance
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        
        # Voting system - using queue (deque)
        self.frame_classifications = deque(maxlen=voting_window)  # Queue of last N classifications
        self.current_verdict = None  # Current classification verdict (None until we have enough data)
        
    def update(self, fake_probability):
        """Update queue with new frame's fake probability and voting system"""
        # Skip if fake_probability is None
        if fake_probability is None:
            return
        
        self.score_history.append(fake_probability)
        
        # Track variance for anomaly detection
        if len(self.score_history) >= 5:
            recent = list(self.score_history)[-5:]
            variance = np.var(recent)
            self.variance_history.append(variance)
        
        # Classify this frame: fake if probability > 0.35, else real
        frame_class = 'FAKE' if fake_probability > 0.35 else 'REAL'
        
        # Add to queue (deque automatically removes oldest if full)
        self.frame_classifications.append(frame_class)
        
        # Update verdict by traversing the queue
        self._update_verdict()
    
    def _update_verdict(self):
        """Traverse the queue and count majority to update verdict"""
        if len(self.frame_classifications) == 0:
            # No data yet
            self.current_verdict = None
            return
        
        # Store previous verdict to detect changes
        previous_verdict = self.current_verdict
        
        # Traverse the queue and count FAKE vs REAL
        fake_count = 0
        real_count = 0
        for classification in self.frame_classifications:
            if classification == 'FAKE':
                fake_count += 1
            else:
                real_count += 1
        
        # Determine new verdict based on majority voting
        if fake_count > real_count:
            new_verdict = 'FAKE'
        else:
            new_verdict = 'REAL'
        
        # Update verdict (either first time or when changed)
        if previous_verdict != new_verdict:
            self.current_verdict = new_verdict
            current_frames = len(self.frame_classifications)
            if previous_verdict is None:
                # First verdict
                if self.current_verdict == 'FAKE':
                    print(f"\n🔴 VERDICT: FAKE ({fake_count}/{current_frames} frames)")
                else:
                    print(f"\n🟢 VERDICT: REAL ({real_count}/{current_frames} frames)")
            else:
                # Verdict changed
                if self.current_verdict == 'FAKE':
                    print(f"\n🔴 VERDICT CHANGED: FAKE ({fake_count}/{current_frames} frames)")
                else:
                    print(f"\n🟢 VERDICT CHANGED: REAL ({real_count}/{current_frames} frames)")
        # If verdict hasn't changed, keep the previous one (no update)
        
    def get_temporal_average(self):
        """Get running average of fake probability"""
        if len(self.score_history) == 0:
            return 0.0
        return sum(self.score_history) / len(self.score_history)
    
    def get_weighted_average(self):
        """Get weighted average (recent frames have more weight)"""
        if len(self.score_history) == 0:
            return 0.0
        
        scores = list(self.score_history)
        weights = np.linspace(0.5, 1.0, len(scores))  # Recent frames weighted more
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / sum(weights)
    
    def get_stability_score(self):
        """Calculate how stable/consistent the predictions are (lower variance = more stable)"""
        if len(self.score_history) < 10:
            return 0.0
        scores = list(self.score_history)
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return 1.0 - min(variance * 4, 1.0)  # Normalize to 0-1, higher is more stable
    
    def detect_anomalies(self):
        """Detect sudden jumps in predictions (deepfake artifacts)"""
        if len(self.variance_history) < 10:
            return 0.0
        
        # High variance = unstable predictions = potential deepfake
        avg_variance = np.mean(list(self.variance_history))
        
        # Normalize to 0-1 range
        anomaly_score = min(avg_variance * 10, 1.0)
        return anomaly_score
    
    def should_trigger_forensic_analysis(self):
        """Determine if we should trigger Layer 3 (Gemini) analysis"""
        if len(self.score_history) < self.window_size // 2:
            return False
            
        avg_score = self.get_temporal_average()
        stability = self.get_stability_score()
        current_time = time.time()
        
        # Trigger if: high average fake score + stable predictions + cooldown passed
        if (avg_score > self.high_confidence_threshold and 
            stability > 0.7 and 
            current_time - self.last_alert_time > self.alert_cooldown):
            self.last_alert_time = current_time
            return True
        return False
    
    def get_confidence_level(self):
        """Get confidence level based on voting system"""
        # Return UNCERTAIN if we don't have enough data yet
        if self.current_verdict is None:
            return 'UNCERTAIN'
        # Return the current verdict from voting system
        return self.current_verdict
    
    def get_voting_stats(self):
        """Get current voting statistics by traversing the queue"""
        fake_count = sum(1 for c in self.frame_classifications if c == 'FAKE')
        real_count = sum(1 for c in self.frame_classifications if c == 'REAL')
        return {
            'fake_count': fake_count,
            'real_count': real_count,
            'total_frames': len(self.frame_classifications)
        }
    
    def reset(self):
        """Reset the tracker - forget all previous verdicts and fake probabilities"""
        # Clear all history
        prev_window_len = len(self.frame_classifications)
        
        self.score_history.clear()
        self.variance_history.clear()
        self.last_alert_time = 0
        
        # Reset voting queue completely
        self.frame_classifications.clear()
        self.current_verdict = None
        
        print("✓ Temporal tracker reset - ALL previous data cleared:")
        print(f"  - Cleared {prev_history_len} previous fake probabilities")
        print(f"  - Cleared queue of {prev_window_len} frame classificationsious data cleared:")
        print(f"  - Cleared {prev_history_len} previous fake probabilities")
        print(f"  - Cleared 10-frame voting window")
        print(f"  - Previous verdict '{prev_verdict}' forgotten")
        print(f"  - Verdict reset to None (UNCERTAIN)")


class DeepfakeDetector:
    """3-Layer Deepfake Detection System with Enhanced Features"""
    
    def __init__(self, enable_gradcam=False, use_tta=True, num_tta_augmentations=3):
        self.enable_gradcam = enable_gradcam
        self.use_tta = use_tta  # Test-Time Augmentation
        self.num_tta_augmentations = num_tta_augmentations
        self.temporal_tracker = TemporalTracker(
            window_size=60, 
            high_confidence_threshold=0.75,
            voting_window=10  # Update verdict every 10 frames
        )
        self.frame_count = 0
        
        # Load calibrator if available
        self.calibrator = None
        calibrator_path = os.path.join(os.path.dirname(__file__), "weights", "calibrator.pkl")
        if os.path.exists(calibrator_path):
            try:
                with open(calibrator_path, 'rb') as f:
                    self.calibrator = pickle.load(f)
                print("✓ Probability calibrator loaded")
            except:
                print("⚠️ Could not load calibrator")
    
    def reset(self):
        """Reset detector state (call when stopping detection)"""
        self.temporal_tracker.reset()
        self.frame_count = 0
        print("=" * 50)
        print("✓ Detector completely reset")
        print("✓ Frame count reset to 0")
        print("✓ Ready for fresh detection session")
        print("=" * 50)
        
    def preprocess_face_quality(self, face_region):
        """Lightweight preprocessing for real-time performance"""
        # Skip expensive quality checks for speed
        processed = face_region.copy()
        
        # Only apply CLAHE for contrast enhancement (fast and effective)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def _single_prediction(self, face_region):
        """Single prediction without augmentation"""
        try:
            # Preprocess face
            input_face = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            input_face = mtcnn(input_face)
            
            if input_face is None:
                return None
            
            input_face = input_face.unsqueeze(0)
            input_face = F.interpolate(input_face, size=(224, 224), mode="bilinear", align_corners=False)
            input_face = input_face.to(DEVICE).to(torch.float32) / 255.0
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
            input_face = (input_face - mean) / std
            
            # Get prediction
            with torch.no_grad():
                logit = model(input_face).squeeze(0)
                output = torch.sigmoid(logit)
                return output.item()
        except:
            return None
    
    def analyze_face_with_tta(self, face_region):
        """Analyze face with Test-Time Augmentation for better accuracy"""
        predictions = []
        
        # Original prediction
        pred = self._single_prediction(face_region)
        if pred is not None:
            predictions.append(pred)
        
        # Augmented predictions
        for _ in range(self.num_tta_augmentations - 1):
            aug_face = face_region.copy()
            
            # Random horizontal flip
            if random.random() > 0.5:
                aug_face = cv2.flip(aug_face, 1)
            
            # Random brightness (±10%)
            brightness = random.uniform(0.9, 1.1)
            aug_face = cv2.convertScaleAbs(aug_face, alpha=brightness, beta=0)
            
            # Random rotation (±3 degrees)
            angle = random.uniform(-3, 3)
            h, w = aug_face.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_face = cv2.warpAffine(aug_face, M, (w, h))
            
            # Get prediction
            pred = self._single_prediction(aug_face)
            if pred is not None:
                predictions.append(pred)
        
        # Average all predictions
        if len(predictions) > 0:
            return np.mean(predictions)
        return None
    
    def apply_calibration(self, raw_prob):
        """Apply probability calibration if available"""
        if self.calibrator is None:
            return raw_prob
        
        try:
            # Calibrate probability
            calibrated = self.calibrator.predict_proba([[raw_prob]])[0][1]
            return calibrated
        except:
            return raw_prob
    
    def analyze_frequency_domain(self, face_region):
        """Analyze face in frequency domain to detect GAN artifacts"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Extract high-frequency energy
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Mask center (low frequencies)
            high_freq_region = magnitude.copy()
            mask_size = min(h, w) // 4
            high_freq_region[center_h-mask_size:center_h+mask_size, 
                           center_w-mask_size:center_w+mask_size] = 0
            
            # Calculate high-frequency ratio
            high_freq_energy = np.sum(high_freq_region)
            total_energy = np.sum(magnitude)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            # Deepfakes typically have lower high-frequency content
            if high_freq_ratio < 0.15:
                return 0.15  # Boost fake probability
            return 0.0
        except:
            return 0.0
    
    def apply_heuristics(self, fake_prob, face_region):
        """Lightweight rule-based adjustments for real-time performance"""
        adjustment = 0.0
        
        # Only check face resolution (very fast)
        h, w = face_region.shape[:2]
        if h < 80 or w < 80:
            adjustment += 0.10  # Low resolution suspicious
        
        # Skip expensive checks for real-time performance
        # (blurriness, smoothness, frequency analysis disabled)
        
        # Clip to valid range
        return np.clip(fake_prob + adjustment, 0, 1)
    
    def analyze_face(self, face_region):
        """Layer 1: Enhanced per-frame analysis with TTA and heuristics"""
        try:
            # Apply adaptive quality preprocessing
            preprocessed = self.preprocess_face_quality(face_region)
            
            # Get prediction with or without TTA
            if self.use_tta:
                fake_probability = self.analyze_face_with_tta(preprocessed)
            else:
                fake_probability = self._single_prediction(preprocessed)
            
            if fake_probability is None:
                return None, None, None
            
            # Apply calibration if available
            fake_probability = self.apply_calibration(fake_probability)
            
            # Apply heuristics (frequency analysis, quality checks)
            fake_probability = self.apply_heuristics(fake_probability, face_region)
            
            # GradCAM disabled for TTA mode (too slow)
            gradcam_img = None
            
            return fake_probability, fake_probability, gradcam_img
            
        except Exception as e:
            print(f"Face analysis error: {e}")
            return None, None, None
    
    def get_box_color(self, confidence_level):
        """Get color based on voting verdict"""
        if confidence_level == 'FAKE':
            return (0, 0, 255)  # Red for fake
        else:
            return (0, 255, 0)  # Green for real
    
    def draw_detection_overlay(self, frame, x, y, w, h, fake_prob, confidence_level):
        """Draw enhanced detection overlay with voting stats"""
        color = self.get_box_color(confidence_level)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Get voting stats
        voting_stats = self.temporal_tracker.get_voting_stats()
        
        # Main label with verdict
        if confidence_level == 'FAKE':
            label = f"FAKE (Frame: {fake_prob*100:.0f}%)"
        else:
            label = f"REAL (Frame: {(1-fake_prob)*100:.0f}%)"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw voting info below box
        if voting_stats['total_frames'] > 0:
            voting_info = f"Votes: F:{voting_stats['fake_count']} R:{voting_stats['real_count']} (Last {voting_stats['total_frames']} frames)"
            cv2.putText(frame, voting_info, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def predict(self, frame):
        """Main prediction function with 3-layer analysis"""
        self.frame_count += 1
        
        # Detect faces
        faces = detect_bounding_box(frame)
        
        trigger_forensic = False
        forensic_frame = None
        
        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            
            # Layer 1: Per-frame analysis
            fake_prob, real_score, gradcam = self.analyze_face(face_region)
            
            if fake_prob is None:
                continue
            
            # Layer 2: Update temporal tracker
            self.temporal_tracker.update(fake_prob)
            confidence_level = self.temporal_tracker.get_confidence_level()
            
            # Check if we should trigger Layer 3
            if self.temporal_tracker.should_trigger_forensic_analysis():
                trigger_forensic = True
                forensic_frame = frame.copy()
            
            # Draw overlay
            frame = self.draw_detection_overlay(frame, x, y, w, h, fake_prob, confidence_level)
            
            # Print detailed info every 10 frames
            if self.frame_count % 10 == 0:
                voting_stats = self.temporal_tracker.get_voting_stats()
                print(f"Frame {self.frame_count} | This Frame: {fake_prob*100:.0f}% | "
                      f"Verdict: {confidence_level} | "
                      f"Votes [F:{voting_stats['fake_count']} R:{voting_stats['real_count']}]")
        
        return frame, trigger_forensic, forensic_frame


# Global detector instance with enhanced features
detector = DeepfakeDetector(
    use_tta=False,             # Disabled for real-time speed
    num_tta_augmentations=1    # Single prediction for speed
)


def predict(frame):
    """Legacy function for backward compatibility"""
    result_frame, _, _ = detector.predict(frame)
    return result_frame


def predict_with_forensics(frame):
    """Enhanced prediction with forensic trigger info"""
    return detector.predict(frame)

