// Content script for capturing tab content and displaying overlay
console.log('Deepfake Detection: Content script loaded and ready');
console.log('Page URL:', window.location.href);

// Global variables (use window to avoid redeclaration)
if (!window.deepfakeDetection) {
  window.deepfakeDetection = {
    overlayIframe: null,
    captureInterval: null,
    isCapturing: false
  };
}

const state = window.deepfakeDetection;

// Create and inject overlay
function createOverlay() {
  if (state.overlayIframe) return;

  state.overlayIframe = document.createElement('iframe');
  state.overlayIframe.id = 'deepfake-detection-overlay';
  state.overlayIframe.src = chrome.runtime.getURL('overlay.html');
  state.overlayIframe.style.cssText = `
    position: fixed;
    top: 0;
    right: 0;
    width: 360px;
    height: 100vh;
    border: none;
    z-index: 999999;
    pointer-events: auto;
  `;
  
  document.body.appendChild(state.overlayIframe);
}

// Remove overlay
function removeOverlay() {
  if (state.overlayIframe) {
    state.overlayIframe.remove();
    state.overlayIframe = null;
  }
}

// Capture current page as image using canvas
async function captureTab() {
  return new Promise((resolve, reject) => {
    try {
      // Find video element on the page
      const video = document.querySelector('video');
      
      if (!video) {
        reject(new Error('No video found on page'));
        return;
      }

      // Check if video is ready
      if (video.readyState < 2) {
        reject(new Error('Video not ready yet'));
        return;
      }

      // Create canvas to capture video frame
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      
      if (canvas.width === 0 || canvas.height === 0) {
        reject(new Error('Video has no dimensions'));
        return;
      }
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert to data URL
      const dataUrl = canvas.toDataURL('image/png');
      resolve(dataUrl);
    } catch (error) {
      reject(error);
    }
  });
}

// Send frame to backend for analysis
async function analyzeFrame(imageDataUrl) {
  try {
    console.log('Sending frame to background script for analysis...');

    // Send to background script to forward to backend
    const result = await chrome.runtime.sendMessage({
      action: 'analyzeFrame',
      imageData: imageDataUrl
    });

    if (result.error) {
      throw new Error(result.error);
    }

    console.log('Analysis result:', result);
    return result;
  } catch (error) {
    console.error('Error analyzing frame:', error);
    throw error;
  }
}

// Start capturing and analyzing
async function startDetection(interval = 1000) {
  if (state.isCapturing) return;

  state.isCapturing = true;
  createOverlay();

  // Wait for overlay to load, then reset it
  setTimeout(() => {
    resetOverlay(); // Reset overlay to clear old fake probability window
  }, 100);

  // Update overlay with initial status
  updateOverlay({ status: 'analyzing' });

  state.captureInterval = setInterval(async () => {
    try {
      // Capture current tab
      const imageDataUrl = await captureTab();

      // Analyze frame
      const result = await analyzeFrame(imageDataUrl);

      // Update overlay with results
      updateOverlay(result);

      // Send results to popup (ignore if popup is closed)
      try {
        chrome.runtime.sendMessage({
          action: 'detectionResult',
          data: result
        });
      } catch (e) {
        // Popup might be closed, ignore
      }

    } catch (error) {
      console.error('Detection error:', error);
      
      // Send error to popup (ignore if popup is closed)
      try {
        chrome.runtime.sendMessage({
          action: 'detectionError',
          error: error.message
        });
      } catch (e) {
        // Popup might be closed, ignore
      }
    }
  }, interval);
}

// Stop detection
async function stopDetection() {
  if (state.captureInterval) {
    clearInterval(state.captureInterval);
    state.captureInterval = null;
  }
  state.isCapturing = false;
  removeOverlay();

  // Reset backend detector state
  try {
    const backendUrl = 'http://localhost:5000';
    await fetch(`${backendUrl}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    console.log('Backend detector reset');
  } catch (error) {
    console.log('Could not reset backend:', error);
  }

  // Send stopped message (ignore if popup is closed)
  try {
    chrome.runtime.sendMessage({ action: 'detectionStopped' });
  } catch (e) {
    // Popup might be closed, ignore
  }
}

// Update overlay with detection results
function updateOverlay(data) {
  if (!state.overlayIframe) return;

  state.overlayIframe.contentWindow.postMessage({
    type: 'updateResults',
    data: data
  }, '*');
}

// Reset overlay display
function resetOverlay() {
  if (!state.overlayIframe) return;

  state.overlayIframe.contentWindow.postMessage({
    type: 'resetDisplay'
  }, '*');
}

// Listen for messages from background script
if (!window.deepfakeDetectionListenerAdded) {
  window.deepfakeDetectionListenerAdded = true;
  
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Content script received message:', message.action);
    
    if (message.action === 'ping') {
      console.log('Ping received - responding ready');
      sendResponse({ success: true, status: 'ready' });
      return false; // Synchronous response
    } 
    
    if (message.action === 'startDetection') {
      console.log('Starting detection with interval:', message.interval);
      const interval = message.interval || 1000;
      startDetection(interval);
      sendResponse({ success: true });
      return false; // Synchronous response
    } 
    
    if (message.action === 'stopDetection') {
      console.log('Stopping detection');
      stopDetection();
      sendResponse({ success: true });
      return false; // Synchronous response
    }
    
    return false; // Synchronous response
  });

  // Handle overlay messages
  window.addEventListener('message', (event) => {
    if (event.data.type === 'overlayClose' || event.data.type === 'overlayStop') {
      stopDetection();
    }
  });

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    stopDetection();
  });
  
  console.log('Deepfake Detection: Message listeners registered');
}
