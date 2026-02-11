// Background service worker for managing tab capture and communication
let activeDetectionTabId = null;

// Convert data URL to Blob
function dataURLtoBlob(dataURL) {
  const parts = dataURL.split(',');
  const mime = parts[0].match(/:(.*?);/)[1];
  const binary = atob(parts[1]);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    array[i] = binary.charCodeAt(i);
  }
  return new Blob([array], { type: mime });
}

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startDetection') {
    handleStartDetection(message.tabId, sendResponse);
    return true; // Keep channel open for async response
  } else if (message.action === 'stopDetection') {
    handleStopDetection(sendResponse);
    return true;
  } else if (message.action === 'analyzeFrame') {
    handleAnalyzeFrame(message.imageData, sendResponse);
    return true; // Keep channel open for async response
  } else if (message.action === 'resetBackend') {
    handleResetBackend(sendResponse);
    return true;
  } else if (message.action === 'detectionResult') {
    // Forward results to popup (ignore if popup is closed)
    chrome.runtime.sendMessage(message).catch(() => {});
  } else if (message.action === 'detectionError') {
    // Forward errors to popup (ignore if popup is closed)
    chrome.runtime.sendMessage(message).catch(() => {});
  } else if (message.action === 'detectionStopped') {
    activeDetectionTabId = null;
    chrome.storage.local.set({ isDetecting: false });
    chrome.runtime.sendMessage(message).catch(() => {});
  }
});

// Analyze a frame by forwarding to backend
async function handleAnalyzeFrame(imageDataUrl, sendResponse) {
  try {
    const settings = await chrome.storage.local.get(['backendUrl']);
    const backendUrl = settings.backendUrl || 'http://localhost:5000';

    // Convert data URL to Blob
    const blob = dataURLtoBlob(imageDataUrl);

    // Send to backend as multipart form data
    const formData = new FormData();
    formData.append('frame', blob, 'frame.png');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

    const response = await fetch(`${backendUrl}/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Backend returned ${response.status}`);
    }

    const data = await response.json();
    sendResponse(data);
  } catch (error) {
    console.error('Frame analysis failed:', error);
    sendResponse({ error: error.message || 'Analysis failed' });
  }
}

// Reset backend detector state
async function handleResetBackend(sendResponse) {
  try {
    const settings = await chrome.storage.local.get(['backendUrl']);
    const backendUrl = settings.backendUrl || 'http://localhost:5000';
    await fetch(`${backendUrl}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    sendResponse({ success: true });
  } catch (error) {
    console.log('Could not reset backend:', error);
    sendResponse({ success: false, error: error.message });
  }
}

// Start detection on a specific tab
async function handleStartDetection(tabId, sendResponse) {
  try {
    // Check if backend is available
    const settings = await chrome.storage.local.get(['backendUrl', 'captureInterval']);
    const backendUrl = settings.backendUrl || 'http://localhost:5000';
    const captureInterval = settings.captureInterval || 1000;

    // Test backend connection
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      let response;
      try {
        response = await fetch(`${backendUrl}/health`, {
          method: 'GET',
          signal: controller.signal
        });
      } catch (e) {
        // If localhost fails, try 127.0.0.1 (IPv6 vs IPv4 issue)
        if (backendUrl.includes('localhost')) {
          const fallbackUrl = backendUrl.replace('localhost', '127.0.0.1');
          response = await fetch(`${fallbackUrl}/health`, {
            method: 'GET',
            signal: controller.signal
          });
          // Save working URL for future use
          await chrome.storage.local.set({ backendUrl: fallbackUrl });
        } else {
          throw e;
        }
      }
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error('Backend not responding');
      }
      
      console.log('Backend health check passed');
    } catch (error) {
      console.error('Backend health check failed:', error);
      sendResponse({ 
        success: false, 
        error: 'Backend server not available. Please start the backend server first.\n\nDetails: ' + error.message
      });
      return;
    }

    // Content script may not be loaded (SPA navigation, extension reload, etc.)
    // Programmatically inject it to be sure — the script guards against re-declaration
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ['content.js']
      });
      console.log('Content script injected successfully');
    } catch (injectionError) {
      console.warn('Content script injection failed (may already be loaded):', injectionError.message);
    }

    // Give the content script a moment to initialize
    await new Promise(resolve => setTimeout(resolve, 500));

    // Send start message to content script
    chrome.tabs.sendMessage(tabId, {
      action: 'startDetection',
      interval: captureInterval
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('Failed to communicate with content script:', chrome.runtime.lastError);
        sendResponse({ 
          success: false, 
          error: 'Could not establish connection. Refresh the page and try again.' 
        });
      } else if (response && response.success) {
        activeDetectionTabId = tabId;
        chrome.storage.local.set({ isDetecting: true });
        sendResponse({ success: true });
      } else {
        sendResponse({ success: false, error: 'Content script failed to start' });
      }
    });

  } catch (error) {
    console.error('Error starting detection:', error);
    sendResponse({ success: false, error: error.message });
  }
}

// Stop detection
function handleStopDetection(sendResponse) {
  if (activeDetectionTabId) {
    chrome.tabs.sendMessage(activeDetectionTabId, {
      action: 'stopDetection'
    }, (response) => {
      activeDetectionTabId = null;
      chrome.storage.local.set({ isDetecting: false });
      sendResponse({ success: true });
    });
  } else {
    chrome.storage.local.set({ isDetecting: false });
    sendResponse({ success: true });
  }
}

// Clean up when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === activeDetectionTabId) {
    activeDetectionTabId = null;
    chrome.storage.local.set({ isDetecting: false });
  }
});

// Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
  // Open popup (default behavior)
});

console.log('Deepfake Detection Extension: Background service worker loaded');
