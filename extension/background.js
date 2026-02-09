// Background service worker for managing tab capture and communication
let activeDetectionTabId = null;

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startDetection') {
    handleStartDetection(message.tabId, sendResponse);
    return true; // Keep channel open for async response
  } else if (message.action === 'stopDetection') {
    handleStopDetection(sendResponse);
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
      
      const response = await fetch(`${backendUrl}/health`, {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error('Backend not responding');
      }
      
      console.log('Backend health check passed');
    } catch (error) {
      console.error('Backend health check failed:', error);
      sendResponse({ 
        success: false, 
        error: 'Backend server not available. Please start the backend server first.' 
      });
      return;
    }

    // Content script is auto-injected via manifest
    // Wait a bit to ensure it's ready
    await new Promise(resolve => setTimeout(resolve, 300));

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
