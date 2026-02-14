// Script for overlay.html to handle updates and interactions
(function() {
  const statusBadge = document.getElementById('overlay-status');
  const modeEl = document.getElementById('overlay-mode');
  const classificationEl = document.getElementById('overlay-classification');
  const confidenceEl = document.getElementById('overlay-confidence');
  const temporalEl = document.getElementById('overlay-temporal');
  const stabilityEl = document.getElementById('overlay-stability');
  const framesEl = document.getElementById('overlay-frames');
  const speedEl = document.getElementById('overlay-speed');
  const closeBtn = document.getElementById('overlay-close');
  const stopBtn = document.getElementById('overlay-stop');

  // Handle close button
  closeBtn.addEventListener('click', () => {
    resetDisplay();
    window.parent.postMessage({ type: 'overlayClose' }, '*');
  });

  // Handle stop button
  stopBtn.addEventListener('click', () => {
    resetDisplay();
    window.parent.postMessage({ type: 'overlayStop' }, '*');
  });

  // Listen for updates from content script
  window.addEventListener('message', (event) => {
    if (event.data.type === 'updateResults') {
      updateDisplay(event.data.data);
    } else if (event.data.type === 'resetDisplay') {
      resetDisplay();
    }
  });

  function updateDisplay(data) {
    if (!data) return;

    // Handle error/disconnected state - don't show stale "REAL" data
    if (data.status === 'error') {
      statusBadge.className = 'status-badge error';
      statusBadge.querySelector('.status-text').textContent = 'Backend Disconnected';
      classificationEl.textContent = 'NO CONNECTION';
      classificationEl.className = 'value error';
      confidenceEl.textContent = '-';
      temporalEl.textContent = '-';
      stabilityEl.textContent = '-';
      speedEl.textContent = '-';
      modeEl.textContent = '-';
      modeEl.className = 'value';
      return;
    }

    // Update status badge
    if (data.status) {
      statusBadge.className = 'status-badge ' + data.status;
      if (data.status === 'analyzing') {
        statusBadge.querySelector('.status-text').textContent = 'Analyzing...';
      }
    }

    // Update analysis mode
    if (data.analysis_mode) {
      const modeLabels = {
        'face+frame': 'Face + Frame',
        'frame_only': 'Frame Only',
      };
      modeEl.textContent = modeLabels[data.analysis_mode] || data.analysis_mode;
      modeEl.className = 'value ' + (data.analysis_mode === 'frame_only' ? 'frame-mode' : 'face-mode');
    }

    // Update classification
    if (data.confidence_level) {
      const classification = data.confidence_level;
      
      if (classification === 'UNCERTAIN') {
        // Still collecting frames — show analyzing
        classificationEl.textContent = 'ANALYZING';
        classificationEl.className = 'value';
        statusBadge.className = 'status-badge analyzing';
        statusBadge.querySelector('.status-text').textContent = 'Analyzing...';
      } else {
        classificationEl.textContent = classification;
        classificationEl.className = 'value ' + classification.toLowerCase().replace('_', '-');

        if (classification === 'FAKE' || classification === 'HIGH_FAKE') {
          statusBadge.className = 'status-badge fake';
          statusBadge.querySelector('.status-text').textContent = 'Deepfake Detected!';
        } else if (classification === 'REAL' || classification === 'HIGH_REAL') {
          statusBadge.className = 'status-badge real';
          statusBadge.querySelector('.status-text').textContent = 'Authentic Video';
        } else {
          statusBadge.className = 'status-badge analyzing';
          statusBadge.querySelector('.status-text').textContent = 'Analyzing...';
        }
      }
    }

    // Update confidence
    if (data.fake_probability !== undefined) {
      const confidence = (data.fake_probability * 100).toFixed(1);
      confidenceEl.textContent = confidence + '%';
    }

    // Update temporal average
    if (data.temporal_average !== undefined) {
      const temporal = (data.temporal_average * 100).toFixed(1);
      temporalEl.textContent = temporal + '%';
    }

    // Update stability score
    if (data.stability_score !== undefined) {
      const stability = (data.stability_score * 100).toFixed(1);
      stabilityEl.textContent = stability + '%';
    }

    // Update frames count
    if (data.frame_count !== undefined) {
      framesEl.textContent = data.frame_count;
    }

    // Update processing speed
    if (data.processing_time_ms !== undefined) {
      speedEl.textContent = data.processing_time_ms + 'ms';
    }
  }

  // Reset display function
  function resetDisplay() {
    statusBadge.className = 'status-badge analyzing';
    statusBadge.querySelector('.status-text').textContent = 'Analyzing...';

    modeEl.textContent = '-';
    modeEl.className = 'value';
    classificationEl.textContent = 'ANALYZING';
    classificationEl.className = 'value';
    confidenceEl.textContent = '-';
    temporalEl.textContent = '-';
    stabilityEl.textContent = '-';
    framesEl.textContent = '0';
    speedEl.textContent = '-';
  }
})();
