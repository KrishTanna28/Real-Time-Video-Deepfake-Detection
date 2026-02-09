# Browser Extension Files

This directory contains all the files needed for the browser extension.

## Files Overview

### Core Files
- **manifest.json** - Extension configuration and permissions
- **background.js** - Service worker for managing tab capture
- **content.js** - Captures tab content and injects overlay
- **popup.html/css/js** - Extension popup control panel

### Overlay Files
- **overlay.html** - Right-side overlay UI
- **overlay.css** - Overlay styling
- **overlay-script.js** - Overlay functionality

### Icons
- **icons/** - Extension icons (16x16, 48x48, 128x128)

## Installation

1. Open Chrome/Edge browser
2. Navigate to `chrome://extensions/` or `edge://extensions/`
3. Enable "Developer mode" (toggle in top-right)
4. Click "Load unpacked"
5. Select this `extension` folder
6. Extension will appear in your toolbar

## Usage

1. Make sure backend server is running (`python backend_server.py`)
2. Click the extension icon in your browser toolbar
3. Click "Start Detection" button
4. Navigate to any page with video content
5. Watch the overlay appear with real-time results!

## Configuration

Settings can be adjusted in the extension popup:
- **Backend URL**: Default `http://localhost:5000`
- **Capture Interval**: Default `1000ms` (1 second)

## Troubleshooting

If the extension doesn't work:
1. Check that backend server is running
2. Reload the extension in `chrome://extensions/`
3. Refresh the webpage with video content
4. Check browser console for errors (F12)

For detailed help, see: [EXTENSION_GUIDE.md](../EXTENSION_GUIDE.md)
