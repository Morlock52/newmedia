# AR/VR Immersive Media Platform 2.0

A production-ready WebXR platform featuring real hand tracking, spatial video playback, mixed reality, and haptic feedback. Fully optimized for Apple Vision Pro and Meta Quest 3.

## 🚀 Features

### ✅ Real WebXR Implementation
- **Native WebXR Device API** - No mocks, real WebXR sessions
- **Cross-platform support** - Works on Vision Pro, Quest 3, and all WebXR devices
- **Polyfill included** - Fallback for older browsers

### 👋 Advanced Hand Tracking
- **Full skeletal tracking** - All 25 joints per hand tracked in real-time
- **Gesture recognition** - Pinch, point, grab, peace, thumbs up, and more
- **Vision Pro support** - Transient-pointer eye + pinch interaction
- **Physics-based interaction** - Natural hand collisions and haptics

### 📹 Spatial Video Player
- **Multiple formats** - Side-by-side, over-under, MV-HEVC (Vision Pro)
- **180°/360° support** - Immersive video experiences
- **Adaptive quality** - Optimized for each platform
- **Cinema environments** - Multiple viewing modes

### 🌍 Mixed Reality
- **Passthrough mode** - Real world view on Quest 3
- **Plane detection** - Surface tracking and anchoring
- **Occlusion handling** - Virtual objects behind real ones
- **Collaborative spaces** - Multi-user support ready

### 🤝 Haptic Feedback
- **Controller vibration** - Contextual feedback patterns
- **Texture simulation** - Different materials feel different
- **Gesture confirmation** - Haptic response for interactions

## 📱 Device Support

### Apple Vision Pro
- ✅ Transient-pointer input (eye + pinch)
- ✅ High-resolution displays optimized
- ✅ Spatial video playback (MV-HEVC ready)
- ⏳ AR mode (waiting for Safari support)

### Meta Quest 3
- ✅ Full hand tracking (all gestures)
- ✅ Mixed reality passthrough
- ✅ Plane and mesh detection
- ✅ Advanced haptic feedback

### Other Devices
- ✅ Pico 4/4 Pro
- ✅ Magic Leap 2
- ✅ HoloLens 2
- ✅ Desktop VR (Vive, Index, etc.)

## 🛠️ Tech Stack

- **Three.js** - 3D graphics and rendering
- **WebXR Device API** - Native XR support
- **ES Modules** - Modern JavaScript
- **Web Audio API** - Spatial audio
- **WebRTC** - Future multiplayer support

## 🚦 Getting Started

### Prerequisites
- Node.js 18+ (for development server)
- HTTPS required for WebXR (localhost is OK)
- WebXR-compatible browser

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd ar-vr-media

# Install dependencies
npm install

# Generate SSL certificate for local HTTPS
npm run generate-cert

# Start development server
npm run dev
```

### Basic Usage

1. Open `https://localhost:8080` in your WebXR browser
2. Click "Enter VR" or "Enter AR" button
3. Select a demo from the navigation
4. Use hand gestures or controllers to interact

## 🎮 Controls

### Vision Pro
- **Look** at objects to target them
- **Pinch** fingers to select/interact
- **Look + Pinch** for all interactions

### Quest 3
- **Point** with index finger
- **Pinch** thumb and index to grab
- **Open palm** to release
- **Peace sign** for menu
- **Thumbs up** to confirm

### Controllers
- **Trigger** - Select/interact
- **Grip** - Grab objects
- **Thumbstick** - Navigate
- **Menu button** - Open menu

## 🏗️ Architecture

```
ar-vr-media/
├── webxr/
│   ├── core/
│   │   ├── webxr-manager-v2.js    # Core WebXR management
│   │   └── app-v2.js              # Main application
│   ├── hand-tracking/
│   │   └── real-hand-tracking.js  # Hand tracking implementation
│   ├── spatial-video/
│   │   └── real-spatial-video.js  # Video player system
│   └── mixed-reality/
│       └── real-mixed-reality.js  # AR/MR features
├── assets/
│   └── 3d-models.js              # Procedural 3D models
├── styles/
│   └── main-v2.css               # Enhanced styles
└── index.html                    # Entry point
```

## 🔧 API Examples

### Starting a VR Session
```javascript
// Simple VR session
await webXRManager.startVR();

// With hand tracking
await webXRManager.startVR([], ['hand-tracking']);
```

### Hand Tracking
```javascript
// Get hand position
const handData = handTracking.getHandData('left');
const indexTip = handData.joints.get('index-finger-tip');

// Add interactable object
handTracking.addInteractableObject(mesh);

// Listen for gestures
window.addEventListener('hand-gesture-start', (e) => {
    console.log(`Gesture: ${e.detail.gesture.type}`);
});
```

### Spatial Video
```javascript
// Load spatial video
await spatialVideoPlayer.loadVideo({
    url: 'video.mp4',
    format: 'side-by-side',
    screen: 'cinema'
});

// Control playback
spatialVideoPlayer.play();
spatialVideoPlayer.setVolume(0.8);
```

### Mixed Reality
```javascript
// Enable passthrough
mixedReality.enablePassthrough();

// Create anchor
const anchorId = await mixedReality.createAnchor(position);

// Place object on surface
mixedReality.placeObject(mesh, position, {
    snapToPlane: true,
    createAnchor: true
});
```

## 🎨 Customization

### Adding New Gestures
```javascript
// In gesture recognizer
gestures.set('custom_gesture', {
    check: (hand) => {
        // Your gesture logic
        return { detected: true, confidence: 0.9 };
    },
    priority: 1
});
```

### Creating Interactive Objects
```javascript
const button = ModelGenerator.createInteractiveButton({
    width: 0.3,
    height: 0.1,
    color: 0x4a90e2,
    text: 'Click Me'
});

// Add interaction
button.addEventListener('select-start', () => {
    console.log('Button pressed!');
});
```

## 🐛 Troubleshooting

### WebXR Not Available
- Ensure HTTPS is enabled (required for WebXR)
- Check browser compatibility
- Enable WebXR flags if needed (Safari)

### Hand Tracking Not Working
- Grant camera permissions
- Ensure good lighting
- Check if device supports hand tracking

### Performance Issues
- Reduce polygon count in scenes
- Lower texture resolutions
- Disable unused features
- Check frame timing in stats

## 📚 Resources

- [WebXR Device API](https://www.w3.org/TR/webxr/)
- [Three.js Documentation](https://threejs.org/docs/)
- [Apple Vision Pro WebXR](https://webkit.org/blog/15162/introducing-natural-input-for-webxr-in-apple-vision-pro/)
- [Meta Quest WebXR](https://developers.meta.com/horizon/documentation/web/webxr-mixed-reality/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test on real XR devices
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Three.js team for the excellent 3D library
- W3C Immersive Web Working Group
- WebXR community for standards development
- Apple and Meta for device support

---

Built with ❤️ for the spatial computing future