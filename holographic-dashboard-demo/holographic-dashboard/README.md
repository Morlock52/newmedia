# 🎬 Holographic Media Dashboard 2025

A futuristic 3D holographic media server dashboard built with Three.js, WebGL, and WebSockets for real-time monitoring and control.

![Holographic Dashboard](screenshot.png)

## ✨ Features

### 🎨 Visual Effects
- **True 3D Holographic Interface** - Floating media cards with depth and parallax
- **WebGL Particle Systems** - Dynamic data flow visualization with thousands of particles
- **Custom Shaders** - Holographic materials, glowing edges, and scanline effects
- **Real-time Audio Visualizer** - Frequency-based 3D bar visualization with reflections
- **Post-processing Effects** - Bloom, chromatic aberration, film grain, and vignette

### 🚀 Interactive Elements
- **3D Media Cards** - Hover effects, click interactions, and smooth animations
- **Gesture Controls** - Orbit camera controls for navigating the 3D space
- **Dynamic UI** - Glass morphism panels with animated borders
- **Real-time Updates** - WebSocket integration for live data streaming

### 📊 Dashboard Features
- **System Statistics** - CPU, GPU, memory, and bandwidth monitoring
- **Media Library** - Browse movies, series, music, and live streams
- **Activity Feed** - Real-time notifications and system events
- **Performance Metrics** - FPS counter and adaptive quality settings

## 🛠️ Technology Stack

- **Three.js r161** - 3D graphics and WebGL rendering
- **GSAP 3.12** - Smooth animations and transitions
- **Socket.io 4.7** - Real-time WebSocket communication
- **Custom WebGL Shaders** - Advanced visual effects
- **Modern CSS** - Glass morphism and holographic styling

## 📦 Installation

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/holographic-media-dashboard.git
cd holographic-media-dashboard
```

2. Install dependencies (for demo server):
```bash
npm install ws
```

3. Start the demo server:
```bash
node demo-server.js
```

4. Open your browser and navigate to:
```
http://localhost:9999
```

## 🎮 Usage

### Navigation
- **Mouse** - Click and drag to rotate the camera
- **Scroll** - Zoom in and out
- **Click Media Cards** - View detailed information

### Controls
- **🎨 Effects Toggle** - Enable/disable post-processing effects
- **✨ Particles Toggle** - Show/hide particle systems
- **🎵 Audio Visualizer** - Toggle music visualization
- **⛶ Fullscreen** - Enter/exit fullscreen mode

### Sections
- **Dashboard** - Overview of all media and system stats
- **Movies** - Browse movie collection
- **TV Series** - View series library
- **Music** - Music collection with audio visualizer
- **Live TV** - Active streams and channels
- **Analytics** - Detailed performance metrics

## 🔧 Configuration

Edit `js/config.js` to customize:

```javascript
CONFIG = {
    // WebSocket connection
    websocket: {
        url: 'ws://localhost:8080',
        reconnectInterval: 5000
    },
    
    // Visual settings
    holographic: {
        glowIntensity: 1.0,
        particleCount: 1000,
        scanlineSpeed: 0.001
    },
    
    // Performance
    performance: {
        shadowsEnabled: true,
        antialias: true,
        adaptiveQuality: true
    }
}
```

## 🚀 Performance Optimization

The dashboard automatically adjusts quality based on device capabilities:

- **High-end GPUs** - Full effects, 2000+ particles, high-res shadows
- **Mid-range** - Moderate effects, 1000 particles, standard shadows
- **Low-end/Mobile** - Reduced effects, 500 particles, no shadows

## 🔌 WebSocket API

The dashboard expects a WebSocket server with the following message types:

### Client → Server
```javascript
{
    type: 'handshake',
    data: { clientType: 'holographic-dashboard' }
}

{
    type: 'request-stats',
    timestamp: Date.now()
}
```

### Server → Client
```javascript
{
    type: 'stats-update',
    data: {
        totalMedia: 2847,
        storageUsed: 47.3,
        activeUsers: 12,
        bandwidth: 450
    }
}

{
    type: 'activity',
    data: {
        icon: '🎬',
        title: 'New movie added',
        description: 'Blade Runner 2049'
    }
}
```

## 🎯 Customization

### Adding Custom Shaders
Create new shaders in `js/shaders.js`:

```javascript
Shaders.myCustomShader = {
    vertexShader: `...`,
    fragmentShader: `...`
}
```

### Creating New Media Types
Add to `js/config.js`:

```javascript
mediaTypes: {
    podcast: { icon: '🎙️', color: 0xFF6B00 }
}
```

### Custom Particle Effects
Modify `js/particles.js` to create new particle behaviors.

## 🌐 Browser Support

- **Chrome 90+** (Recommended)
- **Firefox 88+**
- **Safari 14+**
- **Edge 90+**

Requires WebGL 2.0 support and modern JavaScript features.

## 📱 Mobile Support

The dashboard is optimized for mobile devices with:
- Reduced particle counts
- Simplified shaders
- Touch-friendly controls
- Responsive UI scaling

## 🐛 Troubleshooting

### WebGL Not Supported
Ensure your browser supports WebGL 2.0. Visit [webglreport.com](https://webglreport.com) to check.

### Performance Issues
1. Reduce particle count in config
2. Disable post-processing effects
3. Lower shadow map resolution
4. Enable adaptive quality

### Connection Issues
If WebSocket fails to connect, the dashboard will run in demo mode with simulated data.

## 📄 License

MIT License - feel free to use in your projects!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 🎨 Credits

- Three.js community for amazing WebGL tools
- GSAP for smooth animations
- Inspiration from sci-fi interfaces in movies like Blade Runner, Minority Report, and Iron Man

---

Built with ❤️ for the future of media dashboards