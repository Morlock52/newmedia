# API Documentation

## WebSocket API

The Holographic Media Dashboard communicates with media servers through WebSocket connections for real-time updates.

### Connection

Connect to the WebSocket server:

```javascript
const ws = new WebSocket('ws://localhost:8080');

ws.onopen = () => {
    // Send handshake
    ws.send(JSON.stringify({
        type: 'handshake',
        data: { 
            clientType: 'holographic-dashboard',
            version: '2.0.0'
        }
    }));
};
```

### Message Types

#### Client → Server Messages

##### Handshake
```javascript
{
    type: 'handshake',
    data: {
        clientType: 'holographic-dashboard',
        version: '2.0.0',
        capabilities: ['3d-rendering', 'audio-visualization']
    }
}
```

##### Request Statistics
```javascript
{
    type: 'request-stats',
    timestamp: Date.now(),
    data: {
        categories: ['system', 'media', 'users', 'bandwidth']
    }
}
```

##### Media Action
```javascript
{
    type: 'media-action',
    data: {
        action: 'play' | 'pause' | 'stop' | 'queue',
        mediaId: 'movie-123',
        userId: 'user-456'
    }
}
```

#### Server → Client Messages

##### Statistics Update
```javascript
{
    type: 'stats-update',
    timestamp: Date.now(),
    data: {
        system: {
            cpu: 45.2,          // CPU usage percentage
            memory: 68.5,       // Memory usage percentage
            gpu: 32.1,          // GPU usage percentage
            temperature: 65     // System temperature (°C)
        },
        media: {
            totalItems: 2847,   // Total media items
            totalSize: 47.3,    // Total size in TB
            recentlyAdded: 12,  // Items added in last 24h
            popularGenres: ['Action', 'Sci-Fi', 'Drama']
        },
        users: {
            active: 12,         // Currently active users
            total: 156,         // Total registered users
            concurrent: 8       // Concurrent streams
        },
        bandwidth: {
            current: 450,       // Current usage in Mbps
            peak: 1200,         // Peak usage today
            average: 320        // 24h average
        }
    }
}
```

##### Activity Notification
```javascript
{
    type: 'activity',
    timestamp: Date.now(),
    data: {
        id: 'activity-789',
        icon: '🎬',
        title: 'New movie added',
        description: 'Blade Runner 2049 (2017)',
        category: 'media',
        priority: 'low' | 'medium' | 'high',
        userId: 'user-123'
    }
}
```

##### Media Library Update
```javascript
{
    type: 'media-update',
    data: {
        action: 'added' | 'removed' | 'updated',
        items: [
            {
                id: 'movie-789',
                type: 'movie',
                title: 'Blade Runner 2049',
                year: 2017,
                genre: ['Sci-Fi', 'Drama'],
                rating: 8.0,
                duration: 164,
                size: 12.5,        // Size in GB
                quality: '4K UHD',
                thumbnail: '/thumbs/blade-runner-2049.jpg',
                metadata: {
                    director: 'Denis Villeneuve',
                    cast: ['Ryan Gosling', 'Harrison Ford'],
                    language: 'English',
                    subtitles: ['English', 'Spanish', 'French']
                }
            }
        ]
    }
}
```

##### User Activity
```javascript
{
    type: 'user-activity',
    data: {
        userId: 'user-123',
        action: 'login' | 'logout' | 'stream-start' | 'stream-end',
        details: {
            mediaId: 'movie-456',
            device: 'Samsung TV',
            location: 'Living Room',
            quality: '4K'
        }
    }
}
```

## JavaScript API

### Core Classes

#### HolographicScene

Main 3D scene manager.

```javascript
const scene = new HolographicScene(container, config);

// Methods
scene.init()                    // Initialize the scene
scene.render()                  // Render one frame
scene.resize(width, height)     // Handle window resize
scene.dispose()                 // Clean up resources

// Properties
scene.camera                    // Three.js camera
scene.renderer                  // Three.js renderer
scene.mediaCards               // Array of media card objects
```

#### MediaCard

Individual 3D media card.

```javascript
const card = new MediaCard(mediaData, config);

// Methods
card.show()                     // Animate card into view
card.hide()                     // Animate card out of view
card.update(newData)           // Update card data
card.setHovered(isHovered)     // Set hover state

// Properties
card.mesh                       // Three.js mesh object
card.data                       // Media metadata
card.position                   // 3D position
```

#### ParticleSystem

Particle effects manager.

```javascript
const particles = new ParticleSystem(scene, config);

// Methods
particles.start()               // Start particle animation
particles.stop()                // Stop particle animation
particles.setDataFlow(intensity) // Control data flow visualization
particles.updateColor(color)    // Change particle colors
```

#### AudioVisualizer

Real-time audio visualization.

```javascript
const visualizer = new AudioVisualizer(audioContext, scene);

// Methods
visualizer.connect(audioSource) // Connect audio source
visualizer.start()              // Start visualization
visualizer.stop()               // Stop visualization
visualizer.setStyle(style)      // Change visualization style
```

### Configuration

#### Global Config

```javascript
const CONFIG = {
    // WebSocket settings
    websocket: {
        url: 'ws://localhost:8080',
        reconnectInterval: 5000,
        maxReconnectAttempts: 10
    },
    
    // Visual settings
    holographic: {
        glowIntensity: 1.0,
        scanlineSpeed: 0.001,
        chromaticAberration: 0.002,
        bloomStrength: 1.5
    },
    
    // Particle system
    particles: {
        count: 1000,
        speed: 0.5,
        size: 2.0,
        color: 0x00ffff
    },
    
    // Performance
    performance: {
        shadowsEnabled: true,
        antialias: true,
        adaptiveQuality: true,
        targetFPS: 60
    },
    
    // Media cards
    mediaCards: {
        spacing: 200,
        hoverScale: 1.1,
        animationSpeed: 0.3
    },
    
    // Audio visualizer
    audio: {
        fftSize: 256,
        smoothingTimeConstant: 0.8,
        barCount: 64,
        barHeight: 100
    }
};
```

### Events

#### Custom Events

The dashboard emits custom events for integration:

```javascript
// Listen for scene initialization
document.addEventListener('holographic:scene:ready', (event) => {
    console.log('Scene ready:', event.detail.scene);
});

// Listen for media card interactions
document.addEventListener('holographic:card:click', (event) => {
    console.log('Card clicked:', event.detail.mediaData);
});

// Listen for performance warnings
document.addEventListener('holographic:performance:warning', (event) => {
    console.log('Performance issue:', event.detail.fps);
});

// Listen for WebSocket events
document.addEventListener('holographic:websocket:connected', () => {
    console.log('Connected to media server');
});

document.addEventListener('holographic:websocket:disconnected', () => {
    console.log('Disconnected from media server');
});
```

### Utilities

#### Performance Monitor

```javascript
const monitor = new PerformanceMonitor();

monitor.startFrame();
// ... render operations ...
monitor.endFrame();

const stats = monitor.getStats();
// { fps: 60, frameTime: 16.67, memoryUsage: 45.2 }
```

#### Device Detection

```javascript
const device = DeviceInfo.detect();
// {
//     isMobile: false,
//     hasWebGL2: true,
//     maxTextureSize: 4096,
//     performanceTier: 'high'
// }
```

## Error Handling

### Error Types

- `WebGLError`: WebGL context issues
- `WebSocketError`: Connection problems
- `MediaError`: Media loading failures
- `PerformanceError`: FPS drops or memory issues

### Error Events

```javascript
document.addEventListener('holographic:error', (event) => {
    const { type, message, details } = event.detail;
    
    switch (type) {
        case 'webgl':
            // Handle WebGL errors
            break;
        case 'websocket':
            // Handle connection errors
            break;
        case 'media':
            // Handle media loading errors
            break;
    }
});
```

## Integration Examples

### Custom Media Source

```javascript
// Connect to custom media server
const customWebSocket = new WebSocket('ws://my-server:8080');

customWebSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Update dashboard with custom data
    if (data.type === 'custom-stats') {
        HolographicDashboard.updateStats(data.stats);
    }
};
```

### Plugin System

```javascript
// Register a custom plugin
HolographicDashboard.registerPlugin('myPlugin', {
    init(config) {
        // Plugin initialization
    },
    
    update(deltaTime) {
        // Called each frame
    },
    
    dispose() {
        // Cleanup
    }
});
```

### Theme Customization

```javascript
// Apply custom theme
HolographicDashboard.setTheme({
    primaryColor: 0xff6b00,    // Orange
    secondaryColor: 0x00ff6b,  // Green
    backgroundColor: 0x1a1a2e, // Dark blue
    glowIntensity: 1.2
});
```