# Troubleshooting Guide

## 🚨 Common Issues

### WebGL Issues

#### WebGL Not Supported
**Symptoms**: Black screen or error message about WebGL
**Solutions**:
1. Check WebGL support at [webglreport.com](https://webglreport.com)
2. Update your graphics drivers
3. Enable hardware acceleration in browser settings
4. Try a different browser (Chrome recommended)

#### WebGL Context Lost
**Symptoms**: Sudden black screen during use, console error "WebGL context lost"
**Solutions**:
```javascript
// This is handled automatically, but you can check:
canvas.addEventListener('webglcontextlost', (event) => {
    console.log('WebGL context lost, attempting recovery...');
    event.preventDefault();
});

canvas.addEventListener('webglcontextrestored', () => {
    console.log('WebGL context restored');
    // Dashboard will automatically reinitialize
});
```

#### Poor WebGL Performance
**Symptoms**: Low FPS, stuttering animations, browser freezing
**Solutions**:
1. Reduce particle count in settings
2. Disable post-processing effects
3. Lower shadow map resolution
4. Enable adaptive quality mode
5. Close other browser tabs
6. Check for background processes

### Connection Issues

#### WebSocket Connection Failed
**Symptoms**: "Connection failed" message, no real-time updates
**Solutions**:
1. Check if the WebSocket server is running
2. Verify the WebSocket URL in config
3. Check firewall settings
4. Try a different port
5. Use WSS (secure WebSocket) for HTTPS sites

**Debug Steps**:
```javascript
// Open browser console and check:
console.log('WebSocket URL:', CONFIG.websocket.url);

// Test connection manually:
const testSocket = new WebSocket('ws://your-server:8080');
testSocket.onopen = () => console.log('Test connection successful');
testSocket.onerror = (error) => console.log('Test connection failed:', error);
```

#### Frequent Disconnections
**Symptoms**: Connection drops every few minutes
**Solutions**:
1. Check network stability
2. Increase `reconnectInterval` in config
3. Implement ping/pong heartbeat
4. Check server timeout settings
5. Use a more stable network connection

### Performance Issues

#### Low Frame Rate
**Symptoms**: Choppy animations, FPS counter shows <30 FPS
**Solutions**:
1. **Automatic**: Enable adaptive quality in settings
2. **Manual adjustments**:
   ```javascript
   // Reduce particle count
   CONFIG.particles.count = 500; // from 1000
   
   // Disable expensive effects
   CONFIG.holographic.bloomEnabled = false;
   CONFIG.performance.shadowsEnabled = false;
   
   // Lower render resolution
   renderer.setPixelRatio(1); // from window.devicePixelRatio
   ```

#### High Memory Usage
**Symptoms**: Browser becomes slow, "Out of memory" errors
**Solutions**:
1. Refresh the page periodically
2. Close other browser tabs
3. Reduce texture resolution
4. Enable garbage collection debugging:
   ```javascript
   // Monitor memory usage
   setInterval(() => {
       if (performance.memory) {
           console.log('Memory:', {
               used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024) + 'MB',
               total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024) + 'MB'
           });
       }
   }, 10000);
   ```

### Audio Issues

#### Audio Visualizer Not Working
**Symptoms**: No audio bars, static visualization
**Solutions**:
1. **Browser permissions**: Allow microphone access if using mic input
2. **Audio source**: Ensure audio is playing
3. **Browser compatibility**: Use Chrome or Firefox for best support
4. **HTTPS requirement**: Audio features require HTTPS on production

**Debug Audio**:
```javascript
// Check if audio context is running
console.log('Audio Context State:', audioContext.state);

// Resume if suspended (required by browser policies)
if (audioContext.state === 'suspended') {
    audioContext.resume().then(() => {
        console.log('Audio context resumed');
    });
}
```

#### No Audio Permission
**Symptoms**: Browser asks for microphone permission, visualizer doesn't work
**Solutions**:
1. Click "Allow" when prompted
2. Check browser settings for microphone permissions
3. For media files, no permission needed
4. Use HTTPS for production deployments

### Mobile Issues

#### Poor Mobile Performance
**Symptoms**: Very low FPS on mobile devices
**Solutions**:
1. **Automatic mobile optimization**:
   ```javascript
   // This is handled automatically, but you can force:
   if (DeviceInfo.isMobile()) {
       CONFIG.particles.count = 200;
       CONFIG.performance.shadowsEnabled = false;
       CONFIG.holographic.bloomEnabled = false;
   }
   ```

#### Touch Controls Not Working
**Symptoms**: Cannot rotate camera or interact with cards
**Solutions**:
1. Ensure touch events are enabled
2. Check for CSS `touch-action` conflicts
3. Verify viewport meta tag is present
4. Test on different mobile browsers

#### Mobile Layout Issues
**Symptoms**: UI elements too small or overlapping
**Solutions**:
1. Check viewport configuration:
   ```html
   <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
   ```
2. Verify responsive CSS is loaded
3. Test with browser developer tools mobile emulation

### Loading Issues

#### Assets Not Loading
**Symptoms**: Missing textures, 404 errors in console
**Solutions**:
1. Check file paths in config
2. Verify server is serving static files
3. Check CORS headers for external assets
4. Ensure assets are included in deployment

#### Slow Loading
**Symptoms**: Long loading times, blank screen for extended periods
**Solutions**:
1. Optimize asset sizes
2. Use CDN for libraries
3. Implement progressive loading
4. Add loading progress indicators

## 🔧 Debug Tools

### Browser Developer Tools

#### Console Commands
```javascript
// Check current performance
window.holographicDashboard.getPerformanceStats()

// Force quality change
window.holographicDashboard.setQuality('low')

// Get WebGL info
window.holographicDashboard.getWebGLInfo()

// Check memory usage
window.holographicDashboard.getMemoryUsage()
```

#### Performance Monitoring
1. Open Developer Tools (F12)
2. Go to Performance tab
3. Record while using the dashboard
4. Look for:
   - Frame rate drops
   - Long tasks
   - Memory leaks
   - Excessive garbage collection

### Three.js Debug Tools

#### Stats Panel
Enable the stats panel to monitor performance:
```javascript
// Add to main.js for debugging
import Stats from 'three/examples/jsm/libs/stats.module.js';

const stats = new Stats();
document.body.appendChild(stats.dom);

// In render loop
function animate() {
    stats.begin();
    
    // ... render code ...
    
    stats.end();
    requestAnimationFrame(animate);
}
```

#### WebGL Inspector
Use [webgl-inspector](https://github.com/benvanik/WebGL-Inspector) for detailed WebGL debugging.

### Network Debug

#### WebSocket Testing
```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8080');
ws.onopen = () => console.log('Connected');
ws.onmessage = (msg) => console.log('Received:', msg.data);
ws.onerror = (error) => console.log('Error:', error);
ws.onclose = (event) => console.log('Closed:', event.code, event.reason);
```

## 📱 Browser-Specific Issues

### Chrome
- **Issue**: Autoplay policy blocks audio
- **Solution**: User interaction required before playing audio
- **Workaround**: Add click handler to start audio context

### Firefox
- **Issue**: WebGL performance slightly lower
- **Solution**: Enable `webgl.force-enabled` in about:config
- **Note**: Generally stable, good fallback option

### Safari
- **Issue**: Limited WebGL 2.0 support
- **Solution**: Use WebGL 1.0 fallbacks
- **Note**: Mobile Safari has strict memory limits

### Edge
- **Issue**: Older versions lack full WebGL 2.0
- **Solution**: Detect capabilities and adjust accordingly
- **Note**: Modern Edge (Chromium) works well

## 🔍 Diagnostic Information

### System Requirements Check
```javascript
function checkSystemRequirements() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    
    if (!gl) {
        console.error('WebGL not supported');
        return false;
    }
    
    const info = {
        renderer: gl.getParameter(gl.RENDERER),
        vendor: gl.getParameter(gl.VENDOR),
        version: gl.getParameter(gl.VERSION),
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS)
    };
    
    console.log('WebGL Info:', info);
    return true;
}
```

### Performance Diagnostics
```javascript
function diagnosePerformance() {
    const info = renderer.info;
    
    console.log('Render Info:', {
        triangles: info.render.triangles,
        calls: info.render.calls,
        points: info.render.points,
        lines: info.render.lines
    });
    
    console.log('Memory Info:', {
        geometries: info.memory.geometries,
        textures: info.memory.textures
    });
    
    if (performance.memory) {
        console.log('JS Memory:', {
            used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024) + 'MB',
            total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024) + 'MB',
            limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024) + 'MB'
        });
    }
}
```

## 🆘 Getting Help

### Before Reporting Issues
1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Test in different browsers
4. Gather diagnostic information

### When Reporting Issues
Include this information:
- Browser and version
- Operating system
- Graphics card model
- Console error messages
- Steps to reproduce
- Expected vs actual behavior

### Support Channels
- **GitHub Issues**: Technical problems and bugs
- **Discussions**: Questions and feature requests
- **Discord**: Real-time community support
- **Stack Overflow**: Use tag `holographic-dashboard`

### Emergency Fixes

#### Complete Reset
If the dashboard is completely broken:
1. Clear browser cache and localStorage
2. Disable all browser extensions
3. Reset configuration to defaults
4. Try incognito/private browsing mode

#### Safe Mode
Add `?safe=true` to URL for minimal mode:
- Disables all effects
- Uses fallback renderer
- Minimal particle count
- No post-processing

This should work on any WebGL-capable device.