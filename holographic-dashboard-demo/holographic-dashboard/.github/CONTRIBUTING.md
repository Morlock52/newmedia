# 🤝 Contributing to Holographic Media Dashboard

Welcome to the Holographic Media Dashboard project! We're excited that you want to contribute. This guide will help you get started with contributing to our next-generation 3D holographic media server dashboard.

## 🌟 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [How to Contribute](#-how-to-contribute)
- [Code Guidelines](#-code-guidelines)
- [Testing](#-testing)
- [Performance Guidelines](#-performance-guidelines)
- [Documentation](#-documentation)
- [Community](#-community)

## 🛡️ Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to `conduct@holographic-dashboard.dev`.

## 🚀 Getting Started

### 🎯 What Can You Contribute?

We welcome contributions in many forms:

**🎨 Visual & 3D Graphics:**
- New WebGL shaders and visual effects
- Three.js scene improvements
- Particle system enhancements
- Post-processing effects
- Mobile optimizations

**🎵 Audio Visualization:**
- New visualizer modes
- Audio analysis improvements
- Real-time frequency processing
- Cross-browser audio compatibility

**🖱️ User Interface:**
- UI/UX improvements
- Responsive design enhancements
- Accessibility features
- Touch and gesture controls

**⚡ Performance:**
- Rendering optimizations
- Memory usage improvements
- Loading time reductions
- Device compatibility

**🔌 Integration:**
- WebSocket enhancements
- API improvements
- New data sources
- Server integrations

**📚 Documentation:**
- Setup guides
- API documentation
- Tutorial content
- Troubleshooting guides

### 🎓 Skill Levels Welcome

- **🟢 Beginners**: Documentation, testing, simple UI fixes
- **🟡 Intermediate**: Feature development, bug fixes, performance improvements
- **🔴 Advanced**: Architecture changes, complex 3D graphics, WebGL shaders

## 🛠️ Development Setup

### 📋 Prerequisites

- **Node.js**: Version 14.0 or higher
- **npm**: Version 6.0 or higher
- **Git**: Latest version
- **Modern Browser**: Chrome, Firefox, Safari, or Edge with WebGL 2.0 support

### 🔧 Local Development

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub first
   git clone https://github.com/YOUR_USERNAME/holographic-media-dashboard.git
   cd holographic-media-dashboard
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm start
   # or
   npm run dev
   ```

4. **Open in Browser**
   ```
   http://localhost:9999
   ```

### 🌿 Branch Strategy

- **`main`**: Production-ready code
- **`develop`**: Integration branch for new features
- **`feature/feature-name`**: Individual feature development
- **`fix/issue-description`**: Bug fixes
- **`docs/section-name`**: Documentation updates

```bash
# Create a feature branch
git checkout develop
git pull origin develop
git checkout -b feature/awesome-new-feature

# Create a bug fix branch
git checkout main
git pull origin main
git checkout -b fix/audio-visualizer-bug
```

## 🎯 How to Contribute

### 🐛 Reporting Issues

Before creating an issue:
1. **Search existing issues** to avoid duplicates
2. **Check the troubleshooting guide** in docs
3. **Test in multiple browsers** if possible
4. **Gather system information** (OS, browser, GPU)

Use our issue templates:
- 🐛 **Bug Report**: For functional issues
- ✨ **Feature Request**: For new features
- ⚡ **Performance Issue**: For performance problems
- 📚 **Documentation**: For doc improvements

### 🔀 Pull Request Process

1. **Create an Issue First** (for non-trivial changes)
2. **Fork and Create Branch** from `develop`
3. **Make Your Changes** following our guidelines
4. **Test Thoroughly** on multiple devices/browsers
5. **Update Documentation** if needed
6. **Submit Pull Request** using our template

### 📝 Pull Request Checklist

- [ ] 🎯 **Linked to Issue**: References the related issue number
- [ ] 🧪 **Tested**: Works on desktop and mobile
- [ ] 📱 **Cross-browser**: Tested in Chrome, Firefox, Safari
- [ ] ⚡ **Performance**: No significant performance degradation
- [ ] 📚 **Documentation**: Updated if needed
- [ ] 🎨 **Code Style**: Follows project conventions
- [ ] 🔍 **Self-reviewed**: Code has been self-reviewed
- [ ] 🌐 **WebGL**: WebGL functionality verified (if applicable)

## 📝 Code Guidelines

### 🎨 JavaScript Style

```javascript
// ✅ Good: Use modern ES6+ syntax
const createHolographicMaterial = (config) => {
    const { color, opacity, glowIntensity } = config;
    
    return new THREE.ShaderMaterial({
        uniforms: {
            uColor: { value: new THREE.Color(color) },
            uOpacity: { value: opacity },
            uGlow: { value: glowIntensity }
        },
        vertexShader: HolographicShaders.vertex,
        fragmentShader: HolographicShaders.fragment,
        transparent: true
    });
};

// ✅ Good: Clear naming and documentation
/**
 * Updates the particle system based on audio frequency data
 * @param {Float32Array} frequencyData - Audio frequency analysis
 * @param {number} sensitivity - Sensitivity multiplier (0-2)
 */
const updateAudioParticles = (frequencyData, sensitivity = 1.0) => {
    // Implementation...
};

// ❌ Avoid: Unclear naming and missing documentation
const upd = (data, s) => {
    // What does this do?
};
```

### 🎭 WebGL and Three.js Guidelines

```javascript
// ✅ Good: Proper resource management
const geometry = new THREE.BufferGeometry();
const material = new THREE.ShaderMaterial(/* ... */);
const mesh = new THREE.Mesh(geometry, material);

// Don't forget cleanup
const dispose = () => {
    geometry.dispose();
    material.dispose();
    // Dispose textures, render targets, etc.
};

// ✅ Good: Performance-conscious shader writing
const fragmentShader = `
    precision mediump float;
    
    uniform float uTime;
    uniform vec3 uColor;
    
    varying vec2 vUv;
    
    void main() {
        // Optimize for mobile devices
        vec3 color = uColor * (0.5 + 0.5 * sin(uTime + vUv.x * 10.0));
        gl_FragColor = vec4(color, 1.0);
    }
`;
```

### 🎨 CSS Guidelines

```css
/* ✅ Good: BEM methodology for clarity */
.holographic-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.holographic-card__title {
    font-size: 1.2rem;
    color: #00ffff;
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}

.holographic-card--active {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 255, 255, 0.3);
}

/* ✅ Good: Mobile-first responsive design */
@media (max-width: 768px) {
    .holographic-card {
        backdrop-filter: blur(5px); /* Reduce for performance */
    }
}
```

### 🏗️ File Organization

```
js/
├── core/               # Core functionality
│   ├── renderer.js     # Main Three.js renderer
│   ├── scene.js        # Scene management
│   └── camera.js       # Camera controls
├── effects/            # Visual effects
│   ├── particles.js    # Particle systems
│   ├── shaders.js      # Shader definitions
│   └── post-processing.js
├── ui/                 # User interface
│   ├── controls.js     # UI controls
│   ├── hud.js         # HUD elements
│   └── responsive.js   # Responsive utilities
├── audio/              # Audio processing
│   ├── visualizer.js   # Audio visualization
│   └── analyzer.js     # Frequency analysis
└── utils/              # Utilities
    ├── performance.js  # Performance monitoring
    ├── device.js       # Device detection
    └── math.js         # Math utilities
```

## 🧪 Testing

### 🔍 Manual Testing

**Desktop Testing:**
```bash
# Start the development server
npm start

# Test in multiple browsers
# - Chrome (recommended for development)
# - Firefox
# - Safari
# - Edge
```

**Mobile Testing:**
- Test on actual devices when possible
- Use Chrome DevTools device emulation
- Test touch interactions
- Verify performance on lower-end devices

**WebGL Testing:**
- Test on different GPU vendors (NVIDIA, AMD, Intel)
- Verify fallback behavior for unsupported features
- Check WebGL context loss handling

### ⚡ Performance Testing

```javascript
// Add performance monitoring to your features
const startTime = performance.now();

// Your code here

const endTime = performance.now();
console.log(`Operation took ${endTime - startTime} milliseconds`);

// Monitor memory usage
const memInfo = performance.memory;
console.log(`Used: ${memInfo.usedJSHeapSize / 1048576} MB`);
```

### 🌐 Cross-browser Testing

| Browser | Version | WebGL | Status |
|---------|---------|-------|--------|
| Chrome  | Latest  | 2.0   | ✅ Primary |
| Firefox | Latest  | 2.0   | ✅ Supported |
| Safari  | Latest  | 2.0   | ✅ Supported |
| Edge    | Latest  | 2.0   | ✅ Supported |

## ⚡ Performance Guidelines

### 🎯 Performance Targets

- **Desktop**: 60 FPS with all effects enabled
- **Mobile**: 30 FPS with optimized settings
- **Load Time**: < 3 seconds on broadband
- **Memory**: < 200MB baseline usage

### 🚀 Optimization Tips

**JavaScript Optimization:**
```javascript
// ✅ Good: Object pooling for frequently created objects
class ParticlePool {
    constructor(size) {
        this.pool = [];
        for (let i = 0; i < size; i++) {
            this.pool.push(new Particle());
        }
    }
    
    acquire() {
        return this.pool.pop() || new Particle();
    }
    
    release(particle) {
        particle.reset();
        this.pool.push(particle);
    }
}

// ✅ Good: Efficient animation loops
const animate = () => {
    requestAnimationFrame(animate);
    
    // Only update what changed
    if (needsUpdate) {
        updateScene();
        needsUpdate = false;
    }
    
    renderer.render(scene, camera);
};
```

**WebGL Optimization:**
```javascript
// ✅ Good: Batch similar operations
const updateInstancedMesh = () => {
    // Update all instances at once
    for (let i = 0; i < instanceCount; i++) {
        matrix.setPosition(positions[i]);
        instancedMesh.setMatrixAt(i, matrix);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
};

// ✅ Good: Use appropriate texture sizes
const createOptimizedTexture = (size) => {
    // Power of 2 sizes for better performance
    const textureSize = Math.pow(2, Math.ceil(Math.log2(size)));
    return new THREE.Texture(/* ... */);
};
```

## 📚 Documentation

### 📝 Code Documentation

```javascript
/**
 * Creates a holographic material with customizable properties
 * 
 * @param {Object} config - Configuration object
 * @param {number} config.color - Base color (hex)
 * @param {number} config.opacity - Material opacity (0-1)
 * @param {number} config.glowIntensity - Glow effect strength (0-2)
 * @param {boolean} config.animated - Enable animation
 * @returns {THREE.ShaderMaterial} Configured holographic material
 * 
 * @example
 * const material = createHolographicMaterial({
 *     color: 0x00ffff,
 *     opacity: 0.8,
 *     glowIntensity: 1.2,
 *     animated: true
 * });
 */
```

### 📖 README Updates

When adding new features, update the README:
- Add to feature list
- Update configuration examples
- Include usage instructions
- Add screenshots if visual changes

## 🎨 Specific Contribution Areas

### 🎭 3D Graphics and WebGL

**Adding New Shaders:**
```javascript
// Add to js/shaders.js
Shaders.myNewShader = {
    vertexShader: `
        // Vertex shader code
    `,
    fragmentShader: `
        // Fragment shader code  
    `,
    uniforms: {
        // Shader uniforms
    }
};
```

**Performance Considerations:**
- Use `precision mediump float` for mobile compatibility
- Minimize texture lookups in fragments shaders
- Use built-in functions when possible
- Test on various GPUs

### 🎵 Audio Visualization

**Adding New Visualizers:**
```javascript
// Add to js/audio-visualizer.js
class MyVisualizerMode extends VisualizerMode {
    constructor(scene, config) {
        super(scene, config);
        // Initialize your visualizer
    }
    
    update(frequencyData, timeData) {
        // Update visualization based on audio data
    }
    
    dispose() {
        // Clean up resources
    }
}
```

### 📱 Mobile Optimization

**Key Areas:**
- Reduce particle counts on mobile
- Simplify shaders for lower-end GPUs
- Optimize touch interactions
- Implement proper viewport handling

```javascript
// Example mobile detection and optimization
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

const config = {
    particles: isMobile ? 500 : 2000,
    shadows: !isMobile,
    postProcessing: !isMobile
};
```

## 🏷️ Issue Labels

We use labels to categorize issues and PRs:

**Type Labels:**
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `performance` - Performance related
- `question` - Further information needed

**Component Labels:**
- `3d-graphics` - WebGL/Three.js related
- `ui-ux` - User interface and experience
- `audio` - Audio visualization
- `mobile` - Mobile compatibility
- `websocket` - WebSocket functionality

**Priority Labels:**
- `high-priority` - Critical issues
- `medium-priority` - Important improvements
- `low-priority` - Nice to have features

**Status Labels:**
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `in-progress` - Currently being worked on

## 🌍 Community

### 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

### 🎉 Recognition

We recognize contributors in several ways:
- 📝 Contributors list in README
- 🏷️ Release notes mentions
- 🌟 Special thanks in major releases
- 🎖️ Community showcase features

### 🚀 Becoming a Maintainer

Regular contributors may be invited to become maintainers. Maintainers help with:
- Code review and merging PRs
- Issue triage and labeling
- Release planning and management
- Community management and support

## 📞 Getting Help

**Need Help Getting Started?**
- 📖 Check our [documentation](docs/)
- 🐛 Look at `good first issue` labeled issues
- 💬 Ask questions in GitHub Discussions
- 📧 Email us at `help@holographic-dashboard.dev`

**Found a Security Issue?**
- 🔒 See our [Security Policy](.github/SECURITY.md)
- 📧 Email `security@holographic-dashboard.dev`

## 🙏 Thank You

Thank you for considering contributing to the Holographic Media Dashboard! Every contribution, no matter how small, helps make this project better for everyone.

**🎬 Happy coding, and welcome to the future of media dashboards! ✨**

---

*This contributing guide is living document. Feel free to suggest improvements!*

*Last updated: 2025-01-31*