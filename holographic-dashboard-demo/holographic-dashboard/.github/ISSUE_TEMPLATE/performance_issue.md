---
name: ⚡ Performance Issue
about: Report performance problems with the holographic dashboard
title: '[PERFORMANCE] '
labels: 'performance'
assignees: ''
---

## ⚡ Performance Issue Description
A clear description of the performance problem you're experiencing.

## 📊 Performance Metrics
**Current Performance:**
- FPS: [e.g. 15fps, 30fps, varies between X-Y]
- Loading Time: [e.g. 5 seconds, 30+ seconds]
- Memory Usage: [e.g. 2GB, increases over time]
- CPU Usage: [e.g. 80%, spikes to 100%]

**Expected Performance:**
- Target FPS: [e.g. 60fps, stable 30fps]
- Expected Loading Time: [e.g. under 3 seconds]
- Expected Resource Usage: [reasonable levels]

## 🖥️ System Information
**Hardware:**
- CPU: [e.g. Intel i7-10700K, Apple M1, AMD Ryzen 5 5600X]
- GPU: [e.g. NVIDIA RTX 3070, Intel Iris Xe, AMD RX 6700 XT]
- RAM: [e.g. 16GB DDR4, 8GB LPDDR4]
- Storage: [e.g. NVMe SSD, HDD, eMMC]

**Software:**
- OS: [e.g. Windows 11, macOS 13.0, Ubuntu 22.04]
- Browser: [e.g. Chrome 119, Firefox 119, Safari 16.0]
- WebGL Version: [Check at webglreport.com]

## 🎨 Dashboard Configuration
**Current Settings:**
- Particle Count: [e.g. 2000, 1000, auto-detected]
- Post-processing Effects: [Enabled/Disabled]
- Shadows: [Enabled/Disabled]
- Antialiasing: [Enabled/Disabled]
- Quality Setting: [High/Medium/Low/Auto]

**Which features are active during the performance issue?**
- [ ] 3D Media Cards
- [ ] Particle Systems
- [ ] Audio Visualizer
- [ ] WebGL Shaders
- [ ] Post-processing Effects
- [ ] Real-time Data Updates
- [ ] WebSocket Connection
- [ ] Multiple Browser Tabs

## 🔄 When Does This Occur?
**Performance degradation happens:**
- [ ] Immediately on page load
- [ ] After a few minutes of use
- [ ] When interacting with 3D elements
- [ ] When audio visualizer is active
- [ ] When switching between sections
- [ ] On specific devices only
- [ ] In certain browsers only
- [ ] With many particles visible

## 📱 Device-Specific Issues
**Mobile Devices:**
- Device Model: [e.g. iPhone 14, Samsung Galaxy S23, iPad Pro]
- Network: [WiFi/4G/5G/Slow connection]
- Battery Level: [High/Medium/Low - affects performance on some devices]
- Other apps running: [Many/Few/Background apps closed]

## 🌐 Browser Performance Tools
**Chrome DevTools (Performance tab) findings:**
- Main thread blocked by: [JavaScript/Rendering/Network/Other]
- Memory leaks detected: [Yes/No]
- Frequent garbage collection: [Yes/No]
- Long tasks (>50ms): [How many and what type]

**WebGL Context Information:**
```
// Paste output from browser console:
// console.log(renderer.info)
```

## 📈 Performance Over Time
- [ ] Performance is consistently poor
- [ ] Performance degrades over time (memory leak)
- [ ] Performance is good initially, then drops
- [ ] Performance varies randomly
- [ ] Performance is affected by specific actions

## 🔧 Troubleshooting Steps Tried
**What have you already attempted?**
- [ ] Disabled post-processing effects
- [ ] Reduced particle count
- [ ] Closed other browser tabs
- [ ] Tried different browsers
- [ ] Cleared browser cache
- [ ] Disabled browser extensions
- [ ] Reduced quality settings
- [ ] Tried incognito/private mode

## 📊 Comparison with Other Applications
**How does this compare to other WebGL/3D applications?**
- Similar 3D websites perform: [Better/Worse/Same]
- Other Three.js demos perform: [Better/Worse/Same]
- Local graphics applications perform: [Better/Worse/Same]

## 💡 Potential Solutions
**Any ideas for improving performance?**
- Reduce visual complexity for low-end devices
- Implement level-of-detail (LOD) systems
- Better memory management
- Optimize shader complexity
- Implement frame rate limiting

## 📸 Evidence
**Screenshots/Videos:**
- Performance monitoring screenshots
- Frame rate graphs
- Memory usage over time
- Browser DevTools captures

## 🎯 Priority
**How severely does this impact your experience?**
- [ ] 🔴 Critical - Unusable
- [ ] 🟡 High - Significantly impacts usability
- [ ] 🟢 Medium - Noticeable but manageable
- [ ] 🔵 Low - Minor performance concern

## 🔍 Additional Context
Any other details about the performance issue:
- Network conditions during testing
- Time of day (server load considerations)
- Specific user actions that trigger the issue
- Comparison with previous versions

---

## ✋ Checklist
- [ ] I have tested in multiple browsers (if possible)
- [ ] I have tried basic troubleshooting steps
- [ ] I have checked my system meets minimum requirements
- [ ] I have provided specific performance metrics
- [ ] I have included system specifications
- [ ] I have tested with different quality settings