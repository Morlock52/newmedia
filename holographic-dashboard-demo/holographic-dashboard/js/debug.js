// Debug script to identify initialization issues

console.log('=== Dashboard Debug Info ===');
console.log('THREE.js loaded:', typeof THREE !== 'undefined');
console.log('GSAP loaded:', typeof gsap !== 'undefined');
console.log('Socket.IO loaded:', typeof io !== 'undefined');

// Check if our components are loaded
console.log('\n=== Component Check ===');
console.log('CONFIG:', typeof CONFIG !== 'undefined');
console.log('Utils:', typeof Utils !== 'undefined');
console.log('Shaders:', typeof Shaders !== 'undefined');
console.log('HolographicScene:', typeof HolographicScene !== 'undefined');
console.log('ParticleSystem:', typeof ParticleSystem !== 'undefined');
console.log('MediaCardsManager:', typeof MediaCardsManager !== 'undefined');
console.log('AudioVisualizer:', typeof AudioVisualizer !== 'undefined');
console.log('UIController:', typeof UIController !== 'undefined');
console.log('WebSocketClient:', typeof WebSocketClient !== 'undefined');
console.log('HolographicMediaDashboard:', typeof HolographicMediaDashboard !== 'undefined');

// Check Three.js components
console.log('\n=== Three.js Components ===');
console.log('THREE.OrbitControls:', typeof THREE !== 'undefined' && typeof THREE.OrbitControls !== 'undefined');
console.log('THREE.EffectComposer:', typeof THREE !== 'undefined' && typeof THREE.EffectComposer !== 'undefined');
console.log('THREE.RenderPass:', typeof THREE !== 'undefined' && typeof THREE.RenderPass !== 'undefined');
console.log('THREE.UnrealBloomPass:', typeof THREE !== 'undefined' && typeof THREE.UnrealBloomPass !== 'undefined');
console.log('THREE.ShaderPass:', typeof THREE !== 'undefined' && typeof THREE.ShaderPass !== 'undefined');

// Check DOM elements
console.log('\n=== DOM Elements ===');
console.log('WebGL container:', document.getElementById('webgl-container') !== null);
console.log('Loading screen:', document.getElementById('loading-screen') !== null);
console.log('HUD interface:', document.getElementById('hud-interface') !== null);

// Listen for errors
window.addEventListener('error', (event) => {
    console.error('=== JavaScript Error ===');
    console.error('Message:', event.message);
    console.error('Source:', event.filename);
    console.error('Line:', event.lineno);
    console.error('Column:', event.colno);
    console.error('Error object:', event.error);
});

console.log('\n=== Debug script loaded ===');