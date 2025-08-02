// Utility Functions for Holographic Media Dashboard

const Utils = {
    // Generate unique ID
    generateId: () => {
        return '_' + Math.random().toString(36).substr(2, 9);
    },

    // Clamp value between min and max
    clamp: (value, min, max) => {
        return Math.min(Math.max(value, min), max);
    },

    // Linear interpolation
    lerp: (start, end, factor) => {
        return start + (end - start) * factor;
    },

    // Map value from one range to another
    map: (value, inMin, inMax, outMin, outMax) => {
        return outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin));
    },

    // Random float between min and max
    randomFloat: (min, max) => {
        return Math.random() * (max - min) + min;
    },

    // Random integer between min and max
    randomInt: (min, max) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    },

    // Format bytes to human readable
    formatBytes: (bytes, decimals = 2) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },

    // Format number with commas
    formatNumber: (num) => {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    // Format duration to human readable
    formatDuration: (seconds) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    },

    // Get relative time string
    getRelativeTime: (timestamp) => {
        const now = Date.now();
        const diff = now - timestamp;
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
        if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        return 'Just now';
    },

    // Debounce function
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle function
    throttle: (func, limit) => {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Deep clone object
    deepClone: (obj) => {
        return JSON.parse(JSON.stringify(obj));
    },

    // Merge objects deeply
    deepMerge: (target, ...sources) => {
        if (!sources.length) return target;
        const source = sources.shift();

        if (Utils.isObject(target) && Utils.isObject(source)) {
            for (const key in source) {
                if (Utils.isObject(source[key])) {
                    if (!target[key]) Object.assign(target, { [key]: {} });
                    Utils.deepMerge(target[key], source[key]);
                } else {
                    Object.assign(target, { [key]: source[key] });
                }
            }
        }

        return Utils.deepMerge(target, ...sources);
    },

    // Check if value is object
    isObject: (item) => {
        return item && typeof item === 'object' && !Array.isArray(item);
    },

    // Convert hex color to RGB
    hexToRgb: (hex) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    },

    // Convert RGB to hex
    rgbToHex: (r, g, b) => {
        return '#' + [r, g, b].map(x => {
            const hex = x.toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        }).join('');
    },

    // Create gradient color array
    createGradient: (startColor, endColor, steps) => {
        const start = new THREE.Color(startColor);
        const end = new THREE.Color(endColor);
        const colors = [];

        for (let i = 0; i < steps; i++) {
            const color = new THREE.Color();
            color.lerpColors(start, end, i / (steps - 1));
            colors.push(color.getHex());
        }

        return colors;
    },

    // Easing functions
    easing: {
        linear: (t) => t,
        easeInQuad: (t) => t * t,
        easeOutQuad: (t) => t * (2 - t),
        easeInOutQuad: (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
        easeInCubic: (t) => t * t * t,
        easeOutCubic: (t) => (--t) * t * t + 1,
        easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
        easeInQuart: (t) => t * t * t * t,
        easeOutQuart: (t) => 1 - (--t) * t * t * t,
        easeInOutQuart: (t) => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t,
        easeOutElastic: (t) => {
            const p = 0.3;
            return Math.pow(2, -10 * t) * Math.sin((t - p / 4) * (2 * Math.PI) / p) + 1;
        },
        easeOutBounce: (t) => {
            if (t < (1 / 2.75)) {
                return 7.5625 * t * t;
            } else if (t < (2 / 2.75)) {
                return 7.5625 * (t -= (1.5 / 2.75)) * t + 0.75;
            } else if (t < (2.5 / 2.75)) {
                return 7.5625 * (t -= (2.25 / 2.75)) * t + 0.9375;
            } else {
                return 7.5625 * (t -= (2.625 / 2.75)) * t + 0.984375;
            }
        }
    },

    // Load texture with progress
    loadTexture: (url, onProgress) => {
        return new Promise((resolve, reject) => {
            const loader = new THREE.TextureLoader();
            loader.load(
                url,
                (texture) => resolve(texture),
                (xhr) => {
                    if (onProgress) {
                        const percentComplete = (xhr.loaded / xhr.total) * 100;
                        onProgress(percentComplete);
                    }
                },
                (error) => reject(error)
            );
        });
    },

    // Create canvas texture from text
    createTextTexture: (text, options = {}) => {
        const {
            fontSize = 64,
            fontFamily = 'Arial',
            color = '#FFFFFF',
            backgroundColor = 'transparent',
            padding = 20,
            maxWidth = 512
        } = options;

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Set font to measure text
        context.font = `${fontSize}px ${fontFamily}`;
        const metrics = context.measureText(text);
        
        // Set canvas size
        canvas.width = Math.min(metrics.width + padding * 2, maxWidth);
        canvas.height = fontSize + padding * 2;

        // Fill background
        if (backgroundColor !== 'transparent') {
            context.fillStyle = backgroundColor;
            context.fillRect(0, 0, canvas.width, canvas.height);
        }

        // Draw text
        context.font = `${fontSize}px ${fontFamily}`;
        context.fillStyle = color;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);

        // Create texture
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        return texture;
    },

    // Performance monitor
    performanceMonitor: {
        markers: {},
        
        start: function(name) {
            this.markers[name] = performance.now();
        },
        
        end: function(name) {
            if (this.markers[name]) {
                const duration = performance.now() - this.markers[name];
                delete this.markers[name];
                return duration;
            }
            return 0;
        },
        
        measure: function(name, fn) {
            this.start(name);
            const result = fn();
            const duration = this.end(name);
            console.log(`${name}: ${duration.toFixed(2)}ms`);
            return result;
        }
    },

    // Local storage helper
    storage: {
        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        },
        
        get: (key, defaultValue = null) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Storage error:', e);
                return defaultValue;
            }
        },
        
        remove: (key) => {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        },
        
        clear: () => {
            try {
                localStorage.clear();
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        }
    }
};

// Export for use in other modules
window.Utils = Utils;