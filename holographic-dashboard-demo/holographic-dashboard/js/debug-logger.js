// Comprehensive Debug Logger for Holographic Dashboard
class DebugLogger {
    constructor() {
        this.logs = [];
        this.startTime = performance.now();
        this.initSteps = new Map();
        this.errors = [];
        this.setupGlobalHandlers();
        this.createDebugPanel();
    }

    log(level, message, data = {}) {
        const entry = {
            level,
            message,
            data,
            timestamp: performance.now() - this.startTime,
            time: new Date().toISOString(),
            stack: new Error().stack
        };
        
        this.logs.push(entry);
        
        // Console output with styling
        const styles = {
            debug: 'color: #888',
            info: 'color: #00FFFF',
            warn: 'color: #FFFF00',
            error: 'color: #FF0066; font-weight: bold',
            success: 'color: #00FF00'
        };
        
        console.log(`%c[${level.toUpperCase()}] ${message}`, styles[level] || '', data);
        
        // Update debug panel
        this.updateDebugPanel(entry);
        
        return entry;
    }

    debug(message, data) { return this.log('debug', message, data); }
    info(message, data) { return this.log('info', message, data); }
    warn(message, data) { return this.log('warn', message, data); }
    error(message, data) { return this.log('error', message, data); }
    success(message, data) { return this.log('success', message, data); }

    startStep(stepName, timeout = 5000) {
        const step = {
            name: stepName,
            startTime: performance.now(),
            status: 'running',
            logs: []
        };
        
        this.initSteps.set(stepName, step);
        
        // Set timeout
        step.timeoutId = setTimeout(() => {
            this.handleStepTimeout(stepName);
        }, timeout);
        
        this.info(`Starting: ${stepName}`);
        return step;
    }

    completeStep(stepName, data = {}) {
        const step = this.initSteps.get(stepName);
        if (!step) return;
        
        clearTimeout(step.timeoutId);
        step.status = 'completed';
        step.endTime = performance.now();
        step.duration = step.endTime - step.startTime;
        step.data = data;
        
        this.success(`Completed: ${stepName} (${step.duration.toFixed(2)}ms)`, data);
    }

    failStep(stepName, error) {
        const step = this.initSteps.get(stepName);
        if (!step) return;
        
        clearTimeout(step.timeoutId);
        step.status = 'failed';
        step.error = error;
        
        this.error(`Failed: ${stepName}`, { error: error.message, stack: error.stack });
    }

    handleStepTimeout(stepName) {
        const step = this.initSteps.get(stepName);
        if (!step) return;
        
        step.status = 'timeout';
        this.error(`Timeout: ${stepName} (exceeded ${step.timeout || 5000}ms)`);
        
        // Generate debug report
        this.generateReport();
    }

    setupGlobalHandlers() {
        // Catch unhandled errors
        window.addEventListener('error', (event) => {
            this.error('Global Error', {
                message: event.message,
                filename: event.filename,
                line: event.lineno,
                column: event.colno,
                error: event.error
            });
            
            this.errors.push({
                type: 'global',
                error: event.error,
                timestamp: new Date()
            });
        });

        // Catch unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.error('Unhandled Promise Rejection', {
                reason: event.reason,
                promise: event.promise
            });
            
            this.errors.push({
                type: 'promise',
                reason: event.reason,
                timestamp: new Date()
            });
        });
    }

    checkDependencies() {
        const required = {
            'THREE': window.THREE,
            'HolographicScene': window.HolographicScene,
            'MediaCardsManager': window.MediaCardsManager,
            'AudioVisualizer': window.AudioVisualizer,
            'UIController': window.UIController,
            'WebSocketClient': window.WebSocketClient,
            'Utils': window.Utils,
            'CONFIG': window.CONFIG,
            'Shaders': window.Shaders,
            'ParticleSystem': window.ParticleSystem,
            'DataStreamParticles': window.DataStreamParticles
        };
        
        const missing = [];
        const found = [];
        
        Object.entries(required).forEach(([name, value]) => {
            if (!value) {
                missing.push(name);
            } else {
                found.push(name);
            }
        });
        
        if (missing.length > 0) {
            this.warn('Missing dependencies', { missing, found });
        } else {
            this.success('All dependencies loaded', { found });
        }
        
        return { missing, found, allLoaded: missing.length === 0 };
    }

    checkWebGL() {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        if (!gl) {
            this.error('WebGL not supported');
            return false;
        }
        
        const info = {
            version: gl.getParameter(gl.VERSION),
            vendor: gl.getParameter(gl.VENDOR),
            renderer: gl.getParameter(gl.RENDERER),
            maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
            maxVertexAttributes: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
            maxTextureUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
        };
        
        this.success('WebGL supported', info);
        return info;
    }

    generateReport() {
        const report = {
            timestamp: new Date().toISOString(),
            totalRuntime: performance.now() - this.startTime,
            steps: Array.from(this.initSteps.entries()).map(([name, data]) => ({
                name,
                ...data
            })),
            errors: this.errors,
            dependencies: this.checkDependencies(),
            webgl: this.checkWebGL(),
            environment: {
                userAgent: navigator.userAgent,
                screen: {
                    width: window.screen.width,
                    height: window.screen.height,
                    pixelRatio: window.devicePixelRatio
                },
                memory: performance.memory ? {
                    used: (performance.memory.usedJSHeapSize / 1048576).toFixed(2) + 'MB',
                    total: (performance.memory.totalJSHeapSize / 1048576).toFixed(2) + 'MB'
                } : 'N/A'
            }
        };
        
        console.log('=== INITIALIZATION REPORT ===');
        console.table(report.steps);
        console.log('Full Report:', report);
        
        return report;
    }

    createDebugPanel() {
        const panel = document.createElement('div');
        panel.id = 'debug-panel';
        panel.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-height: 300px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #00FFFF;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            color: #00FFFF;
            overflow-y: auto;
            z-index: 10000;
            display: none;
        `;
        
        panel.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #00FFFF;">Debug Panel</h3>
                <button onclick="window.debugLogger.toggleDebugPanel()" style="
                    background: none;
                    border: 1px solid #00FFFF;
                    color: #00FFFF;
                    cursor: pointer;
                    padding: 2px 8px;
                ">×</button>
            </div>
            <div id="debug-content"></div>
            <div style="margin-top: 10px;">
                <button onclick="window.debugLogger.generateReport()" style="
                    background: #00FFFF;
                    color: #000;
                    border: none;
                    padding: 5px 10px;
                    cursor: pointer;
                    margin-right: 5px;
                ">Generate Report</button>
                <button onclick="window.debugLogger.clearLogs()" style="
                    background: #FF0066;
                    color: #FFF;
                    border: none;
                    padding: 5px 10px;
                    cursor: pointer;
                ">Clear Logs</button>
            </div>
        `;
        
        document.body.appendChild(panel);
        this.debugPanel = panel;
        
        // Add keyboard shortcut (Ctrl+D)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'd') {
                e.preventDefault();
                this.toggleDebugPanel();
            }
        });
    }

    updateDebugPanel(entry) {
        if (!this.debugPanel) return;
        
        const content = document.getElementById('debug-content');
        if (!content) return;
        
        const logEntry = document.createElement('div');
        logEntry.style.cssText = `
            margin: 2px 0;
            padding: 2px;
            border-left: 2px solid ${this.getLevelColor(entry.level)};
            padding-left: 5px;
        `;
        
        logEntry.innerHTML = `
            <span style="color: ${this.getLevelColor(entry.level)};">[${entry.level.toUpperCase()}]</span>
            <span style="color: #888;">${entry.timestamp.toFixed(2)}ms</span>
            ${entry.message}
        `;
        
        content.appendChild(logEntry);
        
        // Keep only last 50 entries
        while (content.children.length > 50) {
            content.removeChild(content.firstChild);
        }
        
        // Auto-scroll to bottom
        content.scrollTop = content.scrollHeight;
    }

    getLevelColor(level) {
        const colors = {
            debug: '#888',
            info: '#00FFFF',
            warn: '#FFFF00',
            error: '#FF0066',
            success: '#00FF00'
        };
        return colors[level] || '#FFF';
    }

    toggleDebugPanel() {
        if (this.debugPanel) {
            this.debugPanel.style.display = this.debugPanel.style.display === 'none' ? 'block' : 'none';
        }
    }

    showDebugPanel() {
        if (this.debugPanel) {
            this.debugPanel.style.display = 'block';
        }
    }

    clearLogs() {
        this.logs = [];
        const content = document.getElementById('debug-content');
        if (content) {
            content.innerHTML = '';
        }
        this.info('Logs cleared');
    }
}

// Create global debug logger instance
window.debugLogger = new DebugLogger();
window.debugLogger.info('Debug Logger initialized');
window.debugLogger.info('Press Ctrl+D to toggle debug panel');