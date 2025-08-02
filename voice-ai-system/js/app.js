class VoiceAISystem {
    constructor() {
        this.speechRecognition = null;
        this.speechSynthesis = null;
        this.nluEngine = null;
        this.mediaController = null;
        this.audioVisualizer = null;
        
        this.isListening = false;
        this.commandHistory = [];
        
        this.initializeSystem();
        this.setupEventListeners();
        this.updateUI();
    }

    async initializeSystem() {
        try {
            // Initialize speech recognition
            this.speechRecognition = new SpeechRecognitionEngine();
            this.setupSpeechRecognitionHandlers();

            // Initialize speech synthesis
            this.speechSynthesis = new SpeechSynthesisEngine();
            await this.loadVoices();

            // Initialize NLU engine
            this.nluEngine = new NaturalLanguageUnderstanding();

            // Initialize media controller
            const videoElement = document.getElementById('videoPlayer');
            this.mediaController = new MediaController(videoElement);

            // Initialize audio visualizer
            const canvas = document.getElementById('visualizerCanvas');
            this.audioVisualizer = new AudioVisualizer(canvas);

            // Check system capabilities
            await this.checkSystemCapabilities();

            console.log('Voice AI System initialized successfully');
            this.updateStatus('ready', 'System ready for voice commands');
            
            // Show static visualization
            this.audioVisualizer.drawStaticVisualization();
            
        } catch (error) {
            console.error('Failed to initialize Voice AI System:', error);
            this.updateStatus('error', `Initialization failed: ${error.message}`);
        }
    }

    setupSpeechRecognitionHandlers() {
        if (!this.speechRecognition) return;

        this.speechRecognition.onStart = () => {
            this.isListening = true;
            this.updateStatus('listening', 'Listening for commands...');
            this.updateUI();
        };

        this.speechRecognition.onResult = (results) => {
            results.forEach(result => {
                if (result.isFinal) {
                    this.processVoiceCommand(result.transcript, result.confidence);
                } else {
                    // Show interim results
                    this.updateStatus('processing', `Hearing: "${result.transcript}"`);
                }
            });
        };

        this.speechRecognition.onError = (error) => {
            console.error('Speech recognition error:', error);
            this.updateStatus('error', error.message);
            this.isListening = false;
            this.updateUI();
            
            // Try to recover from certain errors
            if (error.type === 'no-speech') {
                this.speechSynthesis.speakError("I didn't hear anything. Please try again.");
            }
        };

        this.speechRecognition.onEnd = () => {
            this.isListening = false;
            this.updateStatus('ready', 'Ready for voice commands');
            this.updateUI();
            this.audioVisualizer.disconnectMicrophone();
        };
    }

    async processVoiceCommand(transcript, confidence) {
        try {
            this.updateStatus('processing', `Processing: "${transcript}"`);
            
            // Process command through NLU
            const nluResult = this.nluEngine.processCommand(transcript);
            
            if (!nluResult.success) {
                throw new Error(nluResult.error);
            }

            // Execute media command
            const mediaResult = await this.mediaController.executeCommand(
                nluResult.intent, 
                nluResult.entities
            );

            // Add to command history
            this.addToHistory(transcript, nluResult.intent, true, confidence);

            // Provide voice feedback
            await this.speechSynthesis.speakConfirmation(mediaResult.message);
            
            this.updateStatus('success', `Command executed: ${mediaResult.message}`);
            
        } catch (error) {
            console.error('Command processing error:', error);
            
            // Add failed command to history
            this.addToHistory(transcript, 'UNKNOWN', false, confidence);
            
            // Provide error feedback
            await this.speechSynthesis.speakError(`Sorry, ${error.message}`);
            
            this.updateStatus('error', `Command failed: ${error.message}`);
        }
    }

    async startListening() {
        if (!this.speechRecognition) {
            throw new Error('Speech recognition not available');
        }

        if (this.isListening) {
            console.warn('Already listening');
            return;
        }

        try {
            // Connect microphone to visualizer
            await this.audioVisualizer.connectMicrophone();
            
            // Start speech recognition
            this.speechRecognition.start();
            
        } catch (error) {
            console.error('Failed to start listening:', error);
            this.updateStatus('error', `Failed to start: ${error.message}`);
            throw error;
        }
    }

    stopListening() {
        if (!this.isListening) {
            return;
        }

        this.speechRecognition.stop();
        this.audioVisualizer.disconnectMicrophone();
    }

    setupEventListeners() {
        // Start/Stop buttons
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        startBtn?.addEventListener('click', async () => {
            try {
                await this.startListening();
            } catch (error) {
                console.error('Start listening failed:', error);
            }
        });

        stopBtn?.addEventListener('click', () => {
            this.stopListening();
        });

        // Voice settings
        const voiceSelect = document.getElementById('voiceSelect');
        voiceSelect?.addEventListener('change', (e) => {
            this.speechSynthesis.setVoice(parseInt(e.target.value));
        });

        const pitchSlider = document.getElementById('pitchSlider');
        pitchSlider?.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.speechSynthesis.setPitch(value);
            document.getElementById('pitchValue').textContent = value.toFixed(1);
        });

        const rateSlider = document.getElementById('rateSlider');
        rateSlider?.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.speechSynthesis.setRate(value);
            document.getElementById('rateValue').textContent = value.toFixed(1);
        });

        const confidenceSlider = document.getElementById('confidenceSlider');
        confidenceSlider?.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.speechRecognition.setConfidenceThreshold(value);
            this.nluEngine.setConfidenceThreshold(value);
            document.getElementById('confidenceValue').textContent = value.toFixed(2);
        });

        // Media controller events
        document.addEventListener('mediaController', (event) => {
            this.handleMediaEvent(event.detail);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && e.ctrlKey) {
                e.preventDefault();
                if (this.isListening) {
                    this.stopListening();
                } else {
                    this.startListening();
                }
            }
        });
    }

    async loadVoices() {
        if (!this.speechSynthesis) return;

        // Wait for voices to load
        await new Promise(resolve => {
            const checkVoices = () => {
                const voices = this.speechSynthesis.getVoices();
                if (voices.length > 0) {
                    this.populateVoiceSelect(voices);
                    resolve();
                } else {
                    setTimeout(checkVoices, 100);
                }
            };
            checkVoices();
        });
    }

    populateVoiceSelect(voices) {
        const voiceSelect = document.getElementById('voiceSelect');
        if (!voiceSelect) return;

        voiceSelect.innerHTML = '';
        
        voices.forEach((voice, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${voice.name} (${voice.lang}) ${voice.gender ? '- ' + voice.gender : ''}`;
            if (voice.isDefault) {
                option.selected = true;
            }
            voiceSelect.appendChild(option);
        });
    }

    async checkSystemCapabilities() {
        const capabilities = {
            speechRecognition: SpeechRecognitionEngine.isSupported(),
            speechSynthesis: SpeechSynthesisEngine.isSupported(),
            mediaDevices: 'mediaDevices' in navigator,
            getUserMedia: 'getUserMedia' in navigator.mediaDevices,
            audioContext: 'AudioContext' in window || 'webkitAudioContext' in window
        };

        console.log('System capabilities:', capabilities);

        // Check microphone permission
        if (capabilities.getUserMedia) {
            const micPermission = await this.speechRecognition.checkMicrophonePermission();
            console.log('Microphone permission:', micPermission);
            
            if (micPermission === 'denied') {
                this.updateStatus('warning', 'Microphone access denied. Voice commands will not work.');
            }
        }

        return capabilities;
    }

    updateStatus(type, message) {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const pulse = statusIndicator?.querySelector('.pulse');

        if (statusText) {
            statusText.textContent = message;
        }

        if (pulse) {
            pulse.className = 'pulse';
            
            switch (type) {
                case 'ready':
                    pulse.classList.add('active');
                    break;
                case 'listening':
                    pulse.classList.add('listening');
                    break;
                case 'processing':
                    pulse.classList.add('listening');
                    break;
                case 'success':
                    pulse.classList.add('active');
                    break;
                case 'error':
                case 'warning':
                    pulse.classList.add('error');
                    break;
            }
        }

        console.log(`Status [${type}]: ${message}`);
    }

    updateUI() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (startBtn && stopBtn) {
            startBtn.disabled = this.isListening;
            stopBtn.disabled = !this.isListening;
            
            if (this.isListening) {
                startBtn.textContent = 'Listening...';
            } else {
                startBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="24" height="24">
                        <path d="M12,2A3,3 0 0,1 15,5V11A3,3 0 0,1 12,14A3,3 0 0,1 9,11V5A3,3 0 0,1 12,2M19,11C19,14.53 16.39,17.44 13,17.93V21H11V17.93C7.61,17.44 5,14.53 5,11H7A5,5 0 0,0 12,16A5,5 0 0,0 17,11H19Z" />
                    </svg>
                    Start Listening
                `;
            }
        }
    }

    addToHistory(command, intent, success, confidence) {
        const historyItem = {
            command,
            intent,
            success,
            confidence,
            timestamp: new Date()
        };

        this.commandHistory.unshift(historyItem);
        
        // Keep only last 50 commands
        if (this.commandHistory.length > 50) {
            this.commandHistory = this.commandHistory.slice(0, 50);
        }

        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        historyList.innerHTML = '';

        this.commandHistory.slice(0, 10).forEach(item => {
            const historyElement = document.createElement('div');
            historyElement.className = `history-item ${item.success ? 'success' : 'error'}`;
            
            historyElement.innerHTML = `
                <div>
                    <div class="history-command">"${item.command}"</div>
                    <div class="history-details">
                        Intent: ${item.intent} | Confidence: ${(item.confidence * 100).toFixed(0)}%
                    </div>
                </div>
                <div class="history-time">${item.timestamp.toLocaleTimeString()}</div>
            `;
            
            historyList.appendChild(historyElement);
        });
    }

    handleMediaEvent(eventData) {
        const mediaStatus = document.getElementById('mediaStatus');
        const progressFill = document.getElementById('progressFill');

        if (eventData.type === 'timeupdate' && eventData.status) {
            const progress = eventData.status.progress;
            if (progressFill) {
                progressFill.style.width = `${progress}%`;
            }
            
            if (mediaStatus) {
                const status = eventData.status;
                const state = status.playing ? 'Playing' : 'Paused';
                const time = this.formatTime(status.currentTime);
                const duration = this.formatTime(status.duration);
                mediaStatus.textContent = `${state} - ${time} / ${duration}`;
            }
        }
    }

    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';
        
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }

    // Public API methods
    async testVoiceCommand(command) {
        if (typeof command === 'string') {
            await this.processVoiceCommand(command, 1.0);
        }
    }

    getSystemStatus() {
        return {
            isListening: this.isListening,
            speechRecognitionAvailable: !!this.speechRecognition?.isSupported,
            speechSynthesisAvailable: !!this.speechSynthesis?.isSupported,
            commandHistory: this.commandHistory.slice(0, 10),
            mediaStatus: this.mediaController?.getStatus()
        };
    }

    async speak(text) {
        if (this.speechSynthesis) {
            await this.speechSynthesis.speak(text);
        }
    }

    // Cleanup
    destroy() {
        this.stopListening();
        this.audioVisualizer?.destroy();
        this.speechSynthesis?.stop();
    }
}

// Initialize the system when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAI = new VoiceAISystem();
    
    // Add global error handler
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
        if (window.voiceAI) {
            window.voiceAI.updateStatus('error', 'System error occurred');
        }
    });
    
    // Add unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        if (window.voiceAI) {
            window.voiceAI.updateStatus('error', 'Async operation failed');
        }
    });
});

// Export for testing and external access
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceAISystem;
}