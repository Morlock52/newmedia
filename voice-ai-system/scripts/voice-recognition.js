// Voice Recognition Module
// Handles Web Speech API integration for real-time speech recognition

class VoiceRecognition {
    constructor(options = {}) {
        this.options = {
            continuous: true,
            interimResults: true,
            maxAlternatives: 3,
            language: 'en-US',
            ...options
        };

        this.recognition = null;
        this.isListening = false;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.visualizer = null;
        this.callbacks = {
            onStart: null,
            onResult: null,
            onError: null,
            onEnd: null,
            onAudioLevel: null
        };

        this.init();
    }

    init() {
        // Check for browser support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            console.error('Speech Recognition API not supported');
            return;
        }

        // Initialize speech recognition
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = this.options.continuous;
        this.recognition.interimResults = this.options.interimResults;
        this.recognition.maxAlternatives = this.options.maxAlternatives;
        this.recognition.lang = this.options.language;

        // Set up event handlers
        this.setupEventHandlers();

        // Initialize audio context for visualization
        this.initAudioContext();
    }

    setupEventHandlers() {
        if (!this.recognition) return;

        this.recognition.onstart = () => {
            console.log('Speech recognition started');
            this.isListening = true;
            if (this.callbacks.onStart) {
                this.callbacks.onStart();
            }
        };

        this.recognition.onresult = (event) => {
            const results = this.processResults(event.results);
            if (this.callbacks.onResult) {
                this.callbacks.onResult(results);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;
            
            let errorMessage = 'Speech recognition error';
            switch (event.error) {
                case 'no-speech':
                    errorMessage = 'No speech detected';
                    break;
                case 'audio-capture':
                    errorMessage = 'No microphone found';
                    break;
                case 'not-allowed':
                    errorMessage = 'Microphone permission denied';
                    break;
                case 'network':
                    errorMessage = 'Network error';
                    break;
            }

            if (this.callbacks.onError) {
                this.callbacks.onError({ error: event.error, message: errorMessage });
            }
        };

        this.recognition.onend = () => {
            console.log('Speech recognition ended');
            this.isListening = false;
            if (this.callbacks.onEnd) {
                this.callbacks.onEnd();
            }
        };

        this.recognition.onspeechstart = () => {
            console.log('Speech detected');
        };

        this.recognition.onspeechend = () => {
            console.log('Speech ended');
        };

        this.recognition.onnomatch = () => {
            console.log('No speech match');
        };
    }

    processResults(results) {
        const processedResults = [];
        
        for (let i = results.length - 1; i >= 0; i--) {
            const result = results[i];
            const alternatives = [];

            for (let j = 0; j < result.length; j++) {
                alternatives.push({
                    transcript: result[j].transcript,
                    confidence: result[j].confidence || 0
                });
            }

            processedResults.push({
                transcript: result[0].transcript,
                confidence: result[0].confidence || 0,
                isFinal: result.isFinal,
                alternatives: alternatives,
                timestamp: new Date().toISOString()
            });
        }

        return processedResults;
    }

    async initAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.8;
        } catch (error) {
            console.error('Failed to initialize audio context:', error);
        }
    }

    async startVisualization() {
        if (!this.audioContext || !this.analyser) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);

            // Start visualization loop
            this.visualizeAudio();
        } catch (error) {
            console.error('Failed to start audio visualization:', error);
        }
    }

    visualizeAudio() {
        if (!this.analyser || !this.isListening) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);

        // Calculate average volume
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        const normalizedLevel = average / 255;

        if (this.callbacks.onAudioLevel) {
            this.callbacks.onAudioLevel(normalizedLevel);
        }

        // Continue visualization if still listening
        if (this.isListening) {
            requestAnimationFrame(() => this.visualizeAudio());
        }
    }

    async start() {
        if (!this.recognition) {
            throw new Error('Speech Recognition not initialized');
        }

        if (this.isListening) {
            console.log('Already listening');
            return;
        }

        try {
            // Request microphone permission first
            await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Start recognition
            this.recognition.start();
            
            // Start audio visualization
            await this.startVisualization();
        } catch (error) {
            console.error('Failed to start speech recognition:', error);
            throw error;
        }
    }

    stop() {
        if (!this.recognition || !this.isListening) return;

        this.recognition.stop();
        
        // Stop audio visualization
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
    }

    setLanguage(language) {
        this.options.language = language;
        if (this.recognition) {
            this.recognition.lang = language;
        }
    }

    setCallbacks(callbacks) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }

    setContinuous(continuous) {
        this.options.continuous = continuous;
        if (this.recognition) {
            this.recognition.continuous = continuous;
        }
    }

    setInterimResults(interimResults) {
        this.options.interimResults = interimResults;
        if (this.recognition) {
            this.recognition.interimResults = interimResults;
        }
    }

    setMaxAlternatives(maxAlternatives) {
        this.options.maxAlternatives = maxAlternatives;
        if (this.recognition) {
            this.recognition.maxAlternatives = maxAlternatives;
        }
    }

    // Get supported languages
    static getSupportedLanguages() {
        return [
            { code: 'en-US', name: 'English (US)' },
            { code: 'en-GB', name: 'English (UK)' },
            { code: 'es-ES', name: 'Spanish (Spain)' },
            { code: 'es-MX', name: 'Spanish (Mexico)' },
            { code: 'fr-FR', name: 'French' },
            { code: 'de-DE', name: 'German' },
            { code: 'it-IT', name: 'Italian' },
            { code: 'pt-BR', name: 'Portuguese (Brazil)' },
            { code: 'pt-PT', name: 'Portuguese (Portugal)' },
            { code: 'ru-RU', name: 'Russian' },
            { code: 'zh-CN', name: 'Chinese (Simplified)' },
            { code: 'zh-TW', name: 'Chinese (Traditional)' },
            { code: 'ja-JP', name: 'Japanese' },
            { code: 'ko-KR', name: 'Korean' },
            { code: 'ar-SA', name: 'Arabic' },
            { code: 'hi-IN', name: 'Hindi' },
            { code: 'nl-NL', name: 'Dutch' },
            { code: 'sv-SE', name: 'Swedish' },
            { code: 'da-DK', name: 'Danish' },
            { code: 'no-NO', name: 'Norwegian' },
            { code: 'fi-FI', name: 'Finnish' },
            { code: 'pl-PL', name: 'Polish' },
            { code: 'tr-TR', name: 'Turkish' },
            { code: 'el-GR', name: 'Greek' },
            { code: 'cs-CZ', name: 'Czech' },
            { code: 'hu-HU', name: 'Hungarian' },
            { code: 'ro-RO', name: 'Romanian' },
            { code: 'sk-SK', name: 'Slovak' },
            { code: 'bg-BG', name: 'Bulgarian' },
            { code: 'hr-HR', name: 'Croatian' },
            { code: 'sr-RS', name: 'Serbian' },
            { code: 'uk-UA', name: 'Ukrainian' },
            { code: 'he-IL', name: 'Hebrew' },
            { code: 'th-TH', name: 'Thai' },
            { code: 'vi-VN', name: 'Vietnamese' },
            { code: 'id-ID', name: 'Indonesian' },
            { code: 'ms-MY', name: 'Malay' },
            { code: 'fil-PH', name: 'Filipino' },
            { code: 'ca-ES', name: 'Catalan' },
            { code: 'eu-ES', name: 'Basque' },
            { code: 'gl-ES', name: 'Galician' }
        ];
    }

    // Check if speech recognition is supported
    static isSupported() {
        return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    }

    // Get browser info for compatibility
    static getBrowserInfo() {
        const ua = navigator.userAgent;
        let browserName = 'Unknown';
        let browserVersion = 'Unknown';

        if (ua.indexOf('Chrome') > -1) {
            browserName = 'Chrome';
            browserVersion = ua.match(/Chrome\/(\d+)/)[1];
        } else if (ua.indexOf('Safari') > -1) {
            browserName = 'Safari';
            browserVersion = ua.match(/Version\/(\d+)/)[1];
        } else if (ua.indexOf('Firefox') > -1) {
            browserName = 'Firefox';
            browserVersion = ua.match(/Firefox\/(\d+)/)[1];
        } else if (ua.indexOf('Edge') > -1) {
            browserName = 'Edge';
            browserVersion = ua.match(/Edge\/(\d+)/)[1];
        }

        return {
            name: browserName,
            version: browserVersion,
            userAgent: ua,
            isSupported: this.isSupported()
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceRecognition;
}