class SpeechRecognitionEngine {
    constructor() {
        this.isSupported = this.checkSupport();
        this.recognition = null;
        this.isListening = false;
        this.onResult = null;
        this.onError = null;
        this.onStart = null;
        this.onEnd = null;
        this.confidenceThreshold = 0.7;
        this.language = 'en-US';
        this.continuous = true;
        this.interimResults = true;
        
        this.init();
    }

    checkSupport() {
        return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    }

    init() {
        if (!this.isSupported) {
            console.error('Speech Recognition not supported in this browser');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.setupRecognition();
        this.setupEventHandlers();
    }

    setupRecognition() {
        if (!this.recognition) return;

        this.recognition.continuous = this.continuous;
        this.recognition.interimResults = this.interimResults;
        this.recognition.lang = this.language;
        this.recognition.maxAlternatives = 3;
    }

    setupEventHandlers() {
        if (!this.recognition) return;

        this.recognition.onstart = () => {
            this.isListening = true;
            console.log('Speech recognition started');
            if (this.onStart) this.onStart();
        };

        this.recognition.onresult = (event) => {
            const results = [];
            
            for (let i = 0; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) {
                    const alternative = result[0];
                    if (alternative.confidence >= this.confidenceThreshold) {
                        results.push({
                            transcript: alternative.transcript.trim(),
                            confidence: alternative.confidence,
                            isFinal: true
                        });
                    }
                } else if (this.interimResults) {
                    results.push({
                        transcript: result[0].transcript.trim(),
                        confidence: result[0].confidence,
                        isFinal: false
                    });
                }
            }

            if (results.length > 0 && this.onResult) {
                this.onResult(results);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;
            
            const errorMessages = {
                'no-speech': 'No speech detected. Please try again.',
                'audio-capture': 'Audio capture failed. Check your microphone.',
                'not-allowed': 'Microphone access denied. Please allow microphone access.',
                'network': 'Network error. Please check your connection.',
                'aborted': 'Speech recognition was aborted.',
                'language-not-supported': 'Language not supported.',
                'service-not-allowed': 'Speech recognition service not allowed.'
            };

            const errorMessage = errorMessages[event.error] || `Speech recognition error: ${event.error}`;
            
            if (this.onError) {
                this.onError({
                    type: event.error,
                    message: errorMessage
                });
            }
        };

        this.recognition.onend = () => {
            this.isListening = false;
            console.log('Speech recognition ended');
            if (this.onEnd) this.onEnd();
        };

        this.recognition.onnomatch = () => {
            console.log('No speech match found');
            if (this.onError) {
                this.onError({
                    type: 'no-match',
                    message: 'No recognizable speech found. Please speak clearly.'
                });
            }
        };
    }

    start() {
        if (!this.isSupported) {
            throw new Error('Speech Recognition not supported');
        }

        if (this.isListening) {
            console.warn('Speech recognition is already running');
            return;
        }

        try {
            this.recognition.start();
        } catch (error) {
            console.error('Failed to start speech recognition:', error);
            if (this.onError) {
                this.onError({
                    type: 'start-failed',
                    message: 'Failed to start speech recognition'
                });
            }
        }
    }

    stop() {
        if (!this.recognition || !this.isListening) {
            return;
        }

        try {
            this.recognition.stop();
        } catch (error) {
            console.error('Failed to stop speech recognition:', error);
        }
    }

    abort() {
        if (!this.recognition) {
            return;
        }

        try {
            this.recognition.abort();
            this.isListening = false;
        } catch (error) {
            console.error('Failed to abort speech recognition:', error);
        }
    }

    setLanguage(language) {
        this.language = language;
        if (this.recognition) {
            this.recognition.lang = language;
        }
    }

    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
    }

    setContinuous(continuous) {
        this.continuous = continuous;
        if (this.recognition) {
            this.recognition.continuous = continuous;
        }
    }

    setInterimResults(interim) {
        this.interimResults = interim;
        if (this.recognition) {
            this.recognition.interimResults = interim;
        }
    }

    // Get available languages (browser dependent)
    static getSupportedLanguages() {
        return [
            { code: 'en-US', name: 'English (US)' },
            { code: 'en-GB', name: 'English (UK)' },
            { code: 'es-ES', name: 'Spanish' },
            { code: 'fr-FR', name: 'French' },
            { code: 'de-DE', name: 'German' },
            { code: 'it-IT', name: 'Italian' },
            { code: 'pt-BR', name: 'Portuguese (Brazil)' },
            { code: 'ru-RU', name: 'Russian' },
            { code: 'ja-JP', name: 'Japanese' },
            { code: 'ko-KR', name: 'Korean' },
            { code: 'zh-CN', name: 'Chinese (Mandarin)' }
        ];
    }

    // Check if browser supports speech recognition
    static isSupported() {
        return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    }

    // Get microphone permission status
    async checkMicrophonePermission() {
        try {
            const result = await navigator.permissions.query({ name: 'microphone' });
            return result.state; // 'granted', 'denied', or 'prompt'
        } catch (error) {
            console.warn('Could not check microphone permission:', error);
            return 'unknown';
        }
    }

    // Request microphone access
    async requestMicrophoneAccess() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop()); // Stop the stream
            return true;
        } catch (error) {
            console.error('Microphone access denied:', error);
            return false;
        }
    }
}