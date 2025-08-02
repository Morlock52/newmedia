class SpeechSynthesisEngine {
    constructor() {
        this.isSupported = this.checkSupport();
        this.synth = window.speechSynthesis;
        this.voices = [];
        this.selectedVoice = null;
        this.defaultSettings = {
            pitch: 1,
            rate: 1,
            volume: 1,
            lang: 'en-US'
        };
        this.currentSettings = { ...this.defaultSettings };
        this.isSpeaking = false;
        this.currentUtterance = null;
        
        this.init();
    }

    checkSupport() {
        return 'speechSynthesis' in window;
    }

    init() {
        if (!this.isSupported) {
            console.error('Speech Synthesis not supported in this browser');
            return;
        }

        this.loadVoices();
        this.setupEventHandlers();
    }

    loadVoices() {
        if (!this.synth) return;

        const updateVoices = () => {
            this.voices = this.synth.getVoices();
            
            // Select default voice (prefer English voices)
            if (!this.selectedVoice && this.voices.length > 0) {
                this.selectedVoice = this.voices.find(voice => 
                    voice.lang.startsWith('en') && voice.default
                ) || this.voices.find(voice => 
                    voice.lang.startsWith('en')
                ) || this.voices[0];
            }
        };

        // Load voices immediately if available
        updateVoices();

        // Also listen for voices changed event (some browsers load voices asynchronously)
        if (this.synth.onvoiceschanged !== undefined) {
            this.synth.onvoiceschanged = updateVoices;
        }

        // Fallback for browsers that don't fire the event
        setTimeout(updateVoices, 100);
    }

    setupEventHandlers() {
        if (!this.synth) return;

        // Listen for synthesis events
        this.synth.addEventListener('voiceschanged', () => {
            this.loadVoices();
        });
    }

    speak(text, options = {}) {
        if (!this.isSupported || !text.trim()) {
            return Promise.reject(new Error('Speech synthesis not available or empty text'));
        }

        // Stop any current speech
        this.stop();

        const settings = { ...this.currentSettings, ...options };
        const utterance = new SpeechSynthesisUtterance(text);

        // Configure utterance
        utterance.voice = this.selectedVoice;
        utterance.pitch = settings.pitch;
        utterance.rate = settings.rate;
        utterance.volume = settings.volume;
        utterance.lang = settings.lang;

        return new Promise((resolve, reject) => {
            utterance.onstart = () => {
                this.isSpeaking = true;
                console.log('Speech synthesis started');
            };

            utterance.onend = () => {
                this.isSpeaking = false;
                this.currentUtterance = null;
                console.log('Speech synthesis ended');
                resolve();
            };

            utterance.onerror = (event) => {
                this.isSpeaking = false;
                this.currentUtterance = null;
                console.error('Speech synthesis error:', event.error);
                reject(new Error(`Speech synthesis error: ${event.error}`));
            };

            utterance.onpause = () => {
                console.log('Speech synthesis paused');
            };

            utterance.onresume = () => {
                console.log('Speech synthesis resumed');
            };

            utterance.onmark = (event) => {
                console.log('Speech synthesis mark:', event);
            };

            utterance.onboundary = (event) => {
                console.log('Speech synthesis boundary:', event);
            };

            this.currentUtterance = utterance;
            
            try {
                this.synth.speak(utterance);
            } catch (error) {
                this.isSpeaking = false;
                this.currentUtterance = null;
                reject(error);
            }
        });
    }

    stop() {
        if (!this.synth) return;

        try {
            this.synth.cancel();
            this.isSpeaking = false;
            this.currentUtterance = null;
        } catch (error) {
            console.error('Failed to stop speech synthesis:', error);
        }
    }

    pause() {
        if (!this.synth || !this.isSpeaking) return;

        try {
            this.synth.pause();
        } catch (error) {
            console.error('Failed to pause speech synthesis:', error);
        }
    }

    resume() {
        if (!this.synth) return;

        try {
            this.synth.resume();
        } catch (error) {
            console.error('Failed to resume speech synthesis:', error);
        }
    }

    setVoice(voiceIndex) {
        if (voiceIndex >= 0 && voiceIndex < this.voices.length) {
            this.selectedVoice = this.voices[voiceIndex];
            return true;
        }
        return false;
    }

    setVoiceByName(voiceName) {
        const voice = this.voices.find(v => v.name === voiceName);
        if (voice) {
            this.selectedVoice = voice;
            return true;
        }
        return false;
    }

    setPitch(pitch) {
        this.currentSettings.pitch = Math.max(0, Math.min(2, pitch));
    }

    setRate(rate) {
        this.currentSettings.rate = Math.max(0.1, Math.min(10, rate));
    }

    setVolume(volume) {
        this.currentSettings.volume = Math.max(0, Math.min(1, volume));
    }

    setLanguage(lang) {
        this.currentSettings.lang = lang;
    }

    getVoices() {
        return this.voices.map((voice, index) => ({
            index,
            name: voice.name,
            lang: voice.lang,
            gender: this.guessGender(voice.name),
            isDefault: voice.default,
            isLocal: voice.localService
        }));
    }

    guessGender(voiceName) {
        const name = voiceName.toLowerCase();
        const femaleIndicators = ['female', 'woman', 'girl', 'samantha', 'susan', 'victoria', 'karen', 'zira'];
        const maleIndicators = ['male', 'man', 'boy', 'alex', 'daniel', 'david', 'mark'];
        
        for (const indicator of femaleIndicators) {
            if (name.includes(indicator)) return 'female';
        }
        
        for (const indicator of maleIndicators) {
            if (name.includes(indicator)) return 'male';
        }
        
        return 'unknown';
    }

    getCurrentSettings() {
        return {
            ...this.currentSettings,
            voice: this.selectedVoice ? this.selectedVoice.name : null,
            isSupported: this.isSupported,
            isSpeaking: this.isSpeaking
        };
    }

    resetSettings() {
        this.currentSettings = { ...this.defaultSettings };
    }

    // Utility methods for common responses
    async speakSuccess(message = "Command executed successfully") {
        return this.speak(message, { pitch: 1.1, rate: 1 });
    }

    async speakError(message = "Sorry, I couldn't understand that command") {
        return this.speak(message, { pitch: 0.9, rate: 0.9 });
    }

    async speakConfirmation(action) {
        return this.speak(`${action}`, { pitch: 1, rate: 1.1 });
    }

    async speakHelp() {
        const helpText = "You can say commands like: play, pause, stop, volume up, volume down, skip forward, skip back, mute, unmute, or fullscreen.";
        return this.speak(helpText, { rate: 0.9 });
    }

    // Check if speech synthesis is available and working
    static isSupported() {
        return 'speechSynthesis' in window;
    }

    // Test speech synthesis
    async test(text = "Speech synthesis is working correctly") {
        try {
            await this.speak(text);
            return true;
        } catch (error) {
            console.error('Speech synthesis test failed:', error);
            return false;
        }
    }

    // Get speaking status
    getSpeakingStatus() {
        return {
            isSpeaking: this.isSpeaking,
            isPaused: this.synth ? this.synth.paused : false,
            isPending: this.synth ? this.synth.pending : false,
            currentText: this.currentUtterance ? this.currentUtterance.text : null
        };
    }
}