// Voice Synthesis Module
// Handles Text-to-Speech functionality with multi-language support

class VoiceSynthesis {
    constructor(options = {}) {
        this.options = {
            rate: 1.0,
            pitch: 1.0,
            volume: 1.0,
            voice: null,
            ...options
        };

        this.synthesis = window.speechSynthesis;
        this.voices = [];
        this.currentUtterance = null;
        this.queue = [];
        this.isPaused = false;
        this.callbacks = {
            onStart: null,
            onEnd: null,
            onPause: null,
            onResume: null,
            onError: null,
            onMark: null,
            onWord: null,
            onSentence: null
        };

        this.init();
    }

    init() {
        if (!this.synthesis) {
            console.error('Speech Synthesis API not supported');
            return;
        }

        // Load voices
        this.loadVoices();

        // Handle voice list changes
        if (this.synthesis.onvoiceschanged !== undefined) {
            this.synthesis.onvoiceschanged = () => this.loadVoices();
        }
    }

    loadVoices() {
        this.voices = this.synthesis.getVoices();
        console.log(`Loaded ${this.voices.length} voices`);
        
        // Set default voice if not already set
        if (!this.options.voice && this.voices.length > 0) {
            // Try to find a high-quality voice
            const preferredVoice = this.findPreferredVoice();
            if (preferredVoice) {
                this.options.voice = preferredVoice;
            }
        }
    }

    findPreferredVoice(lang = 'en-US') {
        // Priority order for voice selection
        const priorities = [
            voice => voice.lang === lang && voice.localService === false, // Cloud voices
            voice => voice.lang.startsWith(lang.split('-')[0]) && voice.localService === false,
            voice => voice.lang === lang && voice.localService === true, // Local voices
            voice => voice.lang.startsWith(lang.split('-')[0]) && voice.localService === true,
            voice => voice.default === true
        ];

        for (const priorityFn of priorities) {
            const voice = this.voices.find(priorityFn);
            if (voice) return voice;
        }

        return this.voices[0]; // Fallback to first available voice
    }

    speak(text, options = {}) {
        if (!this.synthesis) {
            console.error('Speech Synthesis not available');
            return Promise.reject(new Error('Speech Synthesis not available'));
        }

        return new Promise((resolve, reject) => {
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Apply options
            utterance.rate = options.rate || this.options.rate;
            utterance.pitch = options.pitch || this.options.pitch;
            utterance.volume = options.volume || this.options.volume;
            utterance.voice = options.voice || this.options.voice;
            utterance.lang = options.lang || (this.options.voice && this.options.voice.lang) || 'en-US';

            // Set up event handlers
            utterance.onstart = (event) => {
                console.log('Speech started');
                this.currentUtterance = utterance;
                if (this.callbacks.onStart) {
                    this.callbacks.onStart(event);
                }
            };

            utterance.onend = (event) => {
                console.log('Speech ended');
                this.currentUtterance = null;
                this.processQueue();
                if (this.callbacks.onEnd) {
                    this.callbacks.onEnd(event);
                }
                resolve();
            };

            utterance.onerror = (event) => {
                console.error('Speech error:', event);
                this.currentUtterance = null;
                if (this.callbacks.onError) {
                    this.callbacks.onError(event);
                }
                reject(event);
            };

            utterance.onpause = (event) => {
                console.log('Speech paused');
                this.isPaused = true;
                if (this.callbacks.onPause) {
                    this.callbacks.onPause(event);
                }
            };

            utterance.onresume = (event) => {
                console.log('Speech resumed');
                this.isPaused = false;
                if (this.callbacks.onResume) {
                    this.callbacks.onResume(event);
                }
            };

            utterance.onmark = (event) => {
                if (this.callbacks.onMark) {
                    this.callbacks.onMark(event);
                }
            };

            utterance.onboundary = (event) => {
                if (event.name === 'word' && this.callbacks.onWord) {
                    this.callbacks.onWord(event);
                } else if (event.name === 'sentence' && this.callbacks.onSentence) {
                    this.callbacks.onSentence(event);
                }
            };

            // Add to queue or speak immediately
            if (options.queue === false || this.queue.length === 0 && !this.currentUtterance) {
                this.synthesis.speak(utterance);
            } else {
                this.queue.push(utterance);
                if (!this.currentUtterance) {
                    this.processQueue();
                }
            }
        });
    }

    processQueue() {
        if (this.queue.length > 0 && !this.currentUtterance) {
            const utterance = this.queue.shift();
            this.synthesis.speak(utterance);
        }
    }

    pause() {
        if (this.synthesis && this.synthesis.speaking && !this.isPaused) {
            this.synthesis.pause();
        }
    }

    resume() {
        if (this.synthesis && this.isPaused) {
            this.synthesis.resume();
        }
    }

    stop() {
        if (this.synthesis) {
            this.synthesis.cancel();
            this.queue = [];
            this.currentUtterance = null;
            this.isPaused = false;
        }
    }

    setRate(rate) {
        this.options.rate = Math.max(0.1, Math.min(10, rate));
    }

    setPitch(pitch) {
        this.options.pitch = Math.max(0, Math.min(2, pitch));
    }

    setVolume(volume) {
        this.options.volume = Math.max(0, Math.min(1, volume));
    }

    setVoice(voice) {
        if (typeof voice === 'string') {
            // Find voice by name
            const foundVoice = this.voices.find(v => v.name === voice);
            if (foundVoice) {
                this.options.voice = foundVoice;
            }
        } else if (voice && voice.name) {
            // Voice object provided
            this.options.voice = voice;
        }
    }

    setCallbacks(callbacks) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }

    getVoices() {
        return this.voices;
    }

    getVoicesByLanguage(lang) {
        return this.voices.filter(voice => voice.lang.startsWith(lang));
    }

    getCurrentVoice() {
        return this.options.voice;
    }

    isSpeaking() {
        return this.synthesis && this.synthesis.speaking;
    }

    isPending() {
        return this.synthesis && this.synthesis.pending;
    }

    // Advanced text processing
    preprocessText(text, options = {}) {
        let processedText = text;

        // Handle abbreviations
        if (options.expandAbbreviations) {
            const abbreviations = {
                'Mr.': 'Mister',
                'Mrs.': 'Missus',
                'Ms.': 'Miss',
                'Dr.': 'Doctor',
                'Prof.': 'Professor',
                'St.': 'Street',
                'Ave.': 'Avenue',
                'etc.': 'et cetera',
                'vs.': 'versus',
                'e.g.': 'for example',
                'i.e.': 'that is',
                'USA': 'U S A',
                'UK': 'U K',
                'CEO': 'C E O',
                'AI': 'A I',
                'API': 'A P I'
            };

            for (const [abbr, expansion] of Object.entries(abbreviations)) {
                const regex = new RegExp(`\\b${abbr.replace('.', '\\.')}\\b`, 'g');
                processedText = processedText.replace(regex, expansion);
            }
        }

        // Handle numbers
        if (options.spellOutNumbers) {
            processedText = this.spellOutNumbers(processedText);
        }

        // Handle emphasis
        if (options.addEmphasis) {
            // Add SSML tags for emphasis
            processedText = processedText.replace(/\*(.+?)\*/g, '<emphasis level="strong">$1</emphasis>');
        }

        // Handle pauses
        if (options.addPauses) {
            processedText = processedText.replace(/\.\s+/g, '. <break time="500ms"/> ');
            processedText = processedText.replace(/,\s+/g, ', <break time="300ms"/> ');
        }

        return processedText;
    }

    spellOutNumbers(text) {
        const ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
        const tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
        const teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];

        return text.replace(/\b(\d{1,3})\b/g, (match, num) => {
            const n = parseInt(num);
            if (n === 0) return 'zero';
            if (n < 10) return ones[n];
            if (n < 20) return teens[n - 10];
            if (n < 100) {
                const tensDigit = Math.floor(n / 10);
                const onesDigit = n % 10;
                return tens[tensDigit] + (onesDigit ? ' ' + ones[onesDigit] : '');
            }
            if (n < 1000) {
                const hundreds = Math.floor(n / 100);
                const remainder = n % 100;
                let result = ones[hundreds] + ' hundred';
                if (remainder > 0) {
                    if (remainder < 10) result += ' and ' + ones[remainder];
                    else if (remainder < 20) result += ' and ' + teens[remainder - 10];
                    else {
                        const tensDigit = Math.floor(remainder / 10);
                        const onesDigit = remainder % 10;
                        result += ' and ' + tens[tensDigit] + (onesDigit ? ' ' + ones[onesDigit] : '');
                    }
                }
                return result;
            }
            return match; // Return original for numbers >= 1000
        });
    }

    // SSML Support
    speakSSML(ssml, options = {}) {
        const utterance = new SpeechSynthesisUtterance();
        utterance.text = ssml;
        
        // Apply options
        utterance.rate = options.rate || this.options.rate;
        utterance.pitch = options.pitch || this.options.pitch;
        utterance.volume = options.volume || this.options.volume;
        utterance.voice = options.voice || this.options.voice;

        return new Promise((resolve, reject) => {
            utterance.onend = () => resolve();
            utterance.onerror = (event) => reject(event);
            this.synthesis.speak(utterance);
        });
    }

    // Voice characteristics analysis
    analyzeVoice(voice) {
        if (!voice) return null;

        return {
            name: voice.name,
            lang: voice.lang,
            localService: voice.localService,
            default: voice.default,
            voiceURI: voice.voiceURI,
            gender: this.detectGender(voice.name),
            quality: voice.localService ? 'standard' : 'premium',
            languageDetails: this.parseLanguage(voice.lang)
        };
    }

    detectGender(voiceName) {
        const femaleIndicators = ['female', 'woman', 'girl', 'lady', 'fem', 'she', 'her'];
        const maleIndicators = ['male', 'man', 'boy', 'guy', 'mas', 'he', 'him'];
        
        const lowerName = voiceName.toLowerCase();
        
        if (femaleIndicators.some(indicator => lowerName.includes(indicator))) {
            return 'female';
        }
        if (maleIndicators.some(indicator => lowerName.includes(indicator))) {
            return 'male';
        }
        
        return 'neutral';
    }

    parseLanguage(langCode) {
        const parts = langCode.split('-');
        return {
            language: parts[0],
            region: parts[1] || null,
            variant: parts[2] || null
        };
    }

    // Export voice list to JSON
    exportVoices() {
        return this.voices.map(voice => this.analyzeVoice(voice));
    }

    // Check if synthesis is supported
    static isSupported() {
        return 'speechSynthesis' in window;
    }

    // Get synthesis capabilities
    static getCapabilities() {
        if (!this.isSupported()) {
            return { supported: false };
        }

        return {
            supported: true,
            canPause: true,
            canResume: true,
            canGetVoices: true,
            hasBoundaryEvents: 'onboundary' in SpeechSynthesisUtterance.prototype,
            hasMarkEvents: 'onmark' in SpeechSynthesisUtterance.prototype,
            maxRate: 10,
            minRate: 0.1,
            maxPitch: 2,
            minPitch: 0,
            maxVolume: 1,
            minVolume: 0
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceSynthesis;
}