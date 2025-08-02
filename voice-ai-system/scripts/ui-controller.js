// UI Controller Module
// Manages all UI interactions and updates

class UIController {
    constructor() {
        this.elements = {};
        this.transcriptEntries = [];
        this.isListening = false;
        this.currentLanguage = 'en-US';
        this.settings = {
            continuous: true,
            interimResults: true,
            autoExecute: false,
            saveTranscripts: false,
            analyticsEnabled: false,
            confidenceThreshold: 0.8,
            maxAlternatives: 3
        };
        
        this.init();
    }

    init() {
        this.cacheElements();
        this.loadSettings();
        this.setupWaveformCanvas();
    }

    cacheElements() {
        // Main controls
        this.elements.micButton = document.getElementById('micButton');
        this.elements.micIcon = this.elements.micButton.querySelector('.mic-icon');
        this.elements.statusIndicator = document.getElementById('statusIndicator');
        this.elements.statusDot = this.elements.statusIndicator.querySelector('.status-dot');
        this.elements.statusText = this.elements.statusIndicator.querySelector('.status-text');
        this.elements.languageSelect = document.getElementById('languageSelect');

        // Transcript
        this.elements.transcriptContainer = document.getElementById('transcriptContainer');
        this.elements.clearTranscriptBtn = document.getElementById('clearTranscriptBtn');
        this.elements.exportTranscriptBtn = document.getElementById('exportTranscriptBtn');

        // Command preview
        this.elements.commandPreview = document.getElementById('commandPreview');
        this.elements.commandIntent = document.getElementById('commandIntent');
        this.elements.commandEntities = document.getElementById('commandEntities');
        this.elements.confidenceBadge = document.getElementById('confidenceBadge');
        this.elements.executeCommandBtn = document.getElementById('executeCommandBtn');
        this.elements.cancelCommandBtn = document.getElementById('cancelCommandBtn');

        // Voice synthesis
        this.elements.ttsInput = document.getElementById('ttsInput');
        this.elements.voiceSelect = document.getElementById('voiceSelect');
        this.elements.rateSlider = document.getElementById('rateSlider');
        this.elements.rateValue = document.getElementById('rateValue');
        this.elements.pitchSlider = document.getElementById('pitchSlider');
        this.elements.pitchValue = document.getElementById('pitchValue');
        this.elements.volumeSlider = document.getElementById('volumeSlider');
        this.elements.volumeValue = document.getElementById('volumeValue');
        this.elements.speakBtn = document.getElementById('speakBtn');

        // Settings
        this.elements.settingsBtn = document.getElementById('settingsBtn');
        this.elements.settingsModal = document.getElementById('settingsModal');
        this.elements.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        this.elements.continuousMode = document.getElementById('continuousMode');
        this.elements.interimResults = document.getElementById('interimResults');
        this.elements.autoExecute = document.getElementById('autoExecute');
        this.elements.saveTranscripts = document.getElementById('saveTranscripts');
        this.elements.analyticsEnabled = document.getElementById('analyticsEnabled');
        this.elements.confidenceThreshold = document.getElementById('confidenceThreshold');
        this.elements.confidenceValue = document.getElementById('confidenceValue');
        this.elements.maxAlternatives = document.getElementById('maxAlternatives');

        // Info
        this.elements.infoBtn = document.getElementById('infoBtn');
        this.elements.infoModal = document.getElementById('infoModal');
        this.elements.closeInfoBtn = document.getElementById('closeInfoBtn');

        // Waveform
        this.elements.waveformCanvas = document.getElementById('waveformCanvas');
    }

    setupWaveformCanvas() {
        if (!this.elements.waveformCanvas) return;

        const resizeCanvas = () => {
            const container = this.elements.waveformCanvas.parentElement;
            this.elements.waveformCanvas.width = container.offsetWidth;
            this.elements.waveformCanvas.height = container.offsetHeight;
        };

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        this.waveformCtx = this.elements.waveformCanvas.getContext('2d');
    }

    loadSettings() {
        // Load settings from localStorage
        const savedSettings = localStorage.getItem('voiceAISettings');
        if (savedSettings) {
            try {
                const parsed = JSON.parse(savedSettings);
                this.settings = { ...this.settings, ...parsed };
                this.applySettings();
            } catch (error) {
                console.error('Error loading settings:', error);
            }
        }
    }

    saveSettings() {
        localStorage.setItem('voiceAISettings', JSON.stringify(this.settings));
    }

    applySettings() {
        // Apply settings to UI elements
        this.elements.continuousMode.checked = this.settings.continuous;
        this.elements.interimResults.checked = this.settings.interimResults;
        this.elements.autoExecute.checked = this.settings.autoExecute;
        this.elements.saveTranscripts.checked = this.settings.saveTranscripts;
        this.elements.analyticsEnabled.checked = this.settings.analyticsEnabled;
        this.elements.confidenceThreshold.value = this.settings.confidenceThreshold;
        this.elements.confidenceValue.textContent = Math.round(this.settings.confidenceThreshold * 100) + '%';
        this.elements.maxAlternatives.value = this.settings.maxAlternatives;
    }

    updateListeningState(isListening) {
        this.isListening = isListening;
        
        if (isListening) {
            this.elements.micButton.classList.add('active');
            this.elements.micIcon.textContent = 'mic';
            this.elements.statusDot.classList.add('active');
            this.elements.statusText.textContent = 'Listening...';
            
            // Clear welcome message if present
            const welcomeMessage = this.elements.transcriptContainer.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
        } else {
            this.elements.micButton.classList.remove('active');
            this.elements.micIcon.textContent = 'mic_off';
            this.elements.statusDot.classList.remove('active');
            this.elements.statusText.textContent = 'Ready';
            
            // Clear waveform
            if (this.waveformCtx) {
                this.waveformCtx.clearRect(0, 0, this.elements.waveformCanvas.width, this.elements.waveformCanvas.height);
            }
        }
    }

    updateStatus(message, type = 'info') {
        this.elements.statusText.textContent = message;
        
        // Update status dot color based on type
        this.elements.statusDot.classList.remove('active', 'error', 'warning');
        
        switch (type) {
            case 'error':
                this.elements.statusDot.style.backgroundColor = 'var(--error-color)';
                break;
            case 'warning':
                this.elements.statusDot.style.backgroundColor = 'var(--warning-color)';
                break;
            case 'success':
                this.elements.statusDot.classList.add('active');
                break;
        }
    }

    addTranscriptEntry(text, type = 'user', isFinal = true) {
        // Check if we should update the last entry or add a new one
        const lastEntry = this.transcriptEntries[this.transcriptEntries.length - 1];
        
        if (!isFinal && lastEntry && lastEntry.type === type && !lastEntry.isFinal) {
            // Update existing interim entry
            lastEntry.text = text;
            lastEntry.element.querySelector('.transcript-text').textContent = text;
            return;
        }

        // Create new entry
        const entry = {
            id: Date.now().toString(),
            text: text,
            type: type,
            isFinal: isFinal,
            timestamp: new Date()
        };

        const entryElement = document.createElement('div');
        entryElement.className = `transcript-entry ${type}${!isFinal ? ' interim' : ''}`;
        entryElement.innerHTML = `
            <div class="transcript-time">${entry.timestamp.toLocaleTimeString()}</div>
            <div class="transcript-text">${text}</div>
        `;

        this.elements.transcriptContainer.appendChild(entryElement);
        entry.element = entryElement;
        
        this.transcriptEntries.push(entry);
        
        // Auto-scroll to bottom
        this.elements.transcriptContainer.scrollTop = this.elements.transcriptContainer.scrollHeight;
        
        // Save transcript if enabled
        if (this.settings.saveTranscripts) {
            this.saveTranscriptToStorage();
        }
    }

    clearTranscript() {
        this.transcriptEntries = [];
        this.elements.transcriptContainer.innerHTML = `
            <div class="welcome-message">
                <span class="material-icons">record_voice_over</span>
                <p>Click the microphone to start speaking</p>
                <p class="hint">Try saying: "What's the weather today?" or "Set a timer for 5 minutes"</p>
            </div>
        `;
        
        if (this.settings.saveTranscripts) {
            localStorage.removeItem('voiceAITranscript');
        }
    }

    exportTranscript() {
        if (this.transcriptEntries.length === 0) {
            alert('No transcript to export');
            return;
        }

        // Create transcript text
        const transcriptText = this.transcriptEntries.map(entry => {
            const time = new Date(entry.timestamp).toLocaleString();
            const speaker = entry.type === 'user' ? 'User' : 'AI';
            return `[${time}] ${speaker}: ${entry.text}`;
        }).join('\n\n');

        // Create and download file
        const blob = new Blob([transcriptText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voice-ai-transcript-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    saveTranscriptToStorage() {
        const transcriptData = this.transcriptEntries.map(entry => ({
            text: entry.text,
            type: entry.type,
            timestamp: entry.timestamp,
            isFinal: entry.isFinal
        }));
        
        localStorage.setItem('voiceAITranscript', JSON.stringify(transcriptData));
    }

    loadTranscriptFromStorage() {
        const saved = localStorage.getItem('voiceAITranscript');
        if (saved) {
            try {
                const transcriptData = JSON.parse(saved);
                transcriptData.forEach(entry => {
                    this.addTranscriptEntry(entry.text, entry.type, entry.isFinal);
                });
            } catch (error) {
                console.error('Error loading transcript:', error);
            }
        }
    }

    showCommandPreview(intent, entities, confidence) {
        this.elements.commandIntent.textContent = intent;
        this.elements.commandEntities.textContent = Object.entries(entities)
            .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
            .join(', ') || 'None';
        this.elements.confidenceBadge.textContent = Math.round(confidence * 100) + '%';
        
        this.elements.commandPreview.style.display = 'block';
        
        // Auto-execute if enabled and confidence is high enough
        if (this.settings.autoExecute && confidence >= this.settings.confidenceThreshold) {
            setTimeout(() => {
                this.elements.executeCommandBtn.click();
            }, 1000);
        }
    }

    hideCommandPreview() {
        this.elements.commandPreview.style.display = 'none';
    }

    updateVoicesList(voices) {
        this.elements.voiceSelect.innerHTML = '';
        
        // Group voices by language
        const voicesByLang = {};
        voices.forEach(voice => {
            const lang = voice.lang.split('-')[0];
            if (!voicesByLang[lang]) {
                voicesByLang[lang] = [];
            }
            voicesByLang[lang].push(voice);
        });

        // Create optgroups
        Object.entries(voicesByLang).forEach(([lang, langVoices]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = this.getLanguageName(lang);
            
            langVoices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.name;
                option.textContent = `${voice.name} (${voice.lang})`;
                option.dataset.lang = voice.lang;
                option.dataset.local = voice.localService;
                
                // Mark premium voices
                if (!voice.localService) {
                    option.textContent += ' ⭐';
                }
                
                optgroup.appendChild(option);
            });
            
            this.elements.voiceSelect.appendChild(optgroup);
        });
    }

    getLanguageName(code) {
        const languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        };
        
        return languages[code] || code.toUpperCase();
    }

    updateSynthesisControls() {
        // Update displayed values
        this.elements.rateValue.textContent = this.elements.rateSlider.value + 'x';
        this.elements.pitchValue.textContent = this.elements.pitchSlider.value + 'x';
        this.elements.volumeValue.textContent = Math.round(this.elements.volumeSlider.value * 100) + '%';
    }

    visualizeAudioLevel(level) {
        if (!this.waveformCtx || !this.isListening) return;

        const canvas = this.elements.waveformCanvas;
        const ctx = this.waveformCtx;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw waveform
        const barWidth = 4;
        const barGap = 2;
        const barCount = Math.floor(width / (barWidth + barGap));
        const centerY = height / 2;

        ctx.fillStyle = 'var(--primary-color)';

        for (let i = 0; i < barCount; i++) {
            const x = i * (barWidth + barGap);
            const randomFactor = 0.5 + Math.random() * 0.5;
            const barHeight = level * height * randomFactor;
            
            ctx.fillRect(x, centerY - barHeight / 2, barWidth, barHeight);
        }
    }

    showModal(modalElement) {
        modalElement.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    hideModal(modalElement) {
        modalElement.classList.remove('show');
        document.body.style.overflow = '';
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span class="material-icons">${this.getNotificationIcon(type)}</span>
            <span>${message}</span>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            padding: 1rem 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-md);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

        // Add to body
        document.body.appendChild(notification);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            'info': 'info',
            'success': 'check_circle',
            'warning': 'warning',
            'error': 'error'
        };
        
        return icons[type] || 'info';
    }

    updateSettingsUI() {
        // Update confidence threshold display
        this.elements.confidenceValue.textContent = Math.round(this.elements.confidenceThreshold.value * 100) + '%';
    }

    highlightLanguage(language) {
        // Update language selector to match current language
        this.elements.languageSelect.value = language;
    }

    enableControls() {
        this.elements.micButton.disabled = false;
        this.elements.languageSelect.disabled = false;
        this.elements.speakBtn.disabled = false;
    }

    disableControls() {
        this.elements.micButton.disabled = true;
        this.elements.languageSelect.disabled = true;
        this.elements.speakBtn.disabled = true;
    }

    showError(error) {
        this.updateStatus(error, 'error');
        this.showNotification(error, 'error');
    }

    showSuccess(message) {
        this.updateStatus(message, 'success');
        this.showNotification(message, 'success');
    }

    getSettings() {
        return this.settings;
    }

    updateSetting(key, value) {
        this.settings[key] = value;
        this.saveSettings();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIController;
}