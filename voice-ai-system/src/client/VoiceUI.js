/**
 * Advanced Voice UI Components
 * Responsive voice interface components for the voice AI system
 */

export class VoiceUI {
  constructor(container, config = {}) {
    this.container = typeof container === 'string' ? document.getElementById(container) : container;
    
    this.config = {
      theme: config.theme || 'dark',
      primaryColor: config.primaryColor || '#007bff',
      accentColor: config.accentColor || '#28a745',
      errorColor: config.errorColor || '#dc3545',
      warningColor: config.warningColor || '#ffc107',
      
      // Voice visualization
      enableWaveform: config.enableWaveform !== false,
      enableEmotionDisplay: config.enableEmotionDisplay !== false,
      enableLanguageSelector: config.enableLanguageSelector !== false,
      
      // Layout
      compactMode: config.compactMode || false,
      showAdvancedControls: config.showAdvancedControls !== false,
      
      ...config
    };

    this.state = {
      isListening: false,
      isConnected: false,
      currentLanguage: 'en-US',
      detectedEmotion: null,
      audioLevel: 0,
      transcription: '',
      translations: new Map(),
      mediaResults: [],
      biometricStatus: null
    };

    this.callbacks = {
      onVoiceStart: null,
      onVoiceStop: null,
      onLanguageChange: null,
      onMediaQuery: null,
      onEditCommand: null,
      onBiometricEnroll: null
    };

    this.components = {};
    this.animationFrame = null;
    
    this.init();
  }

  /**
   * Initialize the voice UI
   */
  init() {
    this.createMainLayout();
    this.createVoiceControls();
    this.createVisualization();
    this.createTranscriptionPanel();
    this.createTranslationPanel();
    this.createMediaPanel();
    this.createSettingsPanel();
    this.createEmotionDisplay();
    this.createBiometricPanel();
    
    this.applyTheme();
    this.setupEventListeners();
    this.startAnimationLoop();
  }

  /**
   * Create main layout structure
   */
  createMainLayout() {
    this.container.innerHTML = `
      <div class="voice-ui ${this.config.theme}" data-compact="${this.config.compactMode}">
        <div class="voice-ui-header">
          <div class="connection-status">
            <div class="status-indicator"></div>
            <span class="status-text">Disconnected</span>
          </div>
          <div class="language-selector-container"></div>
          <div class="settings-toggle">
            <button class="btn-icon" id="settingsToggle" title="Settings">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="currentColor" d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11.03L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.22,8.95 2.27,9.22 2.46,9.37L4.57,11.03C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.22,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.68 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"/>
              </svg>
            </button>
          </div>
        </div>
        
        <div class="voice-ui-main">
          <div class="voice-controls-container"></div>
          <div class="visualization-container"></div>
          <div class="emotion-display-container"></div>
        </div>
        
        <div class="voice-ui-panels">
          <div class="transcription-panel-container"></div>
          <div class="translation-panel-container"></div>
          <div class="media-panel-container"></div>
          <div class="biometric-panel-container"></div>
        </div>
        
        <div class="settings-panel-container" style="display: none;"></div>
      </div>
    `;

    this.components.main = this.container.querySelector('.voice-ui');
    this.components.header = this.container.querySelector('.voice-ui-header');
    this.components.mainArea = this.container.querySelector('.voice-ui-main');
    this.components.panels = this.container.querySelector('.voice-ui-panels');
  }

  /**
   * Create voice control buttons
   */
  createVoiceControls() {
    const container = this.container.querySelector('.voice-controls-container');
    
    container.innerHTML = `
      <div class="voice-controls">
        <button class="voice-btn primary" id="voiceToggle" disabled>
          <div class="voice-btn-content">
            <div class="voice-icon">
              <svg class="mic-icon" viewBox="0 0 24 24" width="32" height="32">
                <path fill="currentColor" d="M12,2A3,3 0 0,1 15,5V11A3,3 0 0,1 12,14A3,3 0 0,1 9,11V5A3,3 0 0,1 12,2M19,11C19,14.53 16.39,17.44 13,17.93V21H11V17.93C7.61,17.44 5,14.53 5,11H7A5,5 0 0,0 12,16A5,5 0 0,0 17,11H19Z"/>
              </svg>
              <svg class="stop-icon" viewBox="0 0 24 24" width="32" height="32" style="display: none;">
                <path fill="currentColor" d="M18,18H6V6H18V18Z"/>
              </svg>
            </div>
            <span class="voice-btn-text">Start Listening</span>
          </div>
          <div class="voice-btn-pulse"></div>
        </button>
        
        ${this.config.showAdvancedControls ? `
          <div class="advanced-controls">
            <button class="btn-secondary" id="biometricBtn" title="Voice Biometric">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="currentColor" d="M9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9A3,3 0 0,0 9,12M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z"/>
              </svg>
            </button>
            
            <button class="btn-secondary" id="mediaBtn" title="Media Library">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="currentColor" d="M12,11A1,1 0 0,0 11,12A1,1 0 0,0 12,13A1,1 0 0,0 13,12A1,1 0 0,0 12,11M12.5,2C17,2 20.5,5.5 20.5,10C20.5,12.5 19.4,14.7 17.7,16.2L12.5,22L7.3,16.2C5.6,14.7 4.5,12.5 4.5,10C4.5,5.5 8,2 12.5,2Z"/>
              </svg>
            </button>
            
            <button class="btn-secondary" id="translateBtn" title="Translation">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="currentColor" d="M12.87,15.07L10.33,12.56L10.36,12.53C12.1,10.59 13.34,8.36 14.07,6H17V4H10V2H8V4H1V6H12.17C11.5,7.92 10.44,9.75 9,11.35C8.07,10.32 7.3,9.19 6.69,8H4.69C5.42,9.63 6.42,11.17 7.67,12.56L2.58,17.58L4,19L9,14L12.11,17.11L12.87,15.07M18.5,10H16.5L12,22H14L15.12,19H19.87L21,22H23L18.5,10M15.88,17L17.5,12.67L19.12,17H15.88Z"/>
              </svg>
            </button>
          </div>
        ` : ''}
      </div>
    `;

    this.components.voiceToggle = container.querySelector('#voiceToggle');
    this.components.biometricBtn = container.querySelector('#biometricBtn');
    this.components.mediaBtn = container.querySelector('#mediaBtn');
    this.components.translateBtn = container.querySelector('#translateBtn');
  }

  /**
   * Create audio visualization
   */
  createVisualization() {
    if (!this.config.enableWaveform) return;

    const container = this.container.querySelector('.visualization-container');
    
    container.innerHTML = `
      <div class="audio-visualization">
        <canvas class="waveform-canvas" width="400" height="100"></canvas>
        <div class="audio-level-meter">
          <div class="level-bar"></div>
        </div>
      </div>
    `;

    this.components.canvas = container.querySelector('.waveform-canvas');
    this.components.levelBar = container.querySelector('.level-bar');
    this.components.canvasContext = this.components.canvas.getContext('2d');
    
    this.setupCanvas();
  }

  /**
   * Create transcription panel
   */
  createTranscriptionPanel() {
    const container = this.container.querySelector('.transcription-panel-container');
    
    container.innerHTML = `
      <div class="panel transcription-panel">
        <div class="panel-header">
          <h3>Transcription</h3>
          <div class="panel-controls">
            <button class="btn-icon" id="clearTranscription" title="Clear">
              <svg viewBox="0 0 24 24" width="16" height="16">
                <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="panel-content">
          <div class="transcription-display">
            <div class="transcription-text" id="transcriptionText">
              <div class="placeholder">Voice transcription will appear here...</div>
            </div>
            <div class="transcription-info">
              <span class="confidence-score">Confidence: <span id="confidenceScore">--</span></span>
              <span class="language-detected">Language: <span id="detectedLanguage">--</span></span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.components.transcriptionText = container.querySelector('#transcriptionText');
    this.components.confidenceScore = container.querySelector('#confidenceScore');
    this.components.detectedLanguage = container.querySelector('#detectedLanguage');
  }

  /**
   * Create translation panel  
   */
  createTranslationPanel() {
    const container = this.container.querySelector('.translation-panel-container');
    
    container.innerHTML = `
      <div class="panel translation-panel">
        <div class="panel-header">
          <h3>Real-time Translation</h3>
          <div class="panel-controls">
            <select class="language-select" id="targetLanguageSelect">
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="it">Italian</option>
              <option value="pt">Portuguese</option>
              <option value="ru">Russian</option>
              <option value="zh">Chinese</option>
              <option value="ja">Japanese</option>
              <option value="ko">Korean</option>
              <option value="ar">Arabic</option>
            </select>
          </div>
        </div>
        <div class="panel-content">
          <div class="translation-display" id="translationDisplay">
            <div class="placeholder">Translations will appear here...</div>
          </div>
        </div>
      </div>
    `;

    this.components.translationDisplay = container.querySelector('#translationDisplay');
    this.components.targetLanguageSelect = container.querySelector('#targetLanguageSelect');
  }

  /**
   * Create media panel
   */
  createMediaPanel() {
    const container = this.container.querySelector('.media-panel-container');
    
    container.innerHTML = `
      <div class="panel media-panel">
        <div class="panel-header">
          <h3>Media Library</h3>
          <div class="panel-controls">
            <input type="text" class="search-input" id="mediaSearchInput" placeholder="Ask about your media...">
            <button class="btn-primary" id="mediaSearchBtn">Search</button>
          </div>
        </div>
        <div class="panel-content">
          <div class="media-results" id="mediaResults">
            <div class="placeholder">
              Try saying: "Find my vacation videos" or "Play jazz music"
            </div>
          </div>
        </div>
      </div>
    `;

    this.components.mediaResults = container.querySelector('#mediaResults');
    this.components.mediaSearchInput = container.querySelector('#mediaSearchInput');
    this.components.mediaSearchBtn = container.querySelector('#mediaSearchBtn');
  }

  /**
   * Create emotion display
   */
  createEmotionDisplay() {
    if (!this.config.enableEmotionDisplay) return;

    const container = this.container.querySelector('.emotion-display-container');
    
    container.innerHTML = `
      <div class="emotion-display">
        <div class="emotion-indicator">
          <div class="emotion-icon" id="emotionIcon">😐</div>
          <div class="emotion-info">
            <div class="emotion-name" id="emotionName">Neutral</div>
            <div class="emotion-confidence" id="emotionConfidence">--</div>
          </div>
        </div>
        <div class="sentiment-bar">
          <div class="sentiment-fill" id="sentimentFill"></div>
          <div class="sentiment-labels">
            <span>Negative</span>
            <span>Neutral</span>
            <span>Positive</span>
          </div>
        </div>
      </div>
    `;

    this.components.emotionIcon = container.querySelector('#emotionIcon');
    this.components.emotionName = container.querySelector('#emotionName');
    this.components.emotionConfidence = container.querySelector('#emotionConfidence'); 
    this.components.sentimentFill = container.querySelector('#sentimentFill');
  }

  /**
   * Create biometric panel
   */
  createBiometricPanel() {
    const container = this.container.querySelector('.biometric-panel-container');
    
    container.innerHTML = `
      <div class="panel biometric-panel" style="display: none;">
        <div class="panel-header">
          <h3>Voice Biometric Authentication</h3>
          <button class="btn-icon panel-close" id="closeBiometric">
            <svg viewBox="0 0 24 24" width="16" height="16">
              <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z"/>
            </svg>
          </button>
        </div>
        <div class="panel-content">
          <div class="biometric-status" id="biometricStatus">
            <div class="status-icon">🔐</div>
            <div class="status-text">Not enrolled</div>
          </div>
          <div class="biometric-controls">
            <button class="btn-primary" id="enrollBiometric">Enroll Voice</button>
            <button class="btn-secondary" id="verifyBiometric" disabled>Verify Identity</button>
          </div>
          <div class="biometric-progress" id="biometricProgress" style="display: none;">
            <div class="progress-bar">
              <div class="progress-fill"></div>
            </div>
            <div class="progress-text">Collecting voice samples...</div>
          </div>
        </div>
      </div>
    `;

    this.components.biometricPanel = container.querySelector('.biometric-panel');
    this.components.biometricStatus = container.querySelector('#biometricStatus');
    this.components.enrollBiometric = container.querySelector('#enrollBiometric');
    this.components.verifyBiometric = container.querySelector('#verifyBiometric');
    this.components.biometricProgress = container.querySelector('#biometricProgress');
  }

  /**
   * Create settings panel
   */
  createSettingsPanel() {
    const container = this.container.querySelector('.settings-panel-container');
    
    container.innerHTML = `
      <div class="panel settings-panel">
        <div class="panel-header">
          <h3>Settings</h3>
          <button class="btn-icon panel-close" id="closeSettings">
            <svg viewBox="0 0 24 24" width="16" height="16">
              <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z"/>
            </svg>
          </button>
        </div>
        <div class="panel-content">
          <div class="settings-section">
            <h4>Voice Settings</h4>
            <div class="setting-item">
              <label for="inputLanguage">Input Language:</label>
              <select id="inputLanguage" class="language-select">
                <option value="en-US">English (US)</option>
                <option value="en-GB">English (UK)</option>
                <option value="es-ES">Spanish</option>
                <option value="fr-FR">French</option>
                <option value="de-DE">German</option>
                <option value="it-IT">Italian</option>
                <option value="pt-PT">Portuguese</option>
                <option value="ru-RU">Russian</option>
                <option value="zh-CN">Chinese</option>
                <option value="ja-JP">Japanese</option>
                <option value="ko-KR">Korean</option>
                <option value="ar-SA">Arabic</option>
              </select>
            </div>
            
            <div class="setting-item">
              <label for="enableEmotionDetection">
                <input type="checkbox" id="enableEmotionDetection" checked>
                Enable Emotion Detection
              </label>
            </div>
            
            <div class="setting-item">
              <label for="enableRealTimeTranslation">
                <input type="checkbox" id="enableRealTimeTranslation" checked>
                Enable Real-time Translation
              </label>
            </div>
          </div>
          
          <div class="settings-section">
            <h4>UI Settings</h4>
            <div class="setting-item">
              <label for="theme">Theme:</label>
              <select id="theme" class="theme-select">
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="auto">Auto</option>
              </select>
            </div>
            
            <div class="setting-item">
              <label for="compactMode">
                <input type="checkbox" id="compactMode">
                Compact Mode
              </label>
            </div>
          </div>
          
          <div class="settings-section">
            <h4>Advanced</h4>
            <div class="setting-item">
              <label for="debugMode">
                <input type="checkbox" id="debugMode">
                Debug Mode
              </label>
            </div>
          </div>
        </div>
      </div>
    `;

    this.components.settingsPanel = container.querySelector('.settings-panel');
    this.components.inputLanguageSelect = container.querySelector('#inputLanguage');
    this.components.themeSelect = container.querySelector('#theme');
  }

  /**
   * Setup canvas for visualization
   */
  setupCanvas() {
    if (!this.components.canvas) return;

    const canvas = this.components.canvas;
    const dpr = window.devicePixelRatio || 1;
    
    // Set actual size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    // Scale context
    const ctx = this.components.canvasContext;
    ctx.scale(dpr, dpr);
    
    // Set canvas CSS size
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Voice toggle button
    if (this.components.voiceToggle) {
      this.components.voiceToggle.addEventListener('click', () => {
        this.toggleVoiceListening();
      });
    }

    // Settings toggle
    const settingsToggle = this.container.querySelector('#settingsToggle');
    if (settingsToggle) {
      settingsToggle.addEventListener('click', () => {
        this.toggleSettings();
      });
    }

    // Close buttons
    const closeButtons = this.container.querySelectorAll('.panel-close');
    closeButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const panel = e.target.closest('.panel');
        if (panel) {
          panel.style.display = 'none';
        }
      });
    });

    // Language changes
    if (this.components.inputLanguageSelect) {
      this.components.inputLanguageSelect.addEventListener('change', (e) => {
        this.handleLanguageChange(e.target.value);
      });
    }

    // Media search
    if (this.components.mediaSearchBtn) {
      this.components.mediaSearchBtn.addEventListener('click', () => {
        this.handleMediaSearch();
      });
    }

    if (this.components.mediaSearchInput) {
      this.components.mediaSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          this.handleMediaSearch();
        }
      });
    }

    // Translation target language
    if (this.components.targetLanguageSelect) {
      this.components.targetLanguageSelect.addEventListener('change', (e) => {
        this.updateTranslationTarget(e.target.value);
      });
    }

    // Advanced control buttons
    if (this.components.biometricBtn) {
      this.components.biometricBtn.addEventListener('click', () => {
        this.toggleBiometricPanel();
      });
    }

    if (this.components.mediaBtn) {
      this.components.mediaBtn.addEventListener('click', () => {
        this.focusMediaPanel();
      });
    }

    // Biometric controls
    if (this.components.enrollBiometric) {
      this.components.enrollBiometric.addEventListener('click', () => {
        this.startBiometricEnrollment();
      });
    }

    if (this.components.verifyBiometric) {
      this.components.verifyBiometric.addEventListener('click', () => {
        this.startBiometricVerification();
      });
    }

    // Window resize
    window.addEventListener('resize', () => {
      this.handleResize();
    });
  }

  /**
   * Apply theme styles
   */
  applyTheme() {
    const root = document.documentElement;
    
    if (this.config.theme === 'dark') {
      root.style.setProperty('--bg-primary', '#1a1a1a');
      root.style.setProperty('--bg-secondary', '#2d2d2d');
      root.style.setProperty('--text-primary', '#ffffff');
      root.style.setProperty('--text-secondary', '#cccccc');
      root.style.setProperty('--border-color', '#404040');
    } else {
      root.style.setProperty('--bg-primary', '#ffffff');
      root.style.setProperty('--bg-secondary', '#f8f9fa');
      root.style.setProperty('--text-primary', '#212529');
      root.style.setProperty('--text-secondary', '#6c757d');
      root.style.setProperty('--border-color', '#dee2e6');
    }
    
    root.style.setProperty('--primary-color', this.config.primaryColor);
    root.style.setProperty('--accent-color', this.config.accentColor);
    root.style.setProperty('--error-color', this.config.errorColor);
    root.style.setProperty('--warning-color', this.config.warningColor);
  }

  /**
   * Start animation loop
   */
  startAnimationLoop() {
    const animate = () => {
      this.updateVisualization();
      this.animationFrame = requestAnimationFrame(animate);
    };
    animate();
  }

  /**
   * Update audio visualization
   */
  updateVisualization() {
    if (!this.components.canvas || !this.config.enableWaveform) return;

    const canvas = this.components.canvas;
    const ctx = this.components.canvasContext;
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw waveform based on audio level
    const audioLevel = this.state.audioLevel;
    const centerY = height / 2;
    const barWidth = 3;
    const barSpacing = 5;
    const numBars = Math.floor(width / (barWidth + barSpacing));

    ctx.fillStyle = this.state.isListening ? this.config.primaryColor : '#666';

    for (let i = 0; i < numBars; i++) {
      const x = i * (barWidth + barSpacing);
      const barHeight = this.state.isListening ? 
        (Math.random() * audioLevel * height * 0.8) + 2 :
        2;
      
      const y = centerY - barHeight / 2;
      
      ctx.fillRect(x, y, barWidth, barHeight);
    }

    // Update audio level meter
    if (this.components.levelBar) {
      const levelPercent = Math.min(audioLevel * 100, 100);
      this.components.levelBar.style.width = levelPercent + '%';
      
      // Change color based on level
      if (levelPercent > 80) {
        this.components.levelBar.style.backgroundColor = this.config.errorColor;
      } else if (levelPercent > 60) {
        this.components.levelBar.style.backgroundColor = this.config.warningColor;
      } else {
        this.components.levelBar.style.backgroundColor = this.config.accentColor;
      }
    }
  }

  /**
   * Toggle voice listening
   */
  toggleVoiceListening() {
    if (!this.state.isConnected) {
      this.showNotification('Please connect to voice service first', 'warning');
      return;
    }

    this.state.isListening = !this.state.isListening;
    this.updateVoiceButton();

    if (this.callbacks.onVoiceStart && this.state.isListening) {
      this.callbacks.onVoiceStart();
    } else if (this.callbacks.onVoiceStop && !this.state.isListening) {
      this.callbacks.onVoiceStop();
    }
  }

  /**
   * Update voice button appearance
   */
  updateVoiceButton() {
    if (!this.components.voiceToggle) return;

    const button = this.components.voiceToggle;
    const micIcon = button.querySelector('.mic-icon');
    const stopIcon = button.querySelector('.stop-icon');
    const text = button.querySelector('.voice-btn-text');
    const pulse = button.querySelector('.voice-btn-pulse');

    if (this.state.isListening) {
      button.classList.add('listening');
      micIcon.style.display = 'none';
      stopIcon.style.display = 'block';
      text.textContent = 'Stop Listening';
      pulse.style.display = 'block';
    } else {
      button.classList.remove('listening');
      micIcon.style.display = 'block';
      stopIcon.style.display = 'none';
      text.textContent = 'Start Listening';
      pulse.style.display = 'none';
    }
  }

  /**
   * Update connection status
   */
  updateConnectionStatus(connected) {
    this.state.isConnected = connected;
    
    const indicator = this.container.querySelector('.status-indicator');
    const text = this.container.querySelector('.status-text');
    
    if (connected) {
      indicator.classList.add('connected');
      text.textContent = 'Connected';
      this.components.voiceToggle.disabled = false;
    } else {
      indicator.classList.remove('connected');
      text.textContent = 'Disconnected';
      this.components.voiceToggle.disabled = true;
      this.state.isListening = false;
      this.updateVoiceButton();
    }
  }

  /**
   * Update transcription display
   */
  updateTranscription(text, confidence, language, isPartial = false) {
    if (!this.components.transcriptionText) return;

    const placeholder = this.components.transcriptionText.querySelector('.placeholder');
    if (placeholder) {
      placeholder.remove();
    }

    if (!isPartial) {
      this.state.transcription = text;
    }

    this.components.transcriptionText.innerHTML = `
      <div class="transcription-final">${this.state.transcription}</div>
      ${isPartial && text ? `<div class="transcription-partial">${text}</div>` : ''}
    `;

    if (this.components.confidenceScore) {
      this.components.confidenceScore.textContent = confidence ? 
        `${Math.round(confidence * 100)}%` : '--';
    }

    if (this.components.detectedLanguage) {
      this.components.detectedLanguage.textContent = language || '--';
    }

    // Auto-scroll
    this.components.transcriptionText.scrollTop = this.components.transcriptionText.scrollHeight;
  }

  /**
   * Update emotion display
   */
  updateEmotion(emotion, confidence, sentiment) {
    if (!this.config.enableEmotionDisplay) return;

    this.state.detectedEmotion = emotion;

    // Emotion icon mapping
    const emotionIcons = {
      joy: '😊',
      happiness: '😄',
      sadness: '😢',
      anger: '😠',
      fear: '😨',
      surprise: '😮',
      disgust: '🤢',
      neutral: '😐',
      excitement: '🤩',
      calm: '😌',
      confused: '🤔'
    };

    if (this.components.emotionIcon) {
      this.components.emotionIcon.textContent = emotionIcons[emotion] || '😐';
    }

    if (this.components.emotionName) {
      this.components.emotionName.textContent = emotion ? 
        emotion.charAt(0).toUpperCase() + emotion.slice(1) : 'Neutral';
    }

    if (this.components.emotionConfidence) {
      this.components.emotionConfidence.textContent = confidence ? 
        `${Math.round(confidence * 100)}%` : '--';
    }

    // Update sentiment bar
    if (this.components.sentimentFill && sentiment !== undefined) {
      const sentimentPercent = ((sentiment + 1) / 2) * 100; // Convert -1 to 1 range to 0-100%
      this.components.sentimentFill.style.width = sentimentPercent + '%';
      
      // Color based on sentiment
      if (sentiment > 0.3) {
        this.components.sentimentFill.style.backgroundColor = this.config.accentColor;
      } else if (sentiment < -0.3) {
        this.components.sentimentFill.style.backgroundColor = this.config.errorColor;
      } else {
        this.components.sentimentFill.style.backgroundColor = this.config.warningColor;
      }
    }
  }

  /**
   * Update audio level
   */
  updateAudioLevel(level) {
    this.state.audioLevel = level;
  }

  /**
   * Add translation
   */
  addTranslation(originalText, translatedText, targetLanguage, confidence) {
    if (!this.components.translationDisplay) return;

    const placeholder = this.components.translationDisplay.querySelector('.placeholder');
    if (placeholder) {
      placeholder.remove();
    }

    const translationItem = document.createElement('div');
    translationItem.className = 'translation-item';
    translationItem.innerHTML = `
      <div class="translation-original">${originalText}</div>
      <div class="translation-arrow">→</div>
      <div class="translation-result">
        <div class="translated-text">${translatedText}</div>
        <div class="translation-info">
          <span class="target-language">${targetLanguage.toUpperCase()}</span>
          <span class="translation-confidence">${Math.round(confidence * 100)}%</span>
        </div>
      </div>
    `;

    this.components.translationDisplay.appendChild(translationItem);

    // Keep only last 5 translations
    const items = this.components.translationDisplay.querySelectorAll('.translation-item');
    if (items.length > 5) {
      items[0].remove();
    }

    // Auto-scroll
    this.components.translationDisplay.scrollTop = this.components.translationDisplay.scrollHeight;
  }

  /**
   * Update media results
   */
  updateMediaResults(results) {
    if (!this.components.mediaResults) return;

    const placeholder = this.components.mediaResults.querySelector('.placeholder');
    if (placeholder) {
      placeholder.remove();
    }

    if (!results || results.length === 0) {
      this.components.mediaResults.innerHTML = '<div class="no-results">No media files found</div>';
      return;
    }

    this.components.mediaResults.innerHTML = results.map(media => `
      <div class="media-item" data-media-id="${media.id}">
        <div class="media-icon">
          ${this.getMediaIcon(media.mediaType)}
        </div>
        <div class="media-info">
          <div class="media-title">${media.fileName}</div>
          <div class="media-details">
            <span class="media-type">${media.mediaType}</span>
            ${media.metadata.duration ? `<span class="media-duration">${this.formatDuration(media.metadata.duration)}</span>` : ''}
            <span class="media-size">${this.formatFileSize(media.size)}</span>
          </div>
          ${media.description ? `<div class="media-description">${media.description}</div>` : ''}
        </div>
        <div class="media-actions">
          <button class="btn-icon play-btn" title="Play">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z"/>
            </svg>
          </button>
        </div>
      </div>
    `).join('');

    // Add click handlers for media items
    this.components.mediaResults.querySelectorAll('.media-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (!e.target.closest('.media-actions')) {
          this.selectMediaItem(item.dataset.mediaId);
        }
      });
    });

    // Add play button handlers
    this.components.mediaResults.querySelectorAll('.play-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const mediaId = e.target.closest('.media-item').dataset.mediaId;
        this.playMedia(mediaId);
      });
    });
  }

  /**
   * Handle language change
   */
  handleLanguageChange(language) {
    this.state.currentLanguage = language;
    
    if (this.callbacks.onLanguageChange) {
      this.callbacks.onLanguageChange(language);
    }
  }

  /**
   * Handle media search
   */
  handleMediaSearch() {
    const query = this.components.mediaSearchInput.value.trim();
    if (!query) return;

    if (this.callbacks.onMediaQuery) {
      this.callbacks.onMediaQuery(query);
    }

    // Clear input
    this.components.mediaSearchInput.value = '';
  }

  /**
   * Toggle settings panel
   */
  toggleSettings() {
    const panel = this.container.querySelector('.settings-panel-container');
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : 'block';
  }

  /**
   * Toggle biometric panel
   */
  toggleBiometricPanel() {
    if (this.components.biometricPanel) {
      const isVisible = this.components.biometricPanel.style.display !== 'none';
      this.components.biometricPanel.style.display = isVisible ? 'none' : 'block';
    }
  }

  /**
   * Focus media panel
   */
  focusMediaPanel() {
    const mediaPanel = this.container.querySelector('.media-panel');
    if (mediaPanel) {
      mediaPanel.scrollIntoView({ behavior: 'smooth' });
      this.components.mediaSearchInput.focus();
    }
  }

  /**
   * Start biometric enrollment
   */
  startBiometricEnrollment() {
    if (this.callbacks.onBiometricEnroll) {
      this.callbacks.onBiometricEnroll('enroll');
    }
    
    this.updateBiometricStatus('enrolling', 'Enrolling voice biometric...');
  }

  /**
   * Start biometric verification
   */
  startBiometricVerification() {
    if (this.callbacks.onBiometricEnroll) {
      this.callbacks.onBiometricEnroll('verify');
    }
    
    this.updateBiometricStatus('verifying', 'Verifying identity...');
  }

  /**
   * Update biometric status
   */
  updateBiometricStatus(status, message) {
    this.state.biometricStatus = status;
    
    if (this.components.biometricStatus) {
      const statusIcon = this.components.biometricStatus.querySelector('.status-icon');
      const statusText = this.components.biometricStatus.querySelector('.status-text');
      
      const icons = {
        'not_enrolled': '🔐',
        'enrolling': '🎙️',
        'enrolled': '✅',
        'verifying': '🔍',
        'verified': '✅',
        'failed': '❌'
      };
      
      statusIcon.textContent = icons[status] || '🔐';
      statusText.textContent = message || status;
    }
    
    // Show/hide progress
    if (this.components.biometricProgress) {
      const shouldShow = ['enrolling', 'verifying'].includes(status);
      this.components.biometricProgress.style.display = shouldShow ? 'block' : 'none';
    }
    
    // Update button states
    if (this.components.enrollBiometric && this.components.verifyBiometric) {
      const isProcessing = ['enrolling', 'verifying'].includes(status);
      const isEnrolled = ['enrolled', 'verified'].includes(status);
      
      this.components.enrollBiometric.disabled = isProcessing;
      this.components.verifyBiometric.disabled = !isEnrolled || isProcessing;
    }
  }

  /**
   * Show notification
   */
  showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let notification = this.container.querySelector('.notification');
    if (!notification) {
      notification = document.createElement('div');
      notification.className = 'notification';
      this.container.appendChild(notification);
    }

    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.display = 'block';

    // Auto-hide after 3 seconds
    setTimeout(() => {
      notification.style.display = 'none';
    }, 3000);
  }

  /**
   * Handle window resize
   */
  handleResize() {
    if (this.components.canvas) {
      this.setupCanvas();
    }
  }

  /**
   * Helper methods
   */
  getMediaIcon(mediaType) {
    const icons = {
      video: `<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M17,10.5V7A1,1 0 0,0 16,6H4A1,1 0 0,0 3,7V17A1,1 0 0,0 4,18H16A1,1 0 0,0 17,17V13.5L21,17.5V6.5L17,10.5Z"/></svg>`,
      audio: `<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M14,3.23V5.29C16.89,6.15 19,8.83 19,12C19,15.17 16.89,17.84 14,18.7V20.77C18,19.86 21,16.28 21,12C21,7.72 18,4.14 14,3.23M16.5,12C16.5,10.23 15.5,8.71 14,7.97V16C15.5,15.29 16.5,13.76 16.5,12M3,9V15H7L12,20V4L7,9H3Z"/></svg>`,
      image: `<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M8.5,13.5L11,16.5L14.5,12L19,18H5M21,19V5C21,3.89 20.1,3 19,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19Z"/></svg>`,
      document: `<svg viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M6,2A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2H6Z"/></svg>`
    };
    return icons[mediaType] || icons.document;
  }

  formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  formatFileSize(bytes) {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  selectMediaItem(mediaId) {
    // Handle media item selection
    console.log('Selected media:', mediaId);
  }

  playMedia(mediaId) {
    // Handle media playback
    console.log('Playing media:', mediaId);
  }

  updateTranslationTarget(language) {
    // Handle translation target language change
    console.log('Translation target:', language);
  }

  /**
   * Set callback functions
   */
  setCallbacks(callbacks) {
    Object.assign(this.callbacks, callbacks);
  }

  /**
   * Cleanup and destroy
   */
  destroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    
    // Remove event listeners
    window.removeEventListener('resize', this.handleResize);
    
    // Clear container
    this.container.innerHTML = '';
  }
}

// CSS styles for the voice UI (can be included in a separate CSS file)
export const voiceUIStyles = `
.voice-ui {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary, #1a1a1a);
  color: var(--text-primary, #ffffff);
  border-radius: 12px;
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.voice-ui-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid var(--border-color, #404040);
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--error-color, #dc3545);
  transition: background-color 0.3s ease;
}

.status-indicator.connected {
  background: var(--accent-color, #28a745);
}

.voice-controls {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.voice-btn {
  position: relative;
  background: var(--primary-color, #007bff);
  border: none;
  border-radius: 50%;
  width: 80px;
  height: 80px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.voice-btn:hover {
  transform: scale(1.05);
}

.voice-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.voice-btn.listening {
  background: var(--error-color, #dc3545);
  animation: pulse 2s infinite;
}

.voice-btn-pulse {
  position: absolute;
  top: -5px;
  left: -5px;
  right: -5px;
  bottom: -5px;
  border: 2px solid var(--primary-color, #007bff);
  border-radius: 50%;
  animation: pulse-ring 2s infinite;
  display: none;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

@keyframes pulse-ring {
  0% { transform: scale(1); opacity: 1; }
  100% { transform: scale(1.5); opacity: 0; }
}

.advanced-controls {
  display: flex;
  gap: 10px;
}

.btn-secondary {
  background: var(--bg-secondary, #2d2d2d);
  border: 1px solid var(--border-color, #404040);
  border-radius: 8px;
  padding: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background: var(--border-color, #404040);
}

.audio-visualization {
  margin: 20px 0;
  text-align: center;
}

.waveform-canvas {
  background: var(--bg-secondary, #2d2d2d);
  border-radius: 8px;
  width: 100%;
  height: 100px;
}

.audio-level-meter {
  width: 100%;
  height: 4px;
  background: var(--bg-secondary, #2d2d2d);
  border-radius: 2px;
  margin-top: 10px;
  overflow: hidden;
}

.level-bar {
  height: 100%;
  background: var(--accent-color, #28a745);
  border-radius: 2px;
  transition: width 0.1s ease;
  width: 0%;
}

.emotion-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  margin: 20px 0;
}

.emotion-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
}

.emotion-icon {
  font-size: 24px;
}

.sentiment-bar {
  width: 200px;
  height: 6px;
  background: var(--bg-secondary, #2d2d2d);
  border-radius: 3px;
  position: relative;
  overflow: hidden;
}

.sentiment-fill {
  height: 100%;
  background: var(--accent-color, #28a745);
  border-radius: 3px;
  transition: width 0.3s ease, background-color 0.3s ease;
  width: 50%;
}

.sentiment-labels {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  opacity: 0.7;
  margin-top: 5px;
}

.panel {
  background: var(--bg-secondary, #2d2d2d);
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid var(--border-color, #404040);
}

.panel-header h3 {
  margin: 0;
  font-size: 16px;
}

.panel-content {
  padding: 20px;
}

.transcription-display {
  min-height: 80px;
}

.transcription-text {
  background: var(--bg-primary, #1a1a1a);
  border-radius: 6px;
  padding: 15px;
  margin-bottom: 10px;
  min-height: 60px;
  max-height: 120px;
  overflow-y: auto;
}

.transcription-final {
  color: var(--text-primary, #ffffff);
}

.transcription-partial {
  color: var(--text-secondary, #cccccc);
  font-style: italic;
}

.transcription-info {
  display: flex;
  gap: 20px;
  font-size: 12px;
  opacity: 0.8;
}

.translation-display {
  max-height: 200px;
  overflow-y: auto;
}

.translation-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  margin-bottom: 10px;
  background: var(--bg-primary, #1a1a1a);
  border-radius: 6px;
}

.translation-arrow {
  color: var(--primary-color, #007bff);
  font-weight: bold;
}

.translation-result {
  flex: 1;
}

.translation-info {
  display: flex;
  gap: 10px;
  font-size: 10px;
  opacity: 0.7;
  margin-top: 5px;
}

.media-results {
  max-height: 300px;
  overflow-y: auto;
}

.media-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px;
  margin-bottom: 10px;
  background: var(--bg-primary, #1a1a1a);
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.media-item:hover {
  background: var(--border-color, #404040);
}

.media-icon {
  color: var(--primary-color, #007bff);
}

.media-info {
  flex: 1;
}

.media-title {
  font-weight: 500;
  margin-bottom: 5px;
}

.media-details {
  display: flex;
  gap: 10px;
  font-size: 12px;
  opacity: 0.7;
}

.media-description {
  font-size: 12px;
  opacity: 0.8;
  margin-top: 5px;
}

.biometric-status {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}

.status-icon {
  font-size: 24px;
}

.biometric-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.btn-primary {
  background: var(--primary-color, #007bff);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 10px 20px;
  cursor: pointer;
  transition: opacity 0.2s ease;
}

.btn-primary:hover {
  opacity: 0.9;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background: var(--bg-primary, #1a1a1a);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 10px;
}

.progress-fill {
  height: 100%;
  background: var(--primary-color, #007bff);
  border-radius: 2px;
  animation: indeterminate 2s infinite;
}

@keyframes indeterminate {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(400%); }
}

.notification {
  position: fixed;
  top: 20px;
  right: 20px;
  background: var(--bg-secondary, #2d2d2d);
  border: 1px solid var(--border-color, #404040);
  border-radius: 6px;
  padding: 15px 20px;
  z-index: 1000;
  display: none;
}

.notification.warning {
  border-color: var(--warning-color, #ffc107);
  background: rgba(255, 193, 7, 0.1);
}

.notification.error {
  border-color: var(--error-color, #dc3545);
  background: rgba(220, 53, 69, 0.1);
}

.notification.success {
  border-color: var(--accent-color, #28a745);
  background: rgba(40, 167, 69, 0.1);
}

.language-select, .search-input {
  background: var(--bg-primary, #1a1a1a);
  color: var(--text-primary, #ffffff);
  border: 1px solid var(--border-color, #404040);
  border-radius: 6px;
  padding: 8px 12px;
}

.search-input {
  flex: 1;
  margin-right: 10px;
}

.placeholder {
  opacity: 0.5;
  font-style: italic;
  text-align: center;
  padding: 20px;
}

.no-results {
  text-align: center;
  opacity: 0.7;
  padding: 20px;
}

/* Compact mode */
.voice-ui[data-compact="true"] {
  padding: 15px;
}

.voice-ui[data-compact="true"] .voice-btn {
  width: 60px;
  height: 60px;
}

.voice-ui[data-compact="true"] .panel-content {
  padding: 15px;
}

/* Responsive design */
@media (max-width: 768px) {
  .voice-ui {
    padding: 15px;
  }
  
  .voice-ui-header {
    flex-direction: column;
    gap: 10px;
  }
  
  .advanced-controls {
    justify-content: center;
  }
  
  .media-item {
    flex-direction: column;
    text-align: center;
  }
  
  .transcription-info,
  .media-details {
    flex-direction: column;
    gap: 5px;
  }
}
`;

export default VoiceUI;