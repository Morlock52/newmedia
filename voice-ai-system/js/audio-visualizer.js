class AudioVisualizer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.bufferLength = 0;
        this.animationId = null;
        this.isActive = false;
        
        this.setupCanvas();
        this.initAudioContext();
    }

    setupCanvas() {
        if (!this.canvas) return;

        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Set up canvas styling
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }

    async initAudioContext() {
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.8;
            
            this.bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(this.bufferLength);
            
            console.log('Audio context initialized');
        } catch (error) {
            console.error('Failed to initialize audio context:', error);
        }
    }

    async connectMicrophone() {
        if (!this.audioContext || !this.analyser) {
            throw new Error('Audio context not initialized');
        }

        try {
            // Resume audio context if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Create source from microphone
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyser);

            this.isActive = true;
            this.startVisualization();
            
            return stream;
        } catch (error) {
            console.error('Failed to connect microphone:', error);
            throw error;
        }
    }

    disconnectMicrophone() {
        this.isActive = false;
        this.stopVisualization();
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.suspend();
        }
    }

    startVisualization() {
        if (!this.isActive || this.animationId) return;

        const animate = () => {
            if (!this.isActive) return;

            this.animationId = requestAnimationFrame(animate);
            this.draw();
        };

        animate();
    }

    stopVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        this.clearCanvas();
    }

    draw() {
        if (!this.analyser || !this.dataArray) return;

        // Get frequency data
        this.analyser.getByteFrequencyData(this.dataArray);

        // Clear canvas
        this.clearCanvas();

        // Draw visualization
        this.drawBars();
        this.drawWaveform();
        this.drawVolumeIndicator();
    }

    drawBars() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        const barWidth = width / this.bufferLength * 2;
        let x = 0;

        // Create gradient
        const gradient = this.ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(0.5, '#8b5cf6');
        gradient.addColorStop(1, '#ec4899');

        this.ctx.fillStyle = gradient;

        for (let i = 0; i < this.bufferLength; i++) {
            const barHeight = (this.dataArray[i] / 255) * height * 0.8;
            
            this.ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
        }
    }

    drawWaveform() {
        if (!this.analyser) return;

        // Get time domain data for waveform
        const timeData = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteTimeDomainData(timeData);

        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        const sliceWidth = width / timeData.length;
        let x = 0;

        for (let i = 0; i < timeData.length; i++) {
            const v = timeData[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        this.ctx.stroke();
    }

    drawVolumeIndicator() {
        if (!this.dataArray) return;

        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const average = sum / this.dataArray.length;
        const normalizedVolume = average / 255;

        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Draw volume circle
        const centerX = width - 40;
        const centerY = 40;
        const maxRadius = 20;
        const radius = normalizedVolume * maxRadius;

        // Outer circle
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, maxRadius, 0, 2 * Math.PI);
        this.ctx.stroke();

        // Volume circle
        const volumeGradient = this.ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        volumeGradient.addColorStop(0, 'rgba(99, 102, 241, 0.8)');
        volumeGradient.addColorStop(1, 'rgba(99, 102, 241, 0.3)');
        
        this.ctx.fillStyle = volumeGradient;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        this.ctx.fill();

        // Volume level text
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.font = '12px system-ui';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            Math.round(normalizedVolume * 100) + '%',
            centerX,
            centerY + 35
        );
    }

    drawSpectrogram() {
        if (!this.dataArray) return;

        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Create frequency bands
        const bands = 32;
        const bandWidth = width / bands;
        const bandData = [];

        // Group frequency data into bands
        const samplesPerBand = Math.floor(this.bufferLength / bands);
        for (let i = 0; i < bands; i++) {
            let sum = 0;
            for (let j = 0; j < samplesPerBand; j++) {
                sum += this.dataArray[i * samplesPerBand + j];
            }
            bandData.push(sum / samplesPerBand);
        }

        // Draw bands
        for (let i = 0; i < bands; i++) {
            const bandHeight = (bandData[i] / 255) * height;
            const hue = (i / bands) * 360;
            
            this.ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.8)`;
            this.ctx.fillRect(
                i * bandWidth,
                height - bandHeight,
                bandWidth - 1,
                bandHeight
            );
        }
    }

    clearCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.ctx.clearRect(0, 0, rect.width, rect.height);
        
        // Draw background
        this.ctx.fillStyle = 'rgba(15, 23, 42, 0.1)';
        this.ctx.fillRect(0, 0, rect.width, rect.height);
    }

    // Static visualization for when not listening
    drawStaticVisualization() {
        this.clearCanvas();
        
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Draw static bars
        const barCount = 32;
        const barWidth = width / barCount;
        
        for (let i = 0; i < barCount; i++) {
            const barHeight = Math.random() * height * 0.3 + 10;
            
            this.ctx.fillStyle = 'rgba(100, 116, 139, 0.3)';
            this.ctx.fillRect(
                i * barWidth,
                height - barHeight,
                barWidth - 1,
                barHeight
            );
        }

        // Draw center line
        this.ctx.strokeStyle = 'rgba(100, 116, 139, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(0, height / 2);
        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();
    }

    // Visualization modes
    setVisualizationMode(mode) {
        this.visualizationMode = mode;
    }

    // Get audio level for voice activity detection
    getAudioLevel() {
        if (!this.dataArray) return 0;

        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        return sum / this.dataArray.length / 255;
    }

    // Check if audio is above threshold (speaking detected)
    isSpeaking(threshold = 0.1) {
        return this.getAudioLevel() > threshold;
    }

    // Cleanup
    destroy() {
        this.stopVisualization();
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
        
        window.removeEventListener('resize', () => this.resizeCanvas());
    }
}