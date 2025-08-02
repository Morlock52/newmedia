class MediaController {
    constructor(mediaElement) {
        this.media = mediaElement;
        this.isFullscreen = false;
        this.lastVolume = 1;
        this.skipAmount = 10; // seconds
        this.volumeStep = 0.1;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        if (!this.media) return;

        // Listen for media events
        this.media.addEventListener('loadstart', () => this.onMediaEvent('loadstart'));
        this.media.addEventListener('loadeddata', () => this.onMediaEvent('loadeddata'));
        this.media.addEventListener('canplay', () => this.onMediaEvent('canplay'));
        this.media.addEventListener('play', () => this.onMediaEvent('play'));
        this.media.addEventListener('pause', () => this.onMediaEvent('pause'));
        this.media.addEventListener('ended', () => this.onMediaEvent('ended'));
        this.media.addEventListener('error', (e) => this.onMediaEvent('error', e));
        this.media.addEventListener('timeupdate', () => this.onMediaEvent('timeupdate'));
        this.media.addEventListener('volumechange', () => this.onMediaEvent('volumechange'));

        // Listen for fullscreen events
        document.addEventListener('fullscreenchange', () => this.onFullscreenChange());
        document.addEventListener('webkitfullscreenchange', () => this.onFullscreenChange());
        document.addEventListener('mozfullscreenchange', () => this.onFullscreenChange());
        document.addEventListener('MSFullscreenChange', () => this.onFullscreenChange());
    }

    // Media Control Methods
    async play() {
        if (!this.media) {
            throw new Error('No media element available');
        }

        try {
            await this.media.play();
            return {
                success: true,
                action: 'play',
                message: 'Playback started'
            };
        } catch (error) {
            console.error('Play failed:', error);
            throw new Error(`Failed to play: ${error.message}`);
        }
    }

    pause() {
        if (!this.media) {
            throw new Error('No media element available');
        }

        try {
            this.media.pause();
            return {
                success: true,
                action: 'pause',
                message: 'Playback paused'
            };
        } catch (error) {
            console.error('Pause failed:', error);
            throw new Error(`Failed to pause: ${error.message}`);
        }
    }

    stop() {
        if (!this.media) {
            throw new Error('No media element available');
        }

        try {
            this.media.pause();
            this.media.currentTime = 0;
            return {
                success: true,
                action: 'stop',
                message: 'Playback stopped'
            };
        } catch (error) {
            console.error('Stop failed:', error);
            throw new Error(`Failed to stop: ${error.message}`);
        }
    }

    // Volume Control
    volumeUp(amount = null) {
        if (!this.media) {
            throw new Error('No media element available');
        }

        const increment = amount || this.volumeStep;
        const newVolume = Math.min(1, this.media.volume + increment);
        this.media.volume = newVolume;

        return {
            success: true,
            action: 'volume_up',
            message: `Volume increased to ${Math.round(newVolume * 100)}%`,
            volume: newVolume
        };
    }

    volumeDown(amount = null) {
        if (!this.media) {
            throw new Error('No media element available');
        }

        const decrement = amount || this.volumeStep;
        const newVolume = Math.max(0, this.media.volume - decrement);
        this.media.volume = newVolume;

        return {
            success: true,
            action: 'volume_down',
            message: `Volume decreased to ${Math.round(newVolume * 100)}%`,
            volume: newVolume
        };
    }

    setVolume(volume) {
        if (!this.media) {
            throw new Error('No media element available');
        }

        const normalizedVolume = Math.max(0, Math.min(1, volume / 100));
        this.media.volume = normalizedVolume;

        return {
            success: true,
            action: 'set_volume',
            message: `Volume set to ${Math.round(normalizedVolume * 100)}%`,
            volume: normalizedVolume
        };
    }

    mute() {
        if (!this.media) {
            throw new Error('No media element available');
        }

        if (!this.media.muted) {
            this.lastVolume = this.media.volume;
            this.media.muted = true;
        }

        return {
            success: true,
            action: 'mute',
            message: 'Audio muted',
            muted: true
        };
    }

    unmute() {
        if (!this.media) {
            throw new Error('No media element available');
        }

        if (this.media.muted) {
            this.media.muted = false;
            if (this.lastVolume > 0) {
                this.media.volume = this.lastVolume;
            }
        }

        return {
            success: true,
            action: 'unmute',
            message: 'Audio unmuted',
            muted: false,
            volume: this.media.volume
        };
    }

    toggleMute() {
        return this.media.muted ? this.unmute() : this.mute();
    }

    // Seeking Controls
    skipForward(seconds = null) {
        if (!this.media) {
            throw new Error('No media element available');
        }

        const skipTime = seconds || this.skipAmount;
        const newTime = Math.min(this.media.duration, this.media.currentTime + skipTime);
        this.media.currentTime = newTime;

        return {
            success: true,
            action: 'skip_forward',
            message: `Skipped forward ${skipTime} seconds`,
            currentTime: newTime,
            skipAmount: skipTime
        };
    }

    skipBackward(seconds = null) {
        if (!this.media) {
            throw new Error('No media element available');
        }

        const skipTime = seconds || this.skipAmount;
        const newTime = Math.max(0, this.media.currentTime - skipTime);
        this.media.currentTime = newTime;

        return {
            success: true,
            action: 'skip_backward',
            message: `Skipped back ${skipTime} seconds`,
            currentTime: newTime,
            skipAmount: skipTime
        };
    }

    seekTo(time) {
        if (!this.media) {
            throw new Error('No media element available');
        }

        const seekTime = Math.max(0, Math.min(this.media.duration, time));
        this.media.currentTime = seekTime;

        return {
            success: true,
            action: 'seek',
            message: `Seeked to ${this.formatTime(seekTime)}`,
            currentTime: seekTime
        };
    }

    // Fullscreen Controls
    async enterFullscreen() {
        if (!this.media) {
            throw new Error('No media element available');
        }

        try {
            if (this.media.requestFullscreen) {
                await this.media.requestFullscreen();
            } else if (this.media.webkitRequestFullscreen) {
                await this.media.webkitRequestFullscreen();
            } else if (this.media.mozRequestFullScreen) {
                await this.media.mozRequestFullScreen();
            } else if (this.media.msRequestFullscreen) {
                await this.media.msRequestFullscreen();
            } else {
                throw new Error('Fullscreen not supported');
            }

            return {
                success: true,
                action: 'fullscreen_enter',
                message: 'Entered fullscreen mode'
            };
        } catch (error) {
            console.error('Fullscreen failed:', error);
            throw new Error(`Failed to enter fullscreen: ${error.message}`);
        }
    }

    async exitFullscreen() {
        try {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                await document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                await document.mozCancelFullScreen();
            } else if (document.msExitFullscreen) {
                await document.msExitFullscreen();
            } else {
                throw new Error('Exit fullscreen not supported');
            }

            return {
                success: true,
                action: 'fullscreen_exit',
                message: 'Exited fullscreen mode'
            };
        } catch (error) {
            console.error('Exit fullscreen failed:', error);
            throw new Error(`Failed to exit fullscreen: ${error.message}`);
        }
    }

    async toggleFullscreen() {
        return this.isFullscreen ? this.exitFullscreen() : this.enterFullscreen();
    }

    // Status and Information
    getStatus() {
        if (!this.media) {
            return {
                available: false,
                error: 'No media element available'
            };
        }

        return {
            available: true,
            playing: !this.media.paused,
            currentTime: this.media.currentTime,
            duration: this.media.duration,
            volume: this.media.volume,
            muted: this.media.muted,
            fullscreen: this.isFullscreen,
            readyState: this.media.readyState,
            networkState: this.media.networkState,
            buffered: this.getBufferedRanges(),
            seekable: this.getSeekableRanges(),
            progress: this.getProgress()
        };
    }

    getProgress() {
        if (!this.media || !this.media.duration) {
            return 0;
        }
        return (this.media.currentTime / this.media.duration) * 100;
    }

    getBufferedRanges() {
        if (!this.media || !this.media.buffered) {
            return [];
        }

        const ranges = [];
        for (let i = 0; i < this.media.buffered.length; i++) {
            ranges.push({
                start: this.media.buffered.start(i),
                end: this.media.buffered.end(i)
            });
        }
        return ranges;
    }

    getSeekableRanges() {
        if (!this.media || !this.media.seekable) {
            return [];
        }

        const ranges = [];
        for (let i = 0; i < this.media.seekable.length; i++) {
            ranges.push({
                start: this.media.seekable.start(i),
                end: this.media.seekable.end(i)
            });
        }
        return ranges;
    }

    // Command Processing
    async executeCommand(intent, entities = {}) {
        try {
            switch (intent) {
                case 'PLAY':
                    return await this.play();
                
                case 'PAUSE':
                    return this.pause();
                
                case 'VOLUME_UP':
                    const upAmount = entities.VOLUME_LEVEL ? entities.VOLUME_LEVEL.value / 100 : null;
                    return this.volumeUp(upAmount);
                
                case 'VOLUME_DOWN':
                    const downAmount = entities.VOLUME_LEVEL ? entities.VOLUME_LEVEL.value / 100 : null;
                    return this.volumeDown(downAmount);
                
                case 'MUTE':
                    return this.mute();
                
                case 'UNMUTE':
                    return this.unmute();
                
                case 'SKIP_FORWARD':
                    const forwardTime = entities.TIME_DURATION ? entities.TIME_DURATION.value : null;
                    return this.skipForward(forwardTime);
                
                case 'SKIP_BACKWARD':
                    const backwardTime = entities.TIME_DURATION ? entities.TIME_DURATION.value : null;
                    return this.skipBackward(backwardTime);
                
                case 'FULLSCREEN_ENTER':
                    return await this.enterFullscreen();
                
                case 'FULLSCREEN_EXIT':
                    return await this.exitFullscreen();
                
                case 'STATUS':
                    return {
                        success: true,
                        action: 'status',
                        message: this.getStatusMessage(),
                        status: this.getStatus()
                    };
                
                default:
                    throw new Error(`Unknown command: ${intent}`);
            }
        } catch (error) {
            return {
                success: false,
                action: intent.toLowerCase(),
                error: error.message
            };
        }
    }

    getStatusMessage() {
        const status = this.getStatus();
        if (!status.available) {
            return 'No media available';
        }

        const state = status.playing ? 'playing' : 'paused';
        const time = this.formatTime(status.currentTime);
        const duration = this.formatTime(status.duration);
        const volume = Math.round(status.volume * 100);
        
        return `Media is ${state}. Time: ${time} of ${duration}. Volume: ${volume}%${status.muted ? ' (muted)' : ''}.`;
    }

    // Event Handlers
    onMediaEvent(eventType, event = null) {
        console.log(`Media event: ${eventType}`, event);
        
        // Dispatch custom events
        const customEvent = new CustomEvent('mediaController', {
            detail: {
                type: eventType,
                status: this.getStatus(),
                originalEvent: event
            }
        });
        
        document.dispatchEvent(customEvent);
    }

    onFullscreenChange() {
        this.isFullscreen = !!(
            document.fullscreenElement ||
            document.webkitFullscreenElement ||
            document.mozFullScreenElement ||
            document.msFullscreenElement
        );
        
        this.onMediaEvent('fullscreenchange', { fullscreen: this.isFullscreen });
    }

    // Utility Methods
    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }

    // Configuration
    setSkipAmount(seconds) {
        this.skipAmount = Math.max(1, seconds);
    }

    setVolumeStep(step) {
        this.volumeStep = Math.max(0.01, Math.min(1, step));
    }
}