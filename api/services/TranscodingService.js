/**
 * TranscodingService - FileFlows GPU transcoding with hardware acceleration
 * Provides video/audio transcoding and optimization for media files
 */

const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');
const EventEmitter = require('events');

const execAsync = promisify(exec);

class TranscodingService extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            fileflowsUrl: config.fileflowsUrl || process.env.FILEFLOWS_URL || 'http://fileflows:5000',
            fileflowsApiKey: config.fileflowsApiKey || process.env.FILEFLOWS_API_KEY,
            ffmpegPath: config.ffmpegPath || '/usr/bin/ffmpeg',
            ffprobePath: config.ffprobePath || '/usr/bin/ffprobe',
            tempDir: config.tempDir || '/tmp/transcoding',
            outputDir: config.outputDir || '/media/transcoded',
            maxConcurrentJobs: config.maxConcurrentJobs || 3,
            enableGPU: config.enableGPU !== false,
            gpuType: config.gpuType || 'nvidia', // nvidia, amd, intel
            qualityPresets: config.qualityPresets || 'medium',
            enableHDR: config.enableHDR || false,
            audioCodec: config.audioCodec || 'aac',
            videoCodec: config.videoCodec || 'h264',
            ...config
        };

        this.jobs = new Map();
        this.queue = [];
        this.activeJobs = new Map();
        this.profiles = new Map();
        this.isInitialized = false;
        this.jobCounter = 0;
        
        this.jobStatus = {
            QUEUED: 'queued',
            PROCESSING: 'processing',
            COMPLETED: 'completed',
            FAILED: 'failed',
            CANCELLED: 'cancelled',
            PAUSED: 'paused'
        };

        this.qualityPresets = {
            ultrafast: { crf: 35, preset: 'ultrafast', profile: 'baseline' },
            superfast: { crf: 30, preset: 'superfast', profile: 'main' },
            veryfast: { crf: 28, preset: 'veryfast', profile: 'main' },
            faster: { crf: 26, preset: 'faster', profile: 'main' },
            fast: { crf: 24, preset: 'fast', profile: 'main' },
            medium: { crf: 23, preset: 'medium', profile: 'high' },
            slow: { crf: 22, preset: 'slow', profile: 'high' },
            slower: { crf: 21, preset: 'slower', profile: 'high' },
            veryslow: { crf: 20, preset: 'veryslow', profile: 'high' }
        };

        this.resolutionPresets = {
            '4k': { width: 3840, height: 2160, bitrate: '15M' },
            '1440p': { width: 2560, height: 1440, bitrate: '10M' },
            '1080p': { width: 1920, height: 1080, bitrate: '5M' },
            '720p': { width: 1280, height: 720, bitrate: '2.5M' },
            '480p': { width: 854, height: 480, bitrate: '1M' },
            '360p': { width: 640, height: 360, bitrate: '600k' }
        };
    }

    /**
     * Initialize Transcoding service
     */
    async initialize() {
        try {
            console.log('🎥 Initializing TranscodingService...');
            
            // Check dependencies
            await this.checkDependencies();
            
            // Create directories
            await this.createDirectories();
            
            // Initialize FileFlows connection
            await this.initializeFileFlows();
            
            // Load transcoding profiles
            await this.loadProfiles();
            
            // Start job processor
            this.startJobProcessor();
            
            this.isInitialized = true;
            this.emit('initialized');
            console.log('✅ TranscodingService initialized successfully');
            
            return { success: true, message: 'TranscodingService initialized' };
        } catch (error) {
            console.error('❌ TranscodingService initialization failed:', error);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Check required dependencies
     */
    async checkDependencies() {
        try {
            // Check FFmpeg
            const { stdout: ffmpegVersion } = await execAsync(`${this.config.ffmpegPath} -version`);
            if (!ffmpegVersion.includes('ffmpeg version')) {
                throw new Error('FFmpeg not found or invalid version');
            }
            
            // Check FFprobe
            const { stdout: ffprobeVersion } = await execAsync(`${this.config.ffprobePath} -version`);
            if (!ffprobeVersion.includes('ffprobe version')) {
                throw new Error('FFprobe not found or invalid version');
            }
            
            // Check GPU capabilities
            if (this.config.enableGPU) {
                await this.checkGPUSupport();
            }
            
            console.log('✅ All dependencies verified');
        } catch (error) {
            console.error('❌ Dependency check failed:', error);
            throw error;
        }
    }

    /**
     * Check GPU support
     */
    async checkGPUSupport() {
        try {
            switch (this.config.gpuType) {
                case 'nvidia':
                    await execAsync('nvidia-smi');
                    console.log('✅ NVIDIA GPU detected');
                    break;
                case 'amd':
                    // Check for AMD GPU
                    console.log('✅ AMD GPU support enabled');
                    break;
                case 'intel':
                    // Check for Intel Quick Sync
                    console.log('✅ Intel Quick Sync support enabled');
                    break;
                default:
                    console.warn('⚠️ Unknown GPU type, disabling hardware acceleration');
                    this.config.enableGPU = false;
            }
        } catch (error) {
            console.warn('⚠️ GPU not available, falling back to CPU transcoding');
            this.config.enableGPU = false;
        }
    }

    /**
     * Create required directories
     */
    async createDirectories() {
        try {
            await fs.mkdir(this.config.tempDir, { recursive: true });
            await fs.mkdir(this.config.outputDir, { recursive: true });
            console.log('✅ Directories created');
        } catch (error) {
            console.error('❌ Directory creation failed:', error);
            throw error;
        }
    }

    /**
     * Initialize FileFlows connection
     */
    async initializeFileFlows() {
        try {
            if (!this.config.fileflowsUrl) {
                console.warn('⚠️ FileFlows URL not configured, using direct FFmpeg');
                return;
            }
            
            // Test FileFlows connection
            const response = await axios.get(`${this.config.fileflowsUrl}/api/system/info`, {
                headers: this.config.fileflowsApiKey ? {
                    'Authorization': `Bearer ${this.config.fileflowsApiKey}`
                } : {},
                timeout: 5000
            });
            
            if (response.data) {
                console.log('✅ FileFlows connection established');
                this.fileflowsAvailable = true;
            }
        } catch (error) {
            console.warn('⚠️ FileFlows not available, using direct FFmpeg:', error.message);
            this.fileflowsAvailable = false;
        }
    }

    /**
     * Load transcoding profiles
     */
    async loadProfiles() {
        try {
            // Define default profiles
            const defaultProfiles = {
                'web_optimized': {
                    name: 'Web Optimized',
                    description: 'Optimized for web streaming',
                    videoCodec: 'h264',
                    audioCodec: 'aac',
                    resolution: '1080p',
                    quality: 'medium',
                    enableGPU: true,
                    format: 'mp4'
                },
                'mobile_friendly': {
                    name: 'Mobile Friendly',
                    description: 'Optimized for mobile devices',
                    videoCodec: 'h264',
                    audioCodec: 'aac',
                    resolution: '720p',
                    quality: 'fast',
                    enableGPU: true,
                    format: 'mp4'
                },
                'high_quality': {
                    name: 'High Quality',
                    description: 'High quality for archival',
                    videoCodec: 'h265',
                    audioCodec: 'aac',
                    resolution: '4k',
                    quality: 'slow',
                    enableGPU: true,
                    format: 'mkv'
                },
                'audio_only': {
                    name: 'Audio Only',
                    description: 'Extract audio only',
                    videoCodec: null,
                    audioCodec: 'mp3',
                    resolution: null,
                    quality: 'medium',
                    enableGPU: false,
                    format: 'mp3'
                }
            };
            
            Object.entries(defaultProfiles).forEach(([id, profile]) => {
                this.profiles.set(id, profile);
            });
            
            console.log(`✅ Transcoding profiles loaded: ${this.profiles.size} profiles`);
        } catch (error) {
            console.error('❌ Profile loading failed:', error);
        }
    }

    /**
     * Start transcoding job
     */
    async startTranscoding(inputFile, options = {}) {
        try {
            const jobId = `job_${++this.jobCounter}_${Date.now()}`;
            
            // Get media info
            const mediaInfo = await this.getMediaInfo(inputFile);
            
            const job = {
                id: jobId,
                inputFile,
                outputFile: options.outputFile || this.generateOutputPath(inputFile, options),
                profile: options.profile || 'web_optimized',
                status: this.jobStatus.QUEUED,
                progress: 0,
                createdAt: new Date(),
                startedAt: null,
                completedAt: null,
                mediaInfo,
                options,
                error: null
            };
            
            this.jobs.set(jobId, job);
            this.queue.push(jobId);
            
            this.emit('jobCreated', job);
            console.log(`✅ Transcoding job created: ${jobId}`);
            
            // Process queue
            this.processQueue();
            
            return {
                success: true,
                jobId,
                job
            };
        } catch (error) {
            console.error('❌ Transcoding job creation failed:', error);
            throw error;
        }
    }

    /**
     * Get media information using FFprobe
     */
    async getMediaInfo(filePath) {
        try {
            const { stdout } = await execAsync(
                `${this.config.ffprobePath} -v quiet -print_format json -show_format -show_streams "${filePath}"`
            );
            
            const info = JSON.parse(stdout);
            
            const videoStream = info.streams.find(s => s.codec_type === 'video');
            const audioStreams = info.streams.filter(s => s.codec_type === 'audio');
            
            return {
                format: info.format,
                duration: parseFloat(info.format.duration) || 0,
                size: parseInt(info.format.size) || 0,
                bitrate: parseInt(info.format.bit_rate) || 0,
                video: videoStream ? {
                    codec: videoStream.codec_name,
                    width: videoStream.width,
                    height: videoStream.height,
                    fps: eval(videoStream.r_frame_rate) || 0,
                    bitrate: parseInt(videoStream.bit_rate) || 0,
                    pixelFormat: videoStream.pix_fmt
                } : null,
                audio: audioStreams.map(stream => ({
                    codec: stream.codec_name,
                    channels: stream.channels,
                    sampleRate: parseInt(stream.sample_rate),
                    bitrate: parseInt(stream.bit_rate) || 0,
                    language: stream.tags?.language || 'und'
                }))
            };
        } catch (error) {
            console.error('❌ Media info extraction failed:', error);
            throw error;
        }
    }

    /**
     * Generate output file path
     */
    generateOutputPath(inputFile, options) {
        const basename = path.basename(inputFile, path.extname(inputFile));
        const profile = this.profiles.get(options.profile || 'web_optimized');
        const extension = profile?.format || 'mp4';
        
        const suffix = options.suffix || `_${options.profile || 'transcoded'}`;
        return path.join(this.config.outputDir, `${basename}${suffix}.${extension}`);
    }

    /**
     * Process transcoding queue
     */
    processQueue() {
        if (this.activeJobs.size >= this.config.maxConcurrentJobs) {
            return; // Max concurrent jobs reached
        }
        
        if (this.queue.length === 0) {
            return; // No jobs in queue
        }
        
        const jobId = this.queue.shift();
        const job = this.jobs.get(jobId);
        
        if (job && job.status === this.jobStatus.QUEUED) {
            this.executeJob(jobId);
        }
    }

    /**
     * Execute transcoding job
     */
    async executeJob(jobId) {
        const job = this.jobs.get(jobId);
        if (!job) return;
        
        try {
            job.status = this.jobStatus.PROCESSING;
            job.startedAt = new Date();
            this.activeJobs.set(jobId, job);
            
            this.emit('jobStarted', job);
            console.log(`🎥 Starting transcoding: ${job.inputFile}`);
            
            if (this.fileflowsAvailable) {
                await this.executeFileFlowsJob(job);
            } else {
                await this.executeFFmpegJob(job);
            }
            
            job.status = this.jobStatus.COMPLETED;
            job.completedAt = new Date();
            job.progress = 100;
            
            this.emit('jobCompleted', job);
            console.log(`✅ Transcoding completed: ${job.outputFile}`);
        } catch (error) {
            job.status = this.jobStatus.FAILED;
            job.error = error.message;
            job.completedAt = new Date();
            
            this.emit('jobFailed', { job, error });
            console.error(`❌ Transcoding failed: ${error.message}`);
        } finally {
            this.activeJobs.delete(jobId);
            
            // Process next job in queue
            setTimeout(() => this.processQueue(), 1000);
        }
    }

    /**
     * Execute job using FileFlows
     */
    async executeFileFlowsJob(job) {
        try {
            const flowData = {
                file: job.inputFile,
                profile: job.profile,
                output: job.outputFile
            };
            
            const response = await axios.post(
                `${this.config.fileflowsUrl}/api/flows/execute`,
                flowData,
                {
                    headers: this.config.fileflowsApiKey ? {
                        'Authorization': `Bearer ${this.config.fileflowsApiKey}`
                    } : {}
                }
            );
            
            if (response.data.flowId) {
                await this.monitorFileFlowsJob(job, response.data.flowId);
            }
        } catch (error) {
            throw new Error(`FileFlows execution failed: ${error.message}`);
        }
    }

    /**
     * Monitor FileFlows job progress
     */
    async monitorFileFlowsJob(job, flowId) {
        const pollInterval = 5000; // 5 seconds
        
        while (job.status === this.jobStatus.PROCESSING) {
            try {
                const response = await axios.get(
                    `${this.config.fileflowsUrl}/api/flows/${flowId}/status`,
                    {
                        headers: this.config.fileflowsApiKey ? {
                            'Authorization': `Bearer ${this.config.fileflowsApiKey}`
                        } : {}
                    }
                );
                
                const status = response.data;
                job.progress = status.progress || 0;
                
                this.emit('jobProgress', { job, progress: job.progress });
                
                if (status.completed) {
                    break;
                } else if (status.failed) {
                    throw new Error(status.error || 'FileFlows job failed');
                }
                
                await new Promise(resolve => setTimeout(resolve, pollInterval));
            } catch (error) {
                throw error;
            }
        }
    }

    /**
     * Execute job using direct FFmpeg
     */
    async executeFFmpegJob(job) {
        try {
            const profile = this.profiles.get(job.profile);
            if (!profile) {
                throw new Error(`Profile not found: ${job.profile}`);
            }
            
            const ffmpegCommand = this.buildFFmpegCommand(job, profile);
            
            console.log(`📝 FFmpeg command: ${ffmpegCommand}`);
            
            await this.executeFFmpegCommand(job, ffmpegCommand);
        } catch (error) {
            throw error;
        }
    }

    /**
     * Build FFmpeg command
     */
    buildFFmpegCommand(job, profile) {
        const quality = this.qualityPresets[profile.quality] || this.qualityPresets.medium;
        const resolution = this.resolutionPresets[profile.resolution];
        
        let command = `${this.config.ffmpegPath} -i "${job.inputFile}"`;
        
        // Hardware acceleration
        if (profile.enableGPU && this.config.enableGPU) {
            switch (this.config.gpuType) {
                case 'nvidia':
                    command += ' -hwaccel cuda -hwaccel_output_format cuda';
                    break;
                case 'intel':
                    command += ' -hwaccel qsv';
                    break;
                case 'amd':
                    command += ' -hwaccel vaapi';
                    break;
            }
        }
        
        // Video encoding
        if (profile.videoCodec) {
            let videoCodec = profile.videoCodec;
            
            // Use hardware-accelerated encoders if available
            if (profile.enableGPU && this.config.enableGPU) {
                switch (this.config.gpuType) {
                    case 'nvidia':
                        videoCodec = videoCodec === 'h264' ? 'h264_nvenc' : 
                                   videoCodec === 'h265' ? 'hevc_nvenc' : videoCodec;
                        break;
                    case 'intel':
                        videoCodec = videoCodec === 'h264' ? 'h264_qsv' : 
                                   videoCodec === 'h265' ? 'hevc_qsv' : videoCodec;
                        break;
                }
            }
            
            command += ` -c:v ${videoCodec}`;
            
            if (resolution) {
                command += ` -vf scale=${resolution.width}:${resolution.height}`;
                command += ` -b:v ${resolution.bitrate}`;
            }
            
            command += ` -crf ${quality.crf} -preset ${quality.preset} -profile:v ${quality.profile}`;
        } else {
            command += ' -vn'; // No video
        }
        
        // Audio encoding
        if (profile.audioCodec) {
            command += ` -c:a ${profile.audioCodec}`;
            
            if (profile.audioCodec === 'aac') {
                command += ' -b:a 128k';
            } else if (profile.audioCodec === 'mp3') {
                command += ' -b:a 192k';
            }
        } else {
            command += ' -an'; // No audio
        }
        
        // Output options
        command += ' -movflags +faststart'; // Optimize for streaming
        command += ` "${job.outputFile}"`;
        
        return command;
    }

    /**
     * Execute FFmpeg command with progress monitoring
     */
    async executeFFmpegCommand(job, command) {
        return new Promise((resolve, reject) => {
            const process = exec(command);
            
            let duration = job.mediaInfo.duration;
            
            process.stderr.on('data', (data) => {
                const progressMatch = data.toString().match(/time=([0-9:.]+)/);
                if (progressMatch && duration > 0) {
                    const timeStr = progressMatch[1];
                    const timeParts = timeStr.split(':');
                    const currentTime = parseFloat(timeParts[0]) * 3600 + 
                                      parseFloat(timeParts[1]) * 60 + 
                                      parseFloat(timeParts[2]);
                    
                    job.progress = Math.min(Math.round((currentTime / duration) * 100), 99);
                    this.emit('jobProgress', { job, progress: job.progress });
                }
            });
            
            process.on('close', (code) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`FFmpeg process exited with code ${code}`));
                }
            });
            
            process.on('error', (error) => {
                reject(error);
            });
        });
    }

    /**
     * Start job processor
     */
    startJobProcessor() {
        // Process queue every 5 seconds
        setInterval(() => {
            this.processQueue();
        }, 5000);
        
        console.log('✅ Job processor started');
    }

    /**
     * Cancel job
     */
    async cancelJob(jobId) {
        try {
            const job = this.jobs.get(jobId);
            if (!job) {
                throw new Error('Job not found');
            }
            
            if (job.status === this.jobStatus.PROCESSING) {
                // Kill the process if it's running
                // Implementation depends on how the job is being executed
            }
            
            job.status = this.jobStatus.CANCELLED;
            job.completedAt = new Date();
            
            this.activeJobs.delete(jobId);
            
            this.emit('jobCancelled', job);
            console.log(`✅ Job cancelled: ${jobId}`);
            
            return { success: true, job };
        } catch (error) {
            console.error('❌ Job cancellation failed:', error);
            throw error;
        }
    }

    /**
     * Get job status
     */
    getJobStatus(jobId) {
        const job = this.jobs.get(jobId);
        if (!job) {
            throw new Error('Job not found');
        }
        
        return {
            id: job.id,
            status: job.status,
            progress: job.progress,
            inputFile: job.inputFile,
            outputFile: job.outputFile,
            profile: job.profile,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: job.completedAt,
            error: job.error,
            duration: job.mediaInfo?.duration || 0
        };
    }

    /**
     * Get all jobs
     */
    getAllJobs() {
        return Array.from(this.jobs.values()).map(job => ({
            id: job.id,
            status: job.status,
            progress: job.progress,
            inputFile: path.basename(job.inputFile),
            outputFile: path.basename(job.outputFile),
            profile: job.profile,
            createdAt: job.createdAt,
            duration: job.completedAt && job.startedAt ? 
                     job.completedAt.getTime() - job.startedAt.getTime() : null
        }));
    }

    /**
     * Get service status
     */
    getStatus() {
        const queuedJobs = Array.from(this.jobs.values())
            .filter(job => job.status === this.jobStatus.QUEUED).length;
        
        const processingJobs = Array.from(this.jobs.values())
            .filter(job => job.status === this.jobStatus.PROCESSING).length;
        
        const completedJobs = Array.from(this.jobs.values())
            .filter(job => job.status === this.jobStatus.COMPLETED).length;
        
        const failedJobs = Array.from(this.jobs.values())
            .filter(job => job.status === this.jobStatus.FAILED).length;
        
        return {
            initialized: this.isInitialized,
            fileflowsAvailable: this.fileflowsAvailable,
            enableGPU: this.config.enableGPU,
            gpuType: this.config.gpuType,
            maxConcurrentJobs: this.config.maxConcurrentJobs,
            activeJobs: this.activeJobs.size,
            queuedJobs,
            processingJobs,
            completedJobs,
            failedJobs,
            totalJobs: this.jobs.size,
            profiles: this.profiles.size,
            config: {
                videoCodec: this.config.videoCodec,
                audioCodec: this.config.audioCodec,
                qualityPresets: this.config.qualityPresets,
                enableHDR: this.config.enableHDR
            },
            lastUpdate: new Date()
        };
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            console.log('🧹 Cleaning up TranscodingService...');
            
            // Cancel all active jobs
            const activeJobIds = Array.from(this.activeJobs.keys());
            await Promise.all(activeJobIds.map(jobId => 
                this.cancelJob(jobId).catch(err => console.warn('Job cancellation failed:', err))
            ));
            
            this.jobs.clear();
            this.queue = [];
            this.activeJobs.clear();
            this.profiles.clear();
            this.removeAllListeners();
            
            this.isInitialized = false;
            console.log('✅ TranscodingService cleanup completed');
        } catch (error) {
            console.error('❌ TranscodingService cleanup failed:', error);
        }
    }
}

module.exports = TranscodingService;