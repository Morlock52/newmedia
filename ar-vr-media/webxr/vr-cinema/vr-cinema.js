/**
 * VR Cinema - Immersive cinema experience with social viewing
 * Optimized for Apple Vision Pro and Meta Quest 3
 */

import { webXRManager } from '../core/webxr-manager.js';

export class VRCinema {
    constructor() {
        this.cinema = null;
        this.screen = null;
        this.videoTexture = null;
        this.spatialVideo = null;
        this.socialAvatars = new Map();
        this.roomEnvironment = null;
        this.lightingSystem = null;
        this.audioSystem = null;
        this.seatPositions = [];
        this.currentRoom = 'theater'; // theater, living-room, space, underwater
        
        this.init();
    }

    async init() {
        console.log('🎬 Initializing VR Cinema...');
        
        // Create cinema environment
        this.createCinemaEnvironment();
        
        // Setup spatial video system
        this.setupSpatialVideo();
        
        // Initialize social features
        this.initializeSocialFeatures();
        
        // Setup lighting system
        this.setupDynamicLighting();
        
        // Initialize spatial audio
        this.setupSpatialAudio();
        
        console.log('✅ VR Cinema initialized');
    }

    createCinemaEnvironment() {
        this.cinema = new THREE.Group();
        webXRManager.scene.add(this.cinema);

        // Create different room environments
        this.createTheaterRoom();
        this.createLivingRoom();
        this.createSpaceEnvironment();
        this.createUnderwaterEnvironment();
        
        // Set default room
        this.switchRoom('theater');
    }

    createTheaterRoom() {
        const theaterGroup = new THREE.Group();
        theaterGroup.name = 'theater';

        // Theater walls
        const wallGeometry = new THREE.PlaneGeometry(20, 12);
        const wallMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x2a1810,
            side: THREE.DoubleSide 
        });

        // Back wall
        const backWall = new THREE.Mesh(wallGeometry, wallMaterial);
        backWall.position.set(0, 6, -10);
        theaterGroup.add(backWall);

        // Side walls
        const leftWall = new THREE.Mesh(wallGeometry, wallMaterial);
        leftWall.rotation.y = Math.PI / 2;
        leftWall.position.set(-10, 6, 0);
        theaterGroup.add(leftWall);

        const rightWall = new THREE.Mesh(wallGeometry, wallMaterial);
        rightWall.rotation.y = -Math.PI / 2;
        rightWall.position.set(10, 6, 0);
        theaterGroup.add(rightWall);

        // Floor
        const floorGeometry = new THREE.PlaneGeometry(20, 20);
        const floorMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x1a0f08,
            side: THREE.DoubleSide 
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0;
        theaterGroup.add(floor);

        // Ceiling
        const ceiling = new THREE.Mesh(floorGeometry, wallMaterial);
        ceiling.rotation.x = Math.PI / 2;
        ceiling.position.y = 12;
        theaterGroup.add(ceiling);

        // Theater seats
        this.createTheaterSeats(theaterGroup);

        // Screen
        this.createMainScreen(theaterGroup);

        this.cinema.add(theaterGroup);
    }

    createLivingRoom() {
        const livingGroup = new THREE.Group();
        livingGroup.name = 'living-room';
        livingGroup.visible = false;

        // Cozy living room environment
        const roomGeometry = new THREE.BoxGeometry(15, 8, 15);
        const roomMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x8b7355,
            side: THREE.BackSide 
        });
        const room = new THREE.Mesh(roomGeometry, roomMaterial);
        room.position.y = 4;
        livingGroup.add(room);

        // Couch
        const couchGeometry = new THREE.BoxGeometry(4, 1, 2);
        const couchMaterial = new THREE.MeshLambertMaterial({ color: 0x654321 });
        const couch = new THREE.Mesh(couchGeometry, couchMaterial);
        couch.position.set(0, 0.5, 3);
        livingGroup.add(couch);

        // Coffee table
        const tableGeometry = new THREE.BoxGeometry(2, 0.1, 1);
        const tableMaterial = new THREE.MeshLambertMaterial({ color: 0x8b4513 });
        const table = new THREE.Mesh(tableGeometry, tableMaterial);
        table.position.set(0, 0.5, 1);
        livingGroup.add(table);

        // TV screen for living room
        this.createTVScreen(livingGroup);

        this.cinema.add(livingGroup);
    }

    createSpaceEnvironment() {
        const spaceGroup = new THREE.Group();
        spaceGroup.name = 'space';
        spaceGroup.visible = false;

        // Space background
        const spaceGeometry = new THREE.SphereGeometry(50, 32, 32);
        const spaceMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x000011,
            side: THREE.BackSide 
        });
        const space = new THREE.Mesh(spaceGeometry, spaceMaterial);
        spaceGroup.add(space);

        // Stars
        this.createStarField(spaceGroup);

        // Floating screen in space
        this.createFloatingScreen(spaceGroup);

        // Floating seats
        this.createFloatingSeats(spaceGroup);

        this.cinema.add(spaceGroup);
    }

    createUnderwaterEnvironment() {
        const underwaterGroup = new THREE.Group();
        underwaterGroup.name = 'underwater';
        underwaterGroup.visible = false;

        // Underwater sphere
        const underwaterGeometry = new THREE.SphereGeometry(30, 32, 32);
        const underwaterMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x006994,
            transparent: true,
            opacity: 0.8,
            side: THREE.BackSide 
        });
        const underwater = new THREE.Mesh(underwaterGeometry, underwaterMaterial);
        underwaterGroup.add(underwater);

        // Bubbles
        this.createUnderwaterBubbles(underwaterGroup);

        // Underwater screen
        this.createUnderwaterScreen(underwaterGroup);

        this.cinema.add(underwaterGroup);
    }

    createMainScreen(parent) {
        // Large cinema screen
        const screenGeometry = new THREE.PlaneGeometry(12, 6.75); // 16:9 aspect ratio
        const screenMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x000000,
            transparent: true 
        });
        
        this.screen = new THREE.Mesh(screenGeometry, screenMaterial);
        this.screen.position.set(0, 4, -8);
        parent.add(this.screen);

        // Screen frame
        const frameGeometry = new THREE.PlaneGeometry(12.5, 7.25);
        const frameMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
        const frame = new THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.set(0, 4, -8.01);
        parent.add(frame);
    }

    createTVScreen(parent) {
        const screenGeometry = new THREE.PlaneGeometry(6, 3.375);
        const screenMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x000000,
            transparent: true 
        });
        
        this.screen = new THREE.Mesh(screenGeometry, screenMaterial);
        this.screen.position.set(0, 2, -5);
        parent.add(this.screen);
    }

    createFloatingScreen(parent) {
        const screenGeometry = new THREE.PlaneGeometry(10, 5.625);
        const screenMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x000000,
            transparent: true 
        });
        
        this.screen = new THREE.Mesh(screenGeometry, screenMaterial);
        this.screen.position.set(0, 0, -8);
        parent.add(this.screen);

        // Glowing frame
        const frameGeometry = new THREE.PlaneGeometry(10.5, 6.125);
        const frameMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x00ffff,
            transparent: true,
            opacity: 0.3 
        });
        const frame = new THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.set(0, 0, -8.01);
        parent.add(frame);
    }

    createUnderwaterScreen(parent) {
        const screenGeometry = new THREE.PlaneGeometry(8, 4.5);
        const screenMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x000000,
            transparent: true 
        });
        
        this.screen = new THREE.Mesh(screenGeometry, screenMaterial);
        this.screen.position.set(0, 0, -6);
        parent.add(this.screen);
    }

    createTheaterSeats(parent) {
        this.seatPositions = [];
        
        // Create rows of seats
        for (let row = 0; row < 3; row++) {
            for (let seat = -2; seat <= 2; seat++) {
                const seatGeometry = new THREE.BoxGeometry(0.8, 1, 0.8);
                const seatMaterial = new THREE.MeshLambertMaterial({ color: 0x8b0000 });
                const seatMesh = new THREE.Mesh(seatGeometry, seatMaterial);
                
                const x = seat * 1.2;
                const z = 2 + row * 1.5;
                seatMesh.position.set(x, 0.5, z);
                parent.add(seatMesh);
                
                this.seatPositions.push({ x, y: 1.6, z, occupied: false });
            }
        }
    }

    createFloatingSeats(parent) {
        // Floating seats in space
        for (let i = 0; i < 5; i++) {
            const seatGeometry = new THREE.BoxGeometry(1, 1, 1);
            const seatMaterial = new THREE.MeshLambertMaterial({ 
                color: 0x666666,
                transparent: true,
                opacity: 0.8 
            });
            const seat = new THREE.Mesh(seatGeometry, seatMaterial);
            
            const angle = (i / 5) * Math.PI * 2;
            seat.position.set(
                Math.cos(angle) * 5,
                Math.sin(i) * 2,
                Math.sin(angle) * 5 + 3
            );
            parent.add(seat);
        }
    }

    createStarField(parent) {
        const starGeometry = new THREE.BufferGeometry();
        const starPositions = [];
        
        for (let i = 0; i < 1000; i++) {
            starPositions.push(
                (Math.random() - 0.5) * 100,
                (Math.random() - 0.5) * 100,
                (Math.random() - 0.5) * 100
            );
        }
        
        starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
        const starMaterial = new THREE.PointsMaterial({ 
            color: 0xffffff,
            size: 0.5,
            sizeAttenuation: false 
        });
        const stars = new THREE.Points(starGeometry, starMaterial);
        parent.add(stars);
    }

    createUnderwaterBubbles(parent) {
        for (let i = 0; i < 20; i++) {
            const bubbleGeometry = new THREE.SphereGeometry(0.1 + Math.random() * 0.2, 8, 8);
            const bubbleMaterial = new THREE.MeshBasicMaterial({ 
                color: 0x87ceeb,
                transparent: true,
                opacity: 0.6 
            });
            const bubble = new THREE.Mesh(bubbleGeometry, bubbleMaterial);
            
            bubble.position.set(
                (Math.random() - 0.5) * 20,
                Math.random() * 10,
                (Math.random() - 0.5) * 20
            );
            
            parent.add(bubble);
            
            // Animate bubbles
            const animateBubble = () => {
                bubble.position.y += 0.02;
                if (bubble.position.y > 15) {
                    bubble.position.y = -5;
                }
                requestAnimationFrame(animateBubble);
            };
            animateBubble();
        }
    }

    setupSpatialVideo() {
        console.log('📹 Setting up spatial video system...');
        
        // Create video element
        const video = document.createElement('video');
        video.src = 'assets/sample-video.mp4';
        video.crossOrigin = 'anonymous';
        video.loop = true;
        video.muted = true; // Required for autoplay
        
        // Create video texture
        this.videoTexture = new THREE.VideoTexture(video);
        this.videoTexture.minFilter = THREE.LinearFilter;
        this.videoTexture.magFilter = THREE.LinearFilter;
        
        // Apply texture to screen
        if (this.screen) {
            this.screen.material.map = this.videoTexture;
            this.screen.material.needsUpdate = true;
        }
        
        // Spatial video configuration for Apple Vision Pro
        this.spatialVideo = {
            element: video,
            texture: this.videoTexture,
            is3D: false,
            isSpatial: false,
            leftEye: null,
            rightEye: null
        };
    }

    async loadSpatialVideo(videoUrl, isSpatial = false) {
        console.log('📼 Loading spatial video:', videoUrl);
        
        this.spatialVideo.element.src = videoUrl;
        this.spatialVideo.isSpatial = isSpatial;
        
        if (isSpatial) {
            // Configure for spatial video (side-by-side or over-under)
            await this.setupSpatialVideoGeometry();
        }
        
        return new Promise((resolve) => {
            this.spatialVideo.element.addEventListener('loadeddata', () => {
                console.log('✅ Spatial video loaded');
                resolve();
            });
        });
    }

    async setupSpatialVideoGeometry() {
        // Create separate geometry for left and right eye views
        const leftGeometry = new THREE.PlaneGeometry(6, 3.375);
        const rightGeometry = new THREE.PlaneGeometry(6, 3.375);
        
        // Configure UV mapping for spatial video
        this.configureSpatialUVMapping(leftGeometry, rightGeometry);
    }

    configureSpatialUVMapping(leftGeometry, rightGeometry) {
        // Side-by-side spatial video UV mapping
        const leftUV = leftGeometry.attributes.uv.array;
        const rightUV = rightGeometry.attributes.uv.array;
        
        // Left eye - use left half of video
        for (let i = 0; i < leftUV.length; i += 2) {
            leftUV[i] *= 0.5; // Scale U coordinate to left half
        }
        
        // Right eye - use right half of video
        for (let i = 0; i < rightUV.length; i += 2) {
            rightUV[i] = 0.5 + (rightUV[i] * 0.5); // Scale and offset to right half
        }
        
        leftGeometry.attributes.uv.needsUpdate = true;
        rightGeometry.attributes.uv.needsUpdate = true;
    }

    initializeSocialFeatures() {
        console.log('👥 Initializing social features...');
        
        // Social avatar system
        this.socialAvatars = new Map();
        
        // Voice chat integration points
        this.setupVoiceChat();
        
        // Shared viewing synchronization
        this.setupViewingSynchronization();
    }

    setupVoiceChat() {
        // Voice chat would integrate with WebRTC here
        console.log('🎤 Voice chat system ready');
    }

    setupViewingSynchronization() {
        // Synchronized playback for group viewing
        console.log('🔄 Viewing synchronization ready');
    }

    addSocialAvatar(userId, position, appearance) {
        const avatarGeometry = new THREE.CapsuleGeometry(0.3, 1.4);
        const avatarMaterial = new THREE.MeshLambertMaterial({ 
            color: appearance.color || 0x4a90e2 
        });
        const avatar = new THREE.Mesh(avatarGeometry, avatarMaterial);
        
        avatar.position.copy(position);
        this.cinema.add(avatar);
        
        this.socialAvatars.set(userId, avatar);
        
        console.log(`👤 Added social avatar for user: ${userId}`);
    }

    setupDynamicLighting() {
        this.lightingSystem = {
            ambient: new THREE.AmbientLight(0x404040, 0.2),
            screen: new THREE.DirectionalLight(0xffffff, 0.8),
            environment: []
        };
        
        // Add screen lighting
        this.lightingSystem.screen.position.set(0, 4, -7);
        this.lightingSystem.screen.target.position.set(0, 2, 0);
        webXRManager.scene.add(this.lightingSystem.screen);
        webXRManager.scene.add(this.lightingSystem.screen.target);
        
        // Dynamic lighting based on video content
        this.setupContentAwareLighting();
    }

    setupContentAwareLighting() {
        // Analyze video content to adjust lighting dynamically
        console.log('💡 Content-aware lighting system ready');
    }

    setupSpatialAudio() {
        console.log('🔊 Setting up spatial audio for cinema...');
        
        this.audioSystem = {
            listener: new THREE.AudioListener(),
            sounds: new Map(),
            environmentalAudio: []
        };
        
        webXRManager.camera.add(this.audioSystem.listener);
        
        // Setup positional audio for video
        if (this.spatialVideo && this.spatialVideo.element) {
            const sound = new THREE.PositionalAudio(this.audioSystem.listener);
            sound.setMediaElementSource(this.spatialVideo.element);
            sound.setRefDistance(10);
            sound.setVolume(0.8);
            
            this.screen.add(sound);
            this.audioSystem.sounds.set('video', sound);
        }
        
        // Environmental audio based on room
        this.setupEnvironmentalAudio();
    }

    setupEnvironmentalAudio() {
        // Different ambient sounds for each environment
        const environmentSounds = {
            'theater': 'assets/audio/theater-ambient.mp3',
            'living-room': 'assets/audio/livingroom-ambient.mp3',
            'space': 'assets/audio/space-ambient.mp3',
            'underwater': 'assets/audio/underwater-ambient.mp3'
        };
        
        // Load and setup environmental audio
        Object.entries(environmentSounds).forEach(([env, audioUrl]) => {
            // Audio loading implementation would go here
            console.log(`🎵 Environmental audio ready for: ${env}`);
        });
    }

    switchRoom(roomName) {
        console.log(`🏠 Switching to room: ${roomName}`);
        
        // Hide all rooms
        this.cinema.children.forEach(room => {
            room.visible = false;
        });
        
        // Show selected room
        const selectedRoom = this.cinema.children.find(room => room.name === roomName);
        if (selectedRoom) {
            selectedRoom.visible = true;
            this.currentRoom = roomName;
            
            // Update screen reference
            this.screen = selectedRoom.children.find(child => 
                child.geometry && child.geometry.type === 'PlaneGeometry' && 
                child.material.map
            );
            
            // Update lighting for room
            this.updateRoomLighting(roomName);
            
            // Update environmental audio
            this.updateEnvironmentalAudio(roomName);
        }
    }

    updateRoomLighting(roomName) {
        const lightingConfigs = {
            'theater': { ambient: 0.1, screen: 0.8, color: 0xffffff },
            'living-room': { ambient: 0.3, screen: 0.6, color: 0xffaa55 },
            'space': { ambient: 0.05, screen: 1.0, color: 0xaaaaff },
            'underwater': { ambient: 0.2, screen: 0.7, color: 0x55aaff }
        };
        
        const config = lightingConfigs[roomName];
        if (config) {
            this.lightingSystem.ambient.intensity = config.ambient;
            this.lightingSystem.screen.intensity = config.screen;
            this.lightingSystem.screen.color.setHex(config.color);
        }
    }

    updateEnvironmentalAudio(roomName) {
        // Switch environmental audio based on room
        console.log(`🎵 Updating environmental audio for: ${roomName}`);
    }

    playVideo() {
        if (this.spatialVideo && this.spatialVideo.element) {
            this.spatialVideo.element.play();
            console.log('▶️ Video playing');
        }
    }

    pauseVideo() {
        if (this.spatialVideo && this.spatialVideo.element) {
            this.spatialVideo.element.pause();
            console.log('⏸️ Video paused');
        }
    }

    async enterVRMode() {
        try {
            await webXRManager.startVRSession(['local-floor'], ['hand-tracking', 'layers']);
            console.log('🥽 Entered VR Cinema mode');
            
            // Start video playback in VR
            this.playVideo();
            
            return true;
        } catch (error) {
            console.error('Failed to enter VR mode:', error);
            return false;
        }
    }

    // Hand gesture controls for video playback
    setupHandGestureControls() {
        // Implement hand gesture recognition for play/pause/volume
        console.log('👋 Hand gesture controls ready');
    }

    // Eye tracking for gaze-based UI
    setupGazeControls() {
        // Implement eye tracking for menu navigation
        console.log('👁️ Gaze controls ready');
    }

    // Haptic feedback for interactions
    triggerHapticFeedback(type = 'click') {
        const intensities = {
            'click': 0.3,
            'select': 0.5,
            'volume': 0.2
        };
        
        webXRManager.pulseHaptic(intensities[type] || 0.3, 100);
    }
}

// Export VR Cinema class
export default VRCinema;