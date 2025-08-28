/**
 * Home Assistant Integration for NEXUS Media Hub
 * Connects to Home Assistant at http://homeassistant.local:8123
 */

const axios = require('axios');
const WebSocket = require('ws');
const EventEmitter = require('events');

class HomeAssistantIntegration extends EventEmitter {
    constructor() {
        super();
        this.baseURL = 'http://homeassistant.local:8123';
        this.wsURL = 'ws://homeassistant.local:8123/api/websocket';
        this.token = process.env.HOME_ASSISTANT_TOKEN || '';
        this.ws = null;
        this.entities = new Map();
        this.messageId = 1;
        this.connected = false;
        
        this.init();
    }

    async init() {
        console.log('🏠 Connecting to Home Assistant...');
        await this.connect();
        await this.subscribeToEvents();
        await this.fetchEntities();
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsURL);

            this.ws.on('open', () => {
                console.log('✅ Connected to Home Assistant WebSocket');
            });

            this.ws.on('message', async (data) => {
                const message = JSON.parse(data);
                
                if (message.type === 'auth_required') {
                    this.authenticate();
                } else if (message.type === 'auth_ok') {
                    console.log('✅ Authenticated with Home Assistant');
                    this.connected = true;
                    resolve();
                } else if (message.type === 'result') {
                    this.handleResult(message);
                } else if (message.type === 'event') {
                    this.handleEvent(message);
                }
            });

            this.ws.on('error', (error) => {
                console.error('❌ Home Assistant WebSocket error:', error);
                reject(error);
            });

            this.ws.on('close', () => {
                console.log('🔌 Disconnected from Home Assistant');
                this.connected = false;
                // Attempt reconnection
                setTimeout(() => this.connect(), 5000);
            });
        });
    }

    authenticate() {
        this.sendMessage({
            type: 'auth',
            access_token: this.token
        });
    }

    sendMessage(message) {
        if (message.type !== 'auth') {
            message.id = this.messageId++;
        }
        this.ws.send(JSON.stringify(message));
        return message.id;
    }

    async subscribeToEvents() {
        this.sendMessage({
            type: 'subscribe_events',
            event_type: 'state_changed'
        });
    }

    async fetchEntities() {
        const id = this.sendMessage({
            type: 'get_states'
        });
        
        // Store promise for result
        return new Promise((resolve) => {
            this.once(`result_${id}`, (result) => {
                if (result.success) {
                    result.result.forEach(entity => {
                        this.entities.set(entity.entity_id, entity);
                    });
                    console.log(`📊 Loaded ${this.entities.size} entities from Home Assistant`);
                }
                resolve();
            });
        });
    }

    handleResult(message) {
        this.emit(`result_${message.id}`, message);
    }

    handleEvent(message) {
        if (message.event.event_type === 'state_changed') {
            const data = message.event.data;
            this.entities.set(data.entity_id, data.new_state);
            this.emit('state_changed', data);
            
            // Emit specific events for media sync
            if (data.entity_id.startsWith('light.')) {
                this.emit('light_changed', data);
            } else if (data.entity_id.startsWith('media_player.')) {
                this.emit('media_player_changed', data);
            }
        }
    }

    /**
     * Control Lights
     */
    async setLights(options = {}) {
        const {
            entity_id = 'light.living_room',
            brightness = 255,
            color = [0, 255, 255], // Cyan for cyberpunk theme
            transition = 1
        } = options;

        return this.callService('light', 'turn_on', {
            entity_id,
            brightness,
            rgb_color: color,
            transition
        });
    }

    async turnOffLights(entity_id = 'light.living_room') {
        return this.callService('light', 'turn_off', { entity_id });
    }

    async syncLightsWithMedia(mediaType) {
        console.log(`🎬 Syncing lights for ${mediaType} content`);
        
        const lightSettings = {
            'horror': { color: [255, 0, 0], brightness: 50 },      // Red, dim
            'action': { color: [255, 165, 0], brightness: 200 },   // Orange, bright
            'romance': { color: [255, 192, 203], brightness: 150 }, // Pink, soft
            'sci-fi': { color: [0, 255, 255], brightness: 180 },   // Cyan
            'comedy': { color: [255, 255, 0], brightness: 255 },   // Yellow, bright
            'drama': { color: [255, 255, 255], brightness: 100 }   // White, dim
        };

        const settings = lightSettings[mediaType] || lightSettings['sci-fi'];
        
        // Apply to all entertainment area lights
        const lights = Array.from(this.entities.keys())
            .filter(id => id.startsWith('light.') && 
                         (id.includes('living') || id.includes('tv') || id.includes('entertainment')));
        
        for (const light of lights) {
            await this.setLights({
                entity_id: light,
                ...settings
            });
        }
    }

    /**
     * Climate Control
     */
    async setTemperature(temperature = 22) {
        return this.callService('climate', 'set_temperature', {
            entity_id: 'climate.living_room',
            temperature
        });
    }

    async setClimateMode(mode = 'cool') {
        return this.callService('climate', 'set_hvac_mode', {
            entity_id: 'climate.living_room',
            hvac_mode: mode
        });
    }

    /**
     * Media Players
     */
    async controlMediaPlayer(action, entity_id = 'media_player.living_room_tv') {
        const actions = {
            'play': 'media_play',
            'pause': 'media_pause',
            'stop': 'media_stop',
            'next': 'media_next_track',
            'previous': 'media_previous_track'
        };

        return this.callService('media_player', actions[action], { entity_id });
    }

    async setVolume(volume, entity_id = 'media_player.living_room_tv') {
        return this.callService('media_player', 'volume_set', {
            entity_id,
            volume_level: volume / 100 // Convert to 0-1 range
        });
    }

    async playMediaOnDevice(media_url, entity_id = 'media_player.living_room_tv') {
        return this.callService('media_player', 'play_media', {
            entity_id,
            media_content_id: media_url,
            media_content_type: 'video'
        });
    }

    /**
     * Security System
     */
    async armSecuritySystem(mode = 'away') {
        return this.callService('alarm_control_panel', `alarm_arm_${mode}`, {
            entity_id: 'alarm_control_panel.home'
        });
    }

    async disarmSecuritySystem(code = '') {
        return this.callService('alarm_control_panel', 'alarm_disarm', {
            entity_id: 'alarm_control_panel.home',
            code
        });
    }

    /**
     * Scenes
     */
    async activateScene(scene) {
        const scenes = {
            'movie_night': 'scene.movie_night',
            'party': 'scene.party_mode',
            'gaming': 'scene.gaming_setup',
            'romantic': 'scene.romantic_evening',
            'cyberpunk': 'scene.cyberpunk_theme'
        };

        return this.callService('scene', 'turn_on', {
            entity_id: scenes[scene] || scene
        });
    }

    /**
     * Automations
     */
    async createMediaAutomation(mediaId, actions) {
        // Create automation for when specific media plays
        const automation = {
            alias: `Media Automation - ${mediaId}`,
            trigger: {
                platform: 'state',
                entity_id: 'media_player.living_room_tv',
                to: 'playing'
            },
            condition: {
                condition: 'template',
                value_template: `{{ state_attr('media_player.living_room_tv', 'media_title') == '${mediaId}' }}`
            },
            action: actions
        };

        // This would normally be sent to HA's automation API
        console.log('Creating automation:', automation);
        return automation;
    }

    /**
     * Smart Speakers Integration
     */
    async announce(message, entity_id = 'media_player.everywhere') {
        return this.callService('tts', 'google_say', {
            entity_id,
            message
        });
    }

    async playMusicOnSpeakers(playlist, entity_id = 'media_player.speakers') {
        return this.callService('media_player', 'play_media', {
            entity_id,
            media_content_id: playlist,
            media_content_type: 'playlist'
        });
    }

    /**
     * Philips Hue Specific
     */
    async setHueScene(scene) {
        return this.callService('hue', 'hue_activate_scene', {
            group_name: 'Living Room',
            scene_name: scene
        });
    }

    async enableAmbilight(enabled = true) {
        // Enable Hue Sync for Ambilight effect
        return this.callService('switch', enabled ? 'turn_on' : 'turn_off', {
            entity_id: 'switch.hue_sync'
        });
    }

    /**
     * Helper method to call Home Assistant services
     */
    async callService(domain, service, service_data = {}) {
        if (!this.connected) {
            throw new Error('Not connected to Home Assistant');
        }

        const id = this.sendMessage({
            type: 'call_service',
            domain,
            service,
            service_data
        });

        return new Promise((resolve, reject) => {
            this.once(`result_${id}`, (result) => {
                if (result.success) {
                    resolve(result.result);
                } else {
                    reject(new Error(result.error?.message || 'Service call failed'));
                }
            });

            // Timeout after 5 seconds
            setTimeout(() => {
                reject(new Error('Service call timeout'));
            }, 5000);
        });
    }

    /**
     * Get current state of an entity
     */
    getEntityState(entity_id) {
        return this.entities.get(entity_id);
    }

    /**
     * Get all entities of a specific type
     */
    getEntitiesByDomain(domain) {
        return Array.from(this.entities.entries())
            .filter(([id]) => id.startsWith(`${domain}.`))
            .map(([id, state]) => ({ entity_id: id, ...state }));
    }

    /**
     * NEXUS-specific integrations
     */
    async startMovieMode(movieTitle, genre) {
        console.log(`🎬 Starting movie mode for "${movieTitle}" (${genre})`);
        
        // 1. Activate movie scene
        await this.activateScene('movie_night');
        
        // 2. Sync lights with genre
        await this.syncLightsWithMedia(genre);
        
        // 3. Set optimal temperature
        await this.setTemperature(21);
        
        // 4. Announce on speakers
        await this.announce(`Now playing ${movieTitle}. Enjoy your movie!`);
        
        // 5. Enable Ambilight if available
        await this.enableAmbilight(true);
        
        // 6. Arm security in home mode
        await this.armSecuritySystem('home');
        
        return {
            status: 'success',
            message: `Movie mode activated for ${movieTitle}`,
            settings: {
                lights: genre,
                temperature: 21,
                security: 'armed_home',
                ambilight: true
            }
        };
    }

    async endMovieMode() {
        console.log('🛑 Ending movie mode');
        
        // Reset to normal scene
        await this.activateScene('normal');
        
        // Turn on normal lights
        await this.setLights({
            color: [255, 255, 255],
            brightness: 200
        });
        
        // Disable Ambilight
        await this.enableAmbilight(false);
        
        return { status: 'success', message: 'Movie mode ended' };
    }

    /**
     * Party Mode with music sync
     */
    async startPartyMode() {
        console.log('🎉 Starting party mode!');
        
        // Activate party scene
        await this.activateScene('party');
        
        // Start color loop on all lights
        const lights = this.getEntitiesByDomain('light');
        for (const light of lights) {
            await this.callService('light', 'turn_on', {
                entity_id: light.entity_id,
                effect: 'colorloop',
                brightness: 255
            });
        }
        
        // Play party playlist
        await this.playMusicOnSpeakers('spotify:playlist:partyvibes');
        
        // Set volume
        await this.setVolume(70);
        
        return { status: 'success', message: 'Party mode activated!' };
    }
}

// Express API endpoints
const express = require('express');
const router = express.Router();
let haIntegration = null;

// Initialize Home Assistant connection
router.use(async (req, res, next) => {
    if (!haIntegration) {
        haIntegration = new HomeAssistantIntegration();
    }
    next();
});

// Get all entities
router.get('/entities', (req, res) => {
    const entities = Array.from(haIntegration.entities.values());
    res.json(entities);
});

// Control lights
router.post('/lights', async (req, res) => {
    try {
        const result = await haIntegration.setLights(req.body);
        res.json({ status: 'success', result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Start movie mode
router.post('/movie-mode', async (req, res) => {
    try {
        const { title, genre } = req.body;
        const result = await haIntegration.startMovieMode(title, genre);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Control climate
router.post('/climate', async (req, res) => {
    try {
        const { temperature } = req.body;
        const result = await haIntegration.setTemperature(temperature);
        res.json({ status: 'success', result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Activate scene
router.post('/scene', async (req, res) => {
    try {
        const { scene } = req.body;
        const result = await haIntegration.activateScene(scene);
        res.json({ status: 'success', result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

module.exports = { router, HomeAssistantIntegration };