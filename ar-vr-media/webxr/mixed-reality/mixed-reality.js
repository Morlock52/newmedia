/**
 * Mixed Reality Collaborative Spaces - Advanced collaborative environments
 * Seamlessly blends virtual and physical worlds for shared experiences
 */

import { webXRManager } from '../core/webxr-manager.js';

export class MixedRealityCollaborativeSpaces {
    constructor() {
        this.collaborativeSpaces = new Map();
        this.participants = new Map();
        this.sharedObjects = new Map();
        this.spatialAnchors = new Map();
        this.realWorldMapping = null;
        this.virtualOverlays = new Map();
        this.communicationSystem = null;
        this.synchronizationEngine = null;
        this.persistenceManager = null;
        
        // Mixed Reality components
        this.passthroughRenderer = null;
        this.occlusionMesh = null;
        this.lightingEstimation = null;
        this.semanticSegmentation = null;
        this.meshGeneration = null;
        
        // Collaboration features
        this.voiceChat = null;
        this.gestureSharing = null;
        this.spatialClipboard = null;
        this.coPresence = null;
        this.sharedWorkspaces = new Map();
        
        this.init();
    }

    async init() {
        console.log('🌍 Initializing Mixed Reality Collaborative Spaces...');
        
        // Initialize core MR systems
        this.initPassthroughRendering();
        this.initRealWorldMapping();
        this.initSpatialAnchoring();
        this.initLightingEstimation();
        this.initOcclusionHandling();
        
        // Initialize collaboration systems
        this.initCommunicationSystem();
        this.initSynchronizationEngine();
        this.initPersistenceManager();
        this.initCoPresenceSystem();
        
        // Initialize collaborative features
        this.initSharedWorkspaces();
        this.initSpatialTools();
        this.initCollaborativeGestures();
        
        console.log('✅ Mixed Reality Collaborative Spaces initialized');
    }

    initPassthroughRendering() {
        console.log('📹 Initializing passthrough rendering...');
        
        this.passthroughRenderer = {
            isEnabled: false,
            cameras: new Map(),
            depthMaps: new Map(),
            colorCorrection: {
                brightness: 1.0,
                contrast: 1.0,
                saturation: 1.0,
                hue: 0.0
            },
            latencyCompensation: true,
            stabilization: true
        };
        
        // Initialize passthrough cameras
        this.initPassthroughCameras();
    }

    initPassthroughCameras() {
        // Simulated passthrough camera setup
        this.passthroughRenderer.cameras.set('front_left', {
            position: new THREE.Vector3(-0.032, 0, 0),
            orientation: new THREE.Quaternion(),
            fov: 110,
            resolution: { width: 1920, height: 1080 },
            fps: 90
        });
        
        this.passthroughRenderer.cameras.set('front_right', {
            position: new THREE.Vector3(0.032, 0, 0),
            orientation: new THREE.Quaternion(),
            fov: 110,
            resolution: { width: 1920, height: 1080 },
            fps: 90
        });
        
        // Additional cameras for full coverage
        this.passthroughRenderer.cameras.set('side_left', {
            position: new THREE.Vector3(-0.05, 0, 0.02),
            orientation: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI/4, 0)),
            fov: 90,
            resolution: { width: 1280, height: 720 },
            fps: 60
        });
        
        this.passthroughRenderer.cameras.set('side_right', {
            position: new THREE.Vector3(0.05, 0, 0.02),
            orientation: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI/4, 0)),
            fov: 90,
            resolution: { width: 1280, height: 720 },
            fps: 60
        });
    }

    initRealWorldMapping() {
        console.log('🗺️ Initializing real world mapping...');
        
        this.realWorldMapping = {
            spatialMesh: new Map(),
            planeSurfaces: new Map(),
            semanticLabels: new Map(),
            pointCloud: null,
            confidenceMap: new Map(),
            updateFrequency: 30, // Hz
            meshQuality: 'medium' // low, medium, high
        };
        
        // Initialize spatial understanding
        this.initSpatialUnderstanding();
    }

    initSpatialUnderstanding() {
        // Spatial mesh generation
        this.meshGeneration = {
            trianglesPerCubicMeter: 1000,
            meshLOD: 3,
            meshSimplification: true,
            meshOptimization: true,
            realTimeUpdate: true
        };
        
        // Semantic segmentation
        this.semanticSegmentation = {
            enabled: true,
            categories: [
                'floor', 'wall', 'ceiling', 'table', 'chair', 'door', 'window',
                'person', 'screen', 'keyboard', 'plant', 'artwork', 'storage'
            ],
            confidence: 0.7,
            neuralNetwork: null
        };
        
        // Start spatial mapping
        this.startSpatialMapping();
    }

    startSpatialMapping() {
        const mapEnvironment = () => {
            if (!webXRManager.xrSession) {
                setTimeout(mapEnvironment, 100);
                return;
            }
            
            // Update spatial mesh
            this.updateSpatialMesh();
            
            // Detect and classify surfaces
            this.updatePlaneSurfaces();
            
            // Update semantic labels
            this.updateSemanticLabels();
            
            // Schedule next update
            setTimeout(mapEnvironment, 1000 / this.realWorldMapping.updateFrequency);
        };
        
        mapEnvironment();
    }

    updateSpatialMesh() {
        // Simulate spatial mesh updates
        const meshId = `mesh_${Date.now()}`;
        
        // Generate mock mesh data
        const vertices = this.generateMockMeshVertices();
        const triangles = this.generateMockMeshTriangles(vertices);
        
        const meshData = {
            id: meshId,
            vertices: vertices,
            triangles: triangles,
            timestamp: Date.now(),
            confidence: 0.8 + Math.random() * 0.2,
            bounds: this.calculateMeshBounds(vertices)
        };
        
        this.realWorldMapping.spatialMesh.set(meshId, meshData);
        
        // Create Three.js mesh for visualization
        this.createVisualizationMesh(meshData);
    }

    generateMockMeshVertices() {
        const vertices = [];
        
        // Generate vertices for a room-like environment
        for (let i = 0; i < 1000; i++) {
            vertices.push({
                position: new THREE.Vector3(
                    (Math.random() - 0.5) * 10,
                    Math.random() * 3,
                    (Math.random() - 0.5) * 10
                ),
                normal: new THREE.Vector3(
                    Math.random() - 0.5,
                    Math.random() - 0.5,
                    Math.random() - 0.5
                ).normalize(),
                confidence: Math.random()
            });
        }
        
        return vertices;
    }

    generateMockMeshTriangles(vertices) {
        const triangles = [];
        
        for (let i = 0; i < vertices.length - 3; i += 3) {
            triangles.push({
                indices: [i, i + 1, i + 2],
                normal: new THREE.Vector3(0, 1, 0),
                area: Math.random() * 0.1
            });
        }
        
        return triangles;
    }

    calculateMeshBounds(vertices) {
        const bounds = new THREE.Box3();
        
        vertices.forEach(vertex => {
            bounds.expandByPoint(vertex.position);
        });
        
        return bounds;
    }

    createVisualizationMesh(meshData) {
        const geometry = new THREE.BufferGeometry();
        
        // Convert vertices to buffer geometry
        const positions = [];
        const normals = [];
        
        meshData.triangles.forEach(triangle => {
            triangle.indices.forEach(index => {
                const vertex = meshData.vertices[index];
                positions.push(vertex.position.x, vertex.position.y, vertex.position.z);
                normals.push(vertex.normal.x, vertex.normal.y, vertex.normal.z);
            });
        });
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        
        const material = new THREE.MeshLambertMaterial({
            color: 0x808080,
            transparent: true,
            opacity: 0.3,
            wireframe: true
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `spatial-mesh-${meshData.id}`;
        mesh.userData.meshData = meshData;
        
        webXRManager.scene.add(mesh);
    }

    updatePlaneSurfaces() {
        // Detect plane surfaces (floors, walls, tables, etc.)
        const planeTypes = ['horizontal', 'vertical'];
        
        planeTypes.forEach(type => {
            const planes = this.detectPlanes(type);
            planes.forEach(plane => {
                this.realWorldMapping.planeSurfaces.set(plane.id, plane);
                this.createPlaneVisualization(plane);
            });
        });
    }

    detectPlanes(type) {
        const planes = [];
        
        // Mock plane detection
        for (let i = 0; i < 3; i++) {
            const plane = {
                id: `plane_${type}_${Date.now()}_${i}`,
                type: type,
                position: new THREE.Vector3(
                    (Math.random() - 0.5) * 8,
                    type === 'horizontal' ? Math.random() * 2 : Math.random() * 3,
                    (Math.random() - 0.5) * 8
                ),
                normal: type === 'horizontal' 
                    ? new THREE.Vector3(0, 1, 0)
                    : new THREE.Vector3(Math.random() - 0.5, 0, Math.random() - 0.5).normalize(),
                size: new THREE.Vector2(1 + Math.random() * 2, 1 + Math.random() * 2),
                confidence: 0.7 + Math.random() * 0.3,
                semanticLabel: this.inferSemanticLabel(type, i)
            };
            
            planes.push(plane);
        }
        
        return planes;
    }

    inferSemanticLabel(planeType, index) {
        if (planeType === 'horizontal') {
            return ['floor', 'table', 'shelf'][index % 3];
        } else {
            return ['wall', 'door', 'window'][index % 3];
        }
    }

    createPlaneVisualization(plane) {
        const geometry = new THREE.PlaneGeometry(plane.size.x, plane.size.y);
        const material = new THREE.MeshBasicMaterial({
            color: this.getSemanticColor(plane.semanticLabel),
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });
        
        const planeMesh = new THREE.Mesh(geometry, material);
        planeMesh.position.copy(plane.position);
        planeMesh.lookAt(plane.position.clone().add(plane.normal));
        planeMesh.name = `plane-${plane.id}`;
        planeMesh.userData.planeData = plane;
        
        webXRManager.scene.add(planeMesh);
    }

    getSemanticColor(label) {
        const colors = {
            floor: 0x8B4513,    // Brown
            wall: 0x708090,     // Slate gray
            ceiling: 0xF0F8FF,  // Alice blue
            table: 0xDEB887,    // Burlywood
            chair: 0x654321,    // Dark brown
            door: 0x8B4513,     // Saddle brown
            window: 0x87CEEB,   // Sky blue
            person: 0xFF69B4,   // Hot pink
            screen: 0x000000,   // Black
            plant: 0x228B22     // Forest green
        };
        
        return colors[label] || 0x808080;
    }

    updateSemanticLabels() {
        // Update semantic understanding of the environment
        if (!this.semanticSegmentation.enabled) return;
        
        // Process spatial mesh and plane data for semantic classification
        this.realWorldMapping.spatialMesh.forEach((mesh, meshId) => {
            const semanticData = this.classifyMeshSemantics(mesh);
            this.realWorldMapping.semanticLabels.set(meshId, semanticData);
        });
    }

    classifyMeshSemantics(mesh) {
        // Mock semantic classification
        const categories = this.semanticSegmentation.categories;
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        
        return {
            category: randomCategory,
            confidence: Math.random(),
            boundingBox: mesh.bounds,
            timestamp: Date.now()
        };
    }

    initSpatialAnchoring() {
        console.log('⚓ Initializing spatial anchoring...');
        
        this.spatialAnchors = new Map();
        this.anchorPersistence = {
            enabled: true,
            cloudSync: true,
            localCache: new Map(),
            maxAnchors: 100
        };
    }

    initLightingEstimation() {
        console.log('💡 Initializing lighting estimation...');
        
        this.lightingEstimation = {
            enabled: true,
            ambientIntensity: 1.0,
            mainLightDirection: new THREE.Vector3(0.3, -0.8, 0.5),
            mainLightIntensity: 1.0,
            colorTemperature: 5500, // Kelvin
            environmentMap: null,
            sphericalHarmonics: null,
            updateFrequency: 10 // Hz
        };
        
        this.startLightingEstimation();
    }

    startLightingEstimation() {
        const estimateLighting = () => {
            if (webXRManager.xrSession) {
                this.updateLightingEstimation();
            }
            
            setTimeout(estimateLighting, 1000 / this.lightingEstimation.updateFrequency);
        };
        
        estimateLighting();
    }

    updateLightingEstimation() {
        // Analyze real-world lighting conditions
        this.lightingEstimation.ambientIntensity = 0.8 + Math.random() * 0.4;
        
        // Update main light direction (sun/dominant light source)
        const time = Date.now() * 0.0001;
        this.lightingEstimation.mainLightDirection.set(
            Math.sin(time) * 0.5,
            -0.8,
            Math.cos(time) * 0.5
        ).normalize();
        
        // Update lighting in scene
        this.applyEstimatedLighting();
    }

    applyEstimatedLighting() {
        // Update ambient lighting
        const ambientLight = webXRManager.scene.getObjectByName('ambient-light');
        if (ambientLight) {
            ambientLight.intensity = this.lightingEstimation.ambientIntensity * 0.4;
        }
        
        // Update directional lighting
        const directionalLight = webXRManager.scene.getObjectByName('directional-light');
        if (directionalLight) {
            directionalLight.position.copy(this.lightingEstimation.mainLightDirection).multiplyScalar(-10);
            directionalLight.intensity = this.lightingEstimation.mainLightIntensity * 0.8;
        }
    }

    initOcclusionHandling() {
        console.log('🫥 Initializing occlusion handling...');
        
        this.occlusionMesh = {
            enabled: true,
            meshes: new Map(),
            depthTexture: null,
            occlusionShader: null
        };
        
        this.createOcclusionShader();
    }

    createOcclusionShader() {
        const occlusionVertexShader = `
            varying vec3 vWorldPosition;
            
            void main() {
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        const occlusionFragmentShader = `
            uniform sampler2D depthTexture;
            uniform mat4 depthProjectionMatrix;
            uniform mat4 depthViewMatrix;
            uniform float depthScale;
            varying vec3 vWorldPosition;
            
            void main() {
                // Convert world position to depth texture coordinates
                vec4 depthCoord = depthProjectionMatrix * depthViewMatrix * vec4(vWorldPosition, 1.0);
                depthCoord.xyz /= depthCoord.w;
                depthCoord.xyz = depthCoord.xyz * 0.5 + 0.5;
                
                // Sample depth texture
                float realDepth = texture2D(depthTexture, depthCoord.xy).r * depthScale;
                float virtualDepth = length(vWorldPosition - cameraPosition);
                
                // Occlude if virtual object is behind real object
                if (virtualDepth > realDepth + 0.01) {
                    discard;
                }
                
                gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
            }
        `;
        
        this.occlusionMesh.occlusionShader = new THREE.ShaderMaterial({
            uniforms: {
                depthTexture: { value: null },
                depthProjectionMatrix: { value: new THREE.Matrix4() },
                depthViewMatrix: { value: new THREE.Matrix4() },
                depthScale: { value: 10.0 }
            },
            vertexShader: occlusionVertexShader,
            fragmentShader: occlusionFragmentShader,
            transparent: true
        });
    }

    initCommunicationSystem() {
        console.log('💬 Initializing communication system...');
        
        this.communicationSystem = {
            voiceChat: null,
            textChat: null,
            spatialAudio: null,
            gestureSync: null,
            dataChannel: null,
            isConnected: false
        };
        
        this.initVoiceChat();
        this.initSpatialAudio();
        this.initGestureSync();
    }

    initVoiceChat() {
        this.communicationSystem.voiceChat = {
            enabled: false,
            spatialAudio: true,
            noiseSupression: true,
            echoCancellation: true,
            participants: new Map()
        };
    }

    initSpatialAudio() {
        this.communicationSystem.spatialAudio = {
            listener: new THREE.AudioListener(),
            sources: new Map(),
            roomImpulseResponse: null,
            spatializer: null
        };
        
        webXRManager.camera.add(this.communicationSystem.spatialAudio.listener);
    }

    initGestureSync() {
        this.communicationSystem.gestureSync = {
            enabled: true,
            syncFrequency: 30, // Hz
            gestureHistory: new Map(),
            compressionEnabled: true
        };
    }

    initSynchronizationEngine() {
        console.log('🔄 Initializing synchronization engine...');
        
        this.synchronizationEngine = {
            clockSync: null,
            stateSync: null,
            conflictResolution: null,
            networkPrediction: null,
            rollbackSystem: null
        };
        
        this.initClockSynchronization();
        this.initStateSynchronization();
        this.initConflictResolution();
    }

    initClockSynchronization() {
        this.synchronizationEngine.clockSync = {
            serverTime: 0,
            localTimeOffset: 0,
            syncAccuracy: 0,
            syncInterval: 10000 // 10 seconds
        };
        
        this.startClockSync();
    }

    startClockSync() {
        const syncClock = () => {
            // Mock network time synchronization
            const networkLatency = 50 + Math.random() * 100; // ms
            this.synchronizationEngine.clockSync.serverTime = Date.now() + networkLatency;
            this.synchronizationEngine.clockSync.localTimeOffset = networkLatency / 2;
            
            setTimeout(syncClock, this.synchronizationEngine.clockSync.syncInterval);
        };
        
        syncClock();
    }

    initStateSynchronization() {
        this.synchronizationEngine.stateSync = {
            updateRate: 20, // Hz
            interpolation: true,
            extrapolation: true,
            compressionEnabled: true,
            deltaCompression: true
        };
    }

    initConflictResolution() {
        this.synchronizationEngine.conflictResolution = {
            strategy: 'timestamp_priority', // timestamp_priority, host_authority, consensus
            conflictHistory: new Map(),
            rollbackBuffer: new Map()
        };
    }

    initPersistenceManager() {
        console.log('💾 Initializing persistence manager...');
        
        this.persistenceManager = {
            localStorage: new Map(),
            cloudStorage: null,
            cachingStrategy: 'lru', // lru, fifo, priority
            maxCacheSize: 100 * 1024 * 1024, // 100MB
            autoSave: true,
            saveInterval: 30000 // 30 seconds
        };
        
        this.startAutoSave();
    }

    startAutoSave() {
        if (!this.persistenceManager.autoSave) return;
        
        const autoSave = () => {
            this.saveCollaborativeState();
            setTimeout(autoSave, this.persistenceManager.saveInterval);
        };
        
        autoSave();
    }

    saveCollaborativeState() {
        const state = {
            spaces: Array.from(this.collaborativeSpaces.entries()),
            participants: Array.from(this.participants.entries()),
            sharedObjects: Array.from(this.sharedObjects.entries()),
            spatialAnchors: Array.from(this.spatialAnchors.entries()),
            timestamp: Date.now()
        };
        
        this.persistenceManager.localStorage.set('collaborative_state', state);
        console.log('💾 Collaborative state saved');
    }

    initCoPresenceSystem() {
        console.log('👥 Initializing co-presence system...');
        
        this.coPresence = {
            avatars: new Map(),
            proximityDetection: true,
            attentionIndicators: true,
            personalSpace: 0.5, // meters
            interactionZones: new Map()
        };
        
        this.initAvatarSystem();
        this.initProximityDetection();
        this.initAttentionTracking();
    }

    initAvatarSystem() {
        this.coPresence.avatarSystem = {
            renderMode: 'realistic', // realistic, stylized, minimal
            animationSystem: null,
            expressionSystem: null,
            clothingSystem: null
        };
    }

    initProximityDetection() {
        const detectProximity = () => {
            this.participants.forEach((participant, id) => {
                const distance = this.calculateDistanceToUser(participant.position);
                participant.proximity = distance;
                
                // Update interaction availability
                participant.canInteract = distance < 2.0; // 2 meter interaction range
            });
            
            setTimeout(detectProximity, 100); // 10 Hz
        };
        
        detectProximity();
    }

    calculateDistanceToUser(position) {
        const userPosition = webXRManager.camera.position;
        return userPosition.distanceTo(position);
    }

    initAttentionTracking() {
        this.coPresence.attentionTracking = {
            enabled: true,
            gazeVisualization: true,
            focusIndicators: true,
            sharedAttention: new Map()
        };
    }

    initSharedWorkspaces() {
        console.log('🏢 Initializing shared workspaces...');
        
        // Create default workspace types
        this.createWorkspaceTemplate('meeting_room', {
            size: { width: 4, height: 3, depth: 4 },
            layout: 'circular_seating',
            features: ['whiteboard', 'screen_sharing', 'document_viewer'],
            maxParticipants: 8
        });
        
        this.createWorkspaceTemplate('design_studio', {
            size: { width: 6, height: 4, depth: 6 },
            layout: 'open_space',
            features: ['3d_modeling', 'material_library', 'collaboration_tools'],
            maxParticipants: 4
        });
        
        this.createWorkspaceTemplate('presentation_hall', {
            size: { width: 8, height: 5, depth: 6 },
            layout: 'auditorium',
            features: ['large_screen', 'podium', 'audience_seating'],
            maxParticipants: 20
        });
    }

    createWorkspaceTemplate(name, config) {
        const template = {
            name: name,
            config: config,
            id: `template_${name}`,
            createInstance: (position, options = {}) => {
                return this.createWorkspaceInstance(template, position, options);
            }
        };
        
        this.sharedWorkspaces.set(name, template);
        console.log(`🏗️ Created workspace template: ${name}`);
    }

    createWorkspaceInstance(template, position, options = {}) {
        const workspaceId = `workspace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const workspace = {
            id: workspaceId,
            template: template,
            position: position.clone(),
            participants: new Map(),
            sharedObjects: new Map(),
            spatialAnchors: new Map(),
            created: Date.now(),
            options: options
        };
        
        // Create workspace geometry
        this.createWorkspaceGeometry(workspace);
        
        // Add workspace features
        this.addWorkspaceFeatures(workspace);
        
        this.collaborativeSpaces.set(workspaceId, workspace);
        
        console.log(`🏢 Created workspace instance: ${workspaceId}`);
        return workspaceId;
    }

    createWorkspaceGeometry(workspace) {
        const config = workspace.template.config;
        const workspaceGroup = new THREE.Group();
        workspaceGroup.name = `workspace-${workspace.id}`;
        
        // Create floor
        const floorGeometry = new THREE.PlaneGeometry(config.size.width, config.size.depth);
        const floorMaterial = new THREE.MeshLambertMaterial({
            color: 0xf0f0f0,
            transparent: true,
            opacity: 0.8
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.01;
        workspaceGroup.add(floor);
        
        // Create boundary walls (optional)
        if (config.layout !== 'open_space') {
            this.createWorkspaceBoundaries(workspaceGroup, config);
        }
        
        // Position workspace
        workspaceGroup.position.copy(workspace.position);
        
        webXRManager.scene.add(workspaceGroup);
        workspace.sceneObject = workspaceGroup;
    }

    createWorkspaceBoundaries(group, config) {
        const wallHeight = config.size.height;
        const wallMaterial = new THREE.MeshLambertMaterial({
            color: 0xe0e0e0,
            transparent: true,
            opacity: 0.6
        });
        
        // Front and back walls
        [config.size.depth / 2, -config.size.depth / 2].forEach((z, index) => {
            const wallGeometry = new THREE.PlaneGeometry(config.size.width, wallHeight);
            const wall = new THREE.Mesh(wallGeometry, wallMaterial);
            wall.position.set(0, wallHeight / 2, z);
            if (index === 1) wall.rotation.y = Math.PI;
            group.add(wall);
        });
        
        // Left and right walls
        [-config.size.width / 2, config.size.width / 2].forEach((x, index) => {
            const wallGeometry = new THREE.PlaneGeometry(config.size.depth, wallHeight);
            const wall = new THREE.Mesh(wallGeometry, wallMaterial);
            wall.position.set(x, wallHeight / 2, 0);
            wall.rotation.y = index === 0 ? Math.PI / 2 : -Math.PI / 2;
            group.add(wall);
        });
    }

    addWorkspaceFeatures(workspace) {
        const features = workspace.template.config.features;
        
        features.forEach(feature => {
            switch (feature) {
                case 'whiteboard':
                    this.addWhiteboard(workspace);
                    break;
                case 'screen_sharing':
                    this.addScreenSharing(workspace);
                    break;
                case 'document_viewer':
                    this.addDocumentViewer(workspace);
                    break;
                case '3d_modeling':
                    this.add3DModelingTools(workspace);
                    break;
                case 'material_library':
                    this.addMaterialLibrary(workspace);
                    break;
                case 'collaboration_tools':
                    this.addCollaborationTools(workspace);
                    break;
                case 'large_screen':
                    this.addLargeScreen(workspace);
                    break;
                case 'podium':
                    this.addPodium(workspace);
                    break;
                case 'audience_seating':
                    this.addAudienceSeating(workspace);
                    break;
            }
        });
    }

    addWhiteboard(workspace) {
        const whiteboardGeometry = new THREE.PlaneGeometry(2, 1.2);
        const whiteboardMaterial = new THREE.MeshLambertMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.9
        });
        
        const whiteboard = new THREE.Mesh(whiteboardGeometry, whiteboardMaterial);
        whiteboard.position.set(0, 1.2, -workspace.template.config.size.depth / 2 + 0.1);
        whiteboard.name = 'whiteboard';
        
        // Add drawing functionality
        whiteboard.userData = {
            isInteractive: true,
            type: 'whiteboard',
            drawingCanvas: this.createDrawingCanvas(),
            strokes: []
        };
        
        workspace.sceneObject.add(whiteboard);
    }

    createDrawingCanvas() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        
        const context = canvas.getContext('2d');
        context.fillStyle = '#ffffff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        
        return { canvas, context, texture };
    }

    addScreenSharing(workspace) {
        const screenGeometry = new THREE.PlaneGeometry(1.6, 0.9);
        const screenMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0.9
        });
        
        const screen = new THREE.Mesh(screenGeometry, screenMaterial);
        screen.position.set(1.5, 1, -workspace.template.config.size.depth / 2 + 0.1);
        screen.name = 'shared-screen';
        
        screen.userData = {
            isInteractive: true,
            type: 'screen_sharing',
            isSharing: false,
            streamSource: null
        };
        
        workspace.sceneObject.add(screen);
    }

    addDocumentViewer(workspace) {
        const viewerGeometry = new THREE.PlaneGeometry(1.2, 1.6);
        const viewerMaterial = new THREE.MeshLambertMaterial({
            color: 0xf8f8f8,
            transparent: true,
            opacity: 0.9
        });
        
        const viewer = new THREE.Mesh(viewerGeometry, viewerMaterial);
        viewer.position.set(-1.5, 1, -workspace.template.config.size.depth / 2 + 0.1);
        viewer.name = 'document-viewer';
        
        viewer.userData = {
            isInteractive: true,
            type: 'document_viewer',
            currentDocument: null,
            pages: []
        };
        
        workspace.sceneObject.add(viewer);
    }

    add3DModelingTools(workspace) {
        // Add 3D modeling toolkit
        const toolkitPosition = new THREE.Vector3(-2, 0.8, 0);
        const toolkit = this.create3DModelingToolkit(toolkitPosition);
        workspace.sceneObject.add(toolkit);
    }

    create3DModelingToolkit(position) {
        const toolkitGroup = new THREE.Group();
        toolkitGroup.position.copy(position);
        toolkitGroup.name = '3d-modeling-toolkit';
        
        // Tool palette
        const paletteGeometry = new THREE.BoxGeometry(0.5, 0.8, 0.1);
        const paletteMaterial = new THREE.MeshLambertMaterial({ color: 0x404040 });
        const palette = new THREE.Mesh(paletteGeometry, paletteMaterial);
        toolkitGroup.add(palette);
        
        // Add modeling tools
        const tools = ['extrude', 'cut', 'smooth', 'paint'];
        tools.forEach((tool, index) => {
            const toolButton = this.createToolButton(tool, index);
            toolButton.position.set(0, 0.3 - index * 0.15, 0.06);
            toolkitGroup.add(toolButton);
        });
        
        return toolkitGroup;
    }

    createToolButton(toolName, index) {
        const buttonGeometry = new THREE.BoxGeometry(0.12, 0.12, 0.02);
        const buttonMaterial = new THREE.MeshLambertMaterial({
            color: 0x4a90e2,
            transparent: true,
            opacity: 0.8
        });
        
        const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
        button.name = `tool-${toolName}`;
        
        button.userData = {
            isInteractive: true,
            type: 'tool_button',
            tool: toolName,
            onSelect: () => this.selectModelingTool(toolName)
        };
        
        return button;
    }

    selectModelingTool(toolName) {
        console.log(`🔧 Selected modeling tool: ${toolName}`);
        // Implement tool selection logic
        webXRManager.pulseHaptic(0.5, 100);
    }

    addMaterialLibrary(workspace) {
        // Add material library panel
        const libraryPosition = new THREE.Vector3(2, 0.8, 0);
        const library = this.createMaterialLibrary(libraryPosition);
        workspace.sceneObject.add(library);
    }

    createMaterialLibrary(position) {
        const libraryGroup = new THREE.Group();
        libraryGroup.position.copy(position);
        libraryGroup.name = 'material-library';
        
        // Library panel
        const panelGeometry = new THREE.PlaneGeometry(1, 1.2);
        const panelMaterial = new THREE.MeshLambertMaterial({ color: 0x303030 });
        const panel = new THREE.Mesh(panelGeometry, panelMaterial);
        libraryGroup.add(panel);
        
        // Material samples
        const materials = ['metal', 'wood', 'plastic', 'glass', 'fabric', 'stone'];
        materials.forEach((material, index) => {
            const sample = this.createMaterialSample(material, index);
            sample.position.set(
                -0.3 + (index % 3) * 0.3,
                0.4 - Math.floor(index / 3) * 0.3,
                0.01
            );
            libraryGroup.add(sample);
        });
        
        return libraryGroup;
    }

    createMaterialSample(materialName, index) {
        const sampleGeometry = new THREE.SphereGeometry(0.08, 16, 16);
        const sampleMaterial = this.getMaterialPreset(materialName);
        
        const sample = new THREE.Mesh(sampleGeometry, sampleMaterial);
        sample.name = `material-${materialName}`;
        
        sample.userData = {
            isInteractive: true,
            type: 'material_sample',
            material: materialName,
            onSelect: () => this.selectMaterial(materialName)
        };
        
        return sample;
    }

    getMaterialPreset(materialName) {
        const presets = {
            metal: new THREE.MeshStandardMaterial({
                color: 0x808080,
                metalness: 0.9,
                roughness: 0.1
            }),
            wood: new THREE.MeshLambertMaterial({
                color: 0x8B4513
            }),
            plastic: new THREE.MeshPhongMaterial({
                color: 0xff4444,
                shininess: 100
            }),
            glass: new THREE.MeshPhysicalMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.3,
                transmission: 0.9
            }),
            fabric: new THREE.MeshLambertMaterial({
                color: 0x4169E1
            }),
            stone: new THREE.MeshLambertMaterial({
                color: 0x696969
            })
        };
        
        return presets[materialName] || new THREE.MeshLambertMaterial({ color: 0x808080 });
    }

    selectMaterial(materialName) {
        console.log(`🎨 Selected material: ${materialName}`);
        // Implement material selection logic
        webXRManager.pulseHaptic(0.4, 80);
    }

    addCollaborationTools(workspace) {
        // Add collaboration toolbar
        const toolbarPosition = new THREE.Vector3(0, 2.5, 0);
        const toolbar = this.createCollaborationToolbar(toolbarPosition);
        workspace.sceneObject.add(toolbar);
    }

    createCollaborationToolbar(position) {
        const toolbarGroup = new THREE.Group();
        toolbarGroup.position.copy(position);
        toolbarGroup.name = 'collaboration-toolbar';
        
        // Toolbar background
        const bgGeometry = new THREE.PlaneGeometry(3, 0.3);
        const bgMaterial = new THREE.MeshLambertMaterial({
            color: 0x2c2c2c,
            transparent: true,
            opacity: 0.8
        });
        const background = new THREE.Mesh(bgGeometry, bgMaterial);
        toolbarGroup.add(background);
        
        // Collaboration tools
        const tools = ['voice_chat', 'screen_share', 'annotation', 'pointer'];
        tools.forEach((tool, index) => {
            const toolButton = this.createCollaborationButton(tool);
            toolButton.position.set(-1.2 + index * 0.8, 0, 0.01);
            toolbarGroup.add(toolButton);
        });
        
        return toolbarGroup;
    }

    createCollaborationButton(toolName) {
        const buttonGeometry = new THREE.CircleGeometry(0.1, 16);
        const buttonMaterial = new THREE.MeshLambertMaterial({
            color: 0x4a90e2,
            transparent: true,
            opacity: 0.9
        });
        
        const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
        button.name = `collab-${toolName}`;
        
        button.userData = {
            isInteractive: true,
            type: 'collaboration_button',
            tool: toolName,
            onSelect: () => this.activateCollaborationTool(toolName)
        };
        
        return button;
    }

    activateCollaborationTool(toolName) {
        console.log(`🤝 Activated collaboration tool: ${toolName}`);
        
        switch (toolName) {
            case 'voice_chat':
                this.toggleVoiceChat();
                break;
            case 'screen_share':
                this.startScreenShare();
                break;
            case 'annotation':
                this.activateAnnotationMode();
                break;
            case 'pointer':
                this.activatePointerMode();
                break;
        }
        
        webXRManager.pulseHaptic(0.6, 120);
    }

    addLargeScreen(workspace) {
        const screenGeometry = new THREE.PlaneGeometry(4, 2.25);
        const screenMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0.95
        });
        
        const screen = new THREE.Mesh(screenGeometry, screenMaterial);
        screen.position.set(0, 2, -workspace.template.config.size.depth / 2 + 0.1);
        screen.name = 'large-presentation-screen';
        
        screen.userData = {
            isInteractive: true,
            type: 'presentation_screen',
            currentSlide: 0,
            slides: []
        };
        
        workspace.sceneObject.add(screen);
    }

    addPodium(workspace) {
        const podiumGeometry = new THREE.BoxGeometry(1, 1.2, 0.6);
        const podiumMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
        
        const podium = new THREE.Mesh(podiumGeometry, podiumMaterial);
        podium.position.set(0, 0.6, workspace.template.config.size.depth / 4);
        podium.name = 'podium';
        
        workspace.sceneObject.add(podium);
    }

    addAudienceSeating(workspace) {
        const rows = 4;
        const seatsPerRow = 5;
        const seatSpacing = 0.8;
        const rowSpacing = 1.2;
        
        for (let row = 0; row < rows; row++) {
            for (let seat = 0; seat < seatsPerRow; seat++) {
                const chair = this.createAudienceChair();
                chair.position.set(
                    (seat - (seatsPerRow - 1) / 2) * seatSpacing,
                    0,
                    -workspace.template.config.size.depth / 4 + row * rowSpacing
                );
                workspace.sceneObject.add(chair);
            }
        }
    }

    createAudienceChair() {
        const chairGroup = new THREE.Group();
        
        // Seat
        const seatGeometry = new THREE.BoxGeometry(0.5, 0.05, 0.5);
        const seatMaterial = new THREE.MeshLambertMaterial({ color: 0x654321 });
        const seat = new THREE.Mesh(seatGeometry, seatMaterial);
        seat.position.y = 0.45;
        chairGroup.add(seat);
        
        // Backrest
        const backGeometry = new THREE.BoxGeometry(0.5, 0.6, 0.05);
        const back = new THREE.Mesh(backGeometry, seatMaterial);
        back.position.set(0, 0.75, -0.22);
        chairGroup.add(back);
        
        // Legs
        for (let i = 0; i < 4; i++) {
            const legGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.45);
            const leg = new THREE.Mesh(legGeometry, seatMaterial);
            const x = i % 2 === 0 ? -0.2 : 0.2;
            const z = i < 2 ? -0.2 : 0.2;
            leg.position.set(x, 0.225, z);
            chairGroup.add(leg);
        }
        
        return chairGroup;
    }

    initSpatialTools() {
        console.log('🛠️ Initializing spatial tools...');
        
        this.spatialTools = {
            spatialClipboard: new Map(),
            measurementTool: null,
            annotationTool: null,
            selectionTool: null,
            transformTool: null
        };
        
        this.initSpatialClipboard();
        this.initMeasurementTool();
        this.initAnnotationTool();
    }

    initSpatialClipboard() {
        this.spatialClipboard = {
            copiedObjects: new Map(),
            clipboardVisualizer: null,
            maxItems: 10,
            autoExpire: true,
            expireTime: 300000 // 5 minutes
        };
    }

    initMeasurementTool() {
        this.spatialTools.measurementTool = {
            isActive: false,
            measurementPoints: [],
            activeMeasurement: null,
            units: 'meters', // meters, centimeters, inches, feet
            precision: 2
        };
    }

    initAnnotationTool() {
        this.spatialTools.annotationTool = {
            isActive: false,
            annotations: new Map(),
            currentAnnotation: null,
            annotationTypes: ['text', 'voice', 'drawing', '3d_model']
        };
    }

    initCollaborativeGestures() {
        console.log('🤲 Initializing collaborative gestures...');
        
        this.collaborativeGestures = {
            sharedGestures: new Map(),
            gestureSync: true,
            gestureRecording: false,
            customGestures: new Map(),
            gestureLibrary: new Map()
        };
        
        this.defineCollaborativeGestures();
    }

    defineCollaborativeGestures() {
        const gestures = {
            point_to_share: {
                description: 'Point to share focus with others',
                pattern: 'index_extended',
                callback: (participant, target) => this.shareAttentionFocus(participant, target)
            },
            gather_around: {
                description: 'Gesture to gather participants around object',
                pattern: 'both_hands_beckoning',
                callback: (participant, location) => this.gatherParticipants(participant, location)
            },
            present_object: {
                description: 'Present object to group',
                pattern: 'open_palms_forward',
                callback: (participant, object) => this.presentObjectToGroup(participant, object)
            },
            request_attention: {
                description: 'Request attention from group',
                pattern: 'raise_hand',
                callback: (participant) => this.requestGroupAttention(participant)
            }
        };
        
        Object.entries(gestures).forEach(([name, gesture]) => {
            this.collaborativeGestures.gestureLibrary.set(name, gesture);
        });
    }

    // Public API Methods

    async createCollaborativeSpace(spaceType, position, options = {}) {
        try {
            const spaceId = this.createWorkspaceInstance(
                this.sharedWorkspaces.get(spaceType),
                position,
                options
            );
            
            console.log(`🌍 Created collaborative space: ${spaceId}`);
            return spaceId;
        } catch (error) {
            console.error('Failed to create collaborative space:', error);
            throw error;
        }
    }

    async joinCollaborativeSpace(spaceId, participantInfo) {
        const space = this.collaborativeSpaces.get(spaceId);
        if (!space) {
            throw new Error(`Collaborative space not found: ${spaceId}`);
        }
        
        const participantId = participantInfo.id || `participant_${Date.now()}`;
        
        const participant = {
            id: participantId,
            info: participantInfo,
            position: new THREE.Vector3(),
            orientation: new THREE.Quaternion(),
            avatar: null,
            joinTime: Date.now(),
            isActive: true
        };
        
        // Create participant avatar
        participant.avatar = this.createParticipantAvatar(participantInfo);
        
        // Add to space and global participants
        space.participants.set(participantId, participant);
        this.participants.set(participantId, participant);
        
        // Sync with other participants
        this.broadcastParticipantJoined(spaceId, participant);
        
        console.log(`👤 Participant joined space ${spaceId}: ${participantId}`);
        return participantId;
    }

    createParticipantAvatar(participantInfo) {
        const avatarGroup = new THREE.Group();
        avatarGroup.name = `avatar-${participantInfo.id}`;
        
        // Avatar body (simplified)
        const bodyGeometry = new THREE.CapsuleGeometry(0.2, 1.4);
        const bodyMaterial = new THREE.MeshLambertMaterial({
            color: participantInfo.avatarColor || 0x4a90e2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        avatarGroup.add(body);
        
        // Avatar head indicator
        const headGeometry = new THREE.SphereGeometry(0.15, 16, 16);
        const headMaterial = new THREE.MeshLambertMaterial({
            color: participantInfo.avatarColor || 0x4a90e2
        });
        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.position.y = 0.85;
        avatarGroup.add(head);
        
        // Name tag
        const nameTag = this.createNameTag(participantInfo.name || 'Anonymous');
        nameTag.position.y = 1.2;
        avatarGroup.add(nameTag);
        
        webXRManager.scene.add(avatarGroup);
        
        return avatarGroup;
    }

    createNameTag(name) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 64;
        
        const context = canvas.getContext('2d');
        context.fillStyle = '#000000';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = '#ffffff';
        context.font = '24px Arial';
        context.textAlign = 'center';
        context.fillText(name, canvas.width / 2, canvas.height / 2 + 8);
        
        const texture = new THREE.CanvasTexture(canvas);
        const geometry = new THREE.PlaneGeometry(0.5, 0.125);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true
        });
        
        const nameTag = new THREE.Mesh(geometry, material);
        nameTag.name = 'name-tag';
        
        return nameTag;
    }

    addSharedObject(spaceId, object, options = {}) {
        const space = this.collaborativeSpaces.get(spaceId);
        if (!space) {
            throw new Error(`Collaborative space not found: ${spaceId}`);
        }
        
        const objectId = options.id || `shared_object_${Date.now()}`;
        
        const sharedObject = {
            id: objectId,
            object: object,
            owner: options.owner || null,
            permissions: options.permissions || 'read_write',
            synchronized: true,
            lastModified: Date.now(),
            version: 1
        };
        
        // Add synchronization behavior
        this.addObjectSynchronization(sharedObject);
        
        space.sharedObjects.set(objectId, sharedObject);
        this.sharedObjects.set(objectId, sharedObject);
        
        // Broadcast to other participants
        this.broadcastObjectAdded(spaceId, sharedObject);
        
        console.log(`📦 Added shared object to space ${spaceId}: ${objectId}`);
        return objectId;
    }

    addObjectSynchronization(sharedObject) {
        const object = sharedObject.object;
        
        // Track position changes
        object.userData.previousPosition = object.position.clone();
        object.userData.previousRotation = object.rotation.clone();
        object.userData.previousScale = object.scale.clone();
        
        // Add change detection
        object.userData.onChange = () => {
            this.onSharedObjectChanged(sharedObject);
        };
    }

    onSharedObjectChanged(sharedObject) {
        sharedObject.lastModified = Date.now();
        sharedObject.version++;
        
        // Broadcast changes to other participants
        this.broadcastObjectChanged(sharedObject);
    }

    createSpatialAnchor(position, orientation, options = {}) {
        const anchorId = options.id || `anchor_${Date.now()}`;
        
        const anchor = {
            id: anchorId,
            position: position.clone(),
            orientation: orientation.clone(),
            persistent: options.persistent || false,
            cloudSync: options.cloudSync || false,
            created: Date.now(),
            confidence: options.confidence || 1.0
        };
        
        this.spatialAnchors.set(anchorId, anchor);
        
        // Create anchor visualization
        this.createAnchorVisualization(anchor);
        
        // Save to persistence if enabled
        if (anchor.persistent) {
            this.persistenceManager.localStorage.set(`anchor_${anchorId}`, anchor);
        }
        
        console.log(`⚓ Created spatial anchor: ${anchorId}`);
        return anchorId;
    }

    createAnchorVisualization(anchor) {
        const anchorGroup = new THREE.Group();
        anchorGroup.name = `anchor-${anchor.id}`;
        
        // Anchor indicator
        const indicatorGeometry = new THREE.OctahedronGeometry(0.05);
        const indicatorMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.8
        });
        const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
        anchorGroup.add(indicator);
        
        // Coordinate axes
        const axesHelper = new THREE.AxesHelper(0.1);
        anchorGroup.add(axesHelper);
        
        anchorGroup.position.copy(anchor.position);
        anchorGroup.quaternion.copy(anchor.orientation);
        
        webXRManager.scene.add(anchorGroup);
        anchor.visualization = anchorGroup;
    }

    enablePassthrough(enabled = true) {
        this.passthroughRenderer.isEnabled = enabled;
        
        if (enabled) {
            console.log('📹 Passthrough rendering enabled');
            this.startPassthroughRendering();
        } else {
            console.log('📹 Passthrough rendering disabled');
            this.stopPassthroughRendering();
        }
    }

    startPassthroughRendering() {
        // Start passthrough camera feeds
        console.log('🎥 Starting passthrough camera feeds...');
        
        // In a real implementation, this would interface with the device's cameras
        // For now, we'll simulate the passthrough effect
        this.simulatePassthroughEffect();
    }

    simulatePassthroughEffect() {
        // Create a subtle background that represents the real world
        const passthroughGeometry = new THREE.SphereGeometry(50, 32, 32);
        const passthroughMaterial = new THREE.MeshBasicMaterial({
            color: 0x404040,
            transparent: true,
            opacity: 0.1,
            side: THREE.BackSide
        });
        
        const passthroughSphere = new THREE.Mesh(passthroughGeometry, passthroughMaterial);
        passthroughSphere.name = 'passthrough-background';
        
        webXRManager.scene.add(passthroughSphere);
    }

    stopPassthroughRendering() {
        const passthroughBackground = webXRManager.scene.getObjectByName('passthrough-background');
        if (passthroughBackground) {
            webXRManager.scene.remove(passthroughBackground);
        }
    }

    // Collaboration Methods
    toggleVoiceChat() {
        const voiceChat = this.communicationSystem.voiceChat;
        voiceChat.enabled = !voiceChat.enabled;
        
        console.log(`🎤 Voice chat ${voiceChat.enabled ? 'enabled' : 'disabled'}`);
        
        if (voiceChat.enabled) {
            this.startVoiceChat();
        } else {
            this.stopVoiceChat();
        }
    }

    startVoiceChat() {
        // Initialize voice chat system
        console.log('🎤 Starting voice chat...');
        
        // In a real implementation, this would use WebRTC
        this.communicationSystem.voiceChat.isActive = true;
    }

    stopVoiceChat() {
        console.log('🎤 Stopping voice chat...');
        this.communicationSystem.voiceChat.isActive = false;
    }

    startScreenShare() {
        console.log('🖥️ Starting screen share...');
        // Implement screen sharing functionality
    }

    activateAnnotationMode() {
        this.spatialTools.annotationTool.isActive = true;
        console.log('✏️ Annotation mode activated');
    }

    activatePointerMode() {
        console.log('👆 Pointer mode activated');
        // Implement 3D pointer functionality
    }

    shareAttentionFocus(participant, target) {
        console.log(`👁️ ${participant.id} shared attention focus on:`, target);
        
        // Create attention indicator
        this.createAttentionIndicator(participant, target);
        
        // Broadcast to other participants
        this.broadcastAttentionShare(participant, target);
    }

    createAttentionIndicator(participant, target) {
        const indicatorGeometry = new THREE.RingGeometry(0.1, 0.15, 16);
        const indicatorMaterial = new THREE.MeshBasicMaterial({
            color: participant.info.avatarColor || 0x4a90e2,
            transparent: true,
            opacity: 0.8
        });
        
        const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
        indicator.position.copy(target);
        indicator.name = `attention-${participant.id}`;
        
        webXRManager.scene.add(indicator);
        
        // Remove after 3 seconds
        setTimeout(() => {
            webXRManager.scene.remove(indicator);
        }, 3000);
    }

    gatherParticipants(participant, location) {
        console.log(`📍 ${participant.id} requested gather at:`, location);
        
        // Create gather point visualization
        this.createGatherPoint(location);
        
        // Broadcast gather request
        this.broadcastGatherRequest(participant, location);
    }

    createGatherPoint(location) {
        const gatherGeometry = new THREE.CylinderGeometry(1, 1, 0.1, 16);
        const gatherMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.3
        });
        
        const gatherPoint = new THREE.Mesh(gatherGeometry, gatherMaterial);
        gatherPoint.position.copy(location);
        gatherPoint.name = 'gather-point';
        
        webXRManager.scene.add(gatherPoint);
        
        // Animate gather point
        const animate = () => {
            gatherPoint.rotation.y += 0.02;
            gatherPoint.scale.setScalar(1 + Math.sin(Date.now() * 0.005) * 0.1);
            requestAnimationFrame(animate);
        };
        animate();
        
        // Remove after 10 seconds
        setTimeout(() => {
            webXRManager.scene.remove(gatherPoint);
        }, 10000);
    }

    presentObjectToGroup(participant, object) {
        console.log(`🎭 ${participant.id} presented object to group:`, object);
        
        // Highlight presented object
        this.highlightPresentedObject(object);
        
        // Broadcast presentation
        this.broadcastObjectPresentation(participant, object);
    }

    highlightPresentedObject(object) {
        // Add presentation highlight effect
        const highlightGeometry = object.geometry.clone();
        const highlightMaterial = new THREE.MeshBasicMaterial({
            color: 0xffff00,
            transparent: true,
            opacity: 0.3,
            wireframe: true
        });
        
        const highlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
        highlight.position.copy(object.position);
        highlight.rotation.copy(object.rotation);
        highlight.scale.copy(object.scale).multiplyScalar(1.1);
        highlight.name = 'presentation-highlight';
        
        webXRManager.scene.add(highlight);
        
        // Remove highlight after 5 seconds
        setTimeout(() => {
            webXRManager.scene.remove(highlight);
        }, 5000);
    }

    requestGroupAttention(participant) {
        console.log(`🙋 ${participant.id} requested group attention`);
        
        // Create attention request indicator
        this.createAttentionRequest(participant);
        
        // Broadcast attention request
        this.broadcastAttentionRequest(participant);
    }

    createAttentionRequest(participant) {
        if (!participant.avatar) return;
        
        const requestGeometry = new THREE.SphereGeometry(0.1, 16, 16);
        const requestMaterial = new THREE.MeshBasicMaterial({
            color: 0xff4444,
            transparent: true,
            opacity: 0.8
        });
        
        const request = new THREE.Mesh(requestGeometry, requestMaterial);
        request.position.set(0, 0.3, 0);
        request.name = 'attention-request';
        
        participant.avatar.add(request);
        
        // Animate attention request
        const animate = () => {
            request.scale.setScalar(1 + Math.sin(Date.now() * 0.01) * 0.3);
            if (request.parent) {
                requestAnimationFrame(animate);
            }
        };
        animate();
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (request.parent) {
                request.parent.remove(request);
            }
        }, 5000);
    }

    // Broadcasting methods (would use real networking in production)
    broadcastParticipantJoined(spaceId, participant) {
        console.log(`📡 Broadcasting participant joined: ${participant.id}`);
    }

    broadcastObjectAdded(spaceId, sharedObject) {
        console.log(`📡 Broadcasting object added: ${sharedObject.id}`);
    }

    broadcastObjectChanged(sharedObject) {
        console.log(`📡 Broadcasting object changed: ${sharedObject.id}`);
    }

    broadcastAttentionShare(participant, target) {
        console.log(`📡 Broadcasting attention share from: ${participant.id}`);
    }

    broadcastGatherRequest(participant, location) {
        console.log(`📡 Broadcasting gather request from: ${participant.id}`);
    }

    broadcastObjectPresentation(participant, object) {
        console.log(`📡 Broadcasting object presentation from: ${participant.id}`);
    }

    broadcastAttentionRequest(participant) {
        console.log(`📡 Broadcasting attention request from: ${participant.id}`);
    }

    // Update method to be called in the main render loop
    updateMixedReality(frame) {
        if (!webXRManager.xrSession) return;
        
        // Update real world mapping
        if (this.realWorldMapping) {
            // Real-time spatial mapping updates would go here
        }
        
        // Update lighting estimation
        if (this.lightingEstimation && this.lightingEstimation.enabled) {
            // Lighting updates are handled in startLightingEstimation
        }
        
        // Update occlusion
        if (this.occlusionMesh && this.occlusionMesh.enabled) {
            this.updateOcclusionMeshes(frame);
        }
        
        // Update participant positions
        this.updateParticipants();
        
        // Update shared objects
        this.updateSharedObjects();
    }

    updateOcclusionMeshes(frame) {
        // Update occlusion meshes based on real-world geometry
        // This would use depth data from the device in a real implementation
    }

    updateParticipants() {
        this.participants.forEach((participant, id) => {
            if (participant.avatar && participant.isActive) {
                // Update participant avatar position and orientation
                // In a real implementation, this would sync with networked data
            }
        });
    }

    updateSharedObjects() {
        this.sharedObjects.forEach((sharedObject, id) => {
            // Update shared object synchronization
            // Check for changes and broadcast updates
        });
    }

    // Cleanup methods
    leaveCollaborativeSpace(spaceId, participantId) {
        const space = this.collaborativeSpaces.get(spaceId);
        if (space && space.participants.has(participantId)) {
            const participant = space.participants.get(participantId);
            
            // Remove avatar
            if (participant.avatar) {
                webXRManager.scene.remove(participant.avatar);
            }
            
            // Remove from space and global participants
            space.participants.delete(participantId);
            this.participants.delete(participantId);
            
            console.log(`👋 Participant left space ${spaceId}: ${participantId}`);
        }
    }

    destroyCollaborativeSpace(spaceId) {
        const space = this.collaborativeSpaces.get(spaceId);
        if (space) {
            // Remove all participants
            space.participants.forEach((participant, id) => {
                this.leaveCollaborativeSpace(spaceId, id);
            });
            
            // Remove scene objects
            if (space.sceneObject) {
                webXRManager.scene.remove(space.sceneObject);
            }
            
            // Remove from storage
            this.collaborativeSpaces.delete(spaceId);
            
            console.log(`🗑️ Destroyed collaborative space: ${spaceId}`);
        }
    }

    getCollaborativeSpaces() {
        return Array.from(this.collaborativeSpaces.keys());
    }

    getParticipants(spaceId = null) {
        if (spaceId) {
            const space = this.collaborativeSpaces.get(spaceId);
            return space ? Array.from(space.participants.keys()) : [];
        }
        return Array.from(this.participants.keys());
    }

    getSharedObjects(spaceId = null) {
        if (spaceId) {
            const space = this.collaborativeSpaces.get(spaceId);
            return space ? Array.from(space.sharedObjects.keys()) : [];
        }
        return Array.from(this.sharedObjects.keys());
    }
}

// Export Mixed Reality Collaborative Spaces class
export default MixedRealityCollaborativeSpaces;