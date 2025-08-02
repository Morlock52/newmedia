/**
 * Real Mixed Reality Implementation
 * Supports Meta Quest 3 passthrough, plane detection, anchors, and collaborative spaces
 * Future-ready for Apple Vision Pro AR when available
 */

import * as THREE from 'three';
import { webXRManager } from '../core/webxr-manager-v2.js';

export class RealMixedReality {
    constructor() {
        // AR/MR features
        this.planes = new Map();
        this.meshes = new Map();
        this.anchors = new Map();
        this.lightEstimation = null;
        
        // Passthrough
        this.passthroughEnabled = false;
        this.passthroughLayer = null;
        
        // Collaborative features
        this.spaces = new Map();
        this.participants = new Map();
        this.sharedObjects = new Map();
        
        // Occlusion handling
        this.occlusionMaterials = new Map();
        this.depthSensing = null;
        
        // Visualization settings
        this.settings = {
            showPlanes: true,
            showMeshes: false,
            showAnchors: true,
            planeOpacity: 0.1,
            meshOpacity: 0.2,
            enableOcclusion: true,
            enableShadows: true
        };
        
        // Materials
        this.materials = {
            plane: new THREE.MeshBasicMaterial({
                color: 0x00ff00,
                transparent: true,
                opacity: 0.1,
                side: THREE.DoubleSide,
                wireframe: true
            }),
            mesh: new THREE.MeshPhongMaterial({
                color: 0x0088ff,
                transparent: true,
                opacity: 0.2,
                wireframe: true
            }),
            anchor: new THREE.MeshBasicMaterial({
                color: 0xff0088,
                emissive: 0xff0088,
                emissiveIntensity: 0.5
            }),
            occlusion: new THREE.MeshBasicMaterial({
                colorWrite: false,
                depthWrite: true,
                depthTest: true
            })
        };
        
        this.init();
    }

    init() {
        console.log('🌍 Initializing Real Mixed Reality System...');
        
        // Listen for WebXR events
        window.addEventListener('xr-session-started', (e) => this.onSessionStarted(e));
        window.addEventListener('xr-session-ended', () => this.onSessionEnded());
        
        // Setup AR features group
        this.arGroup = new THREE.Group();
        this.arGroup.name = 'AR-Features';
        webXRManager.scene.add(this.arGroup);
        
        console.log('✅ Mixed Reality System ready');
    }

    async onSessionStarted(event) {
        const { mode } = event.detail;
        
        if (mode !== 'immersive-ar') {
            console.log('💡 Not in AR mode, MR features limited');
            
            // Still enable passthrough for VR if available
            if (webXRManager.platform.isQuest) {
                await this.enablePassthrough();
            }
            return;
        }
        
        console.log('🎯 AR session started - enabling MR features');
        
        // Enable all AR features
        await this.enableARFeatures();
    }

    async enableARFeatures() {
        // Enable passthrough
        if (webXRManager.platform.isQuest) {
            await this.enablePassthrough();
        }
        
        // Setup plane detection
        if (webXRManager.features.planeDetection) {
            await this.setupPlaneDetection();
        }
        
        // Setup mesh detection
        if (webXRManager.features.meshDetection) {
            await this.setupMeshDetection();
        }
        
        // Setup light estimation
        if (webXRManager.features.lightEstimation) {
            await this.setupLightEstimation();
        }
        
        // Setup depth sensing
        if (webXRManager.features.depthSensing) {
            await this.setupDepthSensing();
        }
        
        // Start update loop
        this.startUpdateLoop();
    }

    async enablePassthrough() {
        if (!webXRManager.xrSession) return;
        
        try {
            // Check if passthrough is supported
            const supported = webXRManager.xrSession.environmentBlendMode === 'alpha-blend' ||
                            webXRManager.xrSession.environmentBlendMode === 'additive';
            
            if (!supported) {
                console.warn('Passthrough not supported in current blend mode');
                return;
            }
            
            // For Meta Quest, passthrough is enabled through the blend mode
            console.log('✅ Passthrough enabled');
            this.passthroughEnabled = true;
            
            // Create passthrough layer if layers are supported
            if (webXRManager.features.layers) {
                await this.createPassthroughLayer();
            }
            
        } catch (error) {
            console.error('Failed to enable passthrough:', error);
        }
    }

    async createPassthroughLayer() {
        // In WebXR, passthrough is typically handled by the UA
        // This is a placeholder for future layer-based passthrough
        console.log('📹 Passthrough layer support detected');
    }

    async setupPlaneDetection() {
        console.log('✈️ Setting up plane detection...');
        
        // Plane detection is done through hit testing and the planes API
        // This will be called in the update loop
        this.planeDetectionEnabled = true;
    }

    async setupMeshDetection() {
        console.log('🔷 Setting up mesh detection...');
        
        // Mesh detection provides detailed environment geometry
        this.meshDetectionEnabled = true;
    }

    async setupLightEstimation() {
        console.log('💡 Setting up light estimation...');
        
        // Light estimation provides environmental lighting information
        this.lightEstimationEnabled = true;
        
        // Create probe light
        this.probeLight = new THREE.DirectionalLight(0xffffff, 1);
        this.probeLight.castShadow = true;
        webXRManager.scene.add(this.probeLight);
    }

    async setupDepthSensing() {
        console.log('📏 Setting up depth sensing...');
        
        // Depth sensing provides per-pixel depth information
        this.depthSensingEnabled = true;
    }

    startUpdateLoop() {
        const update = (timestamp) => {
            if (!webXRManager.isSessionActive()) return;
            
            // Update AR features
            this.updateARFeatures();
            
            // Continue loop
            requestAnimationFrame(update);
        };
        
        requestAnimationFrame(update);
    }

    updateARFeatures() {
        const frame = webXRManager.xrFrame;
        if (!frame) return;
        
        // Update planes
        if (this.planeDetectionEnabled) {
            this.updatePlanes(frame);
        }
        
        // Update meshes
        if (this.meshDetectionEnabled) {
            this.updateMeshes(frame);
        }
        
        // Update light estimation
        if (this.lightEstimationEnabled) {
            this.updateLightEstimation(frame);
        }
        
        // Update anchors
        this.updateAnchors(frame);
    }

    updatePlanes(frame) {
        // Get detected planes
        const detectedPlanes = frame.detectedPlanes;
        if (!detectedPlanes) return;
        
        // Track which planes are still detected
        const currentPlaneIds = new Set();
        
        detectedPlanes.forEach(plane => {
            currentPlaneIds.add(plane.id);
            
            if (!this.planes.has(plane.id)) {
                // New plane detected
                this.onPlaneDetected(plane, frame);
            } else {
                // Update existing plane
                this.updatePlane(plane, frame);
            }
        });
        
        // Remove planes that are no longer detected
        this.planes.forEach((planeData, planeId) => {
            if (!currentPlaneIds.has(planeId)) {
                this.onPlaneLost(planeId);
            }
        });
    }

    onPlaneDetected(plane, frame) {
        console.log('✈️ New plane detected:', plane.orientation);
        
        // Get plane pose
        const pose = frame.getPose(plane.planeSpace, webXRManager.xrRefSpace);
        if (!pose) return;
        
        // Create plane visualization
        const planeGroup = new THREE.Group();
        planeGroup.name = `plane-${plane.id}`;
        
        // Create plane mesh
        const geometry = this.createPlaneGeometry(plane);
        const mesh = new THREE.Mesh(geometry, this.materials.plane.clone());
        mesh.visible = this.settings.showPlanes;
        planeGroup.add(mesh);
        
        // Create occlusion mesh
        if (this.settings.enableOcclusion) {
            const occlusionMesh = new THREE.Mesh(geometry, this.materials.occlusion);
            occlusionMesh.renderOrder = -1;
            planeGroup.add(occlusionMesh);
        }
        
        // Position plane
        planeGroup.position.fromArray(pose.transform.position);
        planeGroup.quaternion.fromArray(pose.transform.orientation);
        
        // Add to scene
        this.arGroup.add(planeGroup);
        
        // Store plane data
        this.planes.set(plane.id, {
            plane,
            group: planeGroup,
            mesh,
            polygon: plane.polygon,
            lastUpdated: Date.now()
        });
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('ar-plane-detected', {
            detail: { plane, pose }
        }));
    }

    updatePlane(plane, frame) {
        const planeData = this.planes.get(plane.id);
        if (!planeData) return;
        
        // Update pose
        const pose = frame.getPose(plane.planeSpace, webXRManager.xrRefSpace);
        if (pose) {
            planeData.group.position.fromArray(pose.transform.position);
            planeData.group.quaternion.fromArray(pose.transform.orientation);
        }
        
        // Update geometry if polygon changed
        if (plane.polygon.length !== planeData.polygon.length) {
            const geometry = this.createPlaneGeometry(plane);
            planeData.mesh.geometry.dispose();
            planeData.mesh.geometry = geometry;
            planeData.polygon = plane.polygon;
        }
        
        planeData.lastUpdated = Date.now();
    }

    onPlaneLost(planeId) {
        console.log('✈️ Plane lost:', planeId);
        
        const planeData = this.planes.get(planeId);
        if (!planeData) return;
        
        // Remove from scene
        this.arGroup.remove(planeData.group);
        
        // Clean up
        planeData.mesh.geometry.dispose();
        this.planes.delete(planeId);
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('ar-plane-lost', {
            detail: { planeId }
        }));
    }

    createPlaneGeometry(plane) {
        // Create geometry from polygon points
        const points = [];
        
        for (let i = 0; i < plane.polygon.length; i++) {
            points.push(new THREE.Vector2(
                plane.polygon[i].x,
                plane.polygon[i].z
            ));
        }
        
        // Create shape
        const shape = new THREE.Shape(points);
        const geometry = new THREE.ShapeGeometry(shape);
        
        // Rotate to match plane orientation
        if (plane.orientation === 'horizontal') {
            geometry.rotateX(-Math.PI / 2);
        } else if (plane.orientation === 'vertical') {
            // Already correct orientation
        }
        
        return geometry;
    }

    updateMeshes(frame) {
        // WebXR mesh detection provides detailed geometry
        const detectedMeshes = frame.detectedMeshes;
        if (!detectedMeshes) return;
        
        detectedMeshes.forEach(mesh => {
            if (!this.meshes.has(mesh.id)) {
                this.onMeshDetected(mesh, frame);
            } else {
                this.updateMesh(mesh, frame);
            }
        });
    }

    onMeshDetected(mesh, frame) {
        console.log('🔷 New mesh detected');
        
        // Create mesh visualization
        const geometry = new THREE.BufferGeometry();
        
        // Set vertices
        const vertices = new Float32Array(mesh.vertices);
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        // Set indices if available
        if (mesh.indices) {
            geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
        }
        
        // Calculate normals
        geometry.computeVertexNormals();
        
        // Create mesh
        const meshObject = new THREE.Mesh(geometry, this.materials.mesh.clone());
        meshObject.visible = this.settings.showMeshes;
        
        // Add to scene
        this.arGroup.add(meshObject);
        
        // Store mesh data
        this.meshes.set(mesh.id, {
            mesh,
            meshObject,
            lastUpdated: Date.now()
        });
    }

    updateMesh(mesh, frame) {
        const meshData = this.meshes.get(mesh.id);
        if (!meshData) return;
        
        // Update vertices if changed
        if (mesh.lastChangedTime > meshData.lastUpdated) {
            const vertices = new Float32Array(mesh.vertices);
            meshData.meshObject.geometry.setAttribute(
                'position',
                new THREE.BufferAttribute(vertices, 3)
            );
            meshData.meshObject.geometry.computeVertexNormals();
            meshData.lastUpdated = Date.now();
        }
    }

    updateLightEstimation(frame) {
        const lightEstimate = frame.getLightEstimate();
        if (!lightEstimate) return;
        
        // Update directional light
        if (lightEstimate.primaryLightDirection) {
            const direction = new THREE.Vector3().fromArray(lightEstimate.primaryLightDirection);
            this.probeLight.position.copy(direction.multiplyScalar(-10));
            this.probeLight.lookAt(0, 0, 0);
        }
        
        // Update light intensity
        if (lightEstimate.primaryLightIntensity) {
            const intensity = lightEstimate.primaryLightIntensity;
            this.probeLight.intensity = intensity.x; // Use red channel as overall intensity
            this.probeLight.color.setRGB(intensity.x, intensity.y, intensity.z);
        }
        
        // Update ambient light
        if (lightEstimate.sphericalHarmonicsCoefficients) {
            // This would require more complex implementation
            // For now, just update ambient based on first coefficient
            const sh = lightEstimate.sphericalHarmonicsCoefficients;
            const ambient = webXRManager.scene.children.find(
                child => child instanceof THREE.AmbientLight
            );
            if (ambient) {
                ambient.intensity = Math.max(sh[0], sh[1], sh[2]) * 0.5;
            }
        }
    }

    updateAnchors(frame) {
        this.anchors.forEach((anchorData, anchorId) => {
            const pose = frame.getPose(anchorData.anchor.anchorSpace, webXRManager.xrRefSpace);
            if (pose) {
                anchorData.object.position.fromArray(pose.transform.position);
                anchorData.object.quaternion.fromArray(pose.transform.orientation);
            }
        });
    }

    // Public API methods

    async createAnchor(position, orientation = null) {
        if (!webXRManager.features.anchors) {
            throw new Error('Anchors not supported');
        }
        
        try {
            // Create anchor at position
            const anchor = await webXRManager.createAnchor(position, orientation || new THREE.Quaternion());
            
            // Create visual representation
            const anchorObject = new THREE.Mesh(
                new THREE.SphereGeometry(0.05, 16, 16),
                this.materials.anchor
            );
            anchorObject.position.copy(position);
            if (orientation) anchorObject.quaternion.copy(orientation);
            
            this.arGroup.add(anchorObject);
            
            // Store anchor data
            const anchorId = `anchor-${Date.now()}`;
            this.anchors.set(anchorId, {
                anchor,
                object: anchorObject,
                created: Date.now()
            });
            
            console.log('⚓ Anchor created:', anchorId);
            
            return anchorId;
            
        } catch (error) {
            console.error('Failed to create anchor:', error);
            throw error;
        }
    }

    removeAnchor(anchorId) {
        const anchorData = this.anchors.get(anchorId);
        if (!anchorData) return;
        
        // Remove from scene
        this.arGroup.remove(anchorData.object);
        
        // Delete anchor
        anchorData.anchor.delete();
        
        // Remove from map
        this.anchors.delete(anchorId);
        
        console.log('⚓ Anchor removed:', anchorId);
    }

    placeObject(object, position, options = {}) {
        const {
            snapToPlane = true,
            createAnchor = true,
            enableOcclusion = true,
            castShadow = true,
            receiveShadow = true
        } = options;
        
        // Snap to nearest plane if requested
        if (snapToPlane) {
            const snappedPosition = this.snapToNearestPlane(position);
            if (snappedPosition) {
                position = snappedPosition;
            }
        }
        
        // Set object position
        object.position.copy(position);
        
        // Configure shadows
        object.castShadow = castShadow;
        object.receiveShadow = receiveShadow;
        
        // Add to AR group
        this.arGroup.add(object);
        
        // Create anchor if requested
        if (createAnchor) {
            this.createAnchor(position).then(anchorId => {
                object.userData.anchorId = anchorId;
            });
        }
        
        return object;
    }

    snapToNearestPlane(position, maxDistance = 0.5) {
        let nearestPlane = null;
        let nearestDistance = maxDistance;
        let snappedPosition = null;
        
        this.planes.forEach(planeData => {
            const planePosition = planeData.group.position;
            const distance = position.distanceTo(planePosition);
            
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestPlane = planeData;
                
                // Project position onto plane
                const planeNormal = new THREE.Vector3(0, 1, 0);
                planeNormal.applyQuaternion(planeData.group.quaternion);
                
                const planePoint = planePosition;
                const pointToPlane = new THREE.Vector3().subVectors(position, planePoint);
                const distance = pointToPlane.dot(planeNormal);
                
                snappedPosition = position.clone().sub(
                    planeNormal.clone().multiplyScalar(distance)
                );
            }
        });
        
        return snappedPosition;
    }

    // Collaborative features

    async createCollaborativeSpace(name, position) {
        const spaceId = `space-${Date.now()}`;
        
        const space = {
            id: spaceId,
            name,
            position: position.clone(),
            participants: new Map(),
            objects: new Map(),
            created: Date.now()
        };
        
        this.spaces.set(spaceId, space);
        
        // Create visual boundary
        const boundary = new THREE.Mesh(
            new THREE.RingGeometry(2, 2.1, 32),
            new THREE.MeshBasicMaterial({
                color: 0x00ff00,
                transparent: true,
                opacity: 0.3
            })
        );
        boundary.rotation.x = -Math.PI / 2;
        boundary.position.copy(position);
        
        this.arGroup.add(boundary);
        space.boundary = boundary;
        
        console.log('🏠 Collaborative space created:', name);
        
        return spaceId;
    }

    async joinCollaborativeSpace(spaceId, participantInfo) {
        const space = this.spaces.get(spaceId);
        if (!space) throw new Error('Space not found');
        
        // Create participant avatar
        const avatar = this.createAvatar(participantInfo);
        avatar.position.copy(space.position);
        avatar.position.x += Math.random() * 2 - 1;
        avatar.position.z += Math.random() * 2 - 1;
        
        this.arGroup.add(avatar);
        
        // Store participant
        space.participants.set(participantInfo.id, {
            info: participantInfo,
            avatar,
            joined: Date.now()
        });
        
        console.log('👤 Joined collaborative space:', participantInfo.name);
    }

    createAvatar(participantInfo) {
        const avatar = new THREE.Group();
        
        // Head
        const head = new THREE.Mesh(
            new THREE.SphereGeometry(0.15, 16, 16),
            new THREE.MeshPhongMaterial({
                color: participantInfo.avatarColor || 0x4a90e2
            })
        );
        head.position.y = 1.6;
        avatar.add(head);
        
        // Body
        const body = new THREE.Mesh(
            new THREE.CylinderGeometry(0.2, 0.15, 0.8, 16),
            new THREE.MeshPhongMaterial({
                color: participantInfo.avatarColor || 0x4a90e2
            })
        );
        body.position.y = 1;
        avatar.add(body);
        
        // Name label
        if (participantInfo.name) {
            // In real implementation, this would be a text sprite
            const label = new THREE.Mesh(
                new THREE.PlaneGeometry(0.5, 0.1),
                new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    transparent: true,
                    opacity: 0.8
                })
            );
            label.position.y = 2;
            avatar.add(label);
        }
        
        return avatar;
    }

    // Settings methods

    setShowPlanes(show) {
        this.settings.showPlanes = show;
        this.planes.forEach(planeData => {
            planeData.mesh.visible = show;
        });
    }

    setShowMeshes(show) {
        this.settings.showMeshes = show;
        this.meshes.forEach(meshData => {
            meshData.meshObject.visible = show;
        });
    }

    setEnableOcclusion(enable) {
        this.settings.enableOcclusion = enable;
        // Update occlusion meshes
    }

    disablePassthrough() {
        this.passthroughEnabled = false;
        console.log('📹 Passthrough disabled');
    }

    onSessionEnded() {
        console.log('🛑 AR session ended');
        
        // Clean up
        this.planes.forEach(planeData => {
            planeData.mesh.geometry.dispose();
        });
        this.planes.clear();
        
        this.meshes.forEach(meshData => {
            meshData.meshObject.geometry.dispose();
        });
        this.meshes.clear();
        
        this.anchors.clear();
        
        // Clear AR group
        while (this.arGroup.children.length > 0) {
            this.arGroup.remove(this.arGroup.children[0]);
        }
        
        this.passthroughEnabled = false;
    }

    // Utility methods

    getDetectedPlanes() {
        return Array.from(this.planes.values());
    }

    getAnchors() {
        return Array.from(this.anchors.values());
    }

    getCollaborativeSpaces() {
        return Array.from(this.spaces.values());
    }

    isPassthroughEnabled() {
        return this.passthroughEnabled;
    }
}

// Export mixed reality system
export const mixedReality = new RealMixedReality();