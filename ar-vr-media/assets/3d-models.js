/**
 * 3D Model Generator
 * Creates procedural 3D models for WebXR demos
 */

import * as THREE from 'three';

export class ModelGenerator {
    static createInteractiveButton(options = {}) {
        const {
            width = 0.3,
            height = 0.1,
            depth = 0.05,
            color = 0x4a90e2,
            text = 'Click Me'
        } = options;
        
        const group = new THREE.Group();
        
        // Button base
        const geometry = new THREE.BoxGeometry(width, height, depth);
        const material = new THREE.MeshPhongMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.1
        });
        
        const button = new THREE.Mesh(geometry, material);
        button.castShadow = true;
        button.receiveShadow = true;
        group.add(button);
        
        // Interaction states
        button.userData = {
            isButton: true,
            originalColor: color,
            hoverColor: 0x6ab0f2,
            pressColor: 0x2a70d2,
            onPress: null,
            onRelease: null
        };
        
        return group;
    }
    
    static createFloatingCrystal(options = {}) {
        const {
            size = 0.2,
            color = 0x00ffff,
            segments = 8
        } = options;
        
        const group = new THREE.Group();
        
        // Crystal geometry
        const geometry = new THREE.OctahedronGeometry(size, 0);
        const material = new THREE.MeshPhysicalMaterial({
            color,
            metalness: 0.2,
            roughness: 0.1,
            transparent: true,
            opacity: 0.8,
            envMapIntensity: 1
        });
        
        const crystal = new THREE.Mesh(geometry, material);
        crystal.castShadow = true;
        group.add(crystal);
        
        // Glow effect
        const glowGeometry = new THREE.OctahedronGeometry(size * 1.2, 0);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.2,
            side: THREE.BackSide
        });
        
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        group.add(glow);
        
        // Animation data
        group.userData = {
            isCrystal: true,
            floatSpeed: 0.5 + Math.random() * 0.5,
            rotationSpeed: 0.01 + Math.random() * 0.02,
            startY: 0
        };
        
        return group;
    }
    
    static createVirtualMonitor(options = {}) {
        const {
            width = 1.6,
            height = 0.9,
            bezelWidth = 0.05,
            screenColor = 0x000000,
            frameColor = 0x333333
        } = options;
        
        const group = new THREE.Group();
        
        // Screen
        const screenGeometry = new THREE.PlaneGeometry(width, height);
        const screenMaterial = new THREE.MeshBasicMaterial({
            color: screenColor,
            emissive: screenColor,
            emissiveIntensity: 0.1
        });
        
        const screen = new THREE.Mesh(screenGeometry, screenMaterial);
        screen.name = 'screen-surface';
        group.add(screen);
        
        // Frame
        const frameWidth = width + bezelWidth * 2;
        const frameHeight = height + bezelWidth * 2;
        const frameGeometry = new THREE.PlaneGeometry(frameWidth, frameHeight);
        const frameMaterial = new THREE.MeshPhongMaterial({
            color: frameColor
        });
        
        const frame = new THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.z = -0.01;
        group.add(frame);
        
        // Stand
        const standGeometry = new THREE.CylinderGeometry(0.15, 0.2, 0.3);
        const standMaterial = new THREE.MeshPhongMaterial({
            color: frameColor
        });
        
        const stand = new THREE.Mesh(standGeometry, standMaterial);
        stand.position.y = -height / 2 - 0.15;
        group.add(stand);
        
        // Base
        const baseGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.05);
        const base = new THREE.Mesh(baseGeometry, standMaterial);
        base.position.y = -height / 2 - 0.3;
        group.add(base);
        
        group.userData = {
            isMonitor: true,
            screen,
            width,
            height
        };
        
        return group;
    }
    
    static createHandTrackingDemo() {
        const group = new THREE.Group();
        
        // Create floating targets for hand interaction
        const colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff];
        const positions = [
            new THREE.Vector3(-0.6, 0, 0),
            new THREE.Vector3(-0.3, 0, 0),
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0.3, 0, 0),
            new THREE.Vector3(0.6, 0, 0)
        ];
        
        positions.forEach((pos, i) => {
            const target = new THREE.Mesh(
                new THREE.SphereGeometry(0.1, 32, 16),
                new THREE.MeshPhongMaterial({
                    color: colors[i],
                    emissive: colors[i],
                    emissiveIntensity: 0.2
                })
            );
            
            target.position.copy(pos);
            target.castShadow = true;
            
            target.userData = {
                isHandTarget: true,
                originalColor: colors[i],
                originalScale: 1,
                isActivated: false
            };
            
            group.add(target);
        });
        
        return group;
    }
    
    static createARPortal(options = {}) {
        const {
            radius = 1,
            color = 0x00ffff,
            destination = 'space'
        } = options;
        
        const group = new THREE.Group();
        
        // Portal ring
        const ringGeometry = new THREE.TorusGeometry(radius, 0.1, 16, 32);
        const ringMaterial = new THREE.MeshPhongMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.5
        });
        
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.castShadow = true;
        group.add(ring);
        
        // Portal surface
        const portalGeometry = new THREE.CircleGeometry(radius * 0.95, 32);
        const portalMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(color) }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                varying vec2 vUv;
                
                void main() {
                    vec2 center = vec2(0.5, 0.5);
                    float dist = distance(vUv, center);
                    
                    // Swirling effect
                    float angle = atan(vUv.y - 0.5, vUv.x - 0.5);
                    float swirl = sin(angle * 5.0 + time * 2.0 - dist * 10.0) * 0.5 + 0.5;
                    
                    // Radial fade
                    float fade = 1.0 - smoothstep(0.0, 0.5, dist);
                    
                    vec3 finalColor = mix(color * 0.5, color, swirl) * fade;
                    gl_FragColor = vec4(finalColor, fade * 0.8);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        const portal = new THREE.Mesh(portalGeometry, portalMaterial);
        portal.position.z = 0.01;
        group.add(portal);
        
        // Particle effects
        const particleCount = 50;
        const particleGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = radius * (0.8 + Math.random() * 0.2);
            
            positions[i * 3] = Math.cos(angle) * r;
            positions[i * 3 + 1] = Math.sin(angle) * r;
            positions[i * 3 + 2] = 0;
            
            velocities[i * 3] = (Math.random() - 0.5) * 0.02;
            velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
            velocities[i * 3 + 2] = Math.random() * 0.05;
        }
        
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            color,
            size: 0.05,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        const particles = new THREE.Points(particleGeometry, particleMaterial);
        group.add(particles);
        
        group.userData = {
            isPortal: true,
            destination,
            portalMaterial,
            particles,
            particleVelocities: velocities
        };
        
        return group;
    }
    
    static createSpaceEnvironment() {
        const group = new THREE.Group();
        
        // Starfield
        const starCount = 1000;
        const starGeometry = new THREE.BufferGeometry();
        const starPositions = new Float32Array(starCount * 3);
        
        for (let i = 0; i < starCount; i++) {
            starPositions[i * 3] = (Math.random() - 0.5) * 100;
            starPositions[i * 3 + 1] = (Math.random() - 0.5) * 100;
            starPositions[i * 3 + 2] = (Math.random() - 0.5) * 100;
        }
        
        starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
        
        const starMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.1,
            transparent: true,
            opacity: 0.8
        });
        
        const stars = new THREE.Points(starGeometry, starMaterial);
        group.add(stars);
        
        // Nebula
        const nebulaGeometry = new THREE.IcosahedronGeometry(30, 3);
        const nebulaMaterial = new THREE.MeshBasicMaterial({
            color: 0x4a0080,
            transparent: true,
            opacity: 0.1,
            side: THREE.BackSide
        });
        
        const nebula = new THREE.Mesh(nebulaGeometry, nebulaMaterial);
        group.add(nebula);
        
        // Planets
        const planets = [
            { radius: 1, color: 0xff6b6b, distance: 10, speed: 0.01 },
            { radius: 0.8, color: 0x4ecdc4, distance: 15, speed: 0.008 },
            { radius: 1.2, color: 0x45b7d1, distance: 20, speed: 0.005 }
        ];
        
        planets.forEach(planetData => {
            const planet = new THREE.Mesh(
                new THREE.SphereGeometry(planetData.radius, 32, 16),
                new THREE.MeshPhongMaterial({
                    color: planetData.color,
                    emissive: planetData.color,
                    emissiveIntensity: 0.1
                })
            );
            
            planet.position.x = planetData.distance;
            planet.castShadow = true;
            planet.receiveShadow = true;
            
            planet.userData = {
                orbitRadius: planetData.distance,
                orbitSpeed: planetData.speed,
                angle: 0
            };
            
            group.add(planet);
        });
        
        group.userData = {
            isSpaceEnvironment: true,
            stars,
            nebula,
            planets: group.children.filter(child => child.userData.orbitRadius)
        };
        
        return group;
    }
    
    static createHapticDemoObject() {
        const group = new THREE.Group();
        
        // Different textures for haptic feedback
        const materials = [
            { 
                name: 'smooth',
                material: new THREE.MeshPhongMaterial({ 
                    color: 0x4a90e2,
                    roughness: 0.1,
                    metalness: 0.8 
                }),
                hapticPattern: { intensity: 0.1, duration: 20 }
            },
            {
                name: 'rough',
                material: new THREE.MeshPhongMaterial({ 
                    color: 0xe24a90,
                    roughness: 0.9,
                    metalness: 0.1 
                }),
                hapticPattern: { intensity: 0.8, duration: 50 }
            },
            {
                name: 'bumpy',
                material: new THREE.MeshPhongMaterial({ 
                    color: 0x90e24a,
                    roughness: 0.5,
                    metalness: 0.5 
                }),
                hapticPattern: { intensity: 0.5, duration: 30, pulse: true }
            }
        ];
        
        materials.forEach((mat, i) => {
            const cube = new THREE.Mesh(
                new THREE.BoxGeometry(0.3, 0.3, 0.3),
                mat.material
            );
            
            cube.position.x = (i - 1) * 0.5;
            cube.castShadow = true;
            cube.receiveShadow = true;
            
            cube.userData = {
                isHapticObject: true,
                hapticPattern: mat.hapticPattern,
                textureName: mat.name
            };
            
            group.add(cube);
        });
        
        return group;
    }
}

// Animation helpers
export class AnimationHelpers {
    static animateFloat(object, amplitude = 0.1, speed = 1) {
        const startY = object.position.y;
        
        const animate = (time) => {
            object.position.y = startY + Math.sin(time * 0.001 * speed) * amplitude;
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
    
    static animateRotation(object, speed = { x: 0, y: 0.01, z: 0 }) {
        const animate = () => {
            object.rotation.x += speed.x;
            object.rotation.y += speed.y;
            object.rotation.z += speed.z;
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
    
    static animatePortal(portal) {
        const animate = (time) => {
            if (portal.userData.portalMaterial) {
                portal.userData.portalMaterial.uniforms.time.value = time * 0.001;
            }
            
            // Animate particles
            if (portal.userData.particles) {
                const positions = portal.userData.particles.geometry.attributes.position.array;
                const velocities = portal.userData.particleVelocities;
                
                for (let i = 0; i < positions.length / 3; i++) {
                    positions[i * 3] += velocities[i * 3];
                    positions[i * 3 + 1] += velocities[i * 3 + 1];
                    positions[i * 3 + 2] += velocities[i * 3 + 2];
                    
                    // Reset if too far
                    if (positions[i * 3 + 2] > 2) {
                        positions[i * 3 + 2] = 0;
                    }
                }
                
                portal.userData.particles.geometry.attributes.position.needsUpdate = true;
            }
            
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
    
    static animateSpace(spaceEnvironment) {
        const animate = () => {
            // Rotate stars slowly
            if (spaceEnvironment.userData.stars) {
                spaceEnvironment.userData.stars.rotation.y += 0.0001;
            }
            
            // Orbit planets
            if (spaceEnvironment.userData.planets) {
                spaceEnvironment.userData.planets.forEach(planet => {
                    planet.userData.angle += planet.userData.orbitSpeed;
                    planet.position.x = Math.cos(planet.userData.angle) * planet.userData.orbitRadius;
                    planet.position.z = Math.sin(planet.userData.angle) * planet.userData.orbitRadius;
                    planet.rotation.y += 0.01;
                });
            }
            
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
}