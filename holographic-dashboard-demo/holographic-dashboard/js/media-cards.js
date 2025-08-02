// 3D Media Cards System

class MediaCard3D {
    constructor(data, index, scene) {
        this.data = data;
        this.index = index;
        this.scene = scene;
        this.mesh = null;
        this.isHovered = false;
        this.targetPosition = new THREE.Vector3();
        this.targetRotation = new THREE.Euler();
        this.originalPosition = new THREE.Vector3();
        
        this.create();
    }

    create() {
        const { cardWidth, cardHeight, cardDepth } = CONFIG.mediaCards;
        
        // Create card group
        this.group = new THREE.Group();
        
        // Card base
        const cardGeometry = new THREE.BoxGeometry(cardWidth, cardHeight, cardDepth);
        const cardMaterial = new THREE.MeshPhongMaterial({
            color: 0x1a1a2e,
            emissive: 0x00FFFF,
            emissiveIntensity: 0.1,
            transparent: true,
            opacity: 0.9
        });
        
        this.mesh = new THREE.Mesh(cardGeometry, cardMaterial);
        this.group.add(this.mesh);
        
        // Holographic frame
        const frameGeometry = new THREE.BoxGeometry(
            cardWidth + 0.2,
            cardHeight + 0.2,
            cardDepth + 0.1
        );
        const frameMaterial = Shaders.createMaterial('holographic', {
            color1: { value: new THREE.Color(0x00FFFF) },
            color2: { value: new THREE.Color(0xFF00FF) },
            color3: { value: new THREE.Color(0xFFFF00) },
            scanlineSpeed: { value: 0.001 },
            glowIntensity: { value: 1.0 },
            hologramAlpha: { value: 0.2 }
        });
        
        const frame = new THREE.Mesh(frameGeometry, frameMaterial);
        frame.position.z = -0.05;
        this.group.add(frame);
        
        // Add thumbnail plane
        const thumbGeometry = new THREE.PlaneGeometry(cardWidth * 0.9, cardHeight * 0.6);
        const thumbMaterial = new THREE.MeshBasicMaterial({
            color: new THREE.Color(CONFIG.mediaTypes[this.data.type]?.color || 0x00FFFF),
            transparent: true,
            opacity: 0.8
        });
        
        const thumbnail = new THREE.Mesh(thumbGeometry, thumbMaterial);
        thumbnail.position.z = cardDepth / 2 + 0.01;
        thumbnail.position.y = cardHeight * 0.15;
        this.group.add(thumbnail);
        
        // Add media type icon
        const iconTexture = Utils.createTextTexture(
            CONFIG.mediaTypes[this.data.type]?.icon || '📁',
            {
                fontSize: 128,
                padding: 32
            }
        );
        
        const iconGeometry = new THREE.PlaneGeometry(2, 2);
        const iconMaterial = new THREE.MeshBasicMaterial({
            map: iconTexture,
            transparent: true,
            opacity: 0.9
        });
        
        const icon = new THREE.Mesh(iconGeometry, iconMaterial);
        icon.position.z = cardDepth / 2 + 0.02;
        icon.position.y = cardHeight * 0.1;
        this.group.add(icon);
        
        // Add title text
        const titleTexture = Utils.createTextTexture(this.data.title || 'Media Item', {
            fontSize: 48,
            fontFamily: 'Orbitron',
            maxWidth: 512,
            padding: 20
        });
        
        const titleGeometry = new THREE.PlaneGeometry(cardWidth * 0.9, 1.5);
        const titleMaterial = new THREE.MeshBasicMaterial({
            map: titleTexture,
            transparent: true
        });
        
        const title = new THREE.Mesh(titleGeometry, titleMaterial);
        title.position.z = cardDepth / 2 + 0.02;
        title.position.y = -cardHeight * 0.35;
        this.group.add(title);
        
        // Add metadata
        if (this.data.metadata) {
            const metaText = `${this.data.metadata.year || ''} • ${this.data.metadata.quality || ''} • ${this.data.metadata.duration || ''}`;
            const metaTexture = Utils.createTextTexture(metaText, {
                fontSize: 32,
                fontFamily: 'Inter',
                color: '#B0B0C0',
                maxWidth: 512
            });
            
            const metaGeometry = new THREE.PlaneGeometry(cardWidth * 0.9, 0.8);
            const metaMaterial = new THREE.MeshBasicMaterial({
                map: metaTexture,
                transparent: true,
                opacity: 0.7
            });
            
            const meta = new THREE.Mesh(metaGeometry, metaMaterial);
            meta.position.z = cardDepth / 2 + 0.02;
            meta.position.y = -cardHeight * 0.45;
            this.group.add(meta);
        }
        
        // Add glow effect
        const glowGeometry = new THREE.PlaneGeometry(cardWidth * 1.2, cardHeight * 1.2);
        const glowMaterial = new THREE.ShaderMaterial({
            uniforms: {
                glowColor: { value: new THREE.Color(0x00FFFF) },
                glowPower: { value: 2.0 },
                glowIntensity: { value: 1.0 }
            },
            vertexShader: Shaders.glowEdges.vertexShader,
            fragmentShader: Shaders.glowEdges.fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        this.glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        this.glowMesh.position.z = -0.5;
        this.glowMesh.visible = false;
        this.group.add(this.glowMesh);
        
        // Add to scene
        this.scene.add(this.group);
        
        // Store original position
        this.originalPosition.copy(this.group.position);
        this.targetPosition.copy(this.group.position);
    }

    setPosition(x, y, z) {
        this.group.position.set(x, y, z);
        this.originalPosition.copy(this.group.position);
        this.targetPosition.copy(this.group.position);
    }

    hover() {
        if (this.isHovered) return;
        this.isHovered = true;
        
        // Animate position
        this.targetPosition.copy(this.originalPosition);
        this.targetPosition.y += CONFIG.mediaCards.hoverHeight;
        
        // Show glow
        this.glowMesh.visible = true;
        
        // Scale up
        if (typeof gsap !== 'undefined') {
            gsap.to(this.group.scale, {
                x: CONFIG.mediaCards.hoverScale,
                y: CONFIG.mediaCards.hoverScale,
                z: CONFIG.mediaCards.hoverScale,
                duration: 0.3,
                ease: "power2.out"
            });
            
            // Brighten material
            gsap.to(this.mesh.material, {
                emissiveIntensity: 0.3,
                duration: 0.3
            });
        } else {
            // Fallback without animation
            this.group.scale.set(CONFIG.mediaCards.hoverScale, CONFIG.mediaCards.hoverScale, CONFIG.mediaCards.hoverScale);
            this.mesh.material.emissiveIntensity = 0.3;
        }
    }

    unhover() {
        if (!this.isHovered) return;
        this.isHovered = false;
        
        // Reset position
        this.targetPosition.copy(this.originalPosition);
        
        // Hide glow
        this.glowMesh.visible = false;
        
        // Scale down
        if (typeof gsap !== 'undefined') {
            gsap.to(this.group.scale, {
                x: 1,
                y: 1,
                z: 1,
                duration: 0.3,
                ease: "power2.out"
            });
            
            // Reset material
            gsap.to(this.mesh.material, {
                emissiveIntensity: 0.1,
                duration: 0.3
            });
        } else {
            // Fallback without animation
            this.group.scale.set(1, 1, 1);
            this.mesh.material.emissiveIntensity = 0.1;
        }
    }

    update(deltaTime) {
        // Smooth position interpolation
        this.group.position.lerp(this.targetPosition, deltaTime * 5);
        
        // Floating animation
        if (!this.isHovered) {
            const time = Date.now() * 0.001;
            this.group.position.y = this.originalPosition.y + Math.sin(time + this.index) * 0.5;
        }
        
        // Rotate slightly
        this.group.rotation.y += deltaTime * 0.1;
        
        // Update shader uniforms
        this.group.traverse((child) => {
            if (child.material && child.material.uniforms && child.material.uniforms.time) {
                child.material.uniforms.time.value += deltaTime;
            }
        });
    }

    dispose() {
        this.group.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
        
        this.scene.remove(this.group);
    }
}

class MediaCardsManager {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.cards = [];
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.hoveredCard = null;
        
        this.init();
    }

    init() {
        // Event listeners
        window.addEventListener('mousemove', this.onMouseMove.bind(this));
        window.addEventListener('click', this.onClick.bind(this));
    }

    loadMediaData(mediaData) {
        // Clear existing cards
        this.clearCards();
        
        // Create new cards
        const { rows, columns, spacing } = CONFIG.mediaCards;
        const startX = -(columns - 1) * spacing / 2;
        const startZ = -(rows - 1) * spacing / 2;
        
        mediaData.forEach((data, index) => {
            const row = Math.floor(index / columns);
            const col = index % columns;
            
            if (row >= rows) return; // Limit to configured rows
            
            const card = new MediaCard3D(data, index, this.scene);
            
            const x = startX + col * spacing;
            const y = 0;
            const z = startZ + row * spacing;
            
            card.setPosition(x, y, z);
            
            // Stagger animation
            if (typeof gsap !== 'undefined') {
                gsap.from(card.group.position, {
                    y: -20,
                    duration: 1,
                    delay: index * 0.1,
                    ease: "power2.out"
                });
                
                gsap.from(card.group.scale, {
                    x: 0,
                    y: 0,
                    z: 0,
                    duration: 0.5,
                    delay: index * 0.1,
                    ease: "back.out(1.7)"
                });
            } else {
                // No animation fallback - cards appear immediately
                card.group.position.y = y;
                card.group.scale.set(1, 1, 1);
            }
            
            this.cards.push(card);
        });
    }

    onMouseMove(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        
        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check intersections
        const intersects = this.raycaster.intersectObjects(
            this.cards.map(card => card.mesh),
            false
        );
        
        // Handle hover
        if (intersects.length > 0) {
            const intersectedMesh = intersects[0].object;
            const card = this.cards.find(c => c.mesh === intersectedMesh);
            
            if (card && card !== this.hoveredCard) {
                if (this.hoveredCard) {
                    this.hoveredCard.unhover();
                }
                card.hover();
                this.hoveredCard = card;
                document.body.style.cursor = 'pointer';
            }
        } else {
            if (this.hoveredCard) {
                this.hoveredCard.unhover();
                this.hoveredCard = null;
                document.body.style.cursor = 'default';
            }
        }
    }

    onClick(event) {
        if (this.hoveredCard) {
            // Dispatch custom event with card data
            const customEvent = new CustomEvent('mediaCardClick', {
                detail: {
                    data: this.hoveredCard.data,
                    card: this.hoveredCard
                }
            });
            window.dispatchEvent(customEvent);
        }
    }

    update(deltaTime) {
        this.cards.forEach(card => {
            card.update(deltaTime);
        });
    }

    clearCards() {
        this.cards.forEach(card => {
            card.dispose();
        });
        this.cards = [];
        this.hoveredCard = null;
    }

    // Generate demo data
    generateDemoData() {
        const types = ['movie', 'series', 'music', 'documentary'];
        const titles = [
            'Blade Runner 2049', 'The Matrix Resurrections', 'Dune Part Two',
            'Interstellar', 'Inception', 'The Mandalorian', 'Westworld',
            'Black Mirror', 'Stranger Things', 'The Expanse', 'Foundation',
            'Cyberpunk: Edgerunners'
        ];
        
        const data = [];
        for (let i = 0; i < 12; i++) {
            data.push({
                id: Utils.generateId(),
                title: titles[i % titles.length],
                type: types[Math.floor(Math.random() * types.length)],
                metadata: {
                    year: 2020 + Math.floor(Math.random() * 5),
                    quality: Math.random() > 0.5 ? '4K HDR' : '1080p',
                    duration: `${90 + Math.floor(Math.random() * 90)} min`
                },
                thumbnail: null // Would be actual image URL
            });
        }
        
        return data;
    }
}

// Export for use in other modules
window.MediaCard3D = MediaCard3D;
window.MediaCardsManager = MediaCardsManager;