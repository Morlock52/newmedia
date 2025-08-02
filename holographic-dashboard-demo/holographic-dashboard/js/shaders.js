// Custom Shaders for Holographic Effects

const Shaders = {
    // Holographic material shader
    holographic: {
        vertexShader: `
            varying vec2 vUv;
            varying vec3 vPosition;
            varying vec3 vNormal;
            
            void main() {
                vUv = uv;
                vPosition = position;
                vNormal = normalize(normalMatrix * normal);
                
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform float time;
            uniform vec3 color1;
            uniform vec3 color2;
            uniform vec3 color3;
            uniform float scanlineSpeed;
            uniform float glowIntensity;
            uniform float hologramAlpha;
            
            varying vec2 vUv;
            varying vec3 vPosition;
            varying vec3 vNormal;
            
            void main() {
                // Holographic color gradient
                vec3 gradient = mix(color1, color2, vUv.y);
                gradient = mix(gradient, color3, sin(vUv.x * 10.0 + time) * 0.5 + 0.5);
                
                // Scanlines
                float scanline = sin(vUv.y * 200.0 + time * scanlineSpeed) * 0.03 + 0.97;
                
                // Fresnel effect
                vec3 viewDirection = normalize(cameraPosition - vPosition);
                float fresnel = pow(1.0 - dot(viewDirection, vNormal), 2.0);
                
                // Glitch effect
                float glitch = step(0.98, sin(time * 43.0)) * step(0.98, sin(time * 17.0 + vUv.y * 5.0));
                
                // Combine effects
                vec3 finalColor = gradient * scanline + fresnel * glowIntensity;
                finalColor += glitch * vec3(1.0, 0.0, 0.5);
                
                float alpha = hologramAlpha * scanline + fresnel * 0.5;
                
                gl_FragColor = vec4(finalColor, alpha);
            }
        `
    },

    // Particle shader for floating data points
    dataParticles: {
        vertexShader: `
            attribute float size;
            attribute vec3 customColor;
            
            varying vec3 vColor;
            varying float vAlpha;
            
            uniform float time;
            
            void main() {
                vColor = customColor;
                
                vec3 pos = position;
                pos.y += sin(time + position.x * 0.5) * 2.0;
                pos.x += cos(time + position.y * 0.5) * 2.0;
                
                vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
                
                vAlpha = 1.0 - smoothstep(100.0, 500.0, -mvPosition.z);
            }
        `,
        fragmentShader: `
            uniform sampler2D pointTexture;
            uniform float time;
            
            varying vec3 vColor;
            varying float vAlpha;
            
            void main() {
                vec2 uv = gl_PointCoord;
                
                // Create circular particle
                float dist = distance(uv, vec2(0.5));
                if (dist > 0.5) discard;
                
                float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
                alpha *= vAlpha;
                
                // Pulsing effect
                alpha *= 0.8 + sin(time * 3.0) * 0.2;
                
                gl_FragColor = vec4(vColor, alpha);
            }
        `
    },

    // Glowing edges shader
    glowEdges: {
        vertexShader: `
            varying vec2 vUv;
            varying vec3 vViewPosition;
            
            void main() {
                vUv = uv;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                vViewPosition = -mvPosition.xyz;
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            uniform vec3 glowColor;
            uniform float glowPower;
            uniform float glowIntensity;
            
            varying vec2 vUv;
            varying vec3 vViewPosition;
            
            void main() {
                float intensity = pow(glowPower / length(vViewPosition), glowIntensity);
                gl_FragColor = vec4(glowColor, intensity);
            }
        `
    },

    // Distortion shader for glitch effects
    distortion: {
        vertexShader: `
            varying vec2 vUv;
            
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform float time;
            uniform float distortionAmount;
            uniform float speed;
            
            varying vec2 vUv;
            
            float random(vec2 co) {
                return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
            }
            
            void main() {
                vec2 uv = vUv;
                
                // Chromatic aberration
                float r = texture2D(tDiffuse, uv + vec2(distortionAmount, 0.0)).r;
                float g = texture2D(tDiffuse, uv).g;
                float b = texture2D(tDiffuse, uv - vec2(distortionAmount, 0.0)).b;
                
                // Glitch distortion
                float glitchTime = floor(time * speed);
                float glitchLine = step(0.99, random(vec2(glitchTime, uv.y)));
                
                if (glitchLine > 0.0) {
                    uv.x += (random(vec2(glitchTime)) - 0.5) * 0.1;
                    r = texture2D(tDiffuse, uv).r;
                    g = texture2D(tDiffuse, uv).g;
                    b = texture2D(tDiffuse, uv).b;
                }
                
                gl_FragColor = vec4(r, g, b, 1.0);
            }
        `
    },

    // Grid floor shader
    gridFloor: {
        vertexShader: `
            varying vec2 vUv;
            varying vec3 vWorldPosition;
            
            void main() {
                vUv = uv;
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * viewMatrix * worldPosition;
            }
        `,
        fragmentShader: `
            uniform float time;
            uniform vec3 gridColor;
            uniform float gridSize;
            uniform float lineWidth;
            uniform float fadeDistance;
            
            varying vec2 vUv;
            varying vec3 vWorldPosition;
            
            float grid(vec2 st, float res) {
                vec2 grid = abs(fract(st * res - 0.5) - 0.5) / fwidth(st * res);
                return min(grid.x, grid.y);
            }
            
            void main() {
                vec2 coord = vWorldPosition.xz / gridSize;
                
                // Create grid lines
                float line = grid(coord, 1.0);
                line = 1.0 - min(line, 1.0);
                line = smoothstep(0.0, lineWidth, line);
                
                // Fade based on distance
                float dist = length(vWorldPosition.xz);
                float fade = 1.0 - smoothstep(fadeDistance * 0.5, fadeDistance, dist);
                
                // Pulsing effect
                float pulse = sin(time * 2.0 + dist * 0.1) * 0.2 + 0.8;
                
                vec3 color = gridColor * line * fade * pulse;
                float alpha = line * fade;
                
                gl_FragColor = vec4(color, alpha);
            }
        `
    },

    // Audio visualizer shader
    audioVisualizer: {
        vertexShader: `
            attribute float audioLevel;
            
            uniform float time;
            uniform float maxHeight;
            
            varying vec3 vColor;
            varying float vAudioLevel;
            
            void main() {
                vAudioLevel = audioLevel;
                
                vec3 pos = position;
                pos.y = audioLevel * maxHeight;
                
                // Wave effect
                pos.y += sin(time * 2.0 + position.x * 0.5) * 0.5;
                
                // Color based on height
                vColor = mix(
                    vec3(0.0, 1.0, 1.0),  // Cyan
                    vec3(1.0, 0.0, 1.0),  // Magenta
                    audioLevel
                );
                
                gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            }
        `,
        fragmentShader: `
            uniform float time;
            
            varying vec3 vColor;
            varying float vAudioLevel;
            
            void main() {
                // Glow effect
                float glow = 0.5 + sin(time * 10.0 * vAudioLevel) * 0.5;
                
                vec3 finalColor = vColor * (1.0 + glow * 0.5);
                float alpha = 0.8 + vAudioLevel * 0.2;
                
                gl_FragColor = vec4(finalColor, alpha);
            }
        `
    },

    // Screen space effects
    screenEffects: {
        scanlines: `
            vec3 scanlines(vec2 uv, vec3 color, float time) {
                float scanline = sin(uv.y * 800.0 + time * 10.0) * 0.04;
                color -= scanline;
                return color;
            }
        `,
        vignette: `
            vec3 vignette(vec2 uv, vec3 color, float darkness, float offset) {
                vec2 center = vec2(0.5);
                float dist = distance(uv, center);
                float vignette = smoothstep(offset, offset - 0.2, dist);
                return mix(color * darkness, color, vignette);
            }
        `,
        chromatic: `
            vec3 chromaticAberration(sampler2D tex, vec2 uv, float amount) {
                float r = texture2D(tex, uv + vec2(amount, 0.0)).r;
                float g = texture2D(tex, uv).g;
                float b = texture2D(tex, uv - vec2(amount, 0.0)).b;
                return vec3(r, g, b);
            }
        `,
        noise: `
            float random(vec2 co) {
                return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
            }
            
            vec3 filmGrain(vec2 uv, vec3 color, float time, float intensity) {
                float noise = random(uv + time) * intensity;
                return color + vec3(noise);
            }
        `
    }
};

// Helper function to create shader material
Shaders.createMaterial = function(shaderName, uniforms = {}) {
    const shader = this[shaderName];
    if (!shader) {
        console.error(`Shader '${shaderName}' not found`);
        return null;
    }

    // Manually merge uniforms since UniformsUtils might not be available
    const mergedUniforms = {
        time: { value: 0 },
        resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        ...uniforms
    };

    return new THREE.ShaderMaterial({
        uniforms: mergedUniforms,
        vertexShader: shader.vertexShader,
        fragmentShader: shader.fragmentShader,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
};

// Export for use in other modules
window.Shaders = Shaders;