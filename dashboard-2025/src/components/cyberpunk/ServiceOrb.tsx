'use client'

import React, { useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import * as THREE from 'three'

interface ServiceOrbProps {
  name: string
  status: 'online' | 'offline' | 'checking' | 'error'
  responseTime?: number
  onClick?: () => void
}

export function ServiceOrb({ name, status, responseTime, onClick }: ServiceOrbProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)

  const statusColors = {
    online: '#00ff88',
    offline: '#ff0040',
    checking: '#ffff00',
    error: '#ff6600'
  }

  const statusColor = statusColors[status]

  useEffect(() => {
    if (!mountRef.current) return

    // Scene setup
    const scene = new THREE.Scene()
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      1,
      0.1,
      1000
    )
    camera.position.z = 3

    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true, 
      antialias: true 
    })
    rendererRef.current = renderer
    renderer.setSize(150, 150)
    renderer.setPixelRatio(window.devicePixelRatio)
    mountRef.current.appendChild(renderer.domElement)

    // Sphere geometry
    const geometry = new THREE.SphereGeometry(1, 32, 32)
    
    // Holographic material
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color(statusColor) }
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        uniform float time;
        
        void main() {
          vUv = uv;
          vNormal = normal;
          
          vec3 pos = position;
          float noise = sin(pos.x * 10.0 + time) * 0.05;
          pos += normal * noise;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 color;
        uniform float time;
        varying vec2 vUv;
        varying vec3 vNormal;
        
        void main() {
          float fresnel = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          vec3 finalColor = color * (0.5 + fresnel);
          
          float scanline = sin(vUv.y * 100.0 + time * 5.0) * 0.1 + 0.9;
          finalColor *= scanline;
          
          float pulse = sin(time * 2.0) * 0.2 + 0.8;
          finalColor *= pulse;
          
          gl_FragColor = vec4(finalColor, 0.8);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide
    })

    const sphere = new THREE.Mesh(geometry, material)
    scene.add(sphere)

    // Wireframe overlay
    const wireframeGeometry = new THREE.IcosahedronGeometry(1.1, 1)
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: statusColor,
      wireframe: true,
      transparent: true,
      opacity: 0.3
    })
    const wireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial)
    scene.add(wireframe)

    // Particle system
    const particlesGeometry = new THREE.BufferGeometry()
    const particleCount = 50
    const positions = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount * 3; i += 3) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.random() * Math.PI
      const radius = 1.5 + Math.random() * 0.5

      positions[i] = radius * Math.sin(phi) * Math.cos(theta)
      positions[i + 1] = radius * Math.sin(phi) * Math.sin(theta)
      positions[i + 2] = radius * Math.cos(phi)
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    
    const particlesMaterial = new THREE.PointsMaterial({
      color: statusColor,
      size: 0.05,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    })

    const particles = new THREE.Points(particlesGeometry, particlesMaterial)
    scene.add(particles)

    // Animation
    let animationId: number
    const animate = () => {
      animationId = requestAnimationFrame(animate)

      sphere.rotation.y += 0.005
      wireframe.rotation.x += 0.003
      wireframe.rotation.y -= 0.002
      particles.rotation.y += 0.001

      material.uniforms.time.value += 0.01

      renderer.render(scene, camera)
    }

    animate()

    return () => {
      cancelAnimationFrame(animationId)
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement)
      }
      
      // Proper cleanup
      geometry.dispose()
      material.dispose()
      wireframeGeometry.dispose()
      wireframeMaterial.dispose()
      particlesGeometry.dispose()
      particlesMaterial.dispose()
      renderer.dispose()
      
      // Clear scene
      while(scene.children.length > 0) {
        scene.remove(scene.children[0])
      }
    }
  }, [statusColor])

  return (
    <motion.div
      className="relative cursor-pointer group"
      onClick={onClick}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      {/* 3D Orb Container */}
      <div 
        ref={mountRef} 
        className="w-[150px] h-[150px] mx-auto"
        style={{
          filter: `drop-shadow(0 0 20px ${statusColor})`
        }}
      />

      {/* Service Info */}
      <div className="text-center mt-4">
        <h3
          className="font-orbitron font-bold text-sm uppercase tracking-wider mb-1"
          style={{
            color: statusColor,
            textShadow: `0 0 10px ${statusColor}`
          }}
        >
          {name}
        </h3>
        
        <div className="flex items-center justify-center gap-2">
          <motion.div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: statusColor }}
            animate={{
              opacity: [1, 0.3, 1],
              scale: [1, 1.2, 1]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
          <span
            className="text-xs font-rajdhani uppercase"
            style={{ color: statusColor }}
          >
            {status}
          </span>
        </div>

        {responseTime && (
          <div className="mt-1">
            <span className="text-xs opacity-60" style={{ color: statusColor }}>
              {responseTime}ms
            </span>
          </div>
        )}
      </div>

      {/* Hover Effect */}
      <motion.div
        className="absolute inset-0 rounded-full pointer-events-none"
        initial={{ opacity: 0 }}
        whileHover={{ opacity: 1 }}
        style={{
          background: `radial-gradient(circle at center, ${statusColor}20, transparent 70%)`,
        }}
      />
    </motion.div>
  )
}