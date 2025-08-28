'use client'

import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Float, Text, Sphere, MeshDistortMaterial } from '@react-three/drei'
// Note: Using regular framer-motion for now
import * as THREE from 'three'

interface MediaOrbProps {
  service: {
    name: string
    port: number
    type: string
    color: string
    status: string
  }
  position: [number, number, number]
}

export function MediaOrb({ service, position }: MediaOrbProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  const [clicked, setClicked] = useState(false)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.2
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3
      
      // Pulse effect when hovered
      if (hovered) {
        meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 8) * 0.1)
      } else {
        meshRef.current.scale.setScalar(1)
      }
    }
  })

  const handleClick = () => {
    setClicked(!clicked)
    // Open service in new tab
    const baseUrl = window.location.hostname
    const serviceUrl = `http://${baseUrl}:${service.port}`
    window.open(serviceUrl, '_blank')
  }

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
      <group position={position}>
        {/* Main service orb */}
        <Sphere
          ref={meshRef}
          args={[0.5, 16, 16]}
          onClick={handleClick}
          onPointerOver={(e) => {
            e.stopPropagation()
            setHovered(true)
            document.body.style.cursor = 'pointer'
          }}
          onPointerOut={() => {
            setHovered(false)
            document.body.style.cursor = 'auto'
          }}
        >
          <MeshDistortMaterial
            color={service.color}
            attach="material"
            distort={hovered ? 0.4 : 0.2}
            speed={hovered ? 5 : 2}
            roughness={0.1}
            metalness={0.8}
            emissive={service.color}
            emissiveIntensity={hovered ? 0.3 : 0.1}
          />
        </Sphere>

        {/* Service name text */}
        <Text
          position={[0, -1, 0]}
          fontSize={0.2}
          color={hovered ? '#FFFFFF' : '#CCCCCC'}
          anchorX="center"
          anchorY="middle"
          font="/fonts/Inter-Bold.woff"
        >
          {service.name}
        </Text>

        {/* Service type indicator */}
        <Text
          position={[0, -1.3, 0]}
          fontSize={0.15}
          color={hovered ? service.color : '#888888'}
          anchorX="center"
          anchorY="middle"
          font="/fonts/Inter-Regular.woff"
        >
          {service.type.toUpperCase()}
        </Text>

        {/* Status indicator */}
        <Sphere args={[0.1, 8, 8]} position={[0.7, 0.7, 0]}>
          <meshStandardMaterial 
            color={service.status === 'online' ? '#00FF00' : '#FF0000'} 
            emissive={service.status === 'online' ? '#004400' : '#440000'}
            emissiveIntensity={0.5}
          />
        </Sphere>

        {/* Connecting lines when hovered */}
        {hovered && (
          <line>
            <bufferGeometry attach="geometry">
              <bufferAttribute
                attach={['attributes', 'position'] as any}
                args={[new Float32Array([0, 0, 0, 0, -2, 0]), 3]}
                count={2}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial attach="material" color={service.color} opacity={0.6} transparent />
          </line>
        )}
      </group>
    </Float>
  )
}