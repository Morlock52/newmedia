import React, { useRef, useMemo, Suspense, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Box as ThreeBox, Sphere, MeshDistortMaterial, Float, Html } from '@react-three/drei'
import * as THREE from 'three'
import { useSpring, animated } from '@react-spring/three'
import { Box, Typography, Chip } from '@mui/material'

interface MediaNode {
  id: string
  type: 'movie' | 'tv' | 'music' | 'book' | 'photo'
  title: string
  position: [number, number, number]
  color: string
  size: number
  connections: string[]
}

const mockMediaNodes: MediaNode[] = [
  { id: '1', type: 'movie', title: 'Recent Movies', position: [2, 0, 0], color: '#ff3366', size: 1, connections: ['2', '3'] },
  { id: '2', type: 'tv', title: 'TV Shows', position: [-2, 0, 0], color: '#00ff88', size: 0.8, connections: ['1', '4'] },
  { id: '3', type: 'music', title: 'Music Library', position: [0, 2, 0], color: '#00ffff', size: 0.9, connections: ['1', '5'] },
  { id: '4', type: 'book', title: 'Audiobooks', position: [0, -2, 0], color: '#ffaa00', size: 0.7, connections: ['2', '5'] },
  { id: '5', type: 'photo', title: 'Photos', position: [0, 0, 2], color: '#ff00ff', size: 0.6, connections: ['3', '4'] },
]

function MediaNode({ node, nodes }: { node: MediaNode; nodes: MediaNode[] }) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const [hovered, setHovered] = useState(false)
  const [clicked, setClicked] = useState(false)

  const { scale } = useSpring({
    scale: clicked ? 1.5 : hovered ? 1.2 : 1,
    config: { mass: 1, tension: 400, friction: 10 }
  })

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01
      meshRef.current.position.y = node.position[1] + Math.sin(state.clock.elapsedTime + node.position[0]) * 0.1
    }
  })

  return (
    <Float speed={2} rotationIntensity={0.2} floatIntensity={0.2}>
      <animated.mesh
        ref={meshRef}
        position={node.position}
        scale={scale}
        onClick={() => setClicked(!clicked)}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <Sphere args={[node.size, 32, 32]}>
          <MeshDistortMaterial
            color={node.color}
            attach="material"
            distort={0.3}
            speed={2}
            roughness={0}
            metalness={0.8}
            emissive={node.color}
            emissiveIntensity={hovered ? 0.5 : 0.2}
          />
        </Sphere>
        <Html
          distanceFactor={10}
          position={[0, node.size + 0.5, 0]}
          style={{
            transition: 'all 0.2s',
            opacity: hovered ? 1 : 0.7,
            transform: `scale(${hovered ? 1.1 : 1})`,
          }}
        >
          <Chip
            label={node.title}
            size="small"
            sx={{
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              color: node.color,
              border: `1px solid ${node.color}`,
              fontSize: '0.7rem',
              backdropFilter: 'blur(10px)',
            }}
          />
        </Html>
      </animated.mesh>
    </Float>
  )
}

function Connections({ nodes }: { nodes: MediaNode[] }) {
  const connections = useMemo(() => {
    const lines: JSX.Element[] = []
    nodes.forEach((node) => {
      node.connections.forEach((targetId) => {
        const targetNode = nodes.find(n => n.id === targetId)
        if (targetNode && node.id < targetId) {
          const points = [
            new THREE.Vector3(...node.position),
            new THREE.Vector3(...targetNode.position),
          ]
          lines.push(
            <line key={`${node.id}-${targetId}`}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#ffffff" opacity={0.2} transparent />
            </line>
          )
        }
      })
    })
    return lines
  }, [nodes])

  return <>{connections}</>
}

function ParticleField() {
  const particlesRef = useRef<THREE.Points>(null!)
  const particleCount = 500
  
  const [positions, colors] = useMemo(() => {
    const positions = new Float32Array(particleCount * 3)
    const colors = new Float32Array(particleCount * 3)
    
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 10
      positions[i * 3 + 1] = (Math.random() - 0.5) * 10
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10
      
      const color = new THREE.Color(`hsl(${Math.random() * 360}, 100%, 50%)`)
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
    }
    
    return [positions, colors]
  }, [])

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.05
      particlesRef.current.rotation.x = state.clock.elapsedTime * 0.03
    }
  })

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particleCount}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.02} vertexColors transparent opacity={0.6} />
    </points>
  )
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00ff88" />
      
      <ParticleField />
      <Connections nodes={mockMediaNodes} />
      
      {mockMediaNodes.map((node) => (
        <MediaNode key={node.id} node={node} nodes={mockMediaNodes} />
      ))}
      
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        maxDistance={10}
        minDistance={3}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </>
  )
}

const MediaSphere: React.FC = () => {
  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      </Canvas>
      
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          left: 16,
          zIndex: 1,
        }}
      >
        <Typography variant="h6" sx={{ color: 'white', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
          Media Universe
        </Typography>
        <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
          Explore your media connections
        </Typography>
      </Box>
    </Box>
  )
}

export default MediaSphere