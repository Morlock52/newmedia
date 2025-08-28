'use client'

import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

export function CyberpunkLayout({ children }: { children: React.ReactNode }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    // Matrix rain effect
    const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    const charArray = chars.split('')
    const fontSize = 14
    const columns = canvas.width / fontSize
    const drops: number[] = []

    for (let i = 0; i < columns; i++) {
      drops[i] = (i * 7) % 100 * -1
    }

    function draw() {
      if (!ctx || !canvas) return
      
      ctx.fillStyle = 'rgba(10, 14, 27, 0.05)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      ctx.fillStyle = '#00ff88'
      ctx.font = fontSize + 'px monospace'

      for (let i = 0; i < drops.length; i++) {
        const text = charArray[Math.floor(Math.random() * charArray.length)]
        ctx.fillText(text, i * fontSize, drops[i] * fontSize)

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0
        }
        drops[i]++
      }
    }

    const interval = setInterval(draw, 35)

    const handleResize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    window.addEventListener('resize', handleResize)

    return () => {
      clearInterval(interval)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <div className="cyberpunk-theme min-h-screen relative overflow-hidden">
      {/* Matrix Rain Background */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none opacity-20"
        style={{ zIndex: 0 }}
      />

      {/* Animated Grid */}
      <div className="cyber-grid" />

      {/* Scanlines */}
      <div className="scanlines" />

      {/* Circuit Pattern */}
      <div className="circuit-bg" />

      {/* Vignette Effect */}
      <div className="fixed inset-0 pointer-events-none" style={{
        background: 'radial-gradient(circle at center, transparent 0%, rgba(10, 14, 27, 0.4) 100%)',
        zIndex: 2
      }} />

      {/* Glitch Lines */}
      <motion.div
        className="fixed inset-0 pointer-events-none"
        style={{ zIndex: 1 }}
        animate={{
          opacity: [0, 0, 1, 0, 0],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          times: [0, 0.9, 0.91, 0.92, 1],
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/20 to-transparent transform skew-x-12" />
      </motion.div>

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  )
}