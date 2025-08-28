'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

interface GlitchTextProps {
  text: string
  className?: string
  color?: string
  size?: 'sm' | 'md' | 'lg' | 'xl' | '2xl'
  continuous?: boolean
}

export function GlitchText({
  text,
  className = '',
  color = '#00ffff',
  size = 'lg',
  continuous = false
}: GlitchTextProps) {
  const [glitchText, setGlitchText] = useState(text)
  const [isGlitching, setIsGlitching] = useState(false)

  const sizeMap = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-2xl',
    xl: 'text-4xl',
    '2xl': 'text-6xl'
  }

  const glitchChars = '!@#$%^&*()_+-=[]{}|;:,.<>?/~`'

  useEffect(() => {
    if (!continuous && !isGlitching) return

    const interval = setInterval(() => {
      const shouldGlitch = continuous || Math.random() > 0.95
      
      if (shouldGlitch) {
        const glitched = text
          .split('')
          .map(char => {
            if (Math.random() > 0.7) {
              return glitchChars[Math.floor(Math.random() * glitchChars.length)]
            }
            return char
          })
          .join('')
        
        setGlitchText(glitched)
        
        setTimeout(() => {
          setGlitchText(text)
        }, 100)
      }
    }, continuous ? 100 : 3000)

    return () => clearInterval(interval)
  }, [text, continuous, isGlitching, glitchChars])

  return (
    <motion.div
      className={`relative font-orbitron font-bold uppercase tracking-wider ${sizeMap[size]} ${className}`}
      style={{ color }}
      onMouseEnter={() => setIsGlitching(true)}
      onMouseLeave={() => setIsGlitching(false)}
    >
      {/* Main text */}
      <span
        style={{
          textShadow: `
            0 0 10px ${color},
            0 0 20px ${color},
            0 0 30px ${color}
          `
        }}
      >
        {glitchText}
      </span>

      {/* Glitch layers */}
      <motion.span
        className="absolute inset-0 opacity-80"
        style={{
          color: '#ff0033',
          textShadow: '2px 2px 0 #ff0033',
          clipPath: 'inset(20% 0 30% 0)',
        }}
        animate={isGlitching || continuous ? {
          x: [-2, 2, -2],
          opacity: [0.8, 0, 0.8],
        } : {}}
        transition={{
          duration: 0.2,
          repeat: Infinity,
        }}
      >
        {text}
      </motion.span>

      <motion.span
        className="absolute inset-0 opacity-80"
        style={{
          color: '#0099ff',
          textShadow: '-2px -2px 0 #0099ff',
          clipPath: 'inset(60% 0 10% 0)',
        }}
        animate={isGlitching || continuous ? {
          x: [2, -2, 2],
          opacity: [0.8, 0, 0.8],
        } : {}}
        transition={{
          duration: 0.2,
          repeat: Infinity,
          delay: 0.1,
        }}
      >
        {text}
      </motion.span>

      {/* Random glitch effect */}
      <motion.span
        className="absolute inset-0"
        style={{
          color: '#00ff00',
          mixBlendMode: 'screen',
        }}
        animate={isGlitching || continuous ? {
          opacity: [0, 0, 1, 0, 0],
          x: [0, -5, 5, -5, 0],
        } : {}}
        transition={{
          duration: 0.5,
          repeat: Infinity,
          times: [0, 0.2, 0.5, 0.8, 1],
        }}
      >
        {text.split('').map((char, i) => (
          <span
            key={i}
            style={{
              opacity: i % 3 === 0 ? 1 : 0,
            }}
          >
            {char}
          </span>
        ))}
      </motion.span>
    </motion.div>
  )
}