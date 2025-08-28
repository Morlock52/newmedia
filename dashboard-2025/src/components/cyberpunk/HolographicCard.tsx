'use client'

import React, { useRef, useEffect } from 'react'
import { motion, useMotionValue, useTransform } from 'framer-motion'

interface HolographicCardProps {
  children: React.ReactNode
  className?: string
  glowColor?: string
  title?: string
  subtitle?: string
}

export function HolographicCard({
  children,
  className = '',
  glowColor = '#00ffff',
  title,
  subtitle
}: HolographicCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)

  const rotateX = useTransform(mouseY, [-0.5, 0.5], [5, -5])
  const rotateY = useTransform(mouseX, [-0.5, 0.5], [-5, 5])

  useEffect(() => {
    const card = cardRef.current
    if (!card) return

    const handleMouseMove = (e: MouseEvent) => {
      const rect = card.getBoundingClientRect()
      const x = (e.clientX - rect.left) / rect.width - 0.5
      const y = (e.clientY - rect.top) / rect.height - 0.5
      mouseX.set(x)
      mouseY.set(y)
    }

    const handleMouseLeave = () => {
      mouseX.set(0)
      mouseY.set(0)
    }

    card.addEventListener('mousemove', handleMouseMove)
    card.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      card.removeEventListener('mousemove', handleMouseMove)
      card.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [mouseX, mouseY])

  return (
    <motion.div
      ref={cardRef}
      className={`relative p-6 rounded-lg overflow-hidden ${className}`}
      style={{
        background: `linear-gradient(135deg, 
          rgba(255, 0, 255, 0.05),
          rgba(0, 255, 255, 0.05),
          rgba(255, 0, 255, 0.05)
        )`,
        border: `1px solid ${glowColor}30`,
        backdropFilter: 'blur(10px)',
        transformStyle: 'preserve-3d',
        perspective: 1000,
      }}
      animate={{
        boxShadow: [
          `0 0 20px ${glowColor}20`,
          `0 0 30px ${glowColor}30`,
          `0 0 20px ${glowColor}20`,
        ],
      }}
      transition={{
        duration: 3,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
      whileHover={{
        scale: 1.02,
        transition: { duration: 0.2 },
      }}
    >
      {/* Holographic gradient overlay */}
      <motion.div
        className="absolute inset-0 opacity-30"
        style={{
          background: `linear-gradient(
            45deg,
            transparent,
            rgba(255, 0, 255, 0.1),
            transparent,
            rgba(0, 255, 255, 0.1),
            transparent
          )`,
          backgroundSize: '200% 200%',
        }}
        animate={{
          backgroundPosition: ['0% 0%', '100% 100%'],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      {/* Scanning line effect */}
      <motion.div
        className="absolute inset-x-0 h-px"
        style={{
          background: `linear-gradient(90deg, transparent, ${glowColor}, transparent)`,
        }}
        initial={{ top: '-1px' }}
        animate={{ top: '100%' }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      {/* Corner decorations */}
      <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2" style={{ borderColor: glowColor }} />
      <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2" style={{ borderColor: glowColor }} />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2" style={{ borderColor: glowColor }} />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2" style={{ borderColor: glowColor }} />

      {/* Header */}
      {(title || subtitle) && (
        <div className="mb-4 pb-4 border-b" style={{ borderColor: `${glowColor}30` }}>
          {title && (
            <h3
              className="text-xl font-orbitron font-bold uppercase tracking-wider mb-1"
              style={{
                color: glowColor,
                textShadow: `0 0 10px ${glowColor}`,
              }}
            >
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="text-sm opacity-70" style={{ color: glowColor }}>
              {subtitle}
            </p>
          )}
        </div>
      )}

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>

      {/* Data stream effect */}
      <div className="absolute bottom-0 left-0 w-full h-1 overflow-hidden">
        <motion.div
          className="h-full"
          style={{
            background: `linear-gradient(90deg, 
              transparent, 
              ${glowColor}, 
              transparent
            )`,
            width: '50%',
          }}
          animate={{
            x: ['-100%', '200%'],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'linear',
          }}
        />
      </div>
    </motion.div>
  )
}