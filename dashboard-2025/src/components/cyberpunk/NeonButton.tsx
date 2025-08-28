'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface NeonButtonProps {
  children: React.ReactNode
  onClick?: () => void
  color?: 'cyan' | 'pink' | 'purple' | 'green' | 'yellow'
  disabled?: boolean
  loading?: boolean
  size?: 'sm' | 'md' | 'lg'
  fullWidth?: boolean
}

export function NeonButton({
  children,
  onClick,
  color = 'cyan',
  disabled = false,
  loading = false,
  size = 'md',
  fullWidth = false
}: NeonButtonProps) {
  const colorMap = {
    cyan: '#00ffff',
    pink: '#ff00ff',
    purple: '#9d00ff',
    green: '#00ff88',
    yellow: '#ffff00'
  }

  const sizeMap = {
    sm: 'px-4 py-2 text-xs',
    md: 'px-6 py-3 text-sm',
    lg: 'px-8 py-4 text-base'
  }

  const neonColor = colorMap[color]

  return (
    <motion.button
      className={`
        relative overflow-hidden
        font-orbitron font-bold uppercase tracking-wider
        border-2 backdrop-blur-sm
        transition-all duration-300 ease-out
        ${sizeMap[size]}
        ${fullWidth ? 'w-full' : ''}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
      style={{
        color: neonColor,
        borderColor: neonColor,
        background: `linear-gradient(45deg, transparent 30%, ${neonColor}10 50%, transparent 70%)`,
        clipPath: 'polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px))',
        textShadow: `0 0 10px ${neonColor}, 0 0 20px ${neonColor}`,
      }}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled ? {
        scale: 1.02,
        boxShadow: `0 0 20px ${neonColor}, 0 0 40px ${neonColor}, inset 0 0 20px ${neonColor}30`,
      } : {}}
      whileTap={!disabled ? { scale: 0.98 } : {}}
    >
      {/* Scanning light effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30"
        initial={{ x: '-100%' }}
        whileHover={{ x: '100%' }}
        transition={{ duration: 0.5 }}
      />

      {/* Button content */}
      <span className="relative z-10 flex items-center justify-center gap-2">
        {loading && (
          <motion.div
            className="w-4 h-4 border-2 border-t-transparent rounded-full"
            style={{ borderColor: `${neonColor} transparent ${neonColor} ${neonColor}` }}
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          />
        )}
        {children}
      </span>

      {/* Corner accents */}
      <div
        className="absolute top-0 left-0 w-2 h-2"
        style={{ borderTop: `2px solid ${neonColor}`, borderLeft: `2px solid ${neonColor}` }}
      />
      <div
        className="absolute top-0 right-0 w-2 h-2"
        style={{ borderTop: `2px solid ${neonColor}`, borderRight: `2px solid ${neonColor}` }}
      />
      <div
        className="absolute bottom-0 left-0 w-2 h-2"
        style={{ borderBottom: `2px solid ${neonColor}`, borderLeft: `2px solid ${neonColor}` }}
      />
      <div
        className="absolute bottom-0 right-0 w-2 h-2"
        style={{ borderBottom: `2px solid ${neonColor}`, borderRight: `2px solid ${neonColor}` }}
      />
    </motion.button>
  )
}