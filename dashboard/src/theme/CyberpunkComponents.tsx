import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from './ThemeProvider';

// Cyberpunk Button Component
interface CyberButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'neon' | 'holographic';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  icon?: React.ReactNode;
  className?: string;
}

export const CyberButton: React.FC<CyberButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  onClick,
  disabled = false,
  loading = false,
  fullWidth = false,
  icon,
  className = ''
}) => {
  const { theme, playSound } = useTheme() as any;
  
  const handleClick = () => {
    if (!disabled && !loading) {
      playSound?.('click');
      onClick?.();
    }
  };
  
  const handleHover = () => {
    if (!disabled && !loading) {
      playSound?.('hover');
    }
  };
  
  const sizeClasses = {
    xs: 'px-2 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
    xl: 'px-8 py-4 text-xl'
  };
  
  const variantClasses = {
    primary: 'cyber-button-primary',
    secondary: 'cyber-button-secondary',
    ghost: 'cyber-button-ghost',
    neon: 'cyber-button-neon',
    holographic: 'cyber-button-holographic'
  };
  
  return (
    <motion.button
      className={`cyber-button ${variantClasses[variant]} ${sizeClasses[size]} ${fullWidth ? 'w-full' : ''} ${className}`}
      onClick={handleClick}
      onHoverStart={handleHover}
      disabled={disabled || loading}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      animate={loading ? { opacity: [1, 0.5, 1] } : {}}
      transition={{ duration: 0.2 }}
    >
      {loading && <span className="loading-spinner" />}
      {icon && <span className="button-icon">{icon}</span>}
      <span>{children}</span>
    </motion.button>
  );
};

// Cyberpunk Card Component
interface CyberCardProps {
  children: React.ReactNode;
  variant?: 'default' | 'glass' | 'neon' | 'holographic' | 'matrix';
  elevation?: 'flat' | 'raised' | 'floating' | 'hovering';
  onClick?: () => void;
  className?: string;
  animated?: boolean;
}

export const CyberCard: React.FC<CyberCardProps> = ({
  children,
  variant = 'default',
  elevation = 'raised',
  onClick,
  className = '',
  animated = true
}) => {
  const variantClasses = {
    default: 'cyber-card',
    glass: 'cyber-card-glass',
    neon: 'cyber-card-neon',
    holographic: 'cyber-card-holographic',
    matrix: 'cyber-card-matrix'
  };
  
  const elevationStyles = {
    flat: { boxShadow: 'none' },
    raised: { boxShadow: '0 4px 6px rgba(0, 255, 255, 0.1)' },
    floating: { boxShadow: '0 10px 30px rgba(0, 255, 255, 0.2)' },
    hovering: { boxShadow: '0 20px 40px rgba(0, 255, 255, 0.3)' }
  };
  
  return (
    <motion.div
      className={`${variantClasses[variant]} ${className}`}
      style={elevationStyles[elevation]}
      onClick={onClick}
      whileHover={animated ? { y: -5, boxShadow: '0 20px 40px rgba(0, 255, 255, 0.4)' } : {}}
      transition={{ duration: 0.3 }}
    >
      {children}
    </motion.div>
  );
};

// Cyberpunk Input Component
interface CyberInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'password' | 'email' | 'number';
  variant?: 'default' | 'neon' | 'glass' | 'minimal' | 'cyber';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  error?: string;
  disabled?: boolean;
  className?: string;
}

export const CyberInput: React.FC<CyberInputProps> = ({
  value,
  onChange,
  placeholder,
  type = 'text',
  variant = 'default',
  size = 'md',
  icon,
  error,
  disabled = false,
  className = ''
}) => {
  const { playSound } = useTheme() as any;
  
  const handleFocus = () => {
    playSound?.('hover');
  };
  
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };
  
  const variantClasses = {
    default: 'cyber-input',
    neon: 'cyber-input-neon',
    glass: 'cyber-input-glass',
    minimal: 'cyber-input-minimal',
    cyber: 'cyber-input-cyber'
  };
  
  return (
    <div className={`cyber-input-wrapper ${className}`}>
      <div className="input-container">
        {icon && <span className="input-icon">{icon}</span>}
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          onFocus={handleFocus}
          className={`${variantClasses[variant]} ${sizeClasses[size]} ${error ? 'error' : ''}`}
        />
      </div>
      {error && (
        <motion.span
          className="error-message"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {error}
        </motion.span>
      )}
    </div>
  );
};

// Cyberpunk Modal Component
interface CyberModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  variant?: 'default' | 'glass' | 'holographic' | 'fullscreen';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  className?: string;
}

export const CyberModal: React.FC<CyberModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  variant = 'default',
  size = 'md',
  className = ''
}) => {
  const { playSound } = useTheme() as any;
  
  const handleClose = () => {
    playSound?.('click');
    onClose();
  };
  
  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-full'
  };
  
  const variantClasses = {
    default: 'cyber-modal',
    glass: 'cyber-modal-glass',
    holographic: 'cyber-modal-holographic',
    fullscreen: 'cyber-modal-fullscreen'
  };
  
  if (!isOpen) return null;
  
  return (
    <motion.div
      className="modal-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={handleClose}
    >
      <motion.div
        className={`${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <div className="modal-header">
            <h2 className="modal-title">{title}</h2>
            <button className="modal-close" onClick={handleClose}>
              ×
            </button>
          </div>
        )}
        <div className="modal-content">
          {children}
        </div>
      </motion.div>
    </motion.div>
  );
};

// Cyberpunk Notification Component
interface CyberNotificationProps {
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error' | 'cyber';
  position?: 'top' | 'bottom' | 'topRight' | 'topLeft' | 'center';
  duration?: number;
  onClose?: () => void;
}

export const CyberNotification: React.FC<CyberNotificationProps> = ({
  message,
  type = 'info',
  position = 'topRight',
  duration = 3000,
  onClose
}) => {
  const { playSound } = useTheme() as any;
  
  React.useEffect(() => {
    playSound?.('notification');
    if (duration > 0) {
      const timer = setTimeout(() => {
        onClose?.();
      }, duration);
      return () => clearTimeout(timer);
    }
  }, []);
  
  const typeClasses = {
    info: 'notification-info',
    success: 'notification-success',
    warning: 'notification-warning',
    error: 'notification-error',
    cyber: 'notification-cyber'
  };
  
  const positionClasses = {
    top: 'top-4 left-1/2 -translate-x-1/2',
    bottom: 'bottom-4 left-1/2 -translate-x-1/2',
    topRight: 'top-4 right-4',
    topLeft: 'top-4 left-4',
    center: 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2'
  };
  
  return (
    <motion.div
      className={`cyber-notification ${typeClasses[type]} ${positionClasses[position]}`}
      initial={{ opacity: 0, scale: 0.8, y: -20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.8, y: -20 }}
    >
      <span className="notification-icon">
        {type === 'success' && '✓'}
        {type === 'error' && '✗'}
        {type === 'warning' && '⚠'}
        {type === 'info' && 'ℹ'}
        {type === 'cyber' && '◈'}
      </span>
      <span className="notification-message">{message}</span>
      {onClose && (
        <button className="notification-close" onClick={onClose}>
          ×
        </button>
      )}
    </motion.div>
  );
};

// Cyberpunk Progress Bar Component
interface CyberProgressProps {
  value: number;
  max?: number;
  variant?: 'default' | 'neon' | 'gradient' | 'pulse';
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const CyberProgress: React.FC<CyberProgressProps> = ({
  value,
  max = 100,
  variant = 'default',
  showLabel = true,
  size = 'md',
  className = ''
}) => {
  const percentage = (value / max) * 100;
  
  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-4'
  };
  
  const variantClasses = {
    default: 'bg-cyan-500',
    neon: 'bg-gradient-to-r from-cyan-500 to-purple-500',
    gradient: 'bg-gradient-to-r from-green-500 via-cyan-500 to-purple-500',
    pulse: 'bg-cyan-500 animate-pulse'
  };
  
  return (
    <div className={`cyber-progress ${className}`}>
      <div className={`progress-bar ${sizeClasses[size]}`}>
        <motion.div
          className={`progress-fill ${variantClasses[variant]}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      {showLabel && (
        <span className="progress-label">{Math.round(percentage)}%</span>
      )}
    </div>
  );
};

// Export all components
export default {
  CyberButton,
  CyberCard,
  CyberInput,
  CyberModal,
  CyberNotification,
  CyberProgress
};