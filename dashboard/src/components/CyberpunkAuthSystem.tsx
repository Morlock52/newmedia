import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './CyberpunkAuthSystem.css';

interface AuthProps {
  onSuccess: (user: any) => void;
  onFailure: (error: string) => void;
}

const CyberpunkAuthSystem: React.FC<AuthProps> = ({ onSuccess, onFailure }) => {
  const [authMode, setAuthMode] = useState<'login' | 'register' | 'biometric'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [mfaCode, setMfaCode] = useState('');
  const [showMFA, setShowMFA] = useState(false);
  const [qrCode, setQrCode] = useState('');
  const [terminalMode, setTerminalMode] = useState(false);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState<string[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (authMode === 'biometric') {
      startBiometricScan();
    }
    
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [authMode]);

  const startBiometricScan = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setTimeout(() => simulateFaceScan(), 1000);
      }
    } catch (error) {
      console.error('Camera access denied:', error);
      setAuthMode('login');
    }
  };

  const simulateFaceScan = () => {
    setIsScanning(true);
    
    if (canvasRef.current && videoRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d')!;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw scan lines and effects
      const drawScanEffect = () => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Add scan lines
        context.strokeStyle = '#00ffff';
        context.lineWidth = 2;
        
        for (let i = 0; i < 5; i++) {
          const y = (Date.now() / 10 + i * 100) % canvas.height;
          context.beginPath();
          context.moveTo(0, y);
          context.lineTo(canvas.width, y);
          context.stroke();
        }
        
        // Add face detection box (simulated)
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const boxSize = 200;
        
        context.strokeStyle = '#ff00ff';
        context.lineWidth = 3;
        context.strokeRect(
          centerX - boxSize / 2,
          centerY - boxSize / 2,
          boxSize,
          boxSize
        );
        
        // Add corner markers
        const cornerLength = 30;
        context.strokeStyle = '#00ff00';
        context.lineWidth = 4;
        
        // Top-left
        context.beginPath();
        context.moveTo(centerX - boxSize / 2, centerY - boxSize / 2 + cornerLength);
        context.lineTo(centerX - boxSize / 2, centerY - boxSize / 2);
        context.lineTo(centerX - boxSize / 2 + cornerLength, centerY - boxSize / 2);
        context.stroke();
        
        // Top-right
        context.beginPath();
        context.moveTo(centerX + boxSize / 2 - cornerLength, centerY - boxSize / 2);
        context.lineTo(centerX + boxSize / 2, centerY - boxSize / 2);
        context.lineTo(centerX + boxSize / 2, centerY - boxSize / 2 + cornerLength);
        context.stroke();
        
        // Bottom-left
        context.beginPath();
        context.moveTo(centerX - boxSize / 2, centerY + boxSize / 2 - cornerLength);
        context.lineTo(centerX - boxSize / 2, centerY + boxSize / 2);
        context.lineTo(centerX - boxSize / 2 + cornerLength, centerY + boxSize / 2);
        context.stroke();
        
        // Bottom-right
        context.beginPath();
        context.moveTo(centerX + boxSize / 2 - cornerLength, centerY + boxSize / 2);
        context.lineTo(centerX + boxSize / 2, centerY + boxSize / 2);
        context.lineTo(centerX + boxSize / 2, centerY + boxSize / 2 - cornerLength);
        context.stroke();
        
        if (isScanning) {
          requestAnimationFrame(drawScanEffect);
        }
      };
      
      drawScanEffect();
      
      // Simulate scan completion
      setTimeout(() => {
        setIsScanning(false);
        handleBiometricAuth();
      }, 3000);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      
      const data = await response.json();
      
      if (data.requiresMFA) {
        setShowMFA(true);
        setQrCode(data.qrCode);
      } else if (data.success) {
        onSuccess(data.user);
      } else {
        onFailure(data.error || 'Authentication failed');
      }
    } catch (error) {
      onFailure('Network error');
    }
  };

  const handleMFASubmit = async () => {
    try {
      const response = await fetch('/api/auth/mfa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, mfaCode })
      });
      
      const data = await response.json();
      
      if (data.success) {
        onSuccess(data.user);
      } else {
        onFailure('Invalid MFA code');
      }
    } catch (error) {
      onFailure('MFA verification failed');
    }
  };

  const handleBiometricAuth = async () => {
    // Simulate biometric authentication
    try {
      const response = await fetch('/api/auth/biometric', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          biometricData: 'simulated_face_data',
          timestamp: Date.now()
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        onSuccess(data.user);
      } else {
        onFailure('Biometric authentication failed');
        setAuthMode('login');
      }
    } catch (error) {
      onFailure('Biometric scan error');
      setAuthMode('login');
    }
  };

  const handleTerminalCommand = (command: string) => {
    const parts = command.split(' ');
    const cmd = parts[0].toLowerCase();
    
    setTerminalHistory(prev => [...prev, `> ${command}`]);
    
    switch (cmd) {
      case 'login':
        if (parts.length === 3) {
          setUsername(parts[1]);
          setPassword(parts[2]);
          setTerminalHistory(prev => [...prev, 'Authenticating...']);
          handleLogin(new Event('submit') as any);
        } else {
          setTerminalHistory(prev => [...prev, 'Usage: login <username> <password>']);
        }
        break;
        
      case 'register':
        if (parts.length === 4) {
          setUsername(parts[1]);
          setEmail(parts[2]);
          setPassword(parts[3]);
          setTerminalHistory(prev => [...prev, 'Creating account...']);
        } else {
          setTerminalHistory(prev => [...prev, 'Usage: register <username> <email> <password>']);
        }
        break;
        
      case 'biometric':
        setAuthMode('biometric');
        setTerminalHistory(prev => [...prev, 'Switching to biometric mode...']);
        break;
        
      case 'clear':
        setTerminalHistory([]);
        break;
        
      case 'help':
        setTerminalHistory(prev => [...prev, 
          'Available commands:',
          '  login <username> <password> - Authenticate user',
          '  register <username> <email> <password> - Create account',
          '  biometric - Switch to biometric authentication',
          '  clear - Clear terminal',
          '  exit - Exit terminal mode'
        ]);
        break;
        
      case 'exit':
        setTerminalMode(false);
        break;
        
      default:
        setTerminalHistory(prev => [...prev, `Command not found: ${cmd}`]);
    }
    
    setTerminalInput('');
  };

  return (
    <div className="cyberpunk-auth-system">
      <div className="auth-container">
        {/* Circuit pattern background */}
        <div className="circuit-pattern"></div>
        
        {/* Glitch effect overlay */}
        <div className="glitch-overlay"></div>
        
        <div className="auth-header">
          <h1 className="auth-title glitch-text" data-text="NEXUS ACCESS">
            NEXUS ACCESS
          </h1>
          <div className="auth-subtitle">Media Server Authentication System</div>
        </div>

        <div className="auth-mode-selector">
          <button
            className={`mode-btn ${authMode === 'login' ? 'active' : ''}`}
            onClick={() => setAuthMode('login')}
          >
            <span className="mode-icon">🔐</span>
            Login
          </button>
          <button
            className={`mode-btn ${authMode === 'register' ? 'active' : ''}`}
            onClick={() => setAuthMode('register')}
          >
            <span className="mode-icon">📝</span>
            Register
          </button>
          <button
            className={`mode-btn ${authMode === 'biometric' ? 'active' : ''}`}
            onClick={() => setAuthMode('biometric')}
          >
            <span className="mode-icon">👤</span>
            Biometric
          </button>
          <button
            className={`mode-btn ${terminalMode ? 'active' : ''}`}
            onClick={() => setTerminalMode(!terminalMode)}
          >
            <span className="mode-icon">💻</span>
            Terminal
          </button>
        </div>

        <AnimatePresence mode="wait">
          {terminalMode ? (
            <motion.div
              key="terminal"
              className="terminal-mode"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <div className="terminal-window" ref={terminalRef}>
                <div className="terminal-header">
                  <span>NEXUS TERMINAL v2.0.25</span>
                </div>
                <div className="terminal-body">
                  {terminalHistory.map((line, index) => (
                    <div key={index} className="terminal-line">
                      {line}
                    </div>
                  ))}
                  <div className="terminal-input-line">
                    <span className="terminal-prompt">nexus@auth:~$ </span>
                    <input
                      type="text"
                      className="terminal-input"
                      value={terminalInput}
                      onChange={(e) => setTerminalInput(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          handleTerminalCommand(terminalInput);
                        }
                      }}
                      autoFocus
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          ) : authMode === 'biometric' ? (
            <motion.div
              key="biometric"
              className="biometric-mode"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <div className="biometric-scanner">
                <video
                  ref={videoRef}
                  className="biometric-video"
                  autoPlay
                  playsInline
                  muted
                />
                <canvas
                  ref={canvasRef}
                  className="biometric-canvas"
                />
                <div className="scanner-overlay">
                  {isScanning && (
                    <>
                      <div className="scan-line"></div>
                      <div className="scan-status">
                        <div className="status-text">SCANNING...</div>
                        <div className="status-progress">
                          <div className="progress-bar"></div>
                        </div>
                      </div>
                      <div className="biometric-data">
                        <div className="data-item">
                          <span className="data-label">IRIS PATTERN:</span>
                          <span className="data-value">ANALYZING...</span>
                        </div>
                        <div className="data-item">
                          <span className="data-label">FACIAL NODES:</span>
                          <span className="data-value">47/50</span>
                        </div>
                        <div className="data-item">
                          <span className="data-label">CONFIDENCE:</span>
                          <span className="data-value">87.3%</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
              <div className="biometric-instructions">
                <p>Position your face within the scanner area</p>
                <p>Authentication will begin automatically</p>
              </div>
            </motion.div>
          ) : showMFA ? (
            <motion.div
              key="mfa"
              className="mfa-mode"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <div className="mfa-container">
                <h2 className="mfa-title">Two-Factor Authentication</h2>
                
                {qrCode && (
                  <div className="qr-code-container">
                    <div className="qr-code" dangerouslySetInnerHTML={{ __html: qrCode }} />
                    <p>Scan with your authenticator app</p>
                  </div>
                )}
                
                <div className="mfa-input-group">
                  <input
                    type="text"
                    className="mfa-input"
                    placeholder="Enter 6-digit code"
                    value={mfaCode}
                    onChange={(e) => setMfaCode(e.target.value)}
                    maxLength={6}
                  />
                </div>
                
                <button
                  className="mfa-submit-btn"
                  onClick={handleMFASubmit}
                  disabled={mfaCode.length !== 6}
                >
                  Verify
                </button>
                
                <button
                  className="mfa-cancel-btn"
                  onClick={() => setShowMFA(false)}
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          ) : (
            <motion.form
              key="form"
              className="auth-form"
              onSubmit={handleLogin}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <div className="input-group">
                <div className="input-icon">👤</div>
                <input
                  type="text"
                  className="cyberpunk-input"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
                <div className="input-glow"></div>
              </div>
              
              {authMode === 'register' && (
                <div className="input-group">
                  <div className="input-icon">📧</div>
                  <input
                    type="email"
                    className="cyberpunk-input"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                  <div className="input-glow"></div>
                </div>
              )}
              
              <div className="input-group">
                <div className="input-icon">🔑</div>
                <input
                  type="password"
                  className="cyberpunk-input"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <div className="input-glow"></div>
              </div>
              
              <button type="submit" className="auth-submit-btn">
                <span className="btn-text">
                  {authMode === 'login' ? 'ACCESS SYSTEM' : 'CREATE ACCOUNT'}
                </span>
                <div className="btn-glow"></div>
              </button>
              
              <div className="auth-options">
                <label className="remember-me">
                  <input type="checkbox" />
                  <span className="checkbox-custom"></span>
                  <span>Remember Me</span>
                </label>
                
                <a href="#" className="forgot-password">
                  Forgot Password?
                </a>
              </div>
            </motion.form>
          )}
        </AnimatePresence>

        {/* Animated circuit lines */}
        <svg className="circuit-svg" width="100%" height="100%">
          <defs>
            <linearGradient id="circuit-gradient">
              <stop offset="0%" stopColor="#00ffff" stopOpacity="0" />
              <stop offset="50%" stopColor="#00ffff" stopOpacity="1" />
              <stop offset="100%" stopColor="#00ffff" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[...Array(5)].map((_, i) => (
            <line
              key={i}
              x1={`${Math.random() * 100}%`}
              y1="0"
              x2={`${Math.random() * 100}%`}
              y2="100%"
              stroke="url(#circuit-gradient)"
              strokeWidth="1"
              className="circuit-line"
              style={{ animationDelay: `${i * 0.5}s` }}
            />
          ))}
        </svg>
      </div>
    </div>
  );
};

export default CyberpunkAuthSystem;