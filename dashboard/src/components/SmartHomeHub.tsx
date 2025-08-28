import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface SmartDevice {
  id: string;
  name: string;
  type: 'light' | 'thermostat' | 'camera' | 'speaker' | 'lock' | 'sensor' | 'tv' | 'appliance';
  status: 'online' | 'offline' | 'updating' | 'error';
  room: string;
  battery?: number;
  value: any;
  icon: string;
  controls: DeviceControl[];
  lastUpdated: Date;
}

interface DeviceControl {
  id: string;
  type: 'toggle' | 'slider' | 'color' | 'preset';
  label: string;
  value: any;
  min?: number;
  max?: number;
  options?: string[];
}

interface AutomationRule {
  id: string;
  name: string;
  trigger: {
    type: 'time' | 'device' | 'sensor' | 'location';
    condition: string;
  };
  actions: Array<{
    deviceId: string;
    action: string;
    value: any;
  }>;
  enabled: boolean;
  lastTriggered?: Date;
}

interface EnergyData {
  current: number;
  today: number;
  thisMonth: number;
  cost: number;
  trend: 'up' | 'down' | 'stable';
}

const SmartHomeHub: React.FC = () => {
  const [devices, setDevices] = useState<SmartDevice[]>([]);
  const [selectedRoom, setSelectedRoom] = useState('all');
  const [selectedDevice, setSelectedDevice] = useState<SmartDevice | null>(null);
  const [automationRules, setAutomationRules] = useState<AutomationRule[]>([]);
  const [energyData, setEnergyData] = useState<EnergyData>({
    current: 0,
    today: 0,
    thisMonth: 0,
    cost: 0,
    trend: 'stable'
  });
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [voiceCommand, setVoiceCommand] = useState('');
  const [weatherData, setWeatherData] = useState({
    temperature: 72,
    humidity: 45,
    condition: 'Clear',
    icon: '☀️'
  });
  const [securityMode, setSecurityMode] = useState<'home' | 'away' | 'night'>('home');
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const speechRef = useRef<SpeechRecognition | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializeSmartHome();
    setupWebSocket();
    setupVoiceRecognition();
    startEnergyVisualization();
    
    // Simulate real-time updates
    const interval = setInterval(updateDeviceStates, 5000);
    return () => clearInterval(interval);
  }, []);

  const initializeSmartHome = () => {
    const smartDevices: SmartDevice[] = [
      {
        id: 'living-room-lights',
        name: 'Living Room Lights',
        type: 'light',
        status: 'online',
        room: 'Living Room',
        value: { brightness: 75, color: '#FFFF00', on: true },
        icon: '💡',
        controls: [
          { id: 'power', type: 'toggle', label: 'Power', value: true },
          { id: 'brightness', type: 'slider', label: 'Brightness', value: 75, min: 0, max: 100 },
          { id: 'color', type: 'color', label: 'Color', value: '#FFFF00' }
        ],
        lastUpdated: new Date()
      },
      {
        id: 'main-thermostat',
        name: 'Main Thermostat',
        type: 'thermostat',
        status: 'online',
        room: 'Living Room',
        value: { temperature: 72, targetTemp: 74, mode: 'heat' },
        icon: '🌡️',
        controls: [
          { id: 'target', type: 'slider', label: 'Target Temp', value: 74, min: 60, max: 85 },
          { id: 'mode', type: 'preset', label: 'Mode', value: 'heat', options: ['heat', 'cool', 'auto', 'off'] }
        ],
        lastUpdated: new Date()
      },
      {
        id: 'front-door-camera',
        name: 'Front Door Camera',
        type: 'camera',
        status: 'online',
        room: 'Entrance',
        battery: 85,
        value: { recording: true, motionDetection: true, nightVision: false },
        icon: '📹',
        controls: [
          { id: 'recording', type: 'toggle', label: 'Recording', value: true },
          { id: 'motion', type: 'toggle', label: 'Motion Detection', value: true },
          { id: 'nightvision', type: 'toggle', label: 'Night Vision', value: false }
        ],
        lastUpdated: new Date()
      },
      {
        id: 'kitchen-speaker',
        name: 'Kitchen Speaker',
        type: 'speaker',
        status: 'online',
        room: 'Kitchen',
        value: { volume: 45, playing: true, source: 'Spotify' },
        icon: '🔊',
        controls: [
          { id: 'volume', type: 'slider', label: 'Volume', value: 45, min: 0, max: 100 },
          { id: 'source', type: 'preset', label: 'Source', value: 'Spotify', options: ['Spotify', 'Apple Music', 'Radio', 'Bluetooth'] }
        ],
        lastUpdated: new Date()
      },
      {
        id: 'front-door-lock',
        name: 'Front Door Lock',
        type: 'lock',
        status: 'online',
        room: 'Entrance',
        battery: 72,
        value: { locked: true, autoLock: true },
        icon: '🔒',
        controls: [
          { id: 'lock', type: 'toggle', label: 'Lock/Unlock', value: true },
          { id: 'autolock', type: 'toggle', label: 'Auto Lock', value: true }
        ],
        lastUpdated: new Date()
      },
      {
        id: 'bedroom-tv',
        name: 'Bedroom TV',
        type: 'tv',
        status: 'offline',
        room: 'Bedroom',
        value: { on: false, channel: 105, volume: 25 },
        icon: '📺',
        controls: [
          { id: 'power', type: 'toggle', label: 'Power', value: false },
          { id: 'volume', type: 'slider', label: 'Volume', value: 25, min: 0, max: 100 }
        ],
        lastUpdated: new Date(Date.now() - 3600000)
      },
      {
        id: 'garage-door',
        name: 'Garage Door',
        type: 'appliance',
        status: 'online',
        room: 'Garage',
        value: { open: false, autoClose: true },
        icon: '🏠',
        controls: [
          { id: 'toggle', type: 'toggle', label: 'Open/Close', value: false },
          { id: 'autoclose', type: 'toggle', label: 'Auto Close', value: true }
        ],
        lastUpdated: new Date()
      },
      {
        id: 'motion-sensor',
        name: 'Living Room Motion',
        type: 'sensor',
        status: 'online',
        room: 'Living Room',
        battery: 91,
        value: { motion: false, lastMotion: new Date(Date.now() - 1800000) },
        icon: '👁️',
        controls: [],
        lastUpdated: new Date()
      }
    ];
    
    const rules: AutomationRule[] = [
      {
        id: 'evening-lights',
        name: 'Evening Lights',
        trigger: { type: 'time', condition: '18:00' },
        actions: [
          { deviceId: 'living-room-lights', action: 'setBrightness', value: 80 },
          { deviceId: 'living-room-lights', action: 'setColor', value: '#FF8800' }
        ],
        enabled: true,
        lastTriggered: new Date(Date.now() - 86400000)
      },
      {
        id: 'away-mode',
        name: 'Away Mode Security',
        trigger: { type: 'location', condition: 'away' },
        actions: [
          { deviceId: 'front-door-lock', action: 'lock', value: true },
          { deviceId: 'front-door-camera', action: 'enableRecording', value: true },
          { deviceId: 'main-thermostat', action: 'setTemp', value: 68 }
        ],
        enabled: true
      },
      {
        id: 'motion-lights',
        name: 'Motion Activated Lights',
        trigger: { type: 'sensor', condition: 'motion detected' },
        actions: [
          { deviceId: 'living-room-lights', action: 'turnOn', value: true }
        ],
        enabled: true,
        lastTriggered: new Date(Date.now() - 1800000)
      }
    ];
    
    setDevices(smartDevices);
    setAutomationRules(rules);
    
    // Initialize energy data
    setEnergyData({
      current: 2.4,
      today: 28.5,
      thisMonth: 845.2,
      cost: 127.80,
      trend: 'down'
    });
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/smart-home');
      
      wsRef.current.onmessage = (event) => {
        const update = JSON.parse(event.data);
        handleDeviceUpdate(update);
      };
    } catch (error) {
      console.warn('Smart home WebSocket not available');
    }
  };

  const setupVoiceRecognition = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      speechRef.current = new SpeechRecognition();
      
      speechRef.current.continuous = false;
      speechRef.current.interimResults = false;
      speechRef.current.lang = 'en-US';
      
      speechRef.current.onresult = (event: any) => {
        const command = event.results[0][0].transcript.toLowerCase();
        processVoiceCommand(command);
      };
      
      speechRef.current.onend = () => {
        setIsVoiceMode(false);
      };
    }
  };

  const handleDeviceUpdate = (update: any) => {
    setDevices(prev => prev.map(device => 
      device.id === update.deviceId 
        ? { ...device, value: { ...device.value, ...update.value }, lastUpdated: new Date() }
        : device
    ));
  };

  const updateDeviceStates = () => {
    setDevices(prev => prev.map(device => {
      const updates: any = {};
      
      // Simulate random state changes
      if (device.type === 'sensor') {
        updates.motion = Math.random() < 0.1;
        if (updates.motion) {
          updates.lastMotion = new Date();
        }
      } else if (device.type === 'thermostat') {
        updates.temperature = device.value.temperature + (Math.random() - 0.5) * 2;
      }
      
      // Simulate battery drain
      if (device.battery) {
        const newBattery = Math.max(0, device.battery - Math.random() * 0.1);
        return { ...device, battery: Math.round(newBattery), value: { ...device.value, ...updates } };
      }
      
      return Object.keys(updates).length > 0 ? { ...device, value: { ...device.value, ...updates } } : device;
    }));
    
    // Update energy data
    setEnergyData(prev => ({
      ...prev,
      current: Math.max(0, prev.current + (Math.random() - 0.5) * 0.5),
      today: prev.today + Math.random() * 0.1
    }));
  };

  const controlDevice = (deviceId: string, controlId: string, value: any) => {
    setDevices(prev => prev.map(device => {
      if (device.id === deviceId) {
        const updatedControls = device.controls.map(control => 
          control.id === controlId ? { ...control, value } : control
        );
        
        const updatedValue = { ...device.value, [controlId]: value };
        
        return {
          ...device,
          controls: updatedControls,
          value: updatedValue,
          lastUpdated: new Date()
        };
      }
      return device;
    }));
  };

  const processVoiceCommand = (command: string) => {
    setVoiceCommand(command);
    
    if (command.includes('turn on') || command.includes('turn off')) {
      const action = command.includes('turn on');
      
      if (command.includes('lights')) {
        const lightDevices = devices.filter(d => d.type === 'light');
        lightDevices.forEach(device => {
          controlDevice(device.id, 'power', action);
        });
      } else if (command.includes('tv')) {
        const tvDevices = devices.filter(d => d.type === 'tv');
        tvDevices.forEach(device => {
          controlDevice(device.id, 'power', action);
        });
      }
    } else if (command.includes('set temperature')) {
      const tempMatch = command.match(/(\d+)/g);
      if (tempMatch) {
        const temp = parseInt(tempMatch[0]);
        const thermostat = devices.find(d => d.type === 'thermostat');
        if (thermostat) {
          controlDevice(thermostat.id, 'target', temp);
        }
      }
    } else if (command.includes('lock') || command.includes('unlock')) {
      const shouldLock = command.includes('lock');
      const lockDevices = devices.filter(d => d.type === 'lock');
      lockDevices.forEach(device => {
        controlDevice(device.id, 'lock', shouldLock);
      });
    }
    
    setTimeout(() => setVoiceCommand(''), 3000);
  };

  const startVoiceCommand = () => {
    if (speechRef.current) {
      setIsVoiceMode(true);
      speechRef.current.start();
    }
  };

  const startEnergyVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const dataPoints: number[] = [];
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Add new data point
      if (dataPoints.length > 50) {
        dataPoints.shift();
      }
      dataPoints.push(energyData.current);
      
      // Draw energy consumption graph
      if (dataPoints.length > 1) {
        ctx.strokeStyle = '#00FFFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < dataPoints.length; i++) {
          const x = (i / (dataPoints.length - 1)) * canvas.width;
          const y = canvas.height - (dataPoints[i] / 5) * canvas.height;
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        
        ctx.stroke();
        
        // Add glow effect
        ctx.shadowColor = '#00FFFF';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      
      requestAnimationFrame(animate);
    };
    
    animate();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#00FF00';
      case 'offline': return '#FF0040';
      case 'updating': return '#FFFF00';
      case 'error': return '#FF6600';
      default: return '#666666';
    }
  };

  const getRoomDevices = () => {
    return selectedRoom === 'all' 
      ? devices 
      : devices.filter(device => device.room === selectedRoom);
  };

  const getRooms = () => {
    return ['all', ...new Set(devices.map(device => device.room))];
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#00FFFF',
      fontFamily: 'Orbitron, monospace',
      minHeight: '100vh',
      padding: '20px',
      position: 'relative'
    }}>
      {/* Smart Grid Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: 'radial-gradient(circle at 25% 25%, #00FFFF 1px, transparent 1px), radial-gradient(circle at 75% 75%, #FF00FF 1px, transparent 1px)',
        backgroundSize: '50px 50px',
        opacity: 0.1,
        pointerEvents: 'none'
      }} />

      {/* Header */}
      <motion.header
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
        }}
      >
        <div>
          <h1 style={{
            fontSize: '3rem',
            margin: 0,
            background: 'linear-gradient(45deg, #00FFFF, #FF00FF, #FFFF00)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 20px #00FFFF',
            animation: 'smartGlow 4s ease-in-out infinite alternate'
          }}>
            SMART HOME HUB
          </h1>
          <div style={{
            display: 'flex',
            gap: '20px',
            marginTop: '10px',
            fontSize: '0.9rem'
          }}>
            <span>Security: {securityMode.toUpperCase()}</span>
            <span>{weatherData.icon} {weatherData.temperature}°F</span>
            <span style={{ color: energyData.trend === 'down' ? '#00FF00' : energyData.trend === 'up' ? '#FF0040' : '#FFFF00' }}>
              Energy: {energyData.current.toFixed(1)}kW
            </span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <button
            onClick={startVoiceCommand}
            disabled={isVoiceMode}
            style={{
              padding: '12px 20px',
              background: isVoiceMode ? 'linear-gradient(45deg, #FF00FF, #FFFF00)' : 'rgba(255,0,255,0.2)',
              border: '2px solid #FF00FF',
              borderRadius: '8px',
              color: isVoiceMode ? '#000' : '#FF00FF',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            {isVoiceMode ? '🎤 LISTENING...' : '🎙️ VOICE CONTROL'}
          </button>
          
          <select
            value={securityMode}
            onChange={(e) => setSecurityMode(e.target.value as any)}
            style={{
              padding: '12px',
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #FFFF00',
              borderRadius: '8px',
              color: '#00FFFF',
              fontSize: '1rem'
            }}
          >
            <option value="home">Home Mode</option>
            <option value="away">Away Mode</option>
            <option value="night">Night Mode</option>
          </select>
        </div>
      </motion.header>

      {/* Voice Command Feedback */}
      <AnimatePresence>
        {voiceCommand && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            style={{
              position: 'fixed',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              background: 'linear-gradient(45deg, #FF00FF, #FFFF00)',
              color: '#000',
              padding: '20px 40px',
              borderRadius: '15px',
              fontSize: '1.2rem',
              fontWeight: 'bold',
              textAlign: 'center',
              zIndex: 1000,
              boxShadow: '0 10px 30px rgba(255,0,255,0.5)'
            }}
          >
            🎤 "{voiceCommand}"
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dashboard Overview */}
      <motion.section
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
        }}
      >
        {[
          { 
            label: 'Online Devices', 
            value: devices.filter(d => d.status === 'online').length, 
            total: devices.length,
            color: '#00FF00', 
            icon: '🟢' 
          },
          { 
            label: 'Energy Usage', 
            value: `${energyData.current.toFixed(1)}kW`, 
            total: `$${energyData.cost.toFixed(2)}/mo`,
            color: '#00FFFF', 
            icon: '⚡' 
          },
          { 
            label: 'Security Status', 
            value: securityMode.toUpperCase(), 
            total: 'All Secure',
            color: '#FFFF00', 
            icon: '🔒' 
          },
          { 
            label: 'Automations', 
            value: automationRules.filter(r => r.enabled).length, 
            total: `${automationRules.length} total`,
            color: '#FF00FF', 
            icon: '🤖' 
          }
        ].map((metric, index) => (
          <div
            key={metric.label}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: `2px solid ${metric.color}`,
              borderRadius: '15px',
              padding: '25px',
              textAlign: 'center',
              position: 'relative',
              overflow: 'hidden',
              boxShadow: `0 0 30px rgba(${metric.color === '#00FF00' ? '0,255,0' : metric.color === '#00FFFF' ? '0,255,255' : metric.color === '#FFFF00' ? '255,255,0' : '255,0,255'},0.3)`
            }}
          >
            <div style={{ fontSize: '3rem', marginBottom: '15px' }}>{metric.icon}</div>
            <div style={{
              fontSize: '2rem',
              fontWeight: 'bold',
              color: metric.color,
              marginBottom: '8px'
            }}>
              {metric.value}
            </div>
            <div style={{ fontSize: '0.9rem', opacity: 0.8, marginBottom: '5px' }}>{metric.label}</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>{metric.total}</div>
            
            {/* Animated accent */}
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '3px',
              background: `linear-gradient(90deg, transparent, ${metric.color}, transparent)`,
              animation: 'smartScan 4s linear infinite'
            }} />
          </div>
        ))}
      </motion.section>

      {/* Room Filter */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4 }}
        style={{
          marginBottom: '30px',
          display: 'flex',
          gap: '10px',
          flexWrap: 'wrap',
          zIndex: 10,
          position: 'relative'
        }}
      >
        {getRooms().map(room => (
          <button
            key={room}
            onClick={() => setSelectedRoom(room)}
            style={{
              padding: '10px 20px',
              background: selectedRoom === room 
                ? 'linear-gradient(45deg, #00FFFF, #FF00FF)' 
                : 'rgba(0,255,255,0.1)',
              border: `2px solid ${selectedRoom === room ? '#00FFFF' : 'rgba(0,255,255,0.3)'}`,
              borderRadius: '25px',
              color: selectedRoom === room ? '#000' : '#00FFFF',
              cursor: 'pointer',
              fontSize: '0.9rem',
              fontWeight: 'bold',
              textTransform: 'capitalize'
            }}
          >
            {room === 'all' ? '🏠 All Rooms' : `📍 ${room}`}
          </button>
        ))}
      </motion.div>

      {/* Device Grid */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '20px',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
        }}
      >
        <AnimatePresence>
          {getRoomDevices().map((device, index) => (
            <motion.div
              key={device.id}
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: -20 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02, y: -5 }}
              style={{
                background: 'rgba(0,0,0,0.8)',
                border: `2px solid ${getStatusColor(device.status)}`,
                borderRadius: '15px',
                padding: '25px',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden'
              }}
              onClick={() => setSelectedDevice(device)}
            >
              {/* Device Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <div style={{
                    fontSize: '3rem',
                    filter: `drop-shadow(0 0 10px ${getStatusColor(device.status)})`
                  }}>
                    {device.icon}
                  </div>
                  <div>
                    <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.2rem' }}>{device.name}</h3>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '5px' }}>
                      <span style={{
                        padding: '3px 8px',
                        background: getStatusColor(device.status),
                        color: device.status === 'online' ? '#000' : '#FFF',
                        borderRadius: '10px',
                        fontSize: '0.8rem',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {device.status}
                      </span>
                      <span style={{ fontSize: '0.9rem', opacity: 0.7 }}>📍 {device.room}</span>
                    </div>
                  </div>
                </div>
                
                {device.battery && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                    color: device.battery > 50 ? '#00FF00' : device.battery > 20 ? '#FFFF00' : '#FF0040'
                  }}>
                    <span style={{ fontSize: '1.2rem' }}>🔋</span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{device.battery}%</span>
                  </div>
                )}
              </div>
              
              {/* Device Controls */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                {device.controls.map(control => (
                  <div key={control.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.9rem', color: '#FFFF00' }}>{control.label}</span>
                    
                    {control.type === 'toggle' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          controlDevice(device.id, control.id, !control.value);
                        }}
                        style={{
                          padding: '6px 12px',
                          background: control.value 
                            ? 'linear-gradient(45deg, #00FF00, #00FFFF)' 
                            : 'rgba(255,0,64,0.2)',
                          border: 'none',
                          borderRadius: '6px',
                          color: control.value ? '#000' : '#FF0040',
                          fontWeight: 'bold',
                          cursor: 'pointer',
                          fontSize: '0.8rem'
                        }}
                      >
                        {control.value ? 'ON' : 'OFF'}
                      </button>
                    )}
                    
                    {control.type === 'slider' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, maxWidth: '150px' }}>
                        <input
                          type="range"
                          min={control.min}
                          max={control.max}
                          value={control.value}
                          onChange={(e) => {
                            e.stopPropagation();
                            controlDevice(device.id, control.id, parseInt(e.target.value));
                          }}
                          style={{
                            flex: 1,
                            accentColor: '#00FFFF'
                          }}
                        />
                        <span style={{ minWidth: '30px', fontSize: '0.9rem', fontWeight: 'bold', color: '#00FFFF' }}>
                          {control.value}
                        </span>
                      </div>
                    )}
                    
                    {control.type === 'preset' && (
                      <select
                        value={control.value}
                        onChange={(e) => {
                          e.stopPropagation();
                          controlDevice(device.id, control.id, e.target.value);
                        }}
                        style={{
                          padding: '6px 10px',
                          background: 'rgba(0,0,0,0.8)',
                          border: '1px solid #FF00FF',
                          borderRadius: '6px',
                          color: '#00FFFF',
                          fontSize: '0.8rem'
                        }}
                      >
                        {control.options?.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    )}
                    
                    {control.type === 'color' && (
                      <input
                        type="color"
                        value={control.value}
                        onChange={(e) => {
                          e.stopPropagation();
                          controlDevice(device.id, control.id, e.target.value);
                        }}
                        style={{
                          width: '40px',
                          height: '30px',
                          border: '2px solid #FF00FF',
                          borderRadius: '6px',
                          cursor: 'pointer'
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>
              
              {/* Last Updated */}
              <div style={{
                marginTop: '15px',
                fontSize: '0.8rem',
                opacity: 0.6,
                textAlign: 'center'
              }}>
                Updated: {device.lastUpdated.toLocaleTimeString()}
              </div>
              
              {/* Status indicator animation */}
              {device.status === 'online' && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  width: '12px',
                  height: '12px',
                  background: '#00FF00',
                  borderRadius: '50%',
                  animation: 'devicePulse 2s infinite'
                }} />
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.section>

      {/* Energy Monitoring */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #00FFFF',
          borderRadius: '15px',
          padding: '30px',
          marginBottom: '30px',
          position: 'relative',
          zIndex: 10
        }}
      >
        <h2 style={{ color: '#00FFFF', marginBottom: '25px' }}>ENERGY MONITORING</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px', alignItems: 'center' }}>
          <div>
            <canvas
              ref={canvasRef}
              width={600}
              height={200}
              style={{
                width: '100%',
                height: '200px',
                border: '1px solid #333',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.5)'
              }}
            />
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {[
              { label: 'Current Usage', value: `${energyData.current.toFixed(1)} kW`, color: '#00FFFF' },
              { label: 'Today', value: `${energyData.today.toFixed(1)} kWh`, color: '#FFFF00' },
              { label: 'This Month', value: `${energyData.thisMonth.toFixed(1)} kWh`, color: '#FF00FF' },
              { label: 'Est. Cost', value: `$${energyData.cost.toFixed(2)}`, color: '#00FF00' }
            ].map(item => (
              <div key={item.label} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px',
                background: 'rgba(0,0,0,0.5)',
                border: `1px solid ${item.color}`,
                borderRadius: '8px'
              }}>
                <span style={{ opacity: 0.8 }}>{item.label}</span>
                <span style={{ color: item.color, fontWeight: 'bold', fontSize: '1.1rem' }}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </motion.section>

      <style jsx>{`
        @keyframes smartGlow {
          0% { filter: hue-rotate(0deg) brightness(1); }
          100% { filter: hue-rotate(180deg) brightness(1.2); }
        }
        
        @keyframes smartScan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        
        @keyframes devicePulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.2); }
        }
      `}</style>
    </div>
  );
};

export default SmartHomeHub;