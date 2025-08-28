import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import './AIAssistantInterface.css';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  actions?: Action[];
  metadata?: {
    service?: string;
    confidence?: number;
    processing?: boolean;
  };
}

interface Action {
  id: string;
  label: string;
  type: 'button' | 'confirm' | 'input';
  action: string;
  params?: any;
  status?: 'pending' | 'executing' | 'completed' | 'failed';
}

interface AICapability {
  name: string;
  description: string;
  icon: string;
  enabled: boolean;
}

const AIAssistantInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiMode, setAiMode] = useState<'chat' | 'command' | 'automation'>('chat');
  const [capabilities, setCapabilities] = useState<AICapability[]>([
    { name: 'Media Management', description: 'Control all media services', icon: '🎬', enabled: true },
    { name: 'Download Control', description: 'Manage downloads and queues', icon: '📥', enabled: true },
    { name: 'System Monitoring', description: 'Monitor service health', icon: '📊', enabled: true },
    { name: 'Automation', description: 'Create and manage automations', icon: '🤖', enabled: true },
    { name: 'Recommendations', description: 'AI-powered content suggestions', icon: '🧠', enabled: true },
    { name: 'Voice Control', description: 'Voice command processing', icon: '🎤', enabled: false }
  ]);
  const [suggestedCommands, setSuggestedCommands] = useState<string[]>([]);
  const [showCapabilities, setShowCapabilities] = useState(false);
  const [typingIndicator, setTypingIndicator] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected'>('connected');
  
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recognitionRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializeWebSocket();
    initializeVoiceRecognition();
    initializeAIVisualization();
    loadInitialMessages();
    generateSuggestions();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initializeWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/ai-assistant');
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('Connected to AI Assistant');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleIncomingMessage(data);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        setTimeout(() => initializeWebSocket(), 5000);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  };

  const initializeVoiceRecognition = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0])
          .map((result: any) => result.transcript)
          .join('');
        
        if (event.results[event.results.length - 1].isFinal) {
          handleVoiceInput(transcript);
        }
      };
      
      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }
  };

  const initializeAIVisualization = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    const drawAIWave = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.02)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const time = Date.now() * 0.001;
      const amplitude = isProcessing ? 30 : 10;
      
      ctx.strokeStyle = isProcessing ? '#ff00ff' : '#00ffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let x = 0; x < canvas.width; x += 5) {
        const y = canvas.height / 2 + 
          Math.sin(x * 0.02 + time) * amplitude +
          Math.sin(x * 0.03 + time * 1.5) * amplitude * 0.5;
        
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
      
      // Add glow effect
      ctx.shadowBlur = 20;
      ctx.shadowColor = isProcessing ? '#ff00ff' : '#00ffff';
      ctx.stroke();
      ctx.shadowBlur = 0;
      
      requestAnimationFrame(drawAIWave);
    };
    
    drawAIWave();
  };

  const loadInitialMessages = () => {
    const welcomeMessage: Message = {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m NEXUS, your AI assistant for the Ultimate Media Server. I can help you manage services, control downloads, monitor system health, and automate tasks. How can I assist you today?',
      timestamp: new Date(),
      metadata: { confidence: 1 }
    };
    
    setMessages([welcomeMessage]);
  };

  const generateSuggestions = () => {
    const suggestions = [
      'Show me the status of all services',
      'What\'s currently downloading?',
      'Search for movies like Blade Runner',
      'Restart Jellyfin server',
      'Show system performance metrics',
      'Add The Matrix to download queue',
      'Enable automated TV show downloads',
      'Optimize streaming quality settings',
      'Check for service updates',
      'Create a backup of all settings'
    ];
    
    setSuggestedCommands(suggestions.sort(() => Math.random() - 0.5).slice(0, 4));
  };

  const handleIncomingMessage = (data: any) => {
    switch (data.type) {
      case 'response':
        addAssistantMessage(data.content, data.actions);
        break;
      case 'status':
        updateMessageStatus(data.messageId, data.status);
        break;
      case 'typing':
        setTypingIndicator(data.isTyping);
        break;
    }
  };

  const handleVoiceInput = (transcript: string) => {
    setInputValue(transcript);
    handleSendMessage(transcript);
  };

  const handleSendMessage = async (messageText?: string) => {
    const text = messageText || inputValue.trim();
    if (!text) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: text,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsProcessing(true);
    setTypingIndicator(true);
    
    // Process the message
    await processUserMessage(text);
  };

  const processUserMessage = async (text: string) => {
    // Simulate AI processing
    setTimeout(() => {
      const response = generateAIResponse(text);
      addAssistantMessage(response.content, response.actions);
      setIsProcessing(false);
      setTypingIndicator(false);
      generateSuggestions(); // Refresh suggestions
    }, 1500);
  };

  const generateAIResponse = (input: string): { content: string; actions?: Action[] } => {
    const lowerInput = input.toLowerCase();
    
    // Service status queries
    if (lowerInput.includes('status') || lowerInput.includes('services')) {
      return {
        content: 'All services are operational. Here\'s the current status:\n\n' +
          '🟢 Jellyfin: Online (45% CPU, 60% Memory)\n' +
          '🟢 Plex: Online (55% CPU, 70% Memory)\n' +
          '🟢 Sonarr: Online (30% CPU, 40% Memory)\n' +
          '🟢 Radarr: Online (25% CPU, 35% Memory)\n' +
          '🟡 Emby: Degraded (60% CPU, 65% Memory)\n' +
          '🔴 Readarr: Offline\n\n' +
          'Would you like me to restart any services?',
        actions: [
          { id: '1', label: 'Restart Emby', type: 'button', action: 'restart_service', params: { service: 'emby' } },
          { id: '2', label: 'Start Readarr', type: 'button', action: 'start_service', params: { service: 'readarr' } },
          { id: '3', label: 'View Details', type: 'button', action: 'view_details' }
        ]
      };
    }
    
    // Download queries
    if (lowerInput.includes('download') || lowerInput.includes('queue')) {
      return {
        content: 'Current download queue:\n\n' +
          '📥 Dune Part Two (2024) - 45% complete - 2.3 GB/5.1 GB\n' +
          '📥 The Last of Us S02E01 - 78% complete - 1.8 GB/2.3 GB\n' +
          '⏸️ Oppenheimer (2023) - Paused at 23%\n' +
          '⏳ Blade Runner 2049 - Queued\n\n' +
          'Total: 4 items | Active: 2 | ETA: 45 minutes',
        actions: [
          { id: '1', label: 'Resume Paused', type: 'button', action: 'resume_downloads' },
          { id: '2', label: 'Priority Mode', type: 'button', action: 'set_priority' },
          { id: '3', label: 'Clear Completed', type: 'button', action: 'clear_completed' }
        ]
      };
    }
    
    // Search queries
    if (lowerInput.includes('search') || lowerInput.includes('find')) {
      const searchTerm = input.replace(/search|find|for/gi, '').trim();
      return {
        content: `Searching for "${searchTerm}" across all media services...\n\n` +
          'Found 12 results:\n' +
          '🎬 Movies (5 results)\n' +
          '📺 TV Shows (4 results)\n' +
          '🎵 Music (3 results)\n\n' +
          'Top matches are ready to stream or download.',
        actions: [
          { id: '1', label: 'View Results', type: 'button', action: 'view_search_results', params: { query: searchTerm } },
          { id: '2', label: 'Download All', type: 'button', action: 'download_all' },
          { id: '3', label: 'Filter Results', type: 'button', action: 'filter_results' }
        ]
      };
    }
    
    // Restart service
    if (lowerInput.includes('restart')) {
      const service = extractServiceName(input);
      return {
        content: `Initiating restart for ${service || 'selected service'}...\n\n` +
          '⚡ Stopping service...\n' +
          '⚡ Clearing cache...\n' +
          '⚡ Starting service...\n' +
          '✅ Service restarted successfully!\n\n' +
          'The service is now back online and functioning normally.',
        actions: [
          { id: '1', label: 'View Logs', type: 'button', action: 'view_logs', params: { service } },
          { id: '2', label: 'Monitor Health', type: 'button', action: 'monitor_health' }
        ]
      };
    }
    
    // Automation queries
    if (lowerInput.includes('automat') || lowerInput.includes('schedule')) {
      return {
        content: 'I can help you set up automation rules. Here are some popular options:\n\n' +
          '🤖 Auto-download new episodes of tracked shows\n' +
          '🤖 Upgrade video quality when available\n' +
          '🤖 Clean up watched content after 30 days\n' +
          '🤖 Backup settings weekly\n' +
          '🤖 Restart services on failure\n\n' +
          'Which automation would you like to configure?',
        actions: [
          { id: '1', label: 'TV Auto-Download', type: 'button', action: 'setup_tv_automation' },
          { id: '2', label: 'Quality Upgrade', type: 'button', action: 'setup_quality_automation' },
          { id: '3', label: 'Custom Rule', type: 'button', action: 'create_custom_automation' }
        ]
      };
    }
    
    // Default response
    return {
      content: 'I understand you want to: ' + input + '\n\n' +
        'I\'m processing your request and will execute the appropriate actions. ' +
        'Is there anything specific you\'d like me to prioritize?',
      actions: [
        { id: '1', label: 'Show Options', type: 'button', action: 'show_options' },
        { id: '2', label: 'Help', type: 'button', action: 'show_help' }
      ]
    };
  };

  const extractServiceName = (input: string): string => {
    const services = ['jellyfin', 'plex', 'emby', 'sonarr', 'radarr', 'lidarr', 'bazarr', 'prowlarr'];
    const found = services.find(service => input.toLowerCase().includes(service));
    return found ? found.charAt(0).toUpperCase() + found.slice(1) : 'Service';
  };

  const addAssistantMessage = (content: string, actions?: Action[]) => {
    const assistantMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content,
      timestamp: new Date(),
      actions,
      metadata: { confidence: 0.95 }
    };
    
    setMessages(prev => [...prev, assistantMessage]);
  };

  const updateMessageStatus = (messageId: string, status: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, metadata: { ...msg.metadata, processing: status === 'processing' } }
        : msg
    ));
  };

  const handleActionClick = async (action: Action) => {
    // Update action status
    setMessages(prev => prev.map(msg => ({
      ...msg,
      actions: msg.actions?.map(a => 
        a.id === action.id ? { ...a, status: 'executing' } : a
      )
    })));
    
    // Execute action
    setTimeout(() => {
      const response = `Executed action: ${action.label}`;
      addSystemMessage(response);
      
      // Update action status to completed
      setMessages(prev => prev.map(msg => ({
        ...msg,
        actions: msg.actions?.map(a => 
          a.id === action.id ? { ...a, status: 'completed' } : a
        )
      })));
    }, 1000);
  };

  const addSystemMessage = (content: string) => {
    const systemMessage: Message = {
      id: Date.now().toString(),
      type: 'system',
      content,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, systemMessage]);
  };

  const toggleVoiceRecognition = () => {
    if (!recognitionRef.current) return;
    
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    handleSendMessage(suggestion);
  };

  const formatMessageContent = (content: string) => {
    return content.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className="ai-assistant-interface cyberpunk-theme">
      <div className="assistant-header">
        <div className="header-left">
          <h1 className="title glitch-text" data-text="NEXUS AI">
            NEXUS AI
          </h1>
          <div className="ai-status">
            <div className={`status-indicator ${connectionStatus}`}></div>
            <span>{connectionStatus === 'connected' ? 'Online' : 'Reconnecting...'}</span>
          </div>
        </div>
        
        <div className="header-controls">
          <div className="mode-selector">
            <button 
              className={`mode-btn ${aiMode === 'chat' ? 'active' : ''}`}
              onClick={() => setAiMode('chat')}
            >
              💬 Chat
            </button>
            <button 
              className={`mode-btn ${aiMode === 'command' ? 'active' : ''}`}
              onClick={() => setAiMode('command')}
            >
              ⚡ Command
            </button>
            <button 
              className={`mode-btn ${aiMode === 'automation' ? 'active' : ''}`}
              onClick={() => setAiMode('automation')}
            >
              🤖 Automation
            </button>
          </div>
          
          <button 
            className="capabilities-btn"
            onClick={() => setShowCapabilities(!showCapabilities)}
          >
            ⚙️ Capabilities
          </button>
        </div>
      </div>

      {/* AI Visualization */}
      <canvas 
        ref={canvasRef}
        className="ai-visualization"
        width={800}
        height={100}
      />

      {/* Capabilities Panel */}
      <AnimatePresence>
        {showCapabilities && (
          <motion.div
            className="capabilities-panel"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h3>AI Capabilities</h3>
            <div className="capabilities-grid">
              {capabilities.map(cap => (
                <div key={cap.name} className={`capability-card ${cap.enabled ? 'enabled' : 'disabled'}`}>
                  <span className="capability-icon">{cap.icon}</span>
                  <div className="capability-info">
                    <h4>{cap.name}</h4>
                    <p>{cap.description}</p>
                  </div>
                  <label className="capability-toggle">
                    <input
                      type="checkbox"
                      checked={cap.enabled}
                      onChange={(e) => {
                        setCapabilities(prev => prev.map(c => 
                          c.name === cap.name ? { ...c, enabled: e.target.checked } : c
                        ));
                      }}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Container */}
      <div className="chat-container" ref={chatContainerRef}>
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              className={`message ${message.type}`}
              initial={{ opacity: 0, x: message.type === 'user' ? 50 : -50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ delay: index * 0.05 }}
            >
              <div className="message-avatar">
                {message.type === 'user' ? '👤' : message.type === 'assistant' ? '🤖' : '⚙️'}
              </div>
              
              <div className="message-content">
                <div className="message-header">
                  <span className="message-sender">
                    {message.type === 'user' ? 'You' : message.type === 'assistant' ? 'NEXUS' : 'System'}
                  </span>
                  <span className="message-time">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                
                <div className="message-text">
                  {formatMessageContent(message.content)}
                </div>
                
                {message.actions && message.actions.length > 0 && (
                  <div className="message-actions">
                    {message.actions.map(action => (
                      <button
                        key={action.id}
                        className={`action-btn ${action.status || ''}`}
                        onClick={() => handleActionClick(action)}
                        disabled={action.status === 'executing' || action.status === 'completed'}
                      >
                        {action.status === 'executing' ? '⏳' : action.status === 'completed' ? '✅' : ''}
                        {action.label}
                      </button>
                    ))}
                  </div>
                )}
                
                {message.metadata && (
                  <div className="message-metadata">
                    {message.metadata.confidence && (
                      <span className="confidence">
                        Confidence: {(message.metadata.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    {message.metadata.service && (
                      <span className="service-tag">{message.metadata.service}</span>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {typingIndicator && (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        )}
      </div>

      {/* Suggested Commands */}
      <div className="suggestions-bar">
        <span className="suggestions-label">Suggestions:</span>
        <div className="suggestions-list">
          {suggestedCommands.map((suggestion, index) => (
            <button
              key={index}
              className="suggestion-chip"
              onClick={() => handleSuggestionClick(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <div className="input-area">
        <button 
          className={`voice-btn ${isListening ? 'listening' : ''}`}
          onClick={toggleVoiceRecognition}
        >
          {isListening ? '🔴' : '🎤'}
        </button>
        
        <input
          ref={inputRef}
          type="text"
          className="message-input"
          placeholder={
            aiMode === 'chat' ? 'Ask me anything...' :
            aiMode === 'command' ? 'Enter command...' :
            'Describe automation...'
          }
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              handleSendMessage();
            }
          }}
        />
        
        <button 
          className="send-btn"
          onClick={() => handleSendMessage()}
          disabled={!inputValue.trim() || isProcessing}
        >
          {isProcessing ? '⏳' : '➤'}
        </button>
      </div>

      {/* Processing Overlay */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            className="processing-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="processing-animation">
              <div className="processing-ring"></div>
              <span>Processing...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AIAssistantInterface;