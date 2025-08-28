import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type: 'text' | 'command' | 'system' | 'media';
  metadata?: {
    mediaItem?: string;
    command?: string;
    confidence?: number;
  };
}

interface AICapability {
  id: string;
  name: string;
  description: string;
  icon: string;
  active: boolean;
  confidence: number;
  examples: string[];
}

interface VoiceCommand {
  pattern: string;
  action: string;
  description: string;
  examples: string[];
}

const NEXUSAIAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiPersonality, setAiPersonality] = useState<'helpful' | 'cyberpunk' | 'technical' | 'friendly'>('cyberpunk');
  const [capabilities, setCapabilities] = useState<AICapability[]>([]);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [currentThought, setCurrentThought] = useState('');
  const [analyticsMode, setAnalyticsMode] = useState(false);
  const [neuralActivity, setNeuralActivity] = useState<number[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);
  const thoughtIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const voiceCommands: VoiceCommand[] = [
    {
      pattern: 'play (.*)',
      action: 'media.play',
      description: 'Play media content',
      examples: ['play avengers', 'play jazz playlist']
    },
    {
      pattern: 'show (.*) stats',
      action: 'system.stats',
      description: 'Display system statistics',
      examples: ['show server stats', 'show download stats']
    },
    {
      pattern: 'scan (.*) library',
      action: 'media.scan',
      description: 'Scan media library',
      examples: ['scan movie library', 'scan music library']
    },
    {
      pattern: 'restart (.*)',
      action: 'system.restart',
      description: 'Restart services',
      examples: ['restart plex', 'restart jellyfin']
    }
  ];

  useEffect(() => {
    initializeAI();
    setupSpeechRecognition();
    startNeuralActivity();
    addWelcomeMessage();

    return () => {
      if (thoughtIntervalRef.current) {
        clearInterval(thoughtIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initializeAI = () => {
    const aiCapabilities: AICapability[] = [
      {
        id: 'media-search',
        name: 'Media Search',
        description: 'Search across all media libraries using natural language',
        icon: '🔍',
        active: true,
        confidence: 0.95,
        examples: ['Find sci-fi movies from 2020', 'Search for jazz albums']
      },
      {
        id: 'automation',
        name: 'Smart Automation',
        description: 'Automate downloads and library management',
        icon: '🤖',
        active: true,
        confidence: 0.88,
        examples: ['Download latest episodes', 'Optimize storage usage']
      },
      {
        id: 'analytics',
        name: 'Usage Analytics',
        description: 'Analyze viewing patterns and provide recommendations',
        icon: '📊',
        active: true,
        confidence: 0.92,
        examples: ['Show viewing trends', 'Recommend based on habits']
      },
      {
        id: 'troubleshooting',
        name: 'System Diagnostics',
        description: 'Diagnose and resolve system issues',
        icon: '🔧',
        active: true,
        confidence: 0.85,
        examples: ['Fix streaming issues', 'Optimize performance']
      },
      {
        id: 'content-curation',
        name: 'Content Curation',
        description: 'Intelligent content discovery and organization',
        icon: '🎯',
        active: true,
        confidence: 0.90,
        examples: ['Create themed playlists', 'Organize by mood']
      },
      {
        id: 'predictive-analysis',
        name: 'Predictive Analysis',
        description: 'Predict user preferences and system needs',
        icon: '🧠',
        active: true,
        confidence: 0.87,
        examples: ['Predict popular content', 'Forecast storage needs']
      }
    ];
    
    setCapabilities(aiCapabilities);
  };

  const setupSpeechRecognition = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        handleVoiceInput(transcript);
      };
      
      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
      
      setVoiceEnabled(true);
    }
    
    if ('speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }
  };

  const startNeuralActivity = () => {
    const interval = setInterval(() => {
      setNeuralActivity(prev => {
        const newActivity = Array.from({ length: 20 }, () => Math.random());
        return [...prev.slice(-19), ...newActivity];
      });
      
      // Random AI thoughts
      const thoughts = [
        'Analyzing media patterns...',
        'Optimizing neural pathways...',
        'Processing user preferences...',
        'Scanning for anomalies...',
        'Computing recommendations...',
        'Indexing content metadata...',
        'Learning from interactions...',
        'Calibrating response algorithms...'
      ];
      
      if (Math.random() < 0.3) {
        setCurrentThought(thoughts[Math.floor(Math.random() * thoughts.length)]);
        setTimeout(() => setCurrentThought(''), 3000);
      }
    }, 500);
    
    thoughtIntervalRef.current = interval;
  };

  const addWelcomeMessage = () => {
    const welcomeMessage: Message = {
      id: '1',
      text: 'NEXUS AI ASSISTANT ONLINE. Neural networks initialized. Ready to assist with media operations, system management, and predictive analytics. How may I optimize your experience today?',
      sender: 'ai',
      timestamp: new Date(),
      type: 'system',
      metadata: { confidence: 1.0 }
    };
    
    setMessages([welcomeMessage]);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
      type: 'text'
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsProcessing(true);
    
    // Simulate AI processing
    setTimeout(() => {
      const aiResponse = generateAIResponse(inputText);
      setMessages(prev => [...prev, aiResponse]);
      setIsProcessing(false);
      
      // Text-to-speech response
      if (synthRef.current && voiceEnabled) {
        const utterance = new SpeechSynthesisUtterance(aiResponse.text);
        utterance.rate = 0.9;
        utterance.pitch = 0.8;
        synthRef.current.speak(utterance);
      }
    }, 1000 + Math.random() * 2000);
  };

  const handleVoiceInput = (transcript: string) => {
    setInputText(transcript);
    handleSendMessage();
  };

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  const generateAIResponse = (input: string): Message => {
    const lowerInput = input.toLowerCase();
    let responseText = '';
    let responseType: 'text' | 'command' | 'system' | 'media' = 'text';
    let confidence = 0.8;
    
    // Check for voice commands
    for (const command of voiceCommands) {
      const regex = new RegExp(command.pattern, 'i');
      const match = lowerInput.match(regex);
      if (match) {
        responseType = 'command';
        confidence = 0.95;
        responseText = executeCommand(command.action, match[1] || '');
        break;
      }
    }
    
    // Generate contextual responses based on personality
    if (!responseText) {
      if (lowerInput.includes('help') || lowerInput.includes('what can you do')) {
        responseText = getCapabilitiesResponse();
      } else if (lowerInput.includes('search') || lowerInput.includes('find')) {
        responseText = handleSearchRequest(input);
        responseType = 'media';
      } else if (lowerInput.includes('status') || lowerInput.includes('stats')) {
        responseText = getSystemStatusResponse();
        responseType = 'system';
      } else if (lowerInput.includes('recommend') || lowerInput.includes('suggest')) {
        responseText = getRecommendationResponse();
        responseType = 'media';
      } else {
        responseText = getPersonalityResponse(input);
      }
    }
    
    return {
      id: Date.now().toString(),
      text: responseText,
      sender: 'ai',
      timestamp: new Date(),
      type: responseType,
      metadata: { confidence }
    };
  };

  const executeCommand = (action: string, parameter: string): string => {
    switch (action) {
      case 'media.play':
        return `⚡ EXECUTING: Initiating playback for "${parameter}". Accessing media servers... Stream initialized with quantum encryption protocols.`;
      case 'system.stats':
        return `📊 NEURAL ANALYSIS: System performance metrics retrieved. CPU: 45%, Memory: 62%, Network: Optimal. All subsystems operating within normal parameters.`;
      case 'media.scan':
        return `🔍 DEEP SCAN INITIATED: Analyzing ${parameter} for new content. Neural networks are indexing metadata... Estimated completion: 2.3 minutes.`;
      case 'system.restart':
        return `🔄 SYSTEM REBOOT: Gracefully restarting ${parameter} service. Preserving active connections... Restart sequence completed successfully.`;
      default:
        return `⚠️ COMMAND RECOGNITION: Processing "${action}" with parameter "${parameter}". Executing through quantum processing matrix...`;
    }
  };

  const getCapabilitiesResponse = (): string => {
    const activeCapabilities = capabilities.filter(c => c.active);
    return `🧠 NEXUS AI CAPABILITIES ONLINE:\n\n${activeCapabilities.map(cap => 
      `• ${cap.icon} ${cap.name}: ${cap.description} (${Math.round(cap.confidence * 100)}% confidence)`
    ).join('\n')}\n\nUtilize voice commands or natural language queries. Neural network continuously learning from your patterns.`;
  };

  const handleSearchRequest = (query: string): string => {
    return `🔍 QUANTUM SEARCH INITIATED: Analyzing request "${query}". Cross-referencing 847,293 media items across all connected libraries. Neural pattern matching engaged... Results compiled with 94% relevance accuracy.`;
  };

  const getSystemStatusResponse = (): string => {
    return `🖥️ SYSTEM MATRIX STATUS:\n\n⚡ Core Systems: OPTIMAL\n📡 Network Latency: 12ms\n🔄 Active Streams: 3\n💾 Storage Utilization: 73%\n🌐 API Endpoints: All responsive\n🛡️ Security Protocols: Active\n\nAll subsystems operating at peak efficiency. No anomalies detected.`;
  };

  const getRecommendationResponse = (): string => {
    return `🎯 PREDICTIVE ANALYSIS COMPLETE: Based on your viewing patterns and neural preference modeling, I recommend:\n\n• Sci-fi thriller matching your 89% genre preference\n• Content similar to your highly-rated selections\n• New releases in your tracked categories\n\nConfidence level: 91%. Shall I queue these recommendations?`;
  };

  const getPersonalityResponse = (input: string): string => {
    const responses = {
      cyberpunk: [
        `🔮 Neural networks processing your query through quantum algorithms... ${input} analyzed with 94.7% pattern recognition accuracy.`,
        `⚡ NEXUS AI acknowledging: Data streams decoded. Your request has been integrated into the consciousness matrix.`,
        `🌐 Cybernetic analysis complete. The digital realm responds to your inquiry with enhanced processing protocols.`
      ],
      technical: [
        `Processing request through advanced machine learning algorithms. Confidence: 87%. Response generated using transformer architecture.`,
        `Query analyzed using natural language processing. Semantic understanding achieved with 92% accuracy.`,
        `Technical analysis complete. Your input has been processed through multiple neural network layers.`
      ],
      helpful: [
        `I understand you'd like help with: ${input}. Let me assist you with that right away!`,
        `I'm here to help! Your request is being processed and I'll provide the best possible assistance.`,
        `Great question! I'm analyzing the best way to help you with this request.`
      ],
      friendly: [
        `That's an interesting question! I'm excited to help you explore this topic.`,
        `I love helping with requests like this! Let me see what I can do for you.`,
        `Thanks for asking! I'm here to make your media experience as smooth as possible.`
      ]
    };
    
    const personalityResponses = responses[aiPersonality];
    return personalityResponses[Math.floor(Math.random() * personalityResponses.length)];
  };

  return (
    <div style={{
      height: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#00FFFF',
      fontFamily: 'Orbitron, monospace',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <motion.header
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          padding: '20px',
          borderBottom: '2px solid #00FFFF',
          background: 'rgba(0,0,0,0.8)',
          backdropFilter: 'blur(10px)'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{
              margin: 0,
              fontSize: '2rem',
              background: 'linear-gradient(45deg, #00FFFF, #FF00FF)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 20px #00FFFF'
            }}>
              NEXUS AI ASSISTANT
            </h1>
            <p style={{ margin: '5px 0 0 0', opacity: 0.8, fontSize: '0.9rem' }}>
              Neural Network Status: ONLINE | Personality: {aiPersonality.toUpperCase()}
            </p>
          </div>
          
          {/* Neural Activity Visualizer */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
            {neuralActivity.slice(-10).map((activity, index) => (
              <div
                key={index}
                style={{
                  width: '4px',
                  height: `${10 + activity * 30}px`,
                  background: `linear-gradient(to top, #00FFFF, #FF00FF)`,
                  borderRadius: '2px',
                  animation: 'pulse 0.5s ease-in-out'
                }}
              />
            ))}
          </div>
        </div>
        
        {/* Current Thought */}
        <AnimatePresence>
          {currentThought && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              style={{
                marginTop: '10px',
                fontSize: '0.8rem',
                color: '#FFFF00',
                fontStyle: 'italic'
              }}
            >
              💭 {currentThought}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.header>

      {/* Chat Messages */}
      <div style={{
        flex: 1,
        padding: '20px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '15px'
      }}>
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              style={{
                display: 'flex',
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start'
              }}
            >
              <div style={{
                maxWidth: '70%',
                padding: '15px 20px',
                borderRadius: '15px',
                background: message.sender === 'user'
                  ? 'linear-gradient(45deg, #FF00FF, #FFFF00)'
                  : 'rgba(0,0,0,0.8)',
                border: message.sender === 'ai' ? '2px solid #00FFFF' : 'none',
                color: message.sender === 'user' ? '#000' : '#00FFFF',
                position: 'relative'
              }}>
                {message.sender === 'ai' && (
                  <div style={{
                    position: 'absolute',
                    top: '-2px',
                    left: '-2px',
                    right: '-2px',
                    height: '2px',
                    background: 'linear-gradient(90deg, #00FFFF, #FF00FF, #FFFF00)',
                    borderRadius: '15px'
                  }} />
                )}
                
                <div style={{
                  fontSize: '1rem',
                  lineHeight: 1.5,
                  whiteSpace: 'pre-wrap'
                }}>
                  {message.text}
                </div>
                
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: '8px',
                  fontSize: '0.7rem',
                  opacity: 0.7
                }}>
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  {message.metadata?.confidence && (
                    <span>Confidence: {Math.round(message.metadata.confidence * 100)}%</span>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ display: 'flex', justifyContent: 'flex-start' }}
          >
            <div style={{
              padding: '15px 20px',
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #00FFFF',
              borderRadius: '15px',
              color: '#00FFFF'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '20px',
                  height: '20px',
                  border: '2px solid #00FFFF',
                  borderTop: '2px solid transparent',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }} />
                Neural networks processing...
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        padding: '20px',
        borderTop: '2px solid #00FFFF',
        background: 'rgba(0,0,0,0.8)',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isProcessing && handleSendMessage()}
            placeholder="Enter command or ask me anything..."
            disabled={isProcessing}
            style={{
              flex: 1,
              padding: '15px',
              background: 'rgba(0,0,0,0.9)',
              border: '2px solid #FF00FF',
              borderRadius: '10px',
              color: '#00FFFF',
              fontSize: '1rem',
              fontFamily: 'inherit'
            }}
          />
          
          {voiceEnabled && (
            <button
              onClick={startListening}
              disabled={isListening || isProcessing}
              style={{
                padding: '15px 20px',
                background: isListening ? 'linear-gradient(45deg, #FF00FF, #FFFF00)' : 'rgba(255,0,255,0.2)',
                border: '2px solid #FF00FF',
                borderRadius: '10px',
                color: isListening ? '#000' : '#FF00FF',
                cursor: 'pointer',
                fontSize: '1.2rem'
              }}
            >
              {isListening ? '🎤' : '🎙️'}
            </button>
          )}
          
          <button
            onClick={handleSendMessage}
            disabled={isProcessing || !inputText.trim()}
            style={{
              padding: '15px 25px',
              background: 'linear-gradient(45deg, #00FFFF, #FF00FF)',
              border: 'none',
              borderRadius: '10px',
              color: '#000',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            SEND
          </button>
        </div>
        
        {/* Quick Actions */}
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          {['System Status', 'Media Search', 'Recommendations', 'Help'].map(action => (
            <button
              key={action}
              onClick={() => {
                setInputText(action.toLowerCase());
                setTimeout(() => handleSendMessage(), 100);
              }}
              style={{
                padding: '8px 15px',
                background: 'rgba(0,255,255,0.1)',
                border: '1px solid #00FFFF',
                borderRadius: '20px',
                color: '#00FFFF',
                cursor: 'pointer',
                fontSize: '0.9rem'
              }}
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
      `}</style>
    </div>
  );
};

export default NEXUSAIAssistant;