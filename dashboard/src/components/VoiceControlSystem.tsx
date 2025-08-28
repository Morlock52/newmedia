import React, { useState, useEffect, useRef, useCallback } from 'react';
import './VoiceControlSystem.css';

interface VoiceCommand {
  command: string;
  action: () => void;
  description: string;
  keywords: string[];
}

interface VoiceVisualizerProps {
  isListening: boolean;
  audioLevel: number;
}

const VoiceControlSystem: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [isWakeWordActive, setIsWakeWordActive] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const [language, setLanguage] = useState('en-US');
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  const recognitionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const WAKE_WORDS = ['hey nexus', 'okay nexus', 'nexus', 'hey media', 'okay media'];

  const voiceCommands: VoiceCommand[] = [
    {
      command: 'play',
      keywords: ['play', 'start', 'resume'],
      action: () => executeCommand('play'),
      description: 'Play media'
    },
    {
      command: 'pause',
      keywords: ['pause', 'stop'],
      action: () => executeCommand('pause'),
      description: 'Pause media'
    },
    {
      command: 'download',
      keywords: ['download', 'get', 'fetch'],
      action: () => executeCommand('download'),
      description: 'Download content'
    },
    {
      command: 'search',
      keywords: ['search', 'find', 'look for'],
      action: () => executeCommand('search'),
      description: 'Search media'
    },
    {
      command: 'show downloads',
      keywords: ['show downloads', 'download status', 'what is downloading'],
      action: () => executeCommand('showDownloads'),
      description: 'Show download queue'
    },
    {
      command: 'show unwatched',
      keywords: ['unwatched', 'new episodes', 'new movies', 'what is new'],
      action: () => executeCommand('showUnwatched'),
      description: 'Show unwatched content'
    },
    {
      command: 'volume up',
      keywords: ['volume up', 'louder', 'increase volume'],
      action: () => executeCommand('volumeUp'),
      description: 'Increase volume'
    },
    {
      command: 'volume down',
      keywords: ['volume down', 'quieter', 'decrease volume'],
      action: () => executeCommand('volumeDown'),
      description: 'Decrease volume'
    },
    {
      command: 'next',
      keywords: ['next', 'skip', 'next episode'],
      action: () => executeCommand('next'),
      description: 'Next item'
    },
    {
      command: 'previous',
      keywords: ['previous', 'back', 'last'],
      action: () => executeCommand('previous'),
      description: 'Previous item'
    },
    {
      command: 'open dashboard',
      keywords: ['dashboard', 'home', 'main screen'],
      action: () => executeCommand('dashboard'),
      description: 'Open dashboard'
    },
    {
      command: 'system status',
      keywords: ['status', 'system status', 'health check'],
      action: () => executeCommand('status'),
      description: 'Check system status'
    }
  ];

  useEffect(() => {
    setupSpeechRecognition();
    setupAudioAnalyser();

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const setupSpeechRecognition = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.error('Speech recognition not supported');
      setFeedbackMessage('Voice control not supported in this browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;
    recognition.maxAlternatives = 3;

    recognition.onstart = () => {
      console.log('Voice recognition started');
      setIsListening(true);
      setFeedbackMessage('Listening...');
    };

    recognition.onresult = (event: any) => {
      const last = event.results.length - 1;
      const transcript = event.results[last][0].transcript.toLowerCase();
      
      setTranscript(transcript);
      
      if (event.results[last].isFinal) {
        processVoiceInput(transcript);
      }
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      setFeedbackMessage(`Error: ${event.error}`);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (isWakeWordActive) {
        recognition.start(); // Restart if wake word is active
      }
    };

    recognitionRef.current = recognition;
  };

  const setupAudioAnalyser = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
      
      analyserRef.current.fftSize = 256;
      microphoneRef.current.connect(analyserRef.current);
      
      updateAudioLevel();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setFeedbackMessage('Microphone access denied');
    }
  };

  const updateAudioLevel = () => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
    setAudioLevel(average / 255);
    
    animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
  };

  const processVoiceInput = (input: string) => {
    // Check for wake word
    const hasWakeWord = WAKE_WORDS.some(word => input.includes(word));
    
    if (!isWakeWordActive && hasWakeWord) {
      setIsWakeWordActive(true);
      setFeedbackMessage('Wake word detected! Listening for commands...');
      speak('Yes, I\'m listening');
      return;
    }

    if (!isWakeWordActive) {
      return; // Ignore if wake word not active
    }

    // Process command
    const matchedCommand = findMatchingCommand(input);
    
    if (matchedCommand) {
      setCommandHistory(prev => [input, ...prev].slice(0, 10));
      matchedCommand.action();
      setIsWakeWordActive(false); // Reset after command
    } else if (input.includes('cancel') || input.includes('never mind')) {
      setIsWakeWordActive(false);
      speak('Cancelled');
      setFeedbackMessage('Command cancelled');
    } else {
      speak('Sorry, I didn\'t understand that command');
      setFeedbackMessage('Command not recognized');
    }
  };

  const findMatchingCommand = (input: string): VoiceCommand | null => {
    for (const command of voiceCommands) {
      for (const keyword of command.keywords) {
        if (input.includes(keyword)) {
          return command;
        }
      }
    }
    return null;
  };

  const executeCommand = async (command: string) => {
    setIsProcessing(true);
    setFeedbackMessage(`Executing: ${command}`);
    
    try {
      const response = await fetch('/api/voice/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, transcript })
      });
      
      const result = await response.json();
      
      if (result.success) {
        speak(result.message || `${command} executed`);
        setFeedbackMessage(result.message || 'Command executed successfully');
      } else {
        speak('Command failed');
        setFeedbackMessage('Command execution failed');
      }
    } catch (error) {
      console.error('Command execution error:', error);
      speak('Error executing command');
      setFeedbackMessage('Error executing command');
    } finally {
      setIsProcessing(false);
    }
  };

  const speak = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 0.8;
    utterance.volume = 1.0;
    
    // Use a robotic/cyberpunk voice if available
    const voices = speechSynthesis.getVoices();
    const synthVoice = voices.find(voice => 
      voice.name.includes('Google UK English Male') || 
      voice.name.includes('Microsoft David') ||
      voice.name.includes('Alex')
    );
    
    if (synthVoice) {
      utterance.voice = synthVoice;
    }
    
    speechSynthesis.speak(utterance);
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsWakeWordActive(false);
    } else {
      recognitionRef.current?.start();
    }
  };

  const VoiceVisualizer: React.FC<VoiceVisualizerProps> = ({ isListening, audioLevel }) => {
    return (
      <div className="voice-visualizer">
        <div className="visualizer-rings">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className={`ring ring-${i + 1} ${isListening ? 'active' : ''}`}
              style={{
                transform: `scale(${1 + audioLevel * (i + 1) * 0.2})`,
                opacity: isListening ? 0.8 - i * 0.15 : 0.2,
              }}
            />
          ))}
        </div>
        
        <div className="waveform-container">
          {[...Array(32)].map((_, i) => (
            <div
              key={i}
              className="waveform-bar"
              style={{
                height: `${10 + audioLevel * 100 * Math.random()}%`,
                animationDelay: `${i * 0.02}s`,
              }}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="voice-control-system cyberpunk-theme">
      <div className="voice-header">
        <h1 className="voice-title glitch-text" data-text="VOICE CONTROL">
          VOICE CONTROL
        </h1>
        
        <div className="wake-word-status">
          <div className={`status-indicator ${isWakeWordActive ? 'active' : ''}`} />
          <span>Wake Word: {isWakeWordActive ? 'Active' : 'Inactive'}</span>
        </div>
      </div>

      <div className="voice-main">
        <div className="voice-button-container">
          <button
            className={`voice-button ${isListening ? 'listening' : ''} ${isWakeWordActive ? 'wake-active' : ''}`}
            onClick={toggleListening}
            disabled={isProcessing}
          >
            <div className="button-inner">
              <div className="mic-icon">
                {isListening ? '🎙️' : '🎤'}
              </div>
              <div className="button-text">
                {isProcessing ? 'Processing...' : isListening ? 'Listening...' : 'Start Voice Control'}
              </div>
            </div>
            
            <VoiceVisualizer isListening={isListening} audioLevel={audioLevel} />
          </button>
        </div>

        <div className="transcript-display">
          <div className="transcript-label">Transcript:</div>
          <div className="transcript-text">
            {transcript || 'Say "Hey Nexus" to start...'}
          </div>
        </div>

        <div className="feedback-display">
          <div className="feedback-message">{feedbackMessage}</div>
        </div>

        <div className="voice-info">
          <div className="language-selector">
            <label>Language:</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="cyberpunk-select"
            >
              <option value="en-US">English (US)</option>
              <option value="en-GB">English (UK)</option>
              <option value="es-ES">Español</option>
              <option value="fr-FR">Français</option>
              <option value="de-DE">Deutsch</option>
              <option value="ja-JP">日本語</option>
              <option value="zh-CN">中文</option>
            </select>
          </div>
        </div>

        <div className="command-list">
          <h2 className="section-title">Available Commands</h2>
          <div className="commands-grid">
            {voiceCommands.map((cmd, index) => (
              <div key={index} className="command-card">
                <div className="command-name">{cmd.command}</div>
                <div className="command-keywords">
                  {cmd.keywords.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="command-history">
          <h2 className="section-title">Command History</h2>
          <div className="history-list">
            {commandHistory.map((cmd, index) => (
              <div key={index} className="history-item">
                <span className="history-index">#{commandHistory.length - index}</span>
                <span className="history-command">{cmd}</span>
                <span className="history-time">{new Date().toLocaleTimeString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cyberpunk decorations */}
      <div className="voice-decorations">
        <div className="scan-line"></div>
        <div className="corner-decoration top-left"></div>
        <div className="corner-decoration top-right"></div>
        <div className="corner-decoration bottom-left"></div>
        <div className="corner-decoration bottom-right"></div>
      </div>
    </div>
  );
};

export default VoiceControlSystem;