import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { io, Socket } from 'socket.io-client';
import Webcam from 'react-webcam';

interface User {
  id: string;
  name: string;
  avatar?: string;
  isHost: boolean;
  status: 'watching' | 'paused' | 'buffering' | 'away';
  reaction?: string;
  color: string;
  permissions: {
    canControl: boolean;
    canChat: boolean;
    canVideo: boolean;
    canAudio: boolean;
  };
  videoStream?: MediaStream;
  audioLevel: number;
  faceDetection?: {
    expressions: any;
    landmarks: any;
  };
}

interface Message {
  id: string;
  userId: string;
  userName: string;
  content: string;
  timestamp: Date;
  type: 'message' | 'system' | 'reaction' | 'event';
}

interface WatchParty {
  id: string;
  name: string;
  mediaTitle: string;
  mediaType: 'movie' | 'episode' | 'video' | 'stream';
  mediaUrl: string;
  thumbnail: string;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  users: User[];
  maxUsers: number;
  privacy: 'public' | 'private' | 'friends';
  code?: string;
}

interface Reaction {
  emoji: string;
  label: string;
  animation: string;
}

const SocialWatchParty: React.FC = () => {
  const [party, setParty] = useState<WatchParty | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [showChat, setShowChat] = useState(true);
  const [showUsers, setShowUsers] = useState(true);
  const [videoSync, setVideoSync] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [videoEnabled, setVideoEnabled] = useState(false);
  const [screenSharing, setScreenSharing] = useState(false);
  const [showReactions, setShowReactions] = useState(false);
  const [showInvite, setShowInvite] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [syncOffset, setSyncOffset] = useState(0);
  const [buffering, setBuffering] = useState(false);
  const [volume, setVolume] = useState(100);
  const [quality, setQuality] = useState<'auto' | '4k' | '1080p' | '720p' | '480p'>('auto');
  const [latency, setLatency] = useState(0);
  const [showEmojis, setShowEmojis] = useState(false);
  const [partyCode, setPartyCode] = useState('');
  const [joinCode, setJoinCode] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isJoining, setIsJoining] = useState(false);
  const [faceDetectionEnabled, setFaceDetectionEnabled] = useState(false);
  const [emotionSync, setEmotionSync] = useState(false);
  const [videoLayout, setVideoLayout] = useState<'grid' | 'sidebar' | 'overlay' | 'theater'>('sidebar');
  const [autoReactions, setAutoReactions] = useState(false);
  const [voiceActivation, setVoiceActivation] = useState(false);
  const [roomTheme, setRoomTheme] = useState<'cyberpunk' | 'neon' | 'retro' | 'minimal'>('cyberpunk');

  const socketRef = useRef<Socket | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<Webcam>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const syncIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const faceApiRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const userVideoRefs = useRef<Map<string, HTMLVideoElement>>(new Map());
  const emotionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const reactions: Reaction[] = [
    { emoji: '😂', label: 'Laugh', animation: 'bounce' },
    { emoji: '😱', label: 'Shocked', animation: 'shake' },
    { emoji: '❤️', label: 'Love', animation: 'pulse' },
    { emoji: '😭', label: 'Cry', animation: 'rain' },
    { emoji: '🔥', label: 'Fire', animation: 'burn' },
    { emoji: '👏', label: 'Clap', animation: 'clap' },
    { emoji: '🤔', label: 'Think', animation: 'rotate' },
    { emoji: '😴', label: 'Sleepy', animation: 'fade' },
    { emoji: '🎉', label: 'Party', animation: 'confetti' },
    { emoji: '💀', label: 'Dead', animation: 'fall' }
  ];

  const quickMessages = [
    "Let's start!",
    "Pause please!",
    "This is awesome!",
    "Wait for me!",
    "LOL 😂",
    "What just happened?",
    "Great scene!",
    "Skip intro?",
    "Volume up!",
    "Anyone else see that?"
  ];

  useEffect(() => {
    initializeSocket();
    return () => {
      cleanupConnections();
    };
  }, []);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  const initializeSocket = () => {
    socketRef.current = io('ws://localhost:8080/watch-party', {
      transports: ['websocket']
    });

    socketRef.current.on('connect', () => {
      setConnectionStatus('connected');
      console.log('Connected to watch party server');
    });

    socketRef.current.on('disconnect', () => {
      setConnectionStatus('disconnected');
    });

    socketRef.current.on('party-update', (data: WatchParty) => {
      setParty(data);
    });

    socketRef.current.on('message', (message: Message) => {
      setMessages(prev => [...prev, message]);
    });

    socketRef.current.on('user-joined', (user: User) => {
      addSystemMessage(`${user.name} joined the party! 🎉`);
      setParty(prev => prev ? {
        ...prev,
        users: [...prev.users, user]
      } : null);
    });

    socketRef.current.on('user-left', (userId: string) => {
      setParty(prev => {
        if (!prev) return null;
        const user = prev.users.find(u => u.id === userId);
        if (user) {
          addSystemMessage(`${user.name} left the party`);
        }
        return {
          ...prev,
          users: prev.users.filter(u => u.id !== userId)
        };
      });
    });

    socketRef.current.on('sync-video', (data: { time: number; playing: boolean }) => {
      if (videoRef.current && videoSync) {
        videoRef.current.currentTime = data.time + syncOffset;
        if (data.playing) {
          videoRef.current.play();
        } else {
          videoRef.current.pause();
        }
      }
    });

    socketRef.current.on('reaction', (data: { userId: string; emoji: string }) => {
      showReactionAnimation(data.userId, data.emoji);
    });

    socketRef.current.on('latency-update', (data: { latency: number }) => {
      setLatency(data.latency);
    });
  };

  const cleanupConnections = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }
    Object.values(peersRef.current).forEach(peer => peer.destroy());
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  };

  const createParty = async (mediaTitle: string, mediaUrl: string) => {
    setIsCreating(true);
    const code = generatePartyCode();
    
    const newParty: WatchParty = {
      id: Date.now().toString(),
      name: `${currentUser?.name}'s Watch Party`,
      mediaTitle,
      mediaType: 'movie',
      mediaUrl,
      thumbnail: '/thumbnails/default.jpg',
      currentTime: 0,
      duration: 0,
      isPlaying: false,
      users: currentUser ? [currentUser] : [],
      maxUsers: 10,
      privacy: 'private',
      code
    };

    socketRef.current?.emit('create-party', newParty);
    setParty(newParty);
    setPartyCode(code);
    setIsCreating(false);
    addSystemMessage(`Party created! Share code: ${code}`);
  };

  const joinParty = (code: string) => {
    setIsJoining(true);
    socketRef.current?.emit('join-party', { code, user: currentUser });
  };

  const leaveParty = () => {
    socketRef.current?.emit('leave-party', { partyId: party?.id, userId: currentUser?.id });
    setParty(null);
    setMessages([]);
  };

  const generatePartyCode = (): string => {
    return Math.random().toString(36).substring(2, 8).toUpperCase();
  };

  const sendMessage = () => {
    if (!inputMessage.trim() || !currentUser) return;

    const message: Message = {
      id: Date.now().toString(),
      userId: currentUser.id,
      userName: currentUser.name,
      content: inputMessage,
      timestamp: new Date(),
      type: 'message'
    };

    socketRef.current?.emit('message', message);
    setMessages(prev => [...prev, message]);
    setInputMessage('');
  };

  const sendReaction = (emoji: string) => {
    if (!currentUser) return;
    
    socketRef.current?.emit('reaction', { userId: currentUser.id, emoji });
    showReactionAnimation(currentUser.id, emoji);
    setShowReactions(false);
  };

  const showReactionAnimation = (userId: string, emoji: string) => {
    const user = party?.users.find(u => u.id === userId);
    if (!user) return;

    // Create floating reaction animation
    const reactionEl = document.createElement('div');
    reactionEl.className = 'floating-reaction';
    reactionEl.textContent = emoji;
    reactionEl.style.left = `${Math.random() * 80 + 10}%`;
    document.querySelector('.video-overlay')?.appendChild(reactionEl);

    setTimeout(() => reactionEl.remove(), 3000);

    // Add to chat
    addSystemMessage(`${user.name} reacted ${emoji}`);
  };

  const addSystemMessage = (content: string) => {
    const message: Message = {
      id: Date.now().toString(),
      userId: 'system',
      userName: 'System',
      content,
      timestamp: new Date(),
      type: 'system'
    };
    setMessages(prev => [...prev, message]);
  };

  const syncVideo = () => {
    if (!videoRef.current || !currentUser?.isHost) return;

    const syncData = {
      time: videoRef.current.currentTime,
      playing: !videoRef.current.paused
    };

    socketRef.current?.emit('sync-video', syncData);
  };

  const toggleVideoSync = () => {
    setVideoSync(!videoSync);
    if (!videoSync) {
      syncVideo();
    }
  };

  const toggleAudioVideo = async (type: 'audio' | 'video') => {
    if (type === 'audio') {
      if (!audioEnabled) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        setAudioEnabled(true);
      } else {
        streamRef.current?.getAudioTracks().forEach(track => track.stop());
        setAudioEnabled(false);
      }
    } else {
      if (!videoEnabled) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;
        setVideoEnabled(true);
      } else {
        streamRef.current?.getVideoTracks().forEach(track => track.stop());
        setVideoEnabled(false);
      }
    }
  };

  const shareScreen = async () => {
    if (!screenSharing) {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        streamRef.current = stream;
        setScreenSharing(true);
        
        stream.getVideoTracks()[0].onended = () => {
          setScreenSharing(false);
        };
      } catch (error) {
        console.error('Screen share failed:', error);
      }
    } else {
      streamRef.current?.getVideoTracks().forEach(track => track.stop());
      setScreenSharing(false);
    }
  };

  const handleVideoControl = (action: 'play' | 'pause' | 'seek' | 'volume') => {
    if (!videoRef.current) return;

    switch (action) {
      case 'play':
        videoRef.current.play();
        break;
      case 'pause':
        videoRef.current.pause();
        break;
    }

    if (currentUser?.isHost) {
      syncVideo();
    }
  };

  const inviteUsers = () => {
    if (party?.code) {
      navigator.clipboard.writeText(party.code);
      addSystemMessage('Party code copied to clipboard!');
    }
  };

  const kickUser = (userId: string) => {
    if (!currentUser?.isHost) return;
    socketRef.current?.emit('kick-user', { partyId: party?.id, userId });
  };

  const promoteToHost = (userId: string) => {
    if (!currentUser?.isHost) return;
    socketRef.current?.emit('promote-host', { partyId: party?.id, userId });
  };

  return (
    <div className="social-watch-party cyberpunk-theme">
      {!party ? (
        // Party Creation/Join Screen
        <div className="party-lobby">
          <div className="lobby-container">
            <h1 className="title glitch-text" data-text="WATCH PARTY">
              WATCH PARTY
            </h1>
            
            <div className="lobby-options">
              <motion.div 
                className="option-card create-party"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <h2>Create Party</h2>
                <input
                  type="text"
                  placeholder="Media Title"
                  className="party-input"
                />
                <input
                  type="text"
                  placeholder="Media URL"
                  className="party-input"
                />
                <button 
                  className="cyber-btn primary"
                  onClick={() => createParty('Blade Runner 2049', 'http://example.com/movie.mp4')}
                  disabled={isCreating}
                >
                  {isCreating ? 'Creating...' : 'Create Party'}
                </button>
              </motion.div>

              <div className="divider">
                <span>OR</span>
              </div>

              <motion.div 
                className="option-card join-party"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <h2>Join Party</h2>
                <input
                  type="text"
                  placeholder="Enter Party Code"
                  value={joinCode}
                  onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
                  className="party-input code-input"
                  maxLength={6}
                />
                <button 
                  className="cyber-btn secondary"
                  onClick={() => joinParty(joinCode)}
                  disabled={isJoining || joinCode.length !== 6}
                >
                  {isJoining ? 'Joining...' : 'Join Party'}
                </button>
              </motion.div>
            </div>

            <div className="public-parties">
              <h3>Public Parties</h3>
              <div className="party-list">
                {[1, 2, 3].map(i => (
                  <div key={i} className="party-item">
                    <div className="party-thumbnail"></div>
                    <div className="party-info">
                      <h4>Cyberpunk 2077 Stream</h4>
                      <span className="party-users">8/10 users</span>
                    </div>
                    <button className="join-btn">Join</button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        // Watch Party Room
        <div className="party-room">
          <div className="video-section">
            <div className="video-container">
              <video
                ref={videoRef}
                className="video-player"
                src={party.mediaUrl}
                onPlay={() => handleVideoControl('play')}
                onPause={() => handleVideoControl('pause')}
                onSeeked={() => syncVideo()}
              />
              
              <div className="video-overlay">
                <div className="video-header">
                  <h2>{party.mediaTitle}</h2>
                  <div className="party-code">
                    Code: <span className="code">{party.code}</span>
                  </div>
                </div>

                <div className="video-controls">
                  <button 
                    className="control-btn"
                    onClick={() => handleVideoControl(party.isPlaying ? 'pause' : 'play')}
                  >
                    {party.isPlaying ? '⏸' : '▶'}
                  </button>
                  
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${(party.currentTime / party.duration) * 100}%` }}
                    />
                  </div>

                  <div className="control-group">
                    <button 
                      className={`control-btn ${videoSync ? 'active' : ''}`}
                      onClick={toggleVideoSync}
                      title="Sync Video"
                    >
                      🔄
                    </button>
                    <button 
                      className="control-btn"
                      onClick={() => setVolume(volume === 0 ? 100 : 0)}
                    >
                      {volume === 0 ? '🔇' : '🔊'}
                    </button>
                    <select 
                      className="quality-select"
                      value={quality}
                      onChange={(e) => setQuality(e.target.value as any)}
                    >
                      <option value="auto">Auto</option>
                      <option value="4k">4K</option>
                      <option value="1080p">1080p</option>
                      <option value="720p">720p</option>
                      <option value="480p">480p</option>
                    </select>
                  </div>
                </div>

                {/* Floating Reactions Container */}
                <div className="reactions-container"></div>
              </div>
            </div>

            {/* User Avatars */}
            <div className="watching-users">
              {party.users.map((user, index) => (
                <motion.div
                  key={user.id}
                  className={`user-avatar ${user.status}`}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  style={{ borderColor: user.color }}
                >
                  <span className="avatar-emoji">👤</span>
                  {user.isHost && <span className="host-badge">👑</span>}
                  <span className="user-name">{user.name}</span>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="sidebar">
            {/* Chat Section */}
            <div className={`chat-section ${showChat ? 'expanded' : 'collapsed'}`}>
              <div className="chat-header">
                <h3>Chat</h3>
                <button onClick={() => setShowChat(!showChat)}>
                  {showChat ? '−' : '+'}
                </button>
              </div>

              {showChat && (
                <>
                  <div className="chat-messages" ref={chatRef}>
                    <AnimatePresence>
                      {messages.map((msg, index) => (
                        <motion.div
                          key={msg.id}
                          className={`message ${msg.type}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                          transition={{ delay: index * 0.02 }}
                        >
                          {msg.type === 'message' && (
                            <>
                              <span className="msg-user" style={{ color: party.users.find(u => u.id === msg.userId)?.color }}>
                                {msg.userName}:
                              </span>
                              <span className="msg-content">{msg.content}</span>
                            </>
                          )}
                          {msg.type === 'system' && (
                            <span className="msg-system">{msg.content}</span>
                          )}
                          <span className="msg-time">
                            {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>

                  <div className="quick-messages">
                    {quickMessages.slice(0, 3).map((msg, index) => (
                      <button
                        key={index}
                        className="quick-msg-btn"
                        onClick={() => {
                          setInputMessage(msg);
                          sendMessage();
                        }}
                      >
                        {msg}
                      </button>
                    ))}
                  </div>

                  <div className="chat-input">
                    <button 
                      className="emoji-btn"
                      onClick={() => setShowEmojis(!showEmojis)}
                    >
                      😊
                    </button>
                    <input
                      type="text"
                      placeholder="Type a message..."
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    />
                    <button onClick={sendMessage} className="send-btn">
                      ➤
                    </button>
                  </div>

                  {showEmojis && (
                    <div className="emoji-picker">
                      {['😊', '😂', '❤️', '👍', '😭', '😱', '🔥', '🎉', '🤔', '😴'].map(emoji => (
                        <button
                          key={emoji}
                          onClick={() => {
                            setInputMessage(inputMessage + emoji);
                            setShowEmojis(false);
                          }}
                        >
                          {emoji}
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Controls Section */}
            <div className="controls-section">
              <div className="control-row">
                <button
                  className={`control-icon ${audioEnabled ? 'active' : ''}`}
                  onClick={() => toggleAudioVideo('audio')}
                  title="Toggle Microphone"
                >
                  {audioEnabled ? '🎤' : '🔇'}
                </button>
                <button
                  className={`control-icon ${videoEnabled ? 'active' : ''}`}
                  onClick={() => toggleAudioVideo('video')}
                  title="Toggle Camera"
                >
                  {videoEnabled ? '📹' : '📷'}
                </button>
                <button
                  className={`control-icon ${screenSharing ? 'active' : ''}`}
                  onClick={shareScreen}
                  title="Share Screen"
                >
                  🖥️
                </button>
                <button
                  className="control-icon"
                  onClick={() => setShowReactions(true)}
                  title="Send Reaction"
                >
                  😄
                </button>
                <button
                  className="control-icon"
                  onClick={inviteUsers}
                  title="Invite Users"
                >
                  ➕
                </button>
                <button
                  className="control-icon leave"
                  onClick={leaveParty}
                  title="Leave Party"
                >
                  🚪
                </button>
              </div>

              <div className="stats-row">
                <span className="stat">
                  <span className="stat-icon">👥</span>
                  {party.users.length}/{party.maxUsers}
                </span>
                <span className="stat">
                  <span className="stat-icon">📡</span>
                  {latency}ms
                </span>
                <span className="stat">
                  <span className="stat-icon">🔄</span>
                  {videoSync ? 'Synced' : 'Manual'}
                </span>
              </div>
            </div>
          </div>

          {/* Reactions Overlay */}
          <AnimatePresence>
            {showReactions && (
              <motion.div
                className="reactions-overlay"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                onClick={() => setShowReactions(false)}
              >
                <div className="reactions-grid">
                  {reactions.map((reaction, index) => (
                    <motion.button
                      key={reaction.emoji}
                      className="reaction-btn"
                      whileHover={{ scale: 1.2 }}
                      whileTap={{ scale: 0.9 }}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        sendReaction(reaction.emoji);
                      }}
                    >
                      <span className="reaction-emoji">{reaction.emoji}</span>
                      <span className="reaction-label">{reaction.label}</span>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export default SocialWatchParty;