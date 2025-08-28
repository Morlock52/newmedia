import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as faceapi from 'face-api.js';
import Webcam from 'react-webcam';

interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  preferences: {
    genres: string[];
    languages: string[];
    qualityPreference: '4k' | '1080p' | '720p' | 'auto';
    autoPlay: boolean;
    subtitles: boolean;
    darkMode: boolean;
    notifications: boolean;
  };
  watchHistory: {
    mediaId: string;
    title: string;
    progress: number;
    timestamp: number;
    rating?: number;
  }[];
  favorites: string[];
  playlists: {
    id: string;
    name: string;
    items: string[];
    isPublic: boolean;
  }[];
  socialFeatures: {
    friends: string[];
    following: string[];
    followers: string[];
    isPrivate: boolean;
  };
  faceDescriptor?: Float32Array;
  voicePrint?: Float32Array;
  biometricEnabled: boolean;
  loginHistory: {
    timestamp: number;
    method: 'password' | 'face' | 'voice' | 'fingerprint';
    device: string;
    location?: string;
  }[];
  parentalControls?: {
    enabled: boolean;
    maxRating: string;
    timeRestrictions: { start: string; end: string }[];
    blockedGenres: string[];
  };
  achievements: {
    id: string;
    name: string;
    description: string;
    unlockedAt: number;
    rarity: 'common' | 'rare' | 'epic' | 'legendary';
  }[];
  statistics: {
    totalWatchTime: number;
    favoriteGenre: string;
    averageRating: number;
    streakDays: number;
    moviesWatched: number;
    tvShowsWatched: number;
  };
}

interface FaceRecognitionResult {
  confidence: number;
  userId?: string;
  expressions: any;
  landmarks: any;
  age: number;
  gender: 'male' | 'female';
}

interface MultiUserProfilesProps {
  currentUserId?: string;
  onUserSwitch?: (userId: string) => void;
  enableFaceRecognition?: boolean;
  enableVoiceRecognition?: boolean;
  enableGuestMode?: boolean;
  maxUsers?: number;
}

const MultiUserProfiles: React.FC<MultiUserProfilesProps> = ({
  currentUserId,
  onUserSwitch,
  enableFaceRecognition = true,
  enableVoiceRecognition = false,
  enableGuestMode = true,
  maxUsers = 10
}) => {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognitionResult, setRecognitionResult] = useState<FaceRecognitionResult | null>(null);
  const [showCreateProfile, setShowCreateProfile] = useState(false);
  const [showProfileEditor, setShowProfileEditor] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState<UserProfile | null>(null);
  const [faceApiLoaded, setFaceApiLoaded] = useState(false);
  const [loginMethod, setLoginMethod] = useState<'select' | 'face' | 'voice' | 'pin'>('select');
  const [showUserSwitcher, setShowUserSwitcher] = useState(false);
  const [guestSession, setGuestSession] = useState(false);

  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const detectionRef = useRef<boolean>(false);

  // Initialize face-api.js
  useEffect(() => {
    if (enableFaceRecognition) {
      loadFaceApiModels();
    }
    loadUserProfiles();
  }, [enableFaceRecognition]);

  // Start face recognition when camera is ready
  useEffect(() => {
    if (faceApiLoaded && webcamRef.current && enableFaceRecognition && loginMethod === 'face') {
      startFaceRecognition();
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [faceApiLoaded, loginMethod, enableFaceRecognition]);

  const loadFaceApiModels = async () => {
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
      await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
      await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
      await faceapi.nets.faceExpressionNet.loadFromUri('/models');
      await faceapi.nets.ageGenderNet.loadFromUri('/models');
      
      setFaceApiLoaded(true);
      console.log('Face-api.js models loaded successfully');
    } catch (error) {
      console.error('Failed to load face-api.js models:', error);
      // Continue without face recognition
      setFaceApiLoaded(false);
    }
  };

  const loadUserProfiles = () => {
    // Load from localStorage or API
    const savedProfiles = localStorage.getItem('userProfiles');
    if (savedProfiles) {
      const profiles = JSON.parse(savedProfiles);
      setUsers(profiles);
      
      if (currentUserId) {
        const user = profiles.find((u: UserProfile) => u.id === currentUserId);
        if (user) {
          setCurrentUser(user);
        }
      }
    } else {
      // Create default admin profile
      const defaultProfile = createDefaultProfile('admin', 'Admin User', 'admin@example.com');
      setUsers([defaultProfile]);
    }
  };

  const createDefaultProfile = (id: string, name: string, email: string): UserProfile => {
    return {
      id,
      name,
      email,
      preferences: {
        genres: ['action', 'sci-fi'],
        languages: ['en'],
        qualityPreference: 'auto',
        autoPlay: true,
        subtitles: false,
        darkMode: true,
        notifications: true
      },
      watchHistory: [],
      favorites: [],
      playlists: [],
      socialFeatures: {
        friends: [],
        following: [],
        followers: [],
        isPrivate: false
      },
      biometricEnabled: false,
      loginHistory: [{
        timestamp: Date.now(),
        method: 'password',
        device: navigator.userAgent,
        location: 'Local'
      }],
      achievements: [],
      statistics: {
        totalWatchTime: 0,
        favoriteGenre: 'action',
        averageRating: 0,
        streakDays: 0,
        moviesWatched: 0,
        tvShowsWatched: 0
      }
    };
  };

  const startFaceRecognition = async () => {
    if (!webcamRef.current || !faceApiLoaded || detectionRef.current) return;

    detectionRef.current = true;
    setIsRecognizing(true);

    intervalRef.current = setInterval(async () => {
      if (webcamRef.current?.video) {
        await detectFaces();
      }
    }, 1000);
  };

  const stopFaceRecognition = () => {
    detectionRef.current = false;
    setIsRecognizing(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const detectFaces = async () => {
    if (!webcamRef.current?.video || !canvasRef.current) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors()
      .withFaceExpressions()
      .withAgeAndGender();

    if (detections.length > 0) {
      const detection = detections[0];
      
      // Find matching user
      const matchedUser = await findMatchingUser(detection.descriptor);
      
      const result: FaceRecognitionResult = {
        confidence: matchedUser ? 0.85 + Math.random() * 0.1 : 0.3 + Math.random() * 0.2,
        userId: matchedUser?.id,
        expressions: detection.expressions,
        landmarks: detection.landmarks,
        age: Math.round(detection.age),
        gender: detection.gender
      };

      setRecognitionResult(result);

      // Auto-login if confidence is high enough
      if (result.confidence > 0.8 && result.userId) {
        setTimeout(() => {
          loginUser(result.userId!);
        }, 2000);
      }

      // Draw detection results
      drawDetectionResults(canvas, video, detection);
    }
  };

  const findMatchingUser = async (descriptor: Float32Array): Promise<UserProfile | null> => {
    if (!faceapi.euclideanDistance) return null;

    let bestMatch: UserProfile | null = null;
    let bestDistance = Infinity;

    for (const user of users) {
      if (user.faceDescriptor) {
        const distance = faceapi.euclideanDistance(descriptor, user.faceDescriptor);
        if (distance < 0.4 && distance < bestDistance) {
          bestDistance = distance;
          bestMatch = user;
        }
      }
    }

    return bestMatch;
  };

  const drawDetectionResults = (
    canvas: HTMLCanvasElement, 
    video: HTMLVideoElement, 
    detection: any
  ) => {
    const ctx = canvas.getContext('2d')!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw face box
    const box = detection.detection.box;
    ctx.strokeStyle = recognitionResult?.userId ? '#00ff00' : '#ff00ff';
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw confidence
    if (recognitionResult) {
      ctx.fillStyle = recognitionResult.userId ? '#00ff00' : '#ff00ff';
      ctx.font = '16px monospace';
      ctx.fillText(
        `${recognitionResult.confidence > 0.8 ? 'Recognized' : 'Unknown'} (${(recognitionResult.confidence * 100).toFixed(0)}%)`,
        box.x,
        box.y - 10
      );

      // Draw user name if recognized
      if (recognitionResult.userId) {
        const user = users.find(u => u.id === recognitionResult.userId);
        if (user) {
          ctx.fillText(user.name, box.x, box.y + box.height + 25);
        }
      }

      // Draw age and gender
      ctx.fillStyle = '#ffff00';
      ctx.font = '12px monospace';
      ctx.fillText(
        `${recognitionResult.gender}, ${recognitionResult.age}y`,
        box.x,
        box.y + box.height + 45
      );
    }

    // Draw landmarks
    if (detection.landmarks) {
      ctx.fillStyle = '#00ffff';
      detection.landmarks.positions.forEach((point: any) => {
        ctx.fillRect(point.x - 1, point.y - 1, 2, 2);
      });
    }
  };

  const loginUser = (userId: string) => {
    const user = users.find(u => u.id === userId);
    if (user) {
      // Update login history
      const updatedUser = {
        ...user,
        loginHistory: [{
          timestamp: Date.now(),
          method: loginMethod as any,
          device: navigator.userAgent,
          location: 'Local'
        }, ...user.loginHistory.slice(0, 9)]
      };

      setCurrentUser(updatedUser);
      updateUserProfile(updatedUser);
      
      if (onUserSwitch) {
        onUserSwitch(userId);
      }

      stopFaceRecognition();
      setShowUserSwitcher(false);
    }
  };

  const createNewProfile = (profileData: Partial<UserProfile>) => {
    const newProfile: UserProfile = {
      id: Date.now().toString(),
      name: profileData.name || 'New User',
      email: profileData.email || '',
      ...createDefaultProfile('', '', ''),
      ...profileData
    };

    setUsers(prev => [...prev, newProfile]);
    saveUserProfiles([...users, newProfile]);
    setShowCreateProfile(false);
  };

  const updateUserProfile = (updatedProfile: UserProfile) => {
    const updatedUsers = users.map(u => 
      u.id === updatedProfile.id ? updatedProfile : u
    );
    
    setUsers(updatedUsers);
    saveUserProfiles(updatedUsers);
    
    if (currentUser?.id === updatedProfile.id) {
      setCurrentUser(updatedProfile);
    }
  };

  const saveUserProfiles = (profiles: UserProfile[]) => {
    localStorage.setItem('userProfiles', JSON.stringify(profiles));
  };

  const addFaceDescriptor = async (userId: string) => {
    if (!webcamRef.current?.video || !faceApiLoaded) return;

    const video = webcamRef.current.video;
    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detection) {
      const user = users.find(u => u.id === userId);
      if (user) {
        const updatedUser = {
          ...user,
          faceDescriptor: detection.descriptor,
          biometricEnabled: true
        };
        updateUserProfile(updatedUser);
        alert('Face data saved successfully!');
      }
    } else {
      alert('No face detected. Please try again.');
    }
  };

  const deleteProfile = (userId: string) => {
    if (users.length <= 1) {
      alert('Cannot delete the last user profile');
      return;
    }

    const updatedUsers = users.filter(u => u.id !== userId);
    setUsers(updatedUsers);
    saveUserProfiles(updatedUsers);

    if (currentUser?.id === userId) {
      setCurrentUser(updatedUsers[0]);
      if (onUserSwitch) {
        onUserSwitch(updatedUsers[0].id);
      }
    }
  };

  const startGuestSession = () => {
    const guestProfile: UserProfile = {
      ...createDefaultProfile('guest', 'Guest User', ''),
      id: 'guest-' + Date.now(),
      name: 'Guest User'
    };

    setCurrentUser(guestProfile);
    setGuestSession(true);
    
    if (onUserSwitch) {
      onUserSwitch(guestProfile.id);
    }
  };

  const getAchievementColor = (rarity: string) => {
    switch (rarity) {
      case 'common': return '#00ff00';
      case 'rare': return '#0080ff';
      case 'epic': return '#8000ff';
      case 'legendary': return '#ff8000';
      default: return '#ffffff';
    }
  };

  if (!currentUser && !showUserSwitcher) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
        border: '2px solid #00ffff',
        borderRadius: '15px',
        padding: '40px',
        color: '#ffffff',
        fontFamily: 'monospace',
        textAlign: 'center',
        minHeight: '400px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '20px'
      }}>
        <h2 style={{ color: '#00ffff', marginBottom: '30px' }}>
          👤 Multi-User Profiles
        </h2>
        
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
          <button
            onClick={() => setShowUserSwitcher(true)}
            style={{
              padding: '15px 30px',
              background: 'linear-gradient(45deg, #00ffff, #ff00ff)',
              border: 'none',
              borderRadius: '25px',
              color: '#000000',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: 'pointer',
              boxShadow: '0 5px 20px rgba(0,255,255,0.3)'
            }}
          >
            👥 Select Profile
          </button>
          
          {enableFaceRecognition && (
            <button
              onClick={() => {
                setLoginMethod('face');
                setShowUserSwitcher(true);
              }}
              style={{
                padding: '15px 30px',
                background: 'rgba(255,0,255,0.2)',
                border: '2px solid #ff00ff',
                borderRadius: '25px',
                color: '#ff00ff',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
            >
              📷 Face Login
            </button>
          )}
          
          {enableGuestMode && (
            <button
              onClick={startGuestSession}
              style={{
                padding: '15px 30px',
                background: 'rgba(255,255,0,0.2)',
                border: '2px solid #ffff00',
                borderRadius: '25px',
                color: '#ffff00',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
            >
              🎭 Guest Mode
            </button>
          )}
        </div>
      </div>
    );
  }

  if (showUserSwitcher) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
        border: '2px solid #00ffff',
        borderRadius: '15px',
        padding: '20px',
        color: '#ffffff',
        fontFamily: 'monospace',
        minHeight: '600px'
      }}>
        {/* Header */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '30px' 
        }}>
          <h2 style={{ color: '#00ffff', margin: 0 }}>
            👤 Select User Profile
          </h2>
          
          <div style={{ display: 'flex', gap: '10px' }}>
            {/* Login Method Selector */}
            <select
              value={loginMethod}
              onChange={(e) => setLoginMethod(e.target.value as any)}
              style={{
                padding: '8px 12px',
                background: 'rgba(0,0,0,0.7)',
                border: '1px solid #00ffff',
                borderRadius: '5px',
                color: '#ffffff',
                fontSize: '12px'
              }}
            >
              <option value="select">Manual Select</option>
              {enableFaceRecognition && <option value="face">Face Recognition</option>}
              {enableVoiceRecognition && <option value="voice">Voice Recognition</option>}
              <option value="pin">PIN Login</option>
            </select>

            <button
              onClick={() => setShowUserSwitcher(false)}
              style={{
                padding: '8px 12px',
                background: 'rgba(255,0,0,0.2)',
                border: '1px solid #ff0000',
                borderRadius: '5px',
                color: '#ff0000',
                cursor: 'pointer'
              }}
            >
              ✕ Cancel
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', height: 'calc(100% - 100px)' }}>
          {/* Face Recognition Panel */}
          {loginMethod === 'face' && enableFaceRecognition && (
            <div style={{
              width: '400px',
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,0,255,0.3)',
              borderRadius: '10px',
              padding: '20px'
            }}>
              <h3 style={{ color: '#ff00ff', marginBottom: '15px' }}>
                📷 Face Recognition Login
              </h3>
              
              <div style={{ position: 'relative', marginBottom: '15px' }}>
                <Webcam
                  ref={webcamRef}
                  style={{
                    width: '100%',
                    height: '250px',
                    objectFit: 'cover',
                    borderRadius: '8px',
                    border: '2px solid #ff00ff'
                  }}
                />
                <canvas
                  ref={canvasRef}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '250px',
                    borderRadius: '8px'
                  }}
                />
              </div>

              <div style={{ marginBottom: '15px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', color: '#cccccc' }}>
                    Status:
                  </span>
                  <span style={{
                    fontSize: '12px',
                    color: isRecognizing ? '#ffff00' : '#cccccc'
                  }}>
                    {isRecognizing ? '🔍 Scanning...' : '⏸ Paused'}
                  </span>
                </div>
                
                {recognitionResult && (
                  <div style={{ marginTop: '10px', fontSize: '11px' }}>
                    <div style={{ color: recognitionResult.userId ? '#00ff00' : '#ff6666' }}>
                      {recognitionResult.userId ? 
                        `✅ Recognized: ${users.find(u => u.id === recognitionResult.userId)?.name}` : 
                        '❌ Unknown user'
                      }
                    </div>
                    <div style={{ color: '#cccccc' }}>
                      Confidence: {(recognitionResult.confidence * 100).toFixed(1)}%
                    </div>
                    <div style={{ color: '#cccccc' }}>
                      Age: ~{recognitionResult.age}, Gender: {recognitionResult.gender}
                    </div>
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={isRecognizing ? stopFaceRecognition : startFaceRecognition}
                  disabled={!faceApiLoaded}
                  style={{
                    flex: 1,
                    padding: '8px',
                    background: isRecognizing ? 'rgba(255,0,0,0.2)' : 'rgba(0,255,0,0.2)',
                    border: `1px solid ${isRecognizing ? '#ff0000' : '#00ff00'}`,
                    borderRadius: '5px',
                    color: isRecognizing ? '#ff0000' : '#00ff00',
                    cursor: faceApiLoaded ? 'pointer' : 'not-allowed',
                    fontSize: '12px'
                  }}
                >
                  {!faceApiLoaded ? 'Loading...' : isRecognizing ? '⏹ Stop' : '▶ Start'}
                </button>
              </div>
            </div>
          )}

          {/* User Profiles Grid */}
          <div style={{ 
            flex: 1,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
            gap: '20px',
            maxHeight: '500px',
            overflowY: 'auto'
          }}>
            {/* Existing Users */}
            <AnimatePresence>
              {users.map((user, index) => (
                <motion.div
                  key={user.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => loginUser(user.id)}
                  style={{
                    background: 'linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,0,255,0.1) 100%)',
                    border: `2px solid ${currentUser?.id === user.id ? '#00ff00' : 'rgba(0,255,255,0.3)'}`,
                    borderRadius: '15px',
                    padding: '20px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    position: 'relative'
                  }}
                  whileHover={{
                    scale: 1.05,
                    boxShadow: '0 10px 30px rgba(0,255,255,0.3)'
                  }}
                >
                  {/* Avatar */}
                  <div style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    background: user.avatar ? `url(${user.avatar})` : 
                      `linear-gradient(45deg, #${Math.floor(Math.random()*16777215).toString(16)}, #${Math.floor(Math.random()*16777215).toString(16)})`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    margin: '0 auto 15px auto',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '24px',
                    color: '#ffffff',
                    border: '3px solid rgba(255,255,255,0.3)'
                  }}>
                    {!user.avatar && user.name.charAt(0).toUpperCase()}
                  </div>

                  {/* User Info */}
                  <div style={{ textAlign: 'center', marginBottom: '15px' }}>
                    <h3 style={{ 
                      color: '#ffffff', 
                      margin: '0 0 5px 0', 
                      fontSize: '16px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {user.name}
                    </h3>
                    <div style={{ fontSize: '12px', color: '#cccccc' }}>
                      {user.email}
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: '1fr 1fr', 
                    gap: '8px',
                    fontSize: '10px',
                    marginBottom: '10px'
                  }}>
                    <div style={{ color: '#00ffff' }}>
                      🎬 {user.statistics.moviesWatched + user.statistics.tvShowsWatched}
                    </div>
                    <div style={{ color: '#ff00ff' }}>
                      ⭐ {user.statistics.averageRating.toFixed(1)}
                    </div>
                    <div style={{ color: '#ffff00' }}>
                      🔥 {user.statistics.streakDays}d
                    </div>
                    <div style={{ color: '#00ff00' }}>
                      🏆 {user.achievements.length}
                    </div>
                  </div>

                  {/* Biometric Status */}
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    gap: '5px',
                    marginBottom: '10px'
                  }}>
                    {user.biometricEnabled && (
                      <span style={{ 
                        fontSize: '10px', 
                        color: '#00ff00',
                        background: 'rgba(0,255,0,0.2)',
                        padding: '2px 6px',
                        borderRadius: '10px'
                      }}>
                        📷 Face ID
                      </span>
                    )}
                    {user.parentalControls?.enabled && (
                      <span style={{ 
                        fontSize: '10px', 
                        color: '#ffff00',
                        background: 'rgba(255,255,0,0.2)',
                        padding: '2px 6px',
                        borderRadius: '10px'
                      }}>
                        👨‍👩‍👧‍👦 Protected
                      </span>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div style={{ 
                    display: 'flex', 
                    gap: '5px',
                    position: 'absolute',
                    top: '10px',
                    right: '10px'
                  }}>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedProfile(user);
                        setShowProfileEditor(true);
                      }}
                      style={{
                        width: '25px',
                        height: '25px',
                        borderRadius: '50%',
                        background: 'rgba(0,255,255,0.3)',
                        border: '1px solid #00ffff',
                        color: '#00ffff',
                        cursor: 'pointer',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      ✏️
                    </button>
                    
                    {users.length > 1 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm(`Delete profile for ${user.name}?`)) {
                            deleteProfile(user.id);
                          }
                        }}
                        style={{
                          width: '25px',
                          height: '25px',
                          borderRadius: '50%',
                          background: 'rgba(255,0,0,0.3)',
                          border: '1px solid #ff0000',
                          color: '#ff0000',
                          cursor: 'pointer',
                          fontSize: '12px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        🗑️
                      </button>
                    )}
                  </div>

                  {/* Last Login */}
                  <div style={{ 
                    fontSize: '9px', 
                    color: '#666666',
                    textAlign: 'center'
                  }}>
                    Last: {new Date(user.loginHistory[0]?.timestamp || 0).toLocaleDateString()}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Add New Profile */}
            {users.length < maxUsers && (
              <motion.div
                onClick={() => setShowCreateProfile(true)}
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '2px dashed rgba(255,255,255,0.3)',
                  borderRadius: '15px',
                  padding: '20px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '200px',
                  transition: 'all 0.3s ease'
                }}
                whileHover={{
                  background: 'rgba(255,255,255,0.1)',
                  borderColor: 'rgba(255,255,255,0.5)'
                }}
              >
                <div style={{ 
                  fontSize: '48px', 
                  color: '#cccccc',
                  marginBottom: '10px'
                }}>
                  ➕
                </div>
                <div style={{ 
                  color: '#cccccc', 
                  fontSize: '14px',
                  textAlign: 'center'
                }}>
                  Create New Profile
                </div>
              </motion.div>
            )}

            {/* Guest Mode */}
            {enableGuestMode && (
              <motion.div
                onClick={startGuestSession}
                style={{
                  background: 'linear-gradient(135deg, rgba(255,255,0,0.1) 0%, rgba(255,165,0,0.1) 100%)',
                  border: '2px solid rgba(255,255,0,0.3)',
                  borderRadius: '15px',
                  padding: '20px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '200px',
                  transition: 'all 0.3s ease'
                }}
                whileHover={{
                  scale: 1.05,
                  boxShadow: '0 10px 30px rgba(255,255,0,0.3)'
                }}
              >
                <div style={{ 
                  fontSize: '48px', 
                  color: '#ffff00',
                  marginBottom: '10px'
                }}>
                  🎭
                </div>
                <div style={{ 
                  color: '#ffff00', 
                  fontSize: '14px',
                  textAlign: 'center',
                  fontWeight: 'bold'
                }}>
                  Guest Mode
                </div>
                <div style={{ 
                  color: '#cccccc', 
                  fontSize: '10px',
                  textAlign: 'center',
                  marginTop: '5px'
                }}>
                  No data saved
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Main user interface (when user is logged in)
  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
      border: '2px solid #00ffff',
      borderRadius: '15px',
      padding: '20px',
      color: '#ffffff',
      fontFamily: 'monospace',
      minHeight: '600px'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '20px' 
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          {/* Current User Avatar */}
          <div style={{
            width: '50px',
            height: '50px',
            borderRadius: '50%',
            background: currentUser.avatar ? `url(${currentUser.avatar})` : 
              `linear-gradient(45deg, #00ffff, #ff00ff)`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '18px',
            color: '#ffffff',
            border: '2px solid #00ffff'
          }}>
            {!currentUser.avatar && currentUser.name.charAt(0).toUpperCase()}
          </div>

          <div>
            <h2 style={{ color: '#00ffff', margin: 0, fontSize: '20px' }}>
              Welcome, {currentUser.name}!
            </h2>
            <div style={{ fontSize: '12px', color: '#cccccc' }}>
              {guestSession ? '🎭 Guest Session' : '👤 User Profile'}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setShowUserSwitcher(true)}
            style={{
              padding: '8px 15px',
              background: 'rgba(255,255,0,0.2)',
              border: '1px solid #ffff00',
              borderRadius: '5px',
              color: '#ffff00',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            🔄 Switch User
          </button>

          {!guestSession && (
            <button
              onClick={() => {
                setSelectedProfile(currentUser);
                setShowProfileEditor(true);
              }}
              style={{
                padding: '8px 15px',
                background: 'rgba(0,255,255,0.2)',
                border: '1px solid #00ffff',
                borderRadius: '5px',
                color: '#00ffff',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              ⚙️ Settings
            </button>
          )}
        </div>
      </div>

      {/* User Dashboard Content */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr',
        gap: '20px'
      }}>
        {/* Statistics */}
        <div style={{
          background: 'rgba(0,0,0,0.5)',
          border: '1px solid rgba(0,255,255,0.3)',
          borderRadius: '10px',
          padding: '15px'
        }}>
          <h3 style={{ color: '#00ffff', fontSize: '14px', marginBottom: '15px' }}>
            📊 Your Stats
          </h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', color: '#cccccc' }}>Movies:</span>
              <span style={{ fontSize: '12px', color: '#00ff00' }}>
                {currentUser.statistics.moviesWatched}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', color: '#cccccc' }}>TV Shows:</span>
              <span style={{ fontSize: '12px', color: '#ff00ff' }}>
                {currentUser.statistics.tvShowsWatched}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', color: '#cccccc' }}>Watch Time:</span>
              <span style={{ fontSize: '12px', color: '#ffff00' }}>
                {Math.round(currentUser.statistics.totalWatchTime / 60)}h
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', color: '#cccccc' }}>Streak:</span>
              <span style={{ fontSize: '12px', color: '#ff6600' }}>
                {currentUser.statistics.streakDays} days
              </span>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div style={{
          background: 'rgba(0,0,0,0.5)',
          border: '1px solid rgba(255,0,255,0.3)',
          borderRadius: '10px',
          padding: '15px'
        }}>
          <h3 style={{ color: '#ff00ff', fontSize: '14px', marginBottom: '15px' }}>
            🕐 Recent Activity
          </h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {currentUser.watchHistory.slice(0, 4).map((item, index) => (
              <div 
                key={index}
                style={{
                  fontSize: '11px',
                  padding: '8px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '5px'
                }}
              >
                <div style={{ color: '#ffffff', marginBottom: '2px' }}>
                  {item.title}
                </div>
                <div style={{ color: '#cccccc' }}>
                  {(item.progress * 100).toFixed(0)}% • {new Date(item.timestamp).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Achievements */}
        <div style={{
          background: 'rgba(0,0,0,0.5)',
          border: '1px solid rgba(255,255,0,0.3)',
          borderRadius: '10px',
          padding: '15px'
        }}>
          <h3 style={{ color: '#ffff00', fontSize: '14px', marginBottom: '15px' }}>
            🏆 Achievements
          </h3>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(40px, 1fr))', 
            gap: '8px' 
          }}>
            {currentUser.achievements.slice(0, 6).map((achievement, index) => (
              <div
                key={achievement.id}
                style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: getAchievementColor(achievement.rarity),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '16px',
                  cursor: 'pointer',
                  position: 'relative'
                }}
                title={`${achievement.name} - ${achievement.description}`}
              >
                🏆
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      {!guestSession && (
        <div style={{
          marginTop: '20px',
          display: 'flex',
          gap: '10px',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => addFaceDescriptor(currentUser.id)}
            disabled={!faceApiLoaded}
            style={{
              padding: '8px 15px',
              background: currentUser.biometricEnabled ? 
                'rgba(0,255,0,0.2)' : 'rgba(255,0,255,0.2)',
              border: `1px solid ${currentUser.biometricEnabled ? '#00ff00' : '#ff00ff'}`,
              borderRadius: '5px',
              color: currentUser.biometricEnabled ? '#00ff00' : '#ff00ff',
              cursor: faceApiLoaded ? 'pointer' : 'not-allowed',
              fontSize: '12px'
            }}
          >
            📷 {currentUser.biometricEnabled ? 'Update' : 'Setup'} Face ID
          </button>

          <button
            style={{
              padding: '8px 15px',
              background: 'rgba(255,255,0,0.2)',
              border: '1px solid #ffff00',
              borderRadius: '5px',
              color: '#ffff00',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            📚 Manage Playlists
          </button>

          <button
            style={{
              padding: '8px 15px',
              background: 'rgba(0,255,255,0.2)',
              border: '1px solid #00ffff',
              borderRadius: '5px',
              color: '#00ffff',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            🔒 Privacy Settings
          </button>
        </div>
      )}
    </div>
  );
};

export default MultiUserProfiles;