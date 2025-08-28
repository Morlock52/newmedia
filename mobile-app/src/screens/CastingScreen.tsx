import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  Animated,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import {
  scanForDevices,
  connectToDevice,
  disconnectFromDevice,
  castMedia,
  controlPlayback,
  setVolume,
} from '../store/slices/castingSlice';

const { width } = Dimensions.get('window');

const CastingScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const {
    availableDevices,
    currentSession,
    isScanning,
    isConnecting,
    isPlaying,
    error,
  } = useSelector((state: RootState) => state.casting);
  
  const { currentlyPlaying } = useSelector((state: RootState) => state.media);
  
  const [scanAnim] = useState(new Animated.Value(0));
  const [fadeAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    // Animate in
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();

    // Start initial scan
    handleScanDevices();
  }, []);

  useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.timing(scanAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        })
      ).start();
    } else {
      scanAnim.stopAnimation();
      scanAnim.setValue(0);
    }
  }, [isScanning]);

  useEffect(() => {
    if (error) {
      Alert.alert('Casting Error', error);
    }
  }, [error]);

  const handleScanDevices = () => {
    dispatch(scanForDevices());
  };

  const handleConnectDevice = (deviceId: string) => {
    dispatch(connectToDevice(deviceId));
  };

  const handleDisconnect = () => {
    if (currentSession) {
      dispatch(disconnectFromDevice());
    }
  };

  const handleCastCurrentMedia = () => {
    if (!currentSession) {
      Alert.alert('No Device Connected', 'Please connect to a casting device first.');
      return;
    }

    if (!currentlyPlaying) {
      Alert.alert('No Media Selected', 'Please select media to cast from the library.');
      return;
    }

    dispatch(castMedia({
      mediaUrl: currentlyPlaying.playUrl || 'https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4',
      title: currentlyPlaying.title,
      subtitle: currentlyPlaying.description,
      imageUrl: currentlyPlaying.poster,
      contentType: 'video/mp4',
    }));
  };

  const handlePlayPause = () => {
    if (currentSession) {
      dispatch(controlPlayback(isPlaying ? 'pause' : 'play'));
    }
  };

  const handleStop = () => {
    if (currentSession) {
      dispatch(controlPlayback('stop'));
    }
  };

  const handleVolumeChange = (volume: number) => {
    if (currentSession) {
      dispatch(setVolume({ volume }));
    }
  };

  const handleMuteToggle = () => {
    if (currentSession) {
      dispatch(setVolume({ muted: !currentSession.muted }));
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'chromecast':
        return 'tv';
      case 'airplay':
        return 'logo-apple';
      case 'dlna':
        return 'desktop';
      default:
        return 'cast';
    }
  };

  const getDeviceColor = (type: string) => {
    switch (type) {
      case 'chromecast':
        return '#4285f4';
      case 'airplay':
        return '#ffffff';
      case 'dlna':
        return '#ffaa00';
      default:
        return '#666699';
    }
  };

  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <LinearGradient
      colors={['#0a0a0f', '#1a1a2e']}
      style={styles.container}
    >
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <Animated.View style={[styles.content, { opacity: fadeAnim }]}>
          
          {/* Current Session */}
          {currentSession ? (
            <View style={styles.sessionContainer}>
              <View style={styles.sessionHeader}>
                <Text style={styles.sessionTitle}>Connected to</Text>
                <TouchableOpacity
                  style={styles.disconnectButton}
                  onPress={handleDisconnect}
                >
                  <Ionicons name="close" size={20} color="#ff6b6b" />
                </TouchableOpacity>
              </View>
              
              <View style={styles.sessionDevice}>
                <LinearGradient
                  colors={['rgba(0, 255, 159, 0.1)', 'rgba(0, 255, 159, 0.05)']}
                  style={styles.sessionDeviceGradient}
                >
                  <Ionicons
                    name={getDeviceIcon(currentSession.type)}
                    size={32}
                    color="#00ff9f"
                  />
                  <Text style={styles.sessionDeviceName}>
                    {currentSession.deviceName}
                  </Text>
                  <Text style={styles.sessionDeviceType}>
                    {currentSession.type.toUpperCase()}
                  </Text>
                </LinearGradient>
              </View>

              {/* Media Control */}
              {currentSession.mediaInfo && (
                <View style={styles.mediaControl}>
                  <Text style={styles.mediaTitle}>
                    {currentSession.mediaInfo.title}
                  </Text>
                  {currentSession.mediaInfo.subtitle && (
                    <Text style={styles.mediaSubtitle}>
                      {currentSession.mediaInfo.subtitle}
                    </Text>
                  )}
                  
                  {/* Progress Bar */}
                  <View style={styles.progressContainer}>
                    <Text style={styles.timeText}>
                      {formatTime(currentSession.mediaInfo.currentTime || 0)}
                    </Text>
                    <View style={styles.progressBar}>
                      <View
                        style={[
                          styles.progressFill,
                          {
                            width: `${
                              ((currentSession.mediaInfo.currentTime || 0) /
                                (currentSession.mediaInfo.duration || 1)) * 100
                            }%`,
                          },
                        ]}
                      />
                    </View>
                    <Text style={styles.timeText}>
                      {formatTime(currentSession.mediaInfo.duration || 0)}
                    </Text>
                  </View>

                  {/* Playback Controls */}
                  <View style={styles.playbackControls}>
                    <TouchableOpacity
                      style={styles.controlButton}
                      onPress={handleStop}
                    >
                      <Ionicons name="stop" size={24} color="#ffffff" />
                    </TouchableOpacity>
                    
                    <TouchableOpacity
                      style={[styles.controlButton, styles.playButton]}
                      onPress={handlePlayPause}
                    >
                      <Ionicons
                        name={isPlaying ? "pause" : "play"}
                        size={28}
                        color="#000000"
                      />
                    </TouchableOpacity>
                    
                    <TouchableOpacity
                      style={styles.controlButton}
                      onPress={handleMuteToggle}
                    >
                      <Ionicons
                        name={currentSession.muted ? "volume-mute" : "volume-high"}
                        size={24}
                        color="#ffffff"
                      />
                    </TouchableOpacity>
                  </View>

                  {/* Volume Control */}
                  <View style={styles.volumeContainer}>
                    <Ionicons name="volume-low" size={16} color="#666699" />
                    <View style={styles.volumeSlider}>
                      <View
                        style={[
                          styles.volumeFill,
                          { width: `${currentSession.volume}%` },
                        ]}
                      />
                      <TouchableOpacity
                        style={[
                          styles.volumeThumb,
                          { left: `${currentSession.volume}%` },
                        ]}
                      />
                    </View>
                    <Ionicons name="volume-high" size={16} color="#666699" />
                  </View>
                </View>
              )}

              {/* Cast Media Button */}
              {!currentSession.mediaInfo && (
                <TouchableOpacity
                  style={styles.castMediaButton}
                  onPress={handleCastCurrentMedia}
                >
                  <LinearGradient
                    colors={['#00ff9f', '#00cc7f']}
                    style={styles.castMediaGradient}
                  >
                    <Ionicons name="cast" size={20} color="#000000" />
                    <Text style={styles.castMediaText}>Cast Current Media</Text>
                  </LinearGradient>
                </TouchableOpacity>
              )}
            </View>
          ) : (
            /* Device Discovery */
            <View style={styles.discoveryContainer}>
              <View style={styles.discoveryHeader}>
                <Text style={styles.discoveryTitle}>Available Devices</Text>
                <TouchableOpacity
                  style={styles.scanButton}
                  onPress={handleScanDevices}
                  disabled={isScanning}
                >
                  <Animated.View
                    style={{
                      transform: [
                        {
                          rotate: scanAnim.interpolate({
                            inputRange: [0, 1],
                            outputRange: ['0deg', '360deg'],
                          }),
                        },
                      ],
                    }}
                  >
                    <Ionicons
                      name="refresh"
                      size={20}
                      color={isScanning ? '#666699' : '#00ff9f'}
                    />
                  </Animated.View>
                </TouchableOpacity>
              </View>

              {isScanning && (
                <View style={styles.scanningIndicator}>
                  <Text style={styles.scanningText}>Scanning for devices...</Text>
                </View>
              )}

              {availableDevices.map((device) => (
                <TouchableOpacity
                  key={device.id}
                  style={styles.deviceCard}
                  onPress={() => handleConnectDevice(device.id)}
                  disabled={isConnecting || device.status !== 'available'}
                >
                  <LinearGradient
                    colors={[
                      device.status === 'available'
                        ? 'rgba(26, 26, 46, 0.8)'
                        : 'rgba(102, 102, 153, 0.3)',
                      device.status === 'available'
                        ? 'rgba(22, 33, 62, 0.6)'
                        : 'rgba(102, 102, 153, 0.1)',
                    ]}
                    style={styles.deviceCardGradient}
                  >
                    <View style={styles.deviceInfo}>
                      <Ionicons
                        name={getDeviceIcon(device.type)}
                        size={24}
                        color={
                          device.status === 'available'
                            ? getDeviceColor(device.type)
                            : '#666699'
                        }
                      />
                      <View style={styles.deviceDetails}>
                        <Text style={styles.deviceName}>{device.name}</Text>
                        <Text style={styles.deviceType}>
                          {device.type.toUpperCase()} • {device.status}
                        </Text>
                        <Text style={styles.deviceCapabilities}>
                          {device.capabilities.join(', ')}
                        </Text>
                      </View>
                    </View>
                    
                    <View style={styles.deviceActions}>
                      {isConnecting ? (
                        <Ionicons name="sync" size={20} color="#666699" />
                      ) : (
                        <Ionicons
                          name="chevron-forward"
                          size={20}
                          color={device.status === 'available' ? '#00ff9f' : '#666699'}
                        />
                      )}
                    </View>
                  </LinearGradient>
                </TouchableOpacity>
              ))}

              {!isScanning && availableDevices.length === 0 && (
                <View style={styles.noDevicesContainer}>
                  <Ionicons name="cast" size={48} color="#666699" />
                  <Text style={styles.noDevicesText}>No devices found</Text>
                  <Text style={styles.noDevicesSubtext}>
                    Make sure your casting devices are on the same network
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Casting Info */}
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Casting Support</Text>
            <View style={styles.infoGrid}>
              <View style={styles.infoCard}>
                <Ionicons name="tv" size={20} color="#4285f4" />
                <Text style={styles.infoText}>Chromecast</Text>
              </View>
              <View style={styles.infoCard}>
                <Ionicons name="logo-apple" size={20} color="#ffffff" />
                <Text style={styles.infoText}>AirPlay</Text>
              </View>
              <View style={styles.infoCard}>
                <Ionicons name="desktop" size={20} color="#ffaa00" />
                <Text style={styles.infoText}>DLNA</Text>
              </View>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  sessionContainer: {
    marginBottom: 24,
  },
  sessionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sessionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  disconnectButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 107, 107, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sessionDevice: {
    marginBottom: 24,
    borderRadius: 12,
    overflow: 'hidden',
  },
  sessionDeviceGradient: {
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#00ff9f40',
  },
  sessionDeviceName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 12,
    textAlign: 'center',
  },
  sessionDeviceType: {
    fontSize: 14,
    color: '#00ff9f',
    marginTop: 4,
    fontWeight: '600',
  },
  mediaControl: {
    backgroundColor: 'rgba(26, 26, 46, 0.8)',
    borderRadius: 12,
    padding: 20,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  mediaTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 8,
  },
  mediaSubtitle: {
    fontSize: 14,
    color: '#666699',
    textAlign: 'center',
    marginBottom: 20,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  timeText: {
    fontSize: 12,
    color: '#666699',
    width: 40,
    textAlign: 'center',
  },
  progressBar: {
    flex: 1,
    height: 4,
    backgroundColor: '#16213e',
    borderRadius: 2,
    marginHorizontal: 12,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#00ff9f',
    borderRadius: 2,
  },
  playbackControls: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  controlButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(102, 102, 153, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 12,
  },
  playButton: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#00ff9f',
    marginHorizontal: 20,
  },
  volumeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  volumeSlider: {
    flex: 1,
    height: 4,
    backgroundColor: '#16213e',
    borderRadius: 2,
    marginHorizontal: 12,
    position: 'relative',
  },
  volumeFill: {
    height: '100%',
    backgroundColor: '#00ff9f',
    borderRadius: 2,
  },
  volumeThumb: {
    position: 'absolute',
    top: -6,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#00ff9f',
    marginLeft: -8,
  },
  castMediaButton: {
    marginTop: 16,
    borderRadius: 8,
    overflow: 'hidden',
  },
  castMediaGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  castMediaText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
    marginLeft: 8,
  },
  discoveryContainer: {
    marginBottom: 24,
  },
  discoveryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  discoveryTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    textShadowColor: '#00ff9f',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
  },
  scanButton: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanningIndicator: {
    padding: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  scanningText: {
    fontSize: 14,
    color: '#666699',
  },
  deviceCard: {
    marginBottom: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  deviceCardGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  deviceInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  deviceDetails: {
    flex: 1,
    marginLeft: 16,
  },
  deviceName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 4,
  },
  deviceType: {
    fontSize: 12,
    color: '#666699',
    marginBottom: 2,
  },
  deviceCapabilities: {
    fontSize: 10,
    color: '#666699',
    textTransform: 'capitalize',
  },
  deviceActions: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  noDevicesContainer: {
    alignItems: 'center',
    padding: 40,
  },
  noDevicesText: {
    fontSize: 16,
    color: '#666699',
    marginTop: 16,
    marginBottom: 8,
  },
  noDevicesSubtext: {
    fontSize: 14,
    color: '#666699',
    textAlign: 'center',
    opacity: 0.7,
  },
  infoContainer: {
    backgroundColor: 'rgba(22, 33, 62, 0.5)',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 12,
    textAlign: 'center',
  },
  infoGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  infoCard: {
    alignItems: 'center',
  },
  infoText: {
    fontSize: 12,
    color: '#666699',
    marginTop: 8,
    textAlign: 'center',
  },
});

export default CastingScreen;