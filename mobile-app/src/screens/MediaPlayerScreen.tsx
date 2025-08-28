import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Video, ResizeMode } from 'expo-av';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';

const { width, height } = Dimensions.get('window');

const MediaPlayerScreen: React.FC = () => {
  const navigation = useNavigation();
  const route = useRoute();
  const { mediaItem } = route.params as any;
  
  const [status, setStatus] = useState<any>({});
  const [controlsVisible, setControlsVisible] = useState(true);
  const [fadeAnim] = useState(new Animated.Value(1));
  
  const hideControlsTimeout = useRef<NodeJS.Timeout | null>(null);

  const toggleControls = () => {
    setControlsVisible(!controlsVisible);
    
    if (hideControlsTimeout.current) {
      clearTimeout(hideControlsTimeout.current);
    }
    
    if (!controlsVisible) {
      hideControlsTimeout.current = setTimeout(() => {
        setControlsVisible(false);
      }, 3000);
    }
  };

  const formatTime = (milliseconds: number) => {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <View style={styles.container}>
      <Video
        style={styles.video}
        source={{
          uri: mediaItem.playUrl || 'https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4',
        }}
        useNativeControls={false}
        resizeMode={ResizeMode.CONTAIN}
        isLooping={false}
        onPlaybackStatusUpdate={setStatus}
        shouldPlay
      />
      
      <TouchableOpacity 
        style={styles.overlay} 
        activeOpacity={1}
        onPress={toggleControls}
      >
        {controlsVisible && (
          <Animated.View style={[styles.controls, { opacity: fadeAnim }]}>
            
            {/* Top Controls */}
            <LinearGradient
              colors={['rgba(0, 0, 0, 0.8)', 'transparent']}
              style={styles.topControls}
            >
              <TouchableOpacity
                style={styles.backButton}
                onPress={() => navigation.goBack()}
              >
                <Ionicons name="arrow-back" size={24} color="#ffffff" />
              </TouchableOpacity>
              
              <View style={styles.mediaInfo}>
                <Text style={styles.mediaTitle} numberOfLines={1}>
                  {mediaItem.title}
                </Text>
                {mediaItem.description && (
                  <Text style={styles.mediaDescription} numberOfLines={1}>
                    {mediaItem.description}
                  </Text>
                )}
              </View>
              
              <TouchableOpacity style={styles.moreButton}>
                <Ionicons name="ellipsis-vertical" size={24} color="#ffffff" />
              </TouchableOpacity>
            </LinearGradient>

            {/* Center Play/Pause */}
            <TouchableOpacity style={styles.centerControls}>
              <View style={styles.playButton}>
                <Ionicons
                  name={status.isPlaying ? "pause" : "play"}
                  size={48}
                  color="#ffffff"
                />
              </View>
            </TouchableOpacity>

            {/* Bottom Controls */}
            <LinearGradient
              colors={['transparent', 'rgba(0, 0, 0, 0.8)']}
              style={styles.bottomControls}
            >
              {/* Progress */}
              <View style={styles.progressContainer}>
                <Text style={styles.timeText}>
                  {formatTime(status.positionMillis || 0)}
                </Text>
                <View style={styles.progressBar}>
                  <View
                    style={[
                      styles.progressFill,
                      {
                        width: `${
                          ((status.positionMillis || 0) / (status.durationMillis || 1)) * 100
                        }%`,
                      },
                    ]}
                  />
                  <View style={styles.progressThumb} />
                </View>
                <Text style={styles.timeText}>
                  {formatTime(status.durationMillis || 0)}
                </Text>
              </View>

              {/* Control Buttons */}
              <View style={styles.controlButtons}>
                <TouchableOpacity style={styles.controlButton}>
                  <Ionicons name="play-skip-back" size={28} color="#ffffff" />
                </TouchableOpacity>
                
                <TouchableOpacity style={styles.controlButton}>
                  <Ionicons name="play-back" size={28} color="#ffffff" />
                </TouchableOpacity>
                
                <TouchableOpacity style={[styles.controlButton, styles.mainPlayButton]}>
                  <Ionicons
                    name={status.isPlaying ? "pause" : "play"}
                    size={32}
                    color="#000000"
                  />
                </TouchableOpacity>
                
                <TouchableOpacity style={styles.controlButton}>
                  <Ionicons name="play-forward" size={28} color="#ffffff" />
                </TouchableOpacity>
                
                <TouchableOpacity style={styles.controlButton}>
                  <Ionicons name="play-skip-forward" size={28} color="#ffffff" />
                </TouchableOpacity>
              </View>

              {/* Additional Controls */}
              <View style={styles.additionalControls}>
                <TouchableOpacity style={styles.additionalButton}>
                  <Ionicons name="chatbox" size={20} color="#ffffff" />
                  <Text style={styles.additionalButtonText}>Subtitles</Text>
                </TouchableOpacity>
                
                <TouchableOpacity style={styles.additionalButton}>
                  <Ionicons name="cast" size={20} color="#ffffff" />
                  <Text style={styles.additionalButtonText}>Cast</Text>
                </TouchableOpacity>
                
                <TouchableOpacity style={styles.additionalButton}>
                  <Ionicons name="settings" size={20} color="#ffffff" />
                  <Text style={styles.additionalButtonText}>Quality</Text>
                </TouchableOpacity>
                
                <TouchableOpacity style={styles.additionalButton}>
                  <Ionicons name="resize" size={20} color="#ffffff" />
                  <Text style={styles.additionalButtonText}>Fullscreen</Text>
                </TouchableOpacity>
              </View>
            </LinearGradient>
          </Animated.View>
        )}
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  video: {
    width: width,
    height: height,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  controls: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topControls: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 50,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  mediaInfo: {
    flex: 1,
    marginHorizontal: 16,
  },
  mediaTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  mediaDescription: {
    fontSize: 14,
    color: '#cccccc',
    marginTop: 4,
  },
  moreButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerControls: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  playButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(0, 255, 159, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
  },
  bottomControls: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  timeText: {
    fontSize: 12,
    color: '#ffffff',
    width: 50,
    textAlign: 'center',
  },
  progressBar: {
    flex: 1,
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 2,
    marginHorizontal: 12,
    position: 'relative',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#00ff9f',
    borderRadius: 2,
  },
  progressThumb: {
    position: 'absolute',
    right: -6,
    top: -6,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#00ff9f',
  },
  controlButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  controlButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 8,
  },
  mainPlayButton: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#00ff9f',
    marginHorizontal: 16,
  },
  additionalControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  additionalButton: {
    alignItems: 'center',
  },
  additionalButtonText: {
    fontSize: 12,
    color: '#ffffff',
    marginTop: 4,
  },
});

export default MediaPlayerScreen;