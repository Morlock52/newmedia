import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  Animated,
  PanResponder,
  Alert,
} from 'react-native';
import { Camera, CameraType } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation, useRoute } from '@react-navigation/native';
import { Canvas } from '@react-three/fiber';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../store';
import { searchMedia } from '../store/slices/mediaSlice';

const { width, height } = Dimensions.get('window');

interface ARMedia {
  id: string;
  title: string;
  type: 'movie' | 'series' | 'music';
  position: { x: number; y: number; z: number };
  distance: number;
  rating: number;
  year: number;
}

const ARViewScreen: React.FC = () => {
  const navigation = useNavigation();
  const route = useRoute();
  const dispatch = useDispatch<AppDispatch>();
  
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [cameraType, setCameraType] = useState(CameraType.back);
  const [isScanning, setIsScanning] = useState(false);
  const [arMedia, setArMedia] = useState<ARMedia[]>([]);
  const [selectedMedia, setSelectedMedia] = useState<ARMedia | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  const scanAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    getCameraPermissions();
    
    // Animate in
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();

    // Initialize AR content
    initializeARContent();
    
    // Start scanning animation
    startScanAnimation();
  }, []);

  const getCameraPermissions = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    setHasPermission(status === 'granted');
  };

  const initializeARContent = () => {
    // Mock AR media items positioned in 3D space
    const mockARMedia: ARMedia[] = [
      {
        id: '1',
        title: 'The Matrix Reloaded',
        type: 'movie',
        position: { x: 0.2, y: 0.3, z: -2 },
        distance: 2.5,
        rating: 8.7,
        year: 2003,
      },
      {
        id: '2',
        title: 'Cyberpunk 2077 OST',
        type: 'music',
        position: { x: -0.3, y: 0.1, z: -1.5 },
        distance: 1.8,
        rating: 9.2,
        year: 2020,
      },
      {
        id: '3',
        title: 'Black Mirror',
        type: 'series',
        position: { x: 0.1, y: -0.2, z: -3 },
        distance: 3.2,
        rating: 8.9,
        year: 2011,
      },
      {
        id: '4',
        title: 'Blade Runner 2049',
        type: 'movie',
        position: { x: -0.2, y: 0.4, z: -2.5 },
        distance: 2.8,
        rating: 8.0,
        year: 2017,
      },
    ];
    
    setArMedia(mockARMedia);
  };

  const startScanAnimation = () => {
    Animated.loop(
      Animated.timing(scanAnim, {
        toValue: 1,
        duration: 2000,
        useNativeDriver: true,
      })
    ).start();

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const handleScan = async () => {
    setIsScanning(true);
    
    // Simulate AR scanning and content discovery
    setTimeout(() => {
      const newMedia: ARMedia = {
        id: `ar_${Date.now()}`,
        title: 'Discovered Content',
        type: 'movie',
        position: { 
          x: (Math.random() - 0.5) * 0.8, 
          y: (Math.random() - 0.5) * 0.6, 
          z: -2 - Math.random() * 2 
        },
        distance: 1.5 + Math.random() * 2,
        rating: 7 + Math.random() * 2,
        year: 2020 + Math.floor(Math.random() * 5),
      };
      
      setArMedia(prev => [...prev, newMedia]);
      setIsScanning(false);
    }, 2000);
  };

  const handleMediaSelect = (media: ARMedia) => {
    setSelectedMedia(media);
    
    // Haptic feedback would be here
    Alert.alert(
      'AR Content Found',
      `${media.title} (${media.year}) - Rating: ${media.rating.toFixed(1)}`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Play', onPress: () => playMedia(media) },
        { text: 'Download', onPress: () => downloadMedia(media) },
      ]
    );
  };

  const playMedia = (media: ARMedia) => {
    navigation.navigate('MediaPlayer' as never, { mediaItem: media } as never);
  };

  const downloadMedia = (media: ARMedia) => {
    // Trigger download
    Alert.alert('Download Started', `${media.title} will be available offline soon.`);
  };

  const getMediaIcon = (type: string) => {
    switch (type) {
      case 'movie':
        return 'film';
      case 'series':
        return 'tv';
      case 'music':
        return 'musical-notes';
      default:
        return 'help';
    }
  };

  const getMediaColor = (type: string) => {
    switch (type) {
      case 'movie':
        return '#00ff9f';
      case 'series':
        return '#ff0080';
      case 'music':
        return '#ffaa00';
      default:
        return '#666699';
    }
  };

  if (hasPermission === null) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-off" size={64} color="#666699" />
        <Text style={styles.permissionText}>Camera access denied</Text>
        <Text style={styles.permissionSubtext}>
          Please enable camera access in settings to use AR features
        </Text>
        <TouchableOpacity style={styles.settingsButton} onPress={() => navigation.goBack()}>
          <Text style={styles.settingsButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <Camera style={styles.camera} type={cameraType}>
        <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
          {/* Header */}
          <View style={styles.header}>
            <TouchableOpacity
              style={styles.backButton}
              onPress={() => navigation.goBack()}
            >
              <Ionicons name="arrow-back" size={24} color="#ffffff" />
            </TouchableOpacity>
            
            <View style={styles.headerCenter}>
              <Text style={styles.headerTitle}>AR Content Finder</Text>
              <Text style={styles.headerSubtitle}>Scan to discover media</Text>
            </View>
            
            <TouchableOpacity
              style={styles.switchButton}
              onPress={() => setCameraType(
                cameraType === CameraType.back ? CameraType.front : CameraType.back
              )}
            >
              <Ionicons name="camera-reverse" size={24} color="#ffffff" />
            </TouchableOpacity>
          </View>

          {/* AR Media Overlays */}
          {arMedia.map((media, index) => {
            const screenX = width * (0.5 + media.position.x);
            const screenY = height * (0.5 + media.position.y);
            
            return (
              <TouchableOpacity
                key={media.id}
                style={[
                  styles.arMediaItem,
                  {
                    left: screenX - 40,
                    top: screenY - 40,
                  },
                ]}
                onPress={() => handleMediaSelect(media)}
                activeOpacity={0.8}
              >
                <Animated.View
                  style={[
                    styles.arMediaContainer,
                    {
                      transform: [
                        { 
                          scale: selectedMedia?.id === media.id ? pulseAnim : 1 
                        }
                      ],
                    },
                  ]}
                >
                  <LinearGradient
                    colors={[
                      `${getMediaColor(media.type)}40`,
                      `${getMediaColor(media.type)}20`,
                    ]}
                    style={styles.arMediaGradient}
                  >
                    <Ionicons
                      name={getMediaIcon(media.type)}
                      size={24}
                      color={getMediaColor(media.type)}
                    />
                    <Text style={styles.arMediaTitle} numberOfLines={1}>
                      {media.title}
                    </Text>
                    <Text style={styles.arMediaDistance}>
                      {media.distance.toFixed(1)}m
                    </Text>
                    <View style={styles.arMediaRating}>
                      <Ionicons name="star" size={12} color="#ffaa00" />
                      <Text style={styles.arMediaRatingText}>
                        {media.rating.toFixed(1)}
                      </Text>
                    </View>
                  </LinearGradient>
                </Animated.View>
              </TouchableOpacity>
            );
          })}

          {/* Scanning Indicator */}
          {isScanning && (
            <Animated.View
              style={[
                styles.scanIndicator,
                {
                  transform: [
                    {
                      rotate: scanAnim.interpolate({
                        inputRange: [0, 1],
                        outputRange: ['0deg', '360deg'],
                      }),
                    },
                  ],
                },
              ]}
            >
              <View style={styles.scanRing} />
            </Animated.View>
          )}

          {/* Bottom Controls */}
          <View style={styles.bottomControls}>
            <TouchableOpacity
              style={styles.scanButton}
              onPress={handleScan}
              disabled={isScanning}
            >
              <LinearGradient
                colors={isScanning ? ['#666699', '#444466'] : ['#00ff9f', '#00cc7f']}
                style={styles.scanButtonGradient}
              >
                {isScanning ? (
                  <Ionicons name="sync" size={28} color="#ffffff" />
                ) : (
                  <Ionicons name="scan" size={28} color="#000000" />
                )}
              </LinearGradient>
            </TouchableOpacity>

            <View style={styles.bottomInfo}>
              <Text style={styles.bottomInfoText}>
                {arMedia.length} items found
              </Text>
              <Text style={styles.bottomInfoSubtext}>
                Tap items to interact
              </Text>
            </View>

            <TouchableOpacity style={styles.settingsButton}>
              <Ionicons name="options" size={24} color="#666699" />
            </TouchableOpacity>
          </View>

          {/* AR Grid Lines */}
          <View style={styles.arGrid}>
            {Array.from({ length: 10 }, (_, i) => (
              <View
                key={`h-${i}`}
                style={[
                  styles.gridLineHorizontal,
                  { top: (height / 10) * i },
                ]}
              />
            ))}
            {Array.from({ length: 8 }, (_, i) => (
              <View
                key={`v-${i}`}
                style={[
                  styles.gridLineVertical,
                  { left: (width / 8) * i },
                ]}
              />
            ))}
          </View>

          {/* Center Crosshair */}
          <View style={styles.crosshair}>
            <View style={styles.crosshairVertical} />
            <View style={styles.crosshairHorizontal} />
          </View>
        </Animated.View>
      </Camera>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0a0a0f',
    padding: 20,
  },
  permissionText: {
    fontSize: 18,
    color: '#ffffff',
    textAlign: 'center',
    marginTop: 20,
  },
  permissionSubtext: {
    fontSize: 14,
    color: '#666699',
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 30,
  },
  header: {
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
  headerCenter: {
    flex: 1,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    textShadowColor: '#00ff9f',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#666699',
    marginTop: 2,
  },
  switchButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  arMediaItem: {
    position: 'absolute',
    width: 80,
    height: 80,
  },
  arMediaContainer: {
    width: '100%',
    height: '100%',
  },
  arMediaGradient: {
    flex: 1,
    borderRadius: 12,
    padding: 8,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  arMediaTitle: {
    fontSize: 10,
    color: '#ffffff',
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 4,
  },
  arMediaDistance: {
    fontSize: 8,
    color: '#666699',
    marginTop: 2,
  },
  arMediaRating: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
  },
  arMediaRatingText: {
    fontSize: 8,
    color: '#ffaa00',
    marginLeft: 2,
    fontWeight: '600',
  },
  scanIndicator: {
    position: 'absolute',
    top: height / 2 - 50,
    left: width / 2 - 50,
    width: 100,
    height: 100,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanRing: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 2,
    borderColor: '#00ff9f',
    borderStyle: 'dashed',
  },
  bottomControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingBottom: 40,
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
  scanButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
  },
  scanButtonGradient: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomInfo: {
    flex: 1,
    alignItems: 'center',
  },
  bottomInfoText: {
    fontSize: 14,
    color: '#ffffff',
    fontWeight: '600',
  },
  bottomInfoSubtext: {
    fontSize: 12,
    color: '#666699',
    marginTop: 2,
  },
  settingsButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  settingsButtonText: {
    color: '#00ff9f',
    fontSize: 16,
    fontWeight: '600',
  },
  arGrid: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.1,
  },
  gridLineHorizontal: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: '#00ff9f',
  },
  gridLineVertical: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: '#00ff9f',
  },
  crosshair: {
    position: 'absolute',
    top: height / 2 - 10,
    left: width / 2 - 10,
    width: 20,
    height: 20,
  },
  crosshairVertical: {
    position: 'absolute',
    left: 9,
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#ff0080',
  },
  crosshairHorizontal: {
    position: 'absolute',
    top: 9,
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: '#ff0080',
  },
});

export default ARViewScreen;