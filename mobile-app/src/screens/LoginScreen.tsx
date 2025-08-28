import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Alert,
  Dimensions,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import {
  loginWithCredentials,
  authenticateWithBiometrics,
  checkBiometricAvailability,
  clearError,
} from '../store/slices/authSlice';
import { useAuth } from '../contexts/AuthContext';

const { width, height } = Dimensions.get('window');

const LoginScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { loading, error, biometricAvailable, biometricEnabled } = useSelector(
    (state: RootState) => state.auth
  );
  const { setupBiometrics } = useAuth();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [fadeAnim] = useState(new Animated.Value(0));
  const [slideAnim] = useState(new Animated.Value(height));

  useEffect(() => {
    // Animate in
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        tension: 50,
        friction: 8,
        useNativeDriver: true,
      }),
    ]).start();

    // Check biometric availability on mount
    dispatch(checkBiometricAvailability());

    // Clear any previous errors
    dispatch(clearError());
  }, []);

  useEffect(() => {
    if (error) {
      Alert.alert('Authentication Error', error, [
        { text: 'OK', onPress: () => dispatch(clearError()) },
      ]);
    }
  }, [error]);

  const handleLogin = async () => {
    if (!username.trim() || !password.trim()) {
      Alert.alert('Error', 'Please enter both username and password');
      return;
    }

    try {
      await dispatch(loginWithCredentials({ username, password })).unwrap();
    } catch (error) {
      // Error handled by useEffect above
    }
  };

  const handleBiometricLogin = async () => {
    if (!biometricAvailable) {
      Alert.alert(
        'Biometric Authentication Unavailable',
        'Biometric authentication is not available on this device or not set up.'
      );
      return;
    }

    try {
      await dispatch(authenticateWithBiometrics()).unwrap();
    } catch (error) {
      // Error handled by useEffect above
    }
  };

  const handleSetupBiometrics = async () => {
    const success = await setupBiometrics();
    if (success) {
      Alert.alert(
        'Biometric Setup Complete',
        'You can now use biometric authentication to log in.'
      );
    } else {
      Alert.alert(
        'Biometric Setup Failed',
        'Unable to set up biometric authentication. Please try again.'
      );
    }
  };

  return (
    <LinearGradient
      colors={['#0a0a0f', '#1a1a2e', '#16213e']}
      style={styles.container}
    >
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoidingView}
      >
        <Animated.View
          style={[
            styles.content,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }],
            },
          ]}
        >
          {/* Logo and Title */}
          <View style={styles.headerContainer}>
            <View style={styles.logoContainer}>
              <View style={styles.logo}>
                <Ionicons name="server" size={40} color="#00ff9f" />
              </View>
            </View>
            <Text style={styles.title}>Media Server</Text>
            <Text style={styles.subtitle}>Remote Control</Text>
          </View>

          {/* Login Form */}
          <View style={styles.formContainer}>
            <View style={styles.inputContainer}>
              <Ionicons name="person-outline" size={20} color="#666699" style={styles.inputIcon} />
              <TextInput
                style={styles.input}
                placeholder="Username"
                placeholderTextColor="#666699"
                value={username}
                onChangeText={setUsername}
                autoCapitalize="none"
                autoCorrect={false}
                returnKeyType="next"
              />
            </View>

            <View style={styles.inputContainer}>
              <Ionicons name="lock-closed-outline" size={20} color="#666699" style={styles.inputIcon} />
              <TextInput
                style={styles.input}
                placeholder="Password"
                placeholderTextColor="#666699"
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                returnKeyType="done"
                onSubmitEditing={handleLogin}
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                style={styles.passwordToggle}
              >
                <Ionicons
                  name={showPassword ? "eye-off-outline" : "eye-outline"}
                  size={20}
                  color="#666699"
                />
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={[styles.loginButton, loading && styles.loginButtonDisabled]}
              onPress={handleLogin}
              disabled={loading}
            >
              <LinearGradient
                colors={['#00ff9f', '#00cc7f']}
                style={styles.buttonGradient}
              >
                {loading ? (
                  <Ionicons name="sync" size={20} color="#000000" />
                ) : (
                  <Text style={styles.loginButtonText}>Sign In</Text>
                )}
              </LinearGradient>
            </TouchableOpacity>

            {/* Biometric Authentication */}
            {biometricAvailable && (
              <View style={styles.biometricContainer}>
                <Text style={styles.dividerText}>or</Text>
                <TouchableOpacity
                  style={styles.biometricButton}
                  onPress={biometricEnabled ? handleBiometricLogin : handleSetupBiometrics}
                  disabled={loading}
                >
                  <Ionicons 
                    name={Platform.OS === 'ios' ? "finger-print" : "fingerprint"} 
                    size={24} 
                    color="#ff0080" 
                  />
                  <Text style={styles.biometricButtonText}>
                    {biometricEnabled ? 'Use Biometric' : 'Setup Biometric'}
                  </Text>
                </TouchableOpacity>
              </View>
            )}
          </View>

          {/* Connection Status */}
          <View style={styles.statusContainer}>
            <View style={styles.statusIndicator}>
              <View style={[styles.statusDot, { backgroundColor: '#ffaa00' }]} />
              <Text style={styles.statusText}>Connecting to localhost:3333</Text>
            </View>
          </View>

          {/* Quick Setup Info */}
          <View style={styles.infoContainer}>
            <Text style={styles.infoText}>
              First time? Use default credentials:
            </Text>
            <Text style={styles.credentialsText}>
              admin / admin
            </Text>
          </View>
        </Animated.View>
      </KeyboardAvoidingView>

      {/* Cyberpunk background grid */}
      <View style={styles.backgroundGrid}>
        {Array.from({ length: 15 }, (_, i) => (
          <View key={i} style={[styles.gridLine, { top: i * (height / 15) }]} />
        ))}
        {Array.from({ length: 10 }, (_, i) => (
          <View key={i} style={[styles.gridLineVertical, { left: i * (width / 10) }]} />
        ))}
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 30,
  },
  headerContainer: {
    alignItems: 'center',
    marginBottom: 50,
  },
  logoContainer: {
    marginBottom: 20,
  },
  logo: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(0, 255, 159, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#00ff9f',
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 8,
    textShadowColor: '#00ff9f',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 15,
  },
  subtitle: {
    fontSize: 16,
    color: '#ff0080',
    textAlign: 'center',
    fontWeight: '600',
  },
  formContainer: {
    marginBottom: 30,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(26, 26, 46, 0.8)',
    borderRadius: 12,
    marginBottom: 16,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: '#16213e',
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
  },
  inputIcon: {
    marginRight: 12,
  },
  input: {
    flex: 1,
    height: 50,
    color: '#ffffff',
    fontSize: 16,
  },
  passwordToggle: {
    padding: 8,
  },
  loginButton: {
    borderRadius: 12,
    overflow: 'hidden',
    marginTop: 16,
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 8,
  },
  loginButtonDisabled: {
    opacity: 0.7,
  },
  buttonGradient: {
    paddingVertical: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loginButtonText: {
    color: '#000000',
    fontSize: 18,
    fontWeight: 'bold',
  },
  biometricContainer: {
    alignItems: 'center',
    marginTop: 30,
  },
  dividerText: {
    color: '#666699',
    fontSize: 14,
    marginBottom: 16,
  },
  biometricButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 0, 128, 0.1)',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: '#ff0080',
    shadowColor: '#ff0080',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
  },
  biometricButtonText: {
    color: '#ff0080',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  statusContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
    shadowColor: '#ffaa00',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 8,
  },
  statusText: {
    color: '#666699',
    fontSize: 14,
  },
  infoContainer: {
    alignItems: 'center',
    backgroundColor: 'rgba(22, 33, 62, 0.5)',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  infoText: {
    color: '#666699',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 8,
  },
  credentialsText: {
    color: '#00ff9f',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  backgroundGrid: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.05,
  },
  gridLine: {
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
});

export default LoginScreen;