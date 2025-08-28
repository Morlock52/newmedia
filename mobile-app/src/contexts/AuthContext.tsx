import React, { createContext, useContext, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import * as SecureStore from 'expo-secure-store';
import { RootState } from '../store';
import { 
  checkBiometricAvailability, 
  authenticateWithBiometrics,
  setToken,
  setBiometricEnabled
} from '../store/slices/authSlice';
import { AppDispatch } from '../store';

interface AuthContextType {
  isLoading: boolean;
  checkStoredAuth: () => Promise<void>;
  setupBiometrics: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const dispatch = useDispatch<AppDispatch>();
  const { isAuthenticated, biometricAvailable, biometricEnabled } = useSelector(
    (state: RootState) => state.auth
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    initializeAuth();
  }, []);

  const initializeAuth = async () => {
    try {
      // Check for stored authentication
      await checkStoredAuth();
      
      // Check biometric availability
      await dispatch(checkBiometricAvailability());
      
      // Check if biometrics were previously enabled
      const biometricStatus = await SecureStore.getItemAsync('biometric_enabled');
      if (biometricStatus === 'true') {
        dispatch(setBiometricEnabled(true));
      }
    } catch (error) {
      console.error('Auth initialization error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const checkStoredAuth = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const userData = await SecureStore.getItemAsync('user_data');
      
      if (token && userData) {
        // Validate token by making a test request
        // You could add token validation logic here
        dispatch(setToken(token));
        
        // If biometrics are enabled and available, prompt for biometric auth
        const biometricStatus = await SecureStore.getItemAsync('biometric_enabled');
        if (biometricStatus === 'true') {
          // Auto-authenticate with biometrics if enabled
          try {
            await dispatch(authenticateWithBiometrics()).unwrap();
          } catch (error) {
            console.warn('Biometric auto-authentication failed:', error);
            // Fall back to token-based auth
          }
        }
      }
    } catch (error) {
      console.error('Error checking stored auth:', error);
      // Clear potentially corrupted auth data
      await SecureStore.deleteItemAsync('auth_token');
      await SecureStore.deleteItemAsync('user_data');
      await SecureStore.deleteItemAsync('biometric_enabled');
    }
  };

  const setupBiometrics = async (): Promise<boolean> => {
    try {
      // Check if biometrics are available
      const result = await dispatch(checkBiometricAvailability()).unwrap();
      
      if (!result.isAvailable || !result.isEnrolled) {
        throw new Error('Biometric authentication is not available or not set up on this device');
      }
      
      // Enable biometric authentication
      await dispatch(authenticateWithBiometrics()).unwrap();
      return true;
    } catch (error) {
      console.error('Biometric setup error:', error);
      return false;
    }
  };

  const contextValue: AuthContextType = {
    isLoading,
    checkStoredAuth,
    setupBiometrics,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};