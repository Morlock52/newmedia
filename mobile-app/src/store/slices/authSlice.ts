import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import * as LocalAuthentication from 'expo-local-authentication';
import * as SecureStore from 'expo-secure-store';
import { apiService } from '../../services/apiService';

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  biometricEnabled: boolean;
  biometricAvailable: boolean;
  loading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  user: null,
  token: null,
  isAuthenticated: false,
  biometricEnabled: false,
  biometricAvailable: false,
  loading: false,
  error: null,
};

// Async thunks
export const checkBiometricAvailability = createAsyncThunk(
  'auth/checkBiometricAvailability',
  async () => {
    const isAvailable = await LocalAuthentication.hasHardwareAsync();
    const isEnrolled = await LocalAuthentication.isEnrolledAsync();
    const supportedTypes = await LocalAuthentication.supportedAuthenticationTypesAsync();
    
    return {
      isAvailable,
      isEnrolled,
      supportedTypes,
    };
  }
);

export const authenticateWithBiometrics = createAsyncThunk(
  'auth/authenticateWithBiometrics',
  async (_, { rejectWithValue }) => {
    try {
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Authenticate to access Media Server',
        fallbackLabel: 'Use Passcode',
        disableDeviceFallback: false,
      });

      if (result.success) {
        // Get stored credentials
        const storedToken = await SecureStore.getItemAsync('auth_token');
        const storedUser = await SecureStore.getItemAsync('user_data');
        
        if (storedToken && storedUser) {
          return {
            token: storedToken,
            user: JSON.parse(storedUser),
          };
        } else {
          throw new Error('No stored credentials found');
        }
      } else {
        throw new Error('Biometric authentication failed');
      }
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Authentication failed');
    }
  }
);

export const loginWithCredentials = createAsyncThunk(
  'auth/loginWithCredentials',
  async ({ username, password }: { username: string; password: string }, { rejectWithValue }) => {
    try {
      const response = await apiService.post('/auth/login', { username, password });
      
      if (response.data.success) {
        const { token, user } = response.data.data;
        
        // Store credentials securely
        await SecureStore.setItemAsync('auth_token', token);
        await SecureStore.setItemAsync('user_data', JSON.stringify(user));
        
        return { token, user };
      } else {
        throw new Error(response.data.error || 'Login failed');
      }
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.error || error.message || 'Login failed');
    }
  }
);

export const enableBiometricAuth = createAsyncThunk(
  'auth/enableBiometricAuth',
  async (_, { rejectWithValue }) => {
    try {
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Enable biometric authentication',
        fallbackLabel: 'Use Passcode',
      });

      if (result.success) {
        await SecureStore.setItemAsync('biometric_enabled', 'true');
        return true;
      } else {
        throw new Error('Biometric setup failed');
      }
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Biometric setup failed');
    }
  }
);

export const logout = createAsyncThunk(
  'auth/logout',
  async () => {
    // Clear secure storage
    await SecureStore.deleteItemAsync('auth_token');
    await SecureStore.deleteItemAsync('user_data');
    await SecureStore.deleteItemAsync('biometric_enabled');
    
    // Call logout endpoint
    try {
      await apiService.post('/auth/logout');
    } catch (error) {
      // Ignore logout errors, clear local data anyway
    }
  }
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setToken: (state, action: PayloadAction<string>) => {
      state.token = action.payload;
    },
    setBiometricEnabled: (state, action: PayloadAction<boolean>) => {
      state.biometricEnabled = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      // Check biometric availability
      .addCase(checkBiometricAvailability.fulfilled, (state, action) => {
        state.biometricAvailable = action.payload.isAvailable && action.payload.isEnrolled;
      })
      
      // Biometric authentication
      .addCase(authenticateWithBiometrics.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(authenticateWithBiometrics.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.token = action.payload.token;
        state.user = action.payload.user;
      })
      .addCase(authenticateWithBiometrics.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })
      
      // Credential login
      .addCase(loginWithCredentials.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(loginWithCredentials.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.token = action.payload.token;
        state.user = action.payload.user;
      })
      .addCase(loginWithCredentials.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })
      
      // Enable biometric auth
      .addCase(enableBiometricAuth.fulfilled, (state) => {
        state.biometricEnabled = true;
      })
      .addCase(enableBiometricAuth.rejected, (state, action) => {
        state.error = action.payload as string;
      })
      
      // Logout
      .addCase(logout.fulfilled, (state) => {
        state.user = null;
        state.token = null;
        state.isAuthenticated = false;
        state.biometricEnabled = false;
        state.error = null;
      });
  },
});

export const { clearError, setToken, setBiometricEnabled } = authSlice.actions;
export default authSlice.reducer;