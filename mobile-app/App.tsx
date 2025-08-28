import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { store, persistor } from './src/store';
import { AuthProvider } from './src/contexts/AuthContext';
import AppNavigator from './src/navigation/AppNavigator';
import { setupNotifications } from './src/services/notificationService';
import { LoadingScreen } from './src/components/LoadingScreen';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StyleSheet } from 'react-native';

export default function App() {
  useEffect(() => {
    setupNotifications();
  }, []);

  return (
    <GestureHandlerRootView style={styles.container}>
      <SafeAreaProvider>
        <Provider store={store}>
          <PersistGate loading={<LoadingScreen />} persistor={persistor}>
            <AuthProvider>
              <NavigationContainer
                theme={{
                  dark: true,
                  colors: {
                    primary: '#00ff9f',
                    background: '#0a0a0f',
                    card: '#1a1a2e',
                    text: '#ffffff',
                    border: '#16213e',
                    notification: '#ff0080',
                  },
                }}
              >
                <AppNavigator />
                <StatusBar style="light" backgroundColor="#0a0a0f" />
              </NavigationContainer>
            </AuthProvider>
          </PersistGate>
        </Provider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});