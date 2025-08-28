import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import { 
  toggleBiometricAuth, 
  setTheme,
  setAutoLockTimeout 
} from '../store/slices/settingsSlice';
import { logout } from '../store/slices/authSlice';

const SettingsScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const settings = useSelector((state: RootState) => state.settings);
  const { user } = useSelector((state: RootState) => state.auth);

  const handleLogout = () => {
    dispatch(logout());
  };

  return (
    <LinearGradient colors={['#0a0a0f', '#1a1a2e']} style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>
          
          {/* User Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Account</Text>
            <View style={styles.userCard}>
              <LinearGradient
                colors={['rgba(0, 255, 159, 0.1)', 'rgba(0, 255, 159, 0.05)']}
                style={styles.userCardGradient}
              >
                <Ionicons name="person-circle" size={48} color="#00ff9f" />
                <View style={styles.userInfo}>
                  <Text style={styles.userName}>{user?.username || 'User'}</Text>
                  <Text style={styles.userEmail}>{user?.email || 'user@example.com'}</Text>
                </View>
              </LinearGradient>
            </View>
          </View>

          {/* Server Settings */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Server</Text>
            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Server URL</Text>
                <Text style={styles.settingValue}>
                  {settings.server.useHttps ? 'https' : 'http'}://{settings.server.baseUrl}:{settings.server.port}
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#666699" />
            </View>
          </View>

          {/* Security */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Security</Text>
            
            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Biometric Authentication</Text>
                <Text style={styles.settingDescription}>Use Face ID or Touch ID</Text>
              </View>
              <Switch
                value={settings.security.biometricAuth}
                onValueChange={() => dispatch(toggleBiometricAuth())}
                trackColor={{ false: '#16213e', true: '#00ff9f40' }}
                thumbColor={settings.security.biometricAuth ? '#00ff9f' : '#666699'}
              />
            </View>

            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Auto Lock</Text>
                <Text style={styles.settingDescription}>
                  Lock after {settings.security.autoLockTimeout} minutes
                </Text>
              </View>
              <Switch
                value={settings.security.autoLock}
                trackColor={{ false: '#16213e', true: '#00ff9f40' }}
                thumbColor={settings.security.autoLock ? '#00ff9f' : '#666699'}
              />
            </View>
          </View>

          {/* App Settings */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>App Settings</Text>
            
            <TouchableOpacity style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Theme</Text>
                <Text style={styles.settingValue}>{settings.app.theme}</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#666699" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Language</Text>
                <Text style={styles.settingValue}>English</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#666699" />
            </TouchableOpacity>
          </View>

          {/* Actions */}
          <View style={styles.section}>
            <TouchableOpacity style={styles.actionButton} onPress={handleLogout}>
              <LinearGradient
                colors={['rgba(255, 107, 107, 0.2)', 'rgba(255, 107, 107, 0.1)']}
                style={styles.actionButtonGradient}
              >
                <Ionicons name="log-out" size={20} color="#ff6b6b" />
                <Text style={styles.actionButtonText}>Sign Out</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollView: { flex: 1 },
  content: { padding: 16 },
  section: { marginBottom: 24 },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
    textShadowColor: '#00ff9f',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
  },
  userCard: { borderRadius: 12, overflow: 'hidden' },
  userCardGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderWidth: 1,
    borderColor: '#00ff9f40',
  },
  userInfo: { flex: 1, marginLeft: 16 },
  userName: { fontSize: 18, fontWeight: 'bold', color: '#ffffff' },
  userEmail: { fontSize: 14, color: '#666699', marginTop: 4 },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(26, 26, 46, 0.5)',
    borderRadius: 8,
    padding: 16,
    marginBottom: 8,
  },
  settingInfo: { flex: 1 },
  settingLabel: { fontSize: 16, color: '#ffffff', fontWeight: '600' },
  settingDescription: { fontSize: 12, color: '#666699', marginTop: 4 },
  settingValue: { fontSize: 14, color: '#00ff9f', marginTop: 4 },
  actionButton: { borderRadius: 8, overflow: 'hidden' },
  actionButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderWidth: 1,
    borderColor: '#ff6b6b40',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ff6b6b',
    marginLeft: 8,
  },
});

export default SettingsScreen;