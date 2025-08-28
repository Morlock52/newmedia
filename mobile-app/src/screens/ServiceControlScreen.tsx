import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import { fetchServices, startService, stopService, restartService } from '../store/slices/mediaSlice';

const ServiceControlScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { services, loading } = useSelector((state: RootState) => state.media);

  useEffect(() => {
    dispatch(fetchServices(true));
  }, []);

  const handleServiceAction = (serviceName: string, action: 'start' | 'stop' | 'restart') => {
    Alert.alert(
      `${action.charAt(0).toUpperCase() + action.slice(1)} Service`,
      `Are you sure you want to ${action} ${serviceName}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: action.charAt(0).toUpperCase() + action.slice(1),
          onPress: () => {
            switch (action) {
              case 'start':
                dispatch(startService(serviceName));
                break;
              case 'stop':
                dispatch(stopService(serviceName));
                break;
              case 'restart':
                dispatch(restartService(serviceName));
                break;
            }
          },
        },
      ]
    );
  };

  const getServiceIcon = (serviceName: string) => {
    const iconMap: { [key: string]: string } = {
      jellyfin: 'tv',
      sonarr: 'film',
      radarr: 'videocam',
      prowlarr: 'search',
      qbittorrent: 'download',
      bazarr: 'chatbox',
      overseerr: 'add-circle',
      jellyseerr: 'add-circle',
    };
    return iconMap[serviceName.toLowerCase()] || 'server';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#00ff9f';
      case 'stopped':
        return '#ff6b6b';
      case 'error':
        return '#ff0080';
      default:
        return '#666699';
    }
  };

  return (
    <LinearGradient colors={['#0a0a0f', '#1a1a2e']} style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>
          <Text style={styles.title}>Service Control</Text>
          
          {services.map((service) => (
            <View key={service.name} style={styles.serviceCard}>
              <LinearGradient
                colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
                style={styles.serviceGradient}
              >
                <View style={styles.serviceHeader}>
                  <View style={styles.serviceInfo}>
                    <Ionicons
                      name={getServiceIcon(service.name) as any}
                      size={24}
                      color={getStatusColor(service.status)}
                    />
                    <View style={styles.serviceDetails}>
                      <Text style={styles.serviceName}>
                        {service.name.charAt(0).toUpperCase() + service.name.slice(1)}
                      </Text>
                      <Text style={styles.serviceVersion}>
                        {service.version ? `v${service.version}` : 'Unknown version'}
                      </Text>
                    </View>
                  </View>
                  
                  <View style={styles.serviceStatus}>
                    <View
                      style={[
                        styles.statusDot,
                        { backgroundColor: getStatusColor(service.status) },
                      ]}
                    />
                    <Text style={styles.statusText}>{service.status}</Text>
                  </View>
                </View>
                
                {service.message && (
                  <Text style={styles.serviceMessage}>{service.message}</Text>
                )}
                
                <View style={styles.serviceActions}>
                  <TouchableOpacity
                    style={[
                      styles.actionButton,
                      service.status === 'running' && styles.actionButtonDisabled,
                    ]}
                    onPress={() => handleServiceAction(service.name, 'start')}
                    disabled={service.status === 'running' || loading.services}
                  >
                    <Ionicons name="play" size={16} color="#00ff9f" />
                    <Text style={styles.actionButtonText}>Start</Text>
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={[
                      styles.actionButton,
                      service.status === 'stopped' && styles.actionButtonDisabled,
                    ]}
                    onPress={() => handleServiceAction(service.name, 'stop')}
                    disabled={service.status === 'stopped' || loading.services}
                  >
                    <Ionicons name="stop" size={16} color="#ff6b6b" />
                    <Text style={styles.actionButtonText}>Stop</Text>
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={styles.actionButton}
                    onPress={() => handleServiceAction(service.name, 'restart')}
                    disabled={loading.services}
                  >
                    <Ionicons name="refresh" size={16} color="#ffaa00" />
                    <Text style={styles.actionButtonText}>Restart</Text>
                  </TouchableOpacity>
                </View>
              </LinearGradient>
            </View>
          ))}
        </View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollView: { flex: 1 },
  content: { padding: 16 },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 20,
    marginTop: 40,
    textAlign: 'center',
  },
  serviceCard: { marginBottom: 16, borderRadius: 12, overflow: 'hidden' },
  serviceGradient: { padding: 16, borderWidth: 1, borderColor: '#16213e' },
  serviceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  serviceInfo: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  serviceDetails: { marginLeft: 12, flex: 1 },
  serviceName: { fontSize: 18, fontWeight: '600', color: '#ffffff' },
  serviceVersion: { fontSize: 12, color: '#666699', marginTop: 2 },
  serviceStatus: { flexDirection: 'row', alignItems: 'center' },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  statusText: { fontSize: 12, color: '#ffffff', textTransform: 'capitalize' },
  serviceMessage: { fontSize: 14, color: '#666699', marginBottom: 12 },
  serviceActions: { flexDirection: 'row', justifyContent: 'space-around' },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(102, 102, 153, 0.2)',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    minWidth: 80,
    justifyContent: 'center',
  },
  actionButtonDisabled: { opacity: 0.5 },
  actionButtonText: { fontSize: 12, color: '#ffffff', marginLeft: 4, fontWeight: '600' },
});

export default ServiceControlScreen;