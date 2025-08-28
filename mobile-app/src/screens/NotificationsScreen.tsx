import React from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import { markAsRead, removeNotification, markAllAsRead } from '../store/slices/notificationsSlice';

const NotificationsScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { notifications, unreadCount } = useSelector((state: RootState) => state.notifications);

  const handleMarkAsRead = (id: string) => {
    dispatch(markAsRead(id));
  };

  const handleRemove = (id: string) => {
    dispatch(removeNotification(id));
  };

  const handleMarkAllAsRead = () => {
    dispatch(markAllAsRead());
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'download_complete':
        return 'download';
      case 'new_content':
        return 'film';
      case 'service_alert':
        return 'warning';
      case 'cast':
        return 'cast';
      default:
        return 'notifications';
    }
  };

  const getNotificationColor = (type: string) => {
    switch (type) {
      case 'download_complete':
        return '#00ff9f';
      case 'new_content':
        return '#ff0080';
      case 'service_alert':
        return '#ffaa00';
      case 'cast':
        return '#0099ff';
      default:
        return '#666699';
    }
  };

  const renderNotification = ({ item }: { item: any }) => (
    <TouchableOpacity
      style={[styles.notificationItem, !item.read && styles.unreadItem]}
      onPress={() => handleMarkAsRead(item.id)}
    >
      <LinearGradient
        colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
        style={styles.notificationGradient}
      >
        <View style={styles.notificationHeader}>
          <Ionicons
            name={getNotificationIcon(item.type)}
            size={20}
            color={getNotificationColor(item.type)}
          />
          <Text style={styles.notificationTime}>
            {new Date(item.timestamp).toLocaleTimeString()}
          </Text>
        </View>
        
        <Text style={styles.notificationTitle}>{item.title}</Text>
        <Text style={styles.notificationBody}>{item.body}</Text>
        
        <TouchableOpacity
          style={styles.removeButton}
          onPress={() => handleRemove(item.id)}
        >
          <Ionicons name="close" size={16} color="#666699" />
        </TouchableOpacity>
      </LinearGradient>
    </TouchableOpacity>
  );

  return (
    <LinearGradient colors={['#0a0a0f', '#1a1a2e']} style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Notifications</Text>
        {unreadCount > 0 && (
          <TouchableOpacity onPress={handleMarkAllAsRead}>
            <Text style={styles.markAllRead}>Mark all as read</Text>
          </TouchableOpacity>
        )}
      </View>
      
      <FlatList
        data={notifications}
        renderItem={renderNotification}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Ionicons name="notifications-off" size={64} color="#666699" />
            <Text style={styles.emptyText}>No notifications</Text>
          </View>
        }
      />
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: 60,
  },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: '#ffffff' },
  markAllRead: { fontSize: 14, color: '#00ff9f' },
  list: { padding: 16, flexGrow: 1 },
  notificationItem: { marginBottom: 12, borderRadius: 8, overflow: 'hidden' },
  unreadItem: { borderLeftWidth: 4, borderLeftColor: '#00ff9f' },
  notificationGradient: { padding: 16, position: 'relative' },
  notificationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  notificationTime: { fontSize: 12, color: '#666699' },
  notificationTitle: { fontSize: 16, fontWeight: '600', color: '#ffffff', marginBottom: 4 },
  notificationBody: { fontSize: 14, color: '#cccccc' },
  removeButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(102, 102, 153, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 60 },
  emptyText: { fontSize: 18, color: '#666699', marginTop: 16 },
});

export default NotificationsScreen;