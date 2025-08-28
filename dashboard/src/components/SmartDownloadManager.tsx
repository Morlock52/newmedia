import React, { useState, useEffect, useCallback } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import './SmartDownloadManager.css';

interface Download {
  id: string;
  name: string;
  size: number;
  progress: number;
  speed: number;
  eta: number;
  status: 'queued' | 'downloading' | 'paused' | 'completed' | 'error';
  priority: 'low' | 'normal' | 'high' | 'critical';
  service: string;
  type: 'movie' | 'tv' | 'music' | 'other';
  seeders?: number;
  leechers?: number;
  uploadSpeed?: number;
  ratio?: number;
  addedAt: Date;
  completedAt?: Date;
}

interface BandwidthAllocation {
  service: string;
  allocated: number;
  used: number;
  limit: number;
}

const SmartDownloadManager: React.FC = () => {
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [bandwidthAllocations, setBandwidthAllocations] = useState<BandwidthAllocation[]>([]);
  const [totalBandwidth, setTotalBandwidth] = useState(100); // Mbps
  const [scheduleEnabled, setScheduleEnabled] = useState(false);
  const [scheduleStart, setScheduleStart] = useState('22:00');
  const [scheduleEnd, setScheduleEnd] = useState('08:00');
  const [duplicateDetection, setDuplicateDetection] = useState(true);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'active' | 'queued' | 'completed'>('all');

  useEffect(() => {
    fetchDownloads();
    fetchBandwidthAllocations();
    
    const interval = setInterval(() => {
      fetchDownloads();
      updateProgress();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const fetchDownloads = async () => {
    try {
      const response = await fetch('/api/downloads');
      const data = await response.json();
      setDownloads(data);
    } catch (error) {
      console.error('Failed to fetch downloads:', error);
    }
  };

  const fetchBandwidthAllocations = async () => {
    try {
      const response = await fetch('/api/downloads/bandwidth');
      const data = await response.json();
      setBandwidthAllocations(data);
    } catch (error) {
      console.error('Failed to fetch bandwidth allocations:', error);
    }
  };

  const updateProgress = () => {
    setDownloads(prev => prev.map(download => {
      if (download.status === 'downloading' && download.progress < 100) {
        const increment = (download.speed / download.size) * 100;
        const newProgress = Math.min(download.progress + increment, 100);
        
        if (newProgress >= 100) {
          return { ...download, progress: 100, status: 'completed', completedAt: new Date() };
        }
        
        return { ...download, progress: newProgress };
      }
      return download;
    }));
  };

  const handlePriorityChange = async (downloadId: string, newPriority: Download['priority']) => {
    try {
      await fetch(`/api/downloads/${downloadId}/priority`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priority: newPriority }),
      });
      
      setDownloads(prev => prev.map(d => 
        d.id === downloadId ? { ...d, priority: newPriority } : d
      ));
    } catch (error) {
      console.error('Failed to update priority:', error);
    }
  };

  const handlePause = async (downloadId: string) => {
    try {
      await fetch(`/api/downloads/${downloadId}/pause`, { method: 'POST' });
      setDownloads(prev => prev.map(d => 
        d.id === downloadId ? { ...d, status: 'paused' } : d
      ));
    } catch (error) {
      console.error('Failed to pause download:', error);
    }
  };

  const handleResume = async (downloadId: string) => {
    try {
      await fetch(`/api/downloads/${downloadId}/resume`, { method: 'POST' });
      setDownloads(prev => prev.map(d => 
        d.id === downloadId ? { ...d, status: 'downloading' } : d
      ));
    } catch (error) {
      console.error('Failed to resume download:', error);
    }
  };

  const handleDelete = async (downloadId: string) => {
    try {
      await fetch(`/api/downloads/${downloadId}`, { method: 'DELETE' });
      setDownloads(prev => prev.filter(d => d.id !== downloadId));
    } catch (error) {
      console.error('Failed to delete download:', error);
    }
  };

  const handleBandwidthChange = async (service: string, newLimit: number) => {
    try {
      await fetch(`/api/downloads/bandwidth/${service}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: newLimit }),
      });
      
      setBandwidthAllocations(prev => prev.map(ba => 
        ba.service === service ? { ...ba, limit: newLimit } : ba
      ));
    } catch (error) {
      console.error('Failed to update bandwidth:', error);
    }
  };

  const onDragEnd = (result: any) => {
    if (!result.destination) return;

    const items = Array.from(downloads);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setDownloads(items);
    // Update order on backend
    updateDownloadOrder(items);
  };

  const updateDownloadOrder = async (items: Download[]) => {
    try {
      await fetch('/api/downloads/order', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ order: items.map(item => item.id) }),
      });
    } catch (error) {
      console.error('Failed to update order:', error);
    }
  };

  const formatSize = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatSpeed = (bytesPerSecond: number) => {
    return formatSize(bytesPerSecond) + '/s';
  };

  const formatETA = (seconds: number) => {
    if (seconds === Infinity || isNaN(seconds)) return '∞';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  const getPriorityColor = (priority: Download['priority']) => {
    switch (priority) {
      case 'critical': return '#ff0040';
      case 'high': return '#ff00ff';
      case 'normal': return '#00ffff';
      case 'low': return '#808080';
    }
  };

  const filteredDownloads = downloads.filter(download => {
    switch (selectedFilter) {
      case 'active':
        return download.status === 'downloading';
      case 'queued':
        return download.status === 'queued' || download.status === 'paused';
      case 'completed':
        return download.status === 'completed';
      default:
        return true;
    }
  });

  return (
    <div className="smart-download-manager cyberpunk-theme">
      <div className="manager-header">
        <h1 className="manager-title glitch-text" data-text="DOWNLOAD MANAGER">
          DOWNLOAD MANAGER
        </h1>
        
        <div className="manager-controls">
          <div className="filter-buttons">
            <button
              className={`filter-btn ${selectedFilter === 'all' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('all')}
            >
              All
            </button>
            <button
              className={`filter-btn ${selectedFilter === 'active' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('active')}
            >
              Active
            </button>
            <button
              className={`filter-btn ${selectedFilter === 'queued' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('queued')}
            >
              Queued
            </button>
            <button
              className={`filter-btn ${selectedFilter === 'completed' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('completed')}
            >
              Completed
            </button>
          </div>
        </div>
      </div>

      <div className="manager-content">
        {/* Bandwidth Allocation */}
        <div className="bandwidth-section">
          <h2 className="section-title">Bandwidth Allocation</h2>
          <div className="bandwidth-total">
            <label>Total Bandwidth:</label>
            <input
              type="range"
              min="10"
              max="1000"
              value={totalBandwidth}
              onChange={(e) => setTotalBandwidth(Number(e.target.value))}
              className="cyberpunk-slider"
            />
            <span className="bandwidth-value">{totalBandwidth} Mbps</span>
          </div>
          
          <div className="bandwidth-allocations">
            {bandwidthAllocations.map(allocation => (
              <div key={allocation.service} className="bandwidth-allocation">
                <span className="service-name">{allocation.service}</span>
                <div className="allocation-bar">
                  <div
                    className="allocation-used"
                    style={{ width: `${(allocation.used / allocation.limit) * 100}%` }}
                  />
                  <input
                    type="range"
                    min="0"
                    max={totalBandwidth}
                    value={allocation.limit}
                    onChange={(e) => handleBandwidthChange(allocation.service, Number(e.target.value))}
                    className="allocation-slider"
                  />
                </div>
                <span className="allocation-value">
                  {allocation.used}/{allocation.limit} Mbps
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Schedule Settings */}
        <div className="schedule-section">
          <h2 className="section-title">Download Schedule</h2>
          <div className="schedule-controls">
            <label className="cyberpunk-checkbox">
              <input
                type="checkbox"
                checked={scheduleEnabled}
                onChange={(e) => setScheduleEnabled(e.target.checked)}
              />
              <span className="checkmark"></span>
              Enable Scheduled Downloads
            </label>
            
            {scheduleEnabled && (
              <div className="schedule-times">
                <div className="time-input">
                  <label>Start:</label>
                  <input
                    type="time"
                    value={scheduleStart}
                    onChange={(e) => setScheduleStart(e.target.value)}
                    className="cyberpunk-input"
                  />
                </div>
                <div className="time-input">
                  <label>End:</label>
                  <input
                    type="time"
                    value={scheduleEnd}
                    onChange={(e) => setScheduleEnd(e.target.value)}
                    className="cyberpunk-input"
                  />
                </div>
              </div>
            )}
          </div>
          
          <label className="cyberpunk-checkbox">
            <input
              type="checkbox"
              checked={duplicateDetection}
              onChange={(e) => setDuplicateDetection(e.target.checked)}
            />
            <span className="checkmark"></span>
            Enable Duplicate Detection
          </label>
        </div>

        {/* Download Queue */}
        <div className="download-queue">
          <h2 className="section-title">Download Queue</h2>
          
          <DragDropContext onDragEnd={onDragEnd}>
            <Droppable droppableId="downloads">
              {(provided) => (
                <div
                  className="downloads-list"
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                >
                  {filteredDownloads.map((download, index) => (
                    <Draggable key={download.id} draggableId={download.id} index={index}>
                      {(provided, snapshot) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          {...provided.dragHandleProps}
                          className={`download-item ${snapshot.isDragging ? 'dragging' : ''}`}
                          style={{
                            ...provided.draggableProps.style,
                            borderColor: getPriorityColor(download.priority),
                          }}
                        >
                          <div className="download-header">
                            <div className="download-info">
                              <h3 className="download-name">{download.name}</h3>
                              <span className="download-service">{download.service}</span>
                              <span className="download-type">{download.type}</span>
                            </div>
                            
                            <div className="download-priority">
                              <select
                                value={download.priority}
                                onChange={(e) => handlePriorityChange(download.id, e.target.value as Download['priority'])}
                                className="priority-select"
                                style={{ borderColor: getPriorityColor(download.priority) }}
                              >
                                <option value="low">Low</option>
                                <option value="normal">Normal</option>
                                <option value="high">High</option>
                                <option value="critical">Critical</option>
                              </select>
                            </div>
                          </div>
                          
                          <div className="download-progress">
                            <div className="progress-bar-container">
                              <div
                                className="progress-bar-fill"
                                style={{
                                  width: `${download.progress}%`,
                                  background: `linear-gradient(90deg, ${getPriorityColor(download.priority)}, #00ffff)`,
                                }}
                              />
                              <span className="progress-text">{Math.round(download.progress)}%</span>
                            </div>
                          </div>
                          
                          <div className="download-stats">
                            <span className="stat">
                              <i className="icon-size"></i> {formatSize(download.size)}
                            </span>
                            <span className="stat">
                              <i className="icon-speed"></i> {formatSpeed(download.speed)}
                            </span>
                            <span className="stat">
                              <i className="icon-eta"></i> {formatETA(download.eta)}
                            </span>
                            {download.seeders !== undefined && (
                              <span className="stat">
                                <i className="icon-seeders"></i> {download.seeders}/{download.leechers}
                              </span>
                            )}
                            {download.ratio !== undefined && (
                              <span className="stat">
                                <i className="icon-ratio"></i> {download.ratio.toFixed(2)}
                              </span>
                            )}
                          </div>
                          
                          <div className="download-actions">
                            {download.status === 'downloading' && (
                              <button
                                className="action-btn pause-btn"
                                onClick={() => handlePause(download.id)}
                              >
                                ⏸
                              </button>
                            )}
                            {download.status === 'paused' && (
                              <button
                                className="action-btn resume-btn"
                                onClick={() => handleResume(download.id)}
                              >
                                ▶
                              </button>
                            )}
                            <button
                              className="action-btn delete-btn"
                              onClick={() => handleDelete(download.id)}
                            >
                              🗑
                            </button>
                          </div>
                        </div>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>
        </div>
      </div>

      {/* Holographic decoration */}
      <div className="holographic-overlay"></div>
    </div>
  );
};

export default SmartDownloadManager;