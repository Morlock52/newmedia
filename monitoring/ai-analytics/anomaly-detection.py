#!/usr/bin/env python3
"""
Advanced AI-Powered Anomaly Detection Engine
Real-time monitoring with machine learning algorithms
"""

import asyncio
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import sqlite3
import aiohttp
import websockets
import yaml

@dataclass
class AnomalyAlert:
    """Anomaly detection alert structure"""
    timestamp: datetime
    service: str
    metric: str
    value: float
    severity: str  # critical, warning, info
    confidence: float
    description: str
    suggested_action: str

class MetricsCollector:
    """Advanced metrics collection from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = []
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session and connections"""
        self.session = aiohttp.ClientSession()
        
    async def collect_prometheus_metrics(self) -> Dict[str, Any]:
        """Collect metrics from Prometheus"""
        try:
            prometheus_url = self.config.get('prometheus_url', 'http://localhost:9090')
            queries = {
                'cpu_usage': 'rate(cpu_usage_total[5m])',
                'memory_usage': 'memory_usage_bytes / memory_total_bytes',
                'disk_io': 'rate(disk_io_bytes[5m])',
                'network_io': 'rate(network_io_bytes[5m])',
                'response_time': 'http_request_duration_seconds',
                'error_rate': 'rate(http_requests_errors_total[5m])',
                'active_connections': 'active_connections_total',
                'transcoding_load': 'transcoding_sessions_active'
            }
            
            metrics = {}
            for metric_name, query in queries.items():
                url = f"{prometheus_url}/api/v1/query?query={query}"
                async with self.session.get(url) as response:
                    data = await response.json()
                    if data.get('status') == 'success':
                        metrics[metric_name] = data['data']['result']
                        
            return metrics
        except Exception as e:
            logging.error(f"Failed to collect Prometheus metrics: {e}")
            return {}
    
    async def collect_docker_stats(self) -> Dict[str, Any]:
        """Collect Docker container statistics"""
        try:
            import docker
            client = docker.from_env()
            
            stats = {}
            for container in client.containers.list():
                container_stats = container.stats(stream=False)
                stats[container.name] = {
                    'cpu_percent': self._calculate_cpu_percent(container_stats),
                    'memory_usage': container_stats['memory_stats'].get('usage', 0),
                    'memory_limit': container_stats['memory_stats'].get('limit', 0),
                    'network_rx': container_stats['networks']['eth0']['rx_bytes'],
                    'network_tx': container_stats['networks']['eth0']['tx_bytes'],
                    'block_read': container_stats['blkio_stats']['io_service_bytes_recursive'][0]['value'],
                    'block_write': container_stats['blkio_stats']['io_service_bytes_recursive'][1]['value']
                }
            
            return stats
        except Exception as e:
            logging.error(f"Failed to collect Docker stats: {e}")
            return {}
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from Docker stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0

class AnomalyDetector:
    """ML-based anomaly detection engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.training_data = {}
        self.detection_thresholds = config.get('thresholds', {
            'cpu_usage': 0.8,
            'memory_usage': 0.9,
            'disk_io': 1e9,  # 1GB/s
            'response_time': 5.0,  # 5 seconds
            'error_rate': 0.05  # 5%
        })
        
    def initialize_models(self):
        """Initialize ML models for different metrics"""
        # Isolation Forest for univariate anomaly detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # DBSCAN for clustering-based anomaly detection
        self.models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # Initialize scalers for each metric type
        metric_types = ['cpu', 'memory', 'disk', 'network', 'response_time', 'error_rate']
        for metric_type in metric_types:
            self.scalers[metric_type] = StandardScaler()
    
    async def train_models(self, historical_data: Dict[str, List[float]]):
        """Train anomaly detection models on historical data"""
        try:
            for metric_name, data in historical_data.items():
                if len(data) < 100:  # Need sufficient data for training
                    continue
                
                # Prepare data
                X = np.array(data).reshape(-1, 1)
                X_scaled = self.scalers[metric_name].fit_transform(X)
                
                # Train isolation forest
                self.models[f'{metric_name}_isolation'] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                ).fit(X_scaled)
                
                # Store training statistics
                self.training_data[metric_name] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'percentile_95': np.percentile(data, 95),
                    'percentile_99': np.percentile(data, 99)
                }
                
                logging.info(f"Trained model for {metric_name} with {len(data)} samples")
                
        except Exception as e:
            logging.error(f"Failed to train models: {e}")
    
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[AnomalyAlert]:
        """Detect anomalies in current metrics"""
        alerts = []
        
        for metric_name, value in current_metrics.items():
            try:
                # Statistical anomaly detection
                stat_anomaly = self._detect_statistical_anomaly(metric_name, value)
                if stat_anomaly:
                    alerts.append(stat_anomaly)
                
                # ML-based anomaly detection
                ml_anomaly = self._detect_ml_anomaly(metric_name, value)
                if ml_anomaly:
                    alerts.append(ml_anomaly)
                
                # Threshold-based detection
                threshold_anomaly = self._detect_threshold_anomaly(metric_name, value)
                if threshold_anomaly:
                    alerts.append(threshold_anomaly)
                    
            except Exception as e:
                logging.error(f"Error detecting anomalies for {metric_name}: {e}")
        
        return alerts
    
    def _detect_statistical_anomaly(self, metric_name: str, value: float) -> Optional[AnomalyAlert]:
        """Detect statistical anomalies using z-score"""
        if metric_name not in self.training_data:
            return None
        
        stats = self.training_data[metric_name]
        z_score = abs((value - stats['mean']) / stats['std']) if stats['std'] > 0 else 0
        
        if z_score > 3:  # 3-sigma rule
            severity = 'critical' if z_score > 4 else 'warning'
            return AnomalyAlert(
                timestamp=datetime.now(),
                service=metric_name.split('_')[0],
                metric=metric_name,
                value=value,
                severity=severity,
                confidence=min(z_score / 4.0, 1.0),
                description=f"Statistical anomaly detected: z-score {z_score:.2f}",
                suggested_action=self._get_suggested_action(metric_name, value)
            )
        
        return None
    
    def _detect_ml_anomaly(self, metric_name: str, value: float) -> Optional[AnomalyAlert]:
        """Detect anomalies using trained ML models"""
        model_key = f'{metric_name}_isolation'
        if model_key not in self.models or metric_name not in self.scalers:
            return None
        
        try:
            # Scale the value
            X = np.array([[value]])
            X_scaled = self.scalers[metric_name].transform(X)
            
            # Predict anomaly
            prediction = self.models[model_key].predict(X_scaled)[0]
            anomaly_score = self.models[model_key].score_samples(X_scaled)[0]
            
            if prediction == -1:  # Anomaly detected
                confidence = abs(anomaly_score)
                severity = 'critical' if confidence > 0.5 else 'warning'
                
                return AnomalyAlert(
                    timestamp=datetime.now(),
                    service=metric_name.split('_')[0],
                    metric=metric_name,
                    value=value,
                    severity=severity,
                    confidence=confidence,
                    description=f"ML anomaly detected: anomaly score {anomaly_score:.3f}",
                    suggested_action=self._get_suggested_action(metric_name, value)
                )
        
        except Exception as e:
            logging.error(f"ML anomaly detection failed for {metric_name}: {e}")
        
        return None
    
    def _detect_threshold_anomaly(self, metric_name: str, value: float) -> Optional[AnomalyAlert]:
        """Detect threshold-based anomalies"""
        if metric_name not in self.detection_thresholds:
            return None
        
        threshold = self.detection_thresholds[metric_name]
        
        if value > threshold:
            severity = 'critical' if value > threshold * 1.5 else 'warning'
            confidence = min(value / threshold, 2.0) / 2.0
            
            return AnomalyAlert(
                timestamp=datetime.now(),
                service=metric_name.split('_')[0],
                metric=metric_name,
                value=value,
                severity=severity,
                confidence=confidence,
                description=f"Threshold exceeded: {value:.2f} > {threshold:.2f}",
                suggested_action=self._get_suggested_action(metric_name, value)
            )
        
        return None
    
    def _get_suggested_action(self, metric_name: str, value: float) -> str:
        """Get suggested action based on metric and value"""
        actions = {
            'cpu_usage': "Consider scaling up instances or optimizing CPU-intensive processes",
            'memory_usage': "Monitor for memory leaks, consider increasing memory allocation",
            'disk_io': "Check for disk bottlenecks, consider faster storage or load balancing",
            'network_io': "Monitor network congestion, consider CDN or bandwidth optimization",
            'response_time': "Check application performance, database queries, and resource usage",
            'error_rate': "Investigate error logs, check service dependencies and configurations",
            'transcoding_load': "Consider adding transcoding capacity or optimizing encoding settings"
        }
        
        return actions.get(metric_name, "Investigate further and monitor trend")

class PredictiveAnalyzer:
    """Predictive analysis for maintenance and capacity planning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prediction_models = {}
        
    def predict_capacity_needs(self, historical_data: Dict[str, List[float]], 
                             horizon_days: int = 30) -> Dict[str, Any]:
        """Predict capacity needs for the next N days"""
        predictions = {}
        
        for metric_name, data in historical_data.items():
            if len(data) < 168:  # Need at least a week of hourly data
                continue
            
            try:
                # Simple linear trend prediction
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 1)
                
                # Predict future values
                future_x = np.arange(len(data), len(data) + horizon_days * 24)
                future_values = np.polyval(coeffs, future_x)
                
                # Calculate confidence intervals
                residuals = data - np.polyval(coeffs, x)
                mse = np.mean(residuals ** 2)
                confidence_interval = 1.96 * np.sqrt(mse)  # 95% CI
                
                predictions[metric_name] = {
                    'trend_slope': coeffs[0],
                    'predicted_values': future_values.tolist(),
                    'confidence_interval': confidence_interval,
                    'current_value': data[-1],
                    'predicted_max': np.max(future_values),
                    'days_to_threshold': self._calculate_days_to_threshold(
                        coeffs, data[-1], metric_name
                    )
                }
                
            except Exception as e:
                logging.error(f"Prediction failed for {metric_name}: {e}")
        
        return predictions
    
    def _calculate_days_to_threshold(self, coeffs: np.ndarray, current_value: float, 
                                   metric_name: str) -> Optional[int]:
        """Calculate days until threshold is reached"""
        thresholds = {
            'cpu_usage': 0.8,
            'memory_usage': 0.9,
            'disk_usage': 0.85,
            'network_usage': 0.8
        }
        
        if metric_name not in thresholds or coeffs[0] <= 0:
            return None
        
        threshold = thresholds[metric_name]
        if current_value >= threshold:
            return 0
        
        # Linear equation: y = mx + b, solve for x when y = threshold
        # threshold = coeffs[0] * x + coeffs[1]
        # x = (threshold - coeffs[1]) / coeffs[0]
        days_to_threshold = (threshold - coeffs[1]) / coeffs[0]
        
        return max(0, int(days_to_threshold / 24))  # Convert hours to days

class RealTimeMonitor:
    """Real-time monitoring orchestrator"""
    
    def __init__(self, config_path: str = '/app/config/monitoring.yml'):
        self.config = self._load_config(config_path)
        self.metrics_collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.predictive_analyzer = PredictiveAnalyzer(self.config)
        self.running = False
        
        # Initialize database
        self.db_path = self.config.get('database_path', '/app/data/monitoring.db')
        self._init_database()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'collection_interval': 30,  # seconds
                'training_interval': 3600,  # 1 hour
                'prediction_interval': 86400,  # 24 hours
                'prometheus_url': 'http://prometheus:9090',
                'database_path': '/app/data/monitoring.db',
                'websocket_port': 8765,
                'thresholds': {
                    'cpu_usage': 0.8,
                    'memory_usage': 0.9,
                    'disk_io': 1e9,
                    'response_time': 5.0,
                    'error_rate': 0.05
                }
            }
    
    def _init_database(self):
        """Initialize SQLite database for storing metrics and alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                service TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                source TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                service TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                prediction_horizon INTEGER NOT NULL,
                predicted_values TEXT NOT NULL,
                confidence_interval REAL NOT NULL,
                days_to_threshold INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def start_monitoring(self):
        """Start the real-time monitoring system"""
        self.running = True
        logging.info("Starting AI-powered monitoring system...")
        
        # Initialize components
        await self.metrics_collector.initialize()
        self.anomaly_detector.initialize_models()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._model_training_loop()),
            asyncio.create_task(self._prediction_loop()),
            asyncio.create_task(self._websocket_server())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"Monitoring system error: {e}")
        finally:
            self.running = False
            if self.metrics_collector.session:
                await self.metrics_collector.session.close()
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection"""
        while self.running:
            try:
                # Collect metrics from various sources
                prometheus_metrics = await self.metrics_collector.collect_prometheus_metrics()
                docker_stats = await self.metrics_collector.collect_docker_stats()
                
                # Store metrics in database
                await self._store_metrics(prometheus_metrics, 'prometheus')
                await self._store_metrics(docker_stats, 'docker')
                
                # Sleep until next collection
                await asyncio.sleep(self.config['collection_interval'])
                
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _anomaly_detection_loop(self):
        """Continuous anomaly detection"""
        while self.running:
            try:
                # Get recent metrics
                current_metrics = await self._get_recent_metrics()
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
                
                # Store and broadcast anomalies
                for anomaly in anomalies:
                    await self._store_anomaly(anomaly)
                    await self._broadcast_anomaly(anomaly)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(5)
    
    async def _model_training_loop(self):
        """Periodic model retraining"""
        while self.running:
            try:
                # Get historical data for training
                historical_data = await self._get_historical_data()
                
                # Retrain models
                await self.anomaly_detector.train_models(historical_data)
                
                logging.info("Models retrained successfully")
                await asyncio.sleep(self.config['training_interval'])
                
            except Exception as e:
                logging.error(f"Model training error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _prediction_loop(self):
        """Periodic predictive analysis"""
        while self.running:
            try:
                # Get historical data
                historical_data = await self._get_historical_data(days=30)
                
                # Generate predictions
                predictions = self.predictive_analyzer.predict_capacity_needs(historical_data)
                
                # Store predictions
                await self._store_predictions(predictions)
                
                logging.info("Predictions updated successfully")
                await asyncio.sleep(self.config['prediction_interval'])
                
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _websocket_server(self):
        """WebSocket server for real-time dashboard updates"""
        async def handle_client(websocket, path):
            try:
                async for message in websocket:
                    # Handle client requests
                    data = json.loads(message)
                    
                    if data.get('type') == 'get_metrics':
                        metrics = await self._get_recent_metrics()
                        await websocket.send(json.dumps({
                            'type': 'metrics_update',
                            'data': metrics
                        }))
                    
                    elif data.get('type') == 'get_anomalies':
                        anomalies = await self._get_recent_anomalies()
                        await websocket.send(json.dumps({
                            'type': 'anomalies_update',
                            'data': anomalies
                        }))
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
        
        port = self.config.get('websocket_port', 8765)
        start_server = websockets.serve(handle_client, "0.0.0.0", port)
        await start_server
        logging.info(f"WebSocket server started on port {port}")
    
    async def _store_metrics(self, metrics: Dict[str, Any], source: str):
        """Store metrics in database"""
        if not metrics:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for service, service_metrics in metrics.items():
                if isinstance(service_metrics, dict):
                    for metric_name, value in service_metrics.items():
                        if isinstance(value, (int, float)):
                            cursor.execute('''
                                INSERT INTO metrics (service, metric_name, value, source)
                                VALUES (?, ?, ?, ?)
                            ''', (service, metric_name, float(value), source))
            
            conn.commit()
        finally:
            conn.close()
    
    async def _store_anomaly(self, anomaly: AnomalyAlert):
        """Store anomaly alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO anomalies 
                (service, metric_name, value, severity, confidence, description, suggested_action)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                anomaly.service,
                anomaly.metric,
                anomaly.value,
                anomaly.severity,
                anomaly.confidence,
                anomaly.description,
                anomaly.suggested_action
            ))
            
            conn.commit()
        finally:
            conn.close()
    
    async def _get_recent_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get recent metrics for anomaly detection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        cursor.execute('''
            SELECT metric_name, AVG(value) as avg_value
            FROM metrics
            WHERE timestamp > ?
            GROUP BY metric_name
        ''', (cutoff_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        return {metric_name: avg_value for metric_name, avg_value in results}
    
    async def _get_historical_data(self, days: int = 7) -> Dict[str, List[float]]:
        """Get historical data for model training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT metric_name, value
            FROM metrics
            WHERE timestamp > ?
            ORDER BY metric_name, timestamp
        ''', (cutoff_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Group by metric name
        data = {}
        for metric_name, value in results:
            if metric_name not in data:
                data[metric_name] = []
            data[metric_name].append(value)
        
        return data
    
    async def _broadcast_anomaly(self, anomaly: AnomalyAlert):
        """Broadcast anomaly to connected WebSocket clients"""
        # This would be implemented with a WebSocket broadcast mechanism
        logging.warning(f"ANOMALY DETECTED: {anomaly.description}")
        
        # Here you could integrate with external alerting systems:
        # - Slack notifications
        # - Email alerts
        # - PagerDuty
        # - Discord webhooks
        pass

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = RealTimeMonitor()
    asyncio.run(monitor.start_monitoring())