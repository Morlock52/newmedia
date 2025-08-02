#!/usr/bin/env python3
"""
Advanced Predictive Maintenance System
AI-powered failure prediction and maintenance scheduling
"""

import asyncio
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import sqlite3
import yaml
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MaintenanceAlert:
    """Predictive maintenance alert structure"""
    timestamp: datetime
    component: str
    predicted_failure_time: datetime
    confidence: float
    failure_type: str
    severity: str  # critical, high, medium, low
    recommended_action: str
    estimated_downtime: int  # minutes
    cost_impact: float
    prerequisites: List[str]

@dataclass
class ComponentHealth:
    """Component health assessment"""
    component_id: str
    component_type: str
    health_score: float  # 0-100
    degradation_rate: float
    remaining_useful_life: int  # days
    risk_factors: List[str]
    maintenance_history: List[Dict]
    performance_trend: str  # improving, stable, degrading

class HealthScoreCalculator:
    """Calculate health scores for system components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.component_weights = config.get('component_weights', {
            'cpu_usage': 0.2,
            'memory_usage': 0.2,
            'disk_usage': 0.15,
            'network_latency': 0.1,
            'error_rate': 0.25,
            'uptime': 0.1
        })
    
    def calculate_health_score(self, metrics: Dict[str, float], 
                             component_type: str) -> float:
        """Calculate overall health score for a component"""
        try:
            score = 100.0
            
            # CPU health contribution
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > 0.8:
                score -= (cpu_usage - 0.8) * 100 * self.component_weights['cpu_usage']
            
            # Memory health contribution
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > 0.85:
                score -= (memory_usage - 0.85) * 100 * self.component_weights['memory_usage']
            
            # Disk health contribution
            disk_usage = metrics.get('disk_usage', 0)
            if disk_usage > 0.9:
                score -= (disk_usage - 0.9) * 100 * self.component_weights['disk_usage']
            
            # Network health contribution
            network_latency = metrics.get('network_latency', 0)
            if network_latency > 100:  # ms
                score -= min(network_latency / 10, 20) * self.component_weights['network_latency']
            
            # Error rate contribution
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.01:  # 1%
                score -= min(error_rate * 1000, 50) * self.component_weights['error_rate']
            
            # Uptime contribution
            uptime = metrics.get('uptime_percentage', 100)
            if uptime < 99:
                score -= (99 - uptime) * 5 * self.component_weights['uptime']
            
            return max(0, min(100, score))
            
        except Exception as e:
            logging.error(f"Health score calculation failed: {e}")
            return 50.0  # Default neutral score
    
    def identify_risk_factors(self, metrics: Dict[str, float], 
                            health_score: float) -> List[str]:
        """Identify risk factors affecting component health"""
        risk_factors = []
        
        if metrics.get('cpu_usage', 0) > 0.8:
            risk_factors.append('High CPU utilization')
        
        if metrics.get('memory_usage', 0) > 0.85:
            risk_factors.append('High memory usage')
        
        if metrics.get('disk_usage', 0) > 0.9:
            risk_factors.append('Low disk space')
        
        if metrics.get('error_rate', 0) > 0.05:
            risk_factors.append('Elevated error rate')
        
        if metrics.get('network_latency', 0) > 200:
            risk_factors.append('High network latency')
        
        if metrics.get('temperature', 0) > 70:  # Celsius
            risk_factors.append('High operating temperature')
        
        if health_score < 70:
            risk_factors.append('Overall health degradation')
        
        return risk_factors

class FailurePredictionModel:
    """Machine learning model for failure prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = [
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency',
            'error_rate', 'uptime_percentage', 'temperature', 'age_days',
            'maintenance_interval_days', 'load_average'
        ]
        
    def prepare_training_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical incidents"""
        try:
            df = pd.DataFrame(historical_data)
            
            # Feature engineering
            df['age_days'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['install_date'])).dt.days
            df['days_since_maintenance'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['last_maintenance'])).dt.days
            
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))
            
            # Prepare features
            X = df[self.feature_columns].values
            
            # Prepare labels (failure occurred within next 7 days)
            y = df['failure_within_7_days'].values
            
            return X, y
            
        except Exception as e:
            logging.error(f"Training data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def train_failure_prediction_model(self, X: np.ndarray, y: np.ndarray, 
                                     component_type: str):
        """Train failure prediction model for specific component type"""
        try:
            if len(X) == 0 or len(y) == 0:
                logging.warning(f"Insufficient data for training {component_type} model")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and scaler
            self.models[component_type] = model
            self.scalers[component_type] = scaler
            
            logging.info(f"Trained {component_type} failure prediction model - Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logging.error(f"Model training failed for {component_type}: {e}")
    
    def predict_failure_probability(self, features: Dict[str, float], 
                                  component_type: str) -> Tuple[float, datetime]:
        """Predict failure probability and estimated time"""
        try:
            if component_type not in self.models:
                return 0.0, datetime.now() + timedelta(days=365)  # Default: low risk
            
            model = self.models[component_type]
            scaler = self.scalers[component_type]
            
            # Prepare features
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            X = np.array([feature_vector])
            X_scaled = scaler.transform(X)
            
            # Predict failure probability
            failure_prob = model.predict_proba(X_scaled)[0][1]  # Probability of failure
            
            # Estimate time to failure based on probability and degradation rate
            degradation_rate = features.get('degradation_rate', 0.01)  # per day
            days_to_failure = max(1, int(-np.log(1 - failure_prob) / degradation_rate))
            
            estimated_failure_time = datetime.now() + timedelta(days=days_to_failure)
            
            return failure_prob, estimated_failure_time
            
        except Exception as e:
            logging.error(f"Failure prediction failed: {e}")
            return 0.0, datetime.now() + timedelta(days=365)

class MaintenanceScheduler:
    """Intelligent maintenance scheduling system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.maintenance_windows = config.get('maintenance_windows', {
            'preferred_days': ['saturday', 'sunday'],
            'preferred_hours': [2, 3, 4, 5],  # 2 AM - 6 AM
            'max_concurrent_maintenance': 2,
            'minimum_notice_hours': 24
        })
        
    def schedule_optimal_maintenance(self, alerts: List[MaintenanceAlert]) -> Dict[str, Any]:
        """Create optimal maintenance schedule considering constraints"""
        try:
            # Sort alerts by urgency (combination of severity and predicted failure time)
            sorted_alerts = sorted(alerts, key=lambda x: (
                self._severity_weight(x.severity),
                (x.predicted_failure_time - datetime.now()).total_seconds()
            ))
            
            schedule = {
                'immediate': [],  # Critical items requiring immediate attention
                'next_window': [],  # Items for next maintenance window
                'future_windows': [],  # Items for future windows
                'deferred': []  # Low priority items that can be deferred
            }
            
            for alert in sorted_alerts:
                time_to_failure = (alert.predicted_failure_time - datetime.now()).total_seconds() / 3600  # hours
                
                if alert.severity == 'critical' or time_to_failure < 24:
                    schedule['immediate'].append(alert)
                elif time_to_failure < 168:  # 1 week
                    schedule['next_window'].append(alert)
                elif time_to_failure < 720:  # 1 month
                    schedule['future_windows'].append(alert)
                else:
                    schedule['deferred'].append(alert)
            
            # Optimize maintenance windows
            optimized_schedule = self._optimize_maintenance_windows(schedule)
            
            return optimized_schedule
            
        except Exception as e:
            logging.error(f"Maintenance scheduling failed: {e}")
            return {'immediate': [], 'next_window': [], 'future_windows': [], 'deferred': []}
    
    def _severity_weight(self, severity: str) -> int:
        """Convert severity to numeric weight for sorting"""
        weights = {'critical': 1, 'high': 2, 'medium': 3, 'low': 4}
        return weights.get(severity, 5)
    
    def _optimize_maintenance_windows(self, schedule: Dict[str, List]) -> Dict[str, Any]:
        """Optimize maintenance timing within available windows"""
        optimized = schedule.copy()
        
        # Calculate optimal maintenance windows
        next_windows = []
        preferred_days = self.maintenance_windows['preferred_days']
        preferred_hours = self.maintenance_windows['preferred_hours']
        
        # Find next available maintenance windows
        current_date = datetime.now()
        for i in range(30):  # Look ahead 30 days
            check_date = current_date + timedelta(days=i)
            day_name = check_date.strftime('%A').lower()
            
            if day_name in preferred_days:
                for hour in preferred_hours:
                    window_start = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    if window_start > current_date:
                        next_windows.append(window_start)
                        if len(next_windows) >= 10:  # Limit to next 10 windows
                            break
            
            if len(next_windows) >= 10:
                break
        
        optimized['suggested_windows'] = [w.isoformat() for w in next_windows]
        optimized['window_utilization'] = self._calculate_window_utilization(schedule)
        
        return optimized
    
    def _calculate_window_utilization(self, schedule: Dict[str, List]) -> Dict[str, Any]:
        """Calculate maintenance window utilization metrics"""
        total_items = sum(len(items) for items in schedule.values() if isinstance(items, list))
        critical_items = len(schedule.get('immediate', []))
        
        return {
            'total_maintenance_items': total_items,
            'critical_items': critical_items,
            'utilization_percentage': min(100, (total_items / 10) * 100),  # Assume capacity of 10 items per window
            'estimated_maintenance_hours': total_items * 2  # Assume 2 hours per item
        }

class PredictiveMaintenanceEngine:
    """Main predictive maintenance orchestrator"""
    
    def __init__(self, config_path: str = '/app/config/predictive-maintenance.yml'):
        self.config = self._load_config(config_path)
        self.health_calculator = HealthScoreCalculator(self.config)
        self.prediction_model = FailurePredictionModel(self.config)
        self.maintenance_scheduler = MaintenanceScheduler(self.config)
        
        # Database setup
        self.db_path = self.config.get('database_path', '/app/data/predictive_maintenance.db')
        self._init_database()
        
        # Component tracking
        self.monitored_components = self.config.get('monitored_components', [
            'jellyfin', 'sonarr', 'radarr', 'prowlarr', 'qbittorrent',
            'grafana', 'prometheus', 'traefik', 'postgres'
        ])
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load predictive maintenance configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'analysis_interval': 3600,  # 1 hour
            'prediction_horizon_days': 30,
            'health_threshold': 70,
            'database_path': '/app/data/predictive_maintenance.db',
            'component_weights': {
                'cpu_usage': 0.2,
                'memory_usage': 0.2,
                'disk_usage': 0.15,
                'network_latency': 0.1,
                'error_rate': 0.25,
                'uptime': 0.1
            },
            'maintenance_windows': {
                'preferred_days': ['saturday', 'sunday'],
                'preferred_hours': [2, 3, 4, 5],
                'max_concurrent_maintenance': 2,
                'minimum_notice_hours': 24
            },
            'monitored_components': [
                'jellyfin', 'sonarr', 'radarr', 'prowlarr', 'qbittorrent',
                'grafana', 'prometheus', 'traefik'
            ]
        }
    
    def _init_database(self):
        """Initialize database for predictive maintenance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Component health history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS component_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component_id TEXT NOT NULL,
                component_type TEXT NOT NULL,
                health_score REAL NOT NULL,
                degradation_rate REAL DEFAULT 0,
                risk_factors TEXT,
                metrics TEXT NOT NULL
            )
        ''')
        
        # Maintenance alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                predicted_failure_time DATETIME NOT NULL,
                confidence REAL NOT NULL,
                failure_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                estimated_downtime INTEGER NOT NULL,
                cost_impact REAL DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Maintenance history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                maintenance_type TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                cost REAL DEFAULT 0,
                outcome TEXT NOT NULL,
                notes TEXT
            )
        ''')
        
        # Failure incidents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                downtime_minutes INTEGER NOT NULL,
                root_cause TEXT,
                resolution TEXT,
                cost_impact REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def analyze_component_health(self, component_id: str, 
                                     metrics: Dict[str, float]) -> ComponentHealth:
        """Analyze health of a specific component"""
        try:
            component_type = self._get_component_type(component_id)
            
            # Calculate health score
            health_score = self.health_calculator.calculate_health_score(metrics, component_type)
            
            # Identify risk factors
            risk_factors = self.health_calculator.identify_risk_factors(metrics, health_score)
            
            # Calculate degradation rate from historical data
            degradation_rate = await self._calculate_degradation_rate(component_id)
            
            # Estimate remaining useful life
            remaining_life = await self._estimate_remaining_life(component_id, health_score, degradation_rate)
            
            # Get maintenance history
            maintenance_history = await self._get_maintenance_history(component_id)
            
            # Determine performance trend
            performance_trend = await self._determine_performance_trend(component_id)
            
            # Store health data
            await self._store_component_health(component_id, component_type, health_score, 
                                             degradation_rate, risk_factors, metrics)
            
            return ComponentHealth(
                component_id=component_id,
                component_type=component_type,
                health_score=health_score,
                degradation_rate=degradation_rate,
                remaining_useful_life=remaining_life,
                risk_factors=risk_factors,
                maintenance_history=maintenance_history,
                performance_trend=performance_trend
            )
            
        except Exception as e:
            logging.error(f"Component health analysis failed for {component_id}: {e}")
            return ComponentHealth(
                component_id=component_id,
                component_type="unknown",
                health_score=50.0,
                degradation_rate=0.01,
                remaining_useful_life=365,
                risk_factors=[],
                maintenance_history=[],
                performance_trend="unknown"
            )
    
    async def predict_maintenance_needs(self, components: List[ComponentHealth]) -> List[MaintenanceAlert]:
        """Predict maintenance needs for components"""
        alerts = []
        
        for component in components:
            try:
                # Prepare features for prediction
                features = {
                    'health_score': component.health_score,
                    'degradation_rate': component.degradation_rate,
                    'age_days': 365,  # Default age
                    'maintenance_interval_days': 30,  # Default interval
                    'cpu_usage': 0.5,  # Default values - should come from metrics
                    'memory_usage': 0.5,
                    'disk_usage': 0.5,
                    'network_latency': 50,
                    'error_rate': 0.01,
                    'uptime_percentage': 99.0,
                    'temperature': 45,
                    'load_average': 1.0
                }
                
                # Predict failure probability and time
                failure_prob, failure_time = self.prediction_model.predict_failure_probability(
                    features, component.component_type
                )
                
                # Generate alert if probability is significant
                if failure_prob > 0.3 or component.health_score < self.config['health_threshold']:
                    severity = self._determine_alert_severity(failure_prob, component.health_score)
                    
                    alert = MaintenanceAlert(
                        timestamp=datetime.now(),
                        component=component.component_id,
                        predicted_failure_time=failure_time,
                        confidence=failure_prob,
                        failure_type=self._predict_failure_type(component),
                        severity=severity,
                        recommended_action=self._get_recommended_action(component, failure_prob),
                        estimated_downtime=self._estimate_downtime(component.component_type),
                        cost_impact=self._estimate_cost_impact(component.component_type),
                        prerequisites=self._get_maintenance_prerequisites(component.component_type)
                    )
                    
                    alerts.append(alert)
                    await self._store_maintenance_alert(alert)
                    
            except Exception as e:
                logging.error(f"Maintenance prediction failed for {component.component_id}: {e}")
        
        return alerts
    
    async def generate_maintenance_schedule(self, alerts: List[MaintenanceAlert]) -> Dict[str, Any]:
        """Generate optimal maintenance schedule"""
        try:
            schedule = self.maintenance_scheduler.schedule_optimal_maintenance(alerts)
            
            # Add cost-benefit analysis
            schedule['cost_analysis'] = self._calculate_cost_benefit(alerts)
            
            # Add resource requirements
            schedule['resource_requirements'] = self._calculate_resource_requirements(alerts)
            
            return schedule
            
        except Exception as e:
            logging.error(f"Maintenance scheduling failed: {e}")
            return {}
    
    def _get_component_type(self, component_id: str) -> str:
        """Determine component type from component ID"""
        type_mapping = {
            'jellyfin': 'media_server',
            'sonarr': 'arr_service',
            'radarr': 'arr_service',
            'lidarr': 'arr_service',
            'prowlarr': 'indexer',
            'qbittorrent': 'download_client',
            'grafana': 'monitoring',
            'prometheus': 'monitoring',
            'traefik': 'proxy',
            'postgres': 'database'
        }
        
        return type_mapping.get(component_id, 'generic')
    
    async def _calculate_degradation_rate(self, component_id: str) -> float:
        """Calculate component degradation rate from historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get health scores from last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            cursor.execute('''
                SELECT health_score, timestamp FROM component_health
                WHERE component_id = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (component_id, cutoff_date))
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) < 2:
                return 0.01  # Default degradation rate
            
            # Calculate linear regression slope
            health_scores = [r[0] for r in results]
            timestamps = [(datetime.fromisoformat(r[1]) - datetime.now()).total_seconds() for r in results]
            
            if len(health_scores) > 1:
                slope = np.polyfit(timestamps, health_scores, 1)[0]
                return max(0, -slope / 86400)  # Convert to per-day degradation
            
            return 0.01
            
        except Exception as e:
            logging.error(f"Degradation rate calculation failed: {e}")
            return 0.01
    
    async def _estimate_remaining_life(self, component_id: str, health_score: float, 
                                     degradation_rate: float) -> int:
        """Estimate remaining useful life in days"""
        try:
            if degradation_rate <= 0:
                return 365  # Default to 1 year if no degradation
            
            # Calculate days until health score reaches critical threshold (20)
            critical_threshold = 20
            if health_score <= critical_threshold:
                return 0
            
            days_to_critical = (health_score - critical_threshold) / degradation_rate
            return max(0, int(days_to_critical))
            
        except Exception:
            return 365
    
    def _determine_alert_severity(self, failure_prob: float, health_score: float) -> str:
        """Determine alert severity based on probability and health score"""
        if failure_prob > 0.8 or health_score < 20:
            return 'critical'
        elif failure_prob > 0.6 or health_score < 40:
            return 'high'
        elif failure_prob > 0.4 or health_score < 60:
            return 'medium'
        else:
            return 'low'
    
    def _predict_failure_type(self, component: ComponentHealth) -> str:
        """Predict most likely failure type based on risk factors"""
        risk_factors = component.risk_factors
        
        if 'High CPU utilization' in risk_factors:
            return 'performance_degradation'
        elif 'High memory usage' in risk_factors:
            return 'memory_exhaustion'
        elif 'Low disk space' in risk_factors:
            return 'storage_failure'
        elif 'Elevated error rate' in risk_factors:
            return 'service_failure'
        elif 'High network latency' in risk_factors:
            return 'network_issues'
        else:
            return 'general_degradation'
    
    async def start_predictive_maintenance(self):
        """Start the predictive maintenance system"""
        logging.info("Starting Predictive Maintenance Engine...")
        
        while True:
            try:
                # Collect current metrics for all components
                component_metrics = await self._collect_component_metrics()
                
                # Analyze health for each component
                component_health = []
                for component_id, metrics in component_metrics.items():
                    health = await self.analyze_component_health(component_id, metrics)
                    component_health.append(health)
                
                # Predict maintenance needs
                maintenance_alerts = await self.predict_maintenance_needs(component_health)
                
                # Generate maintenance schedule
                if maintenance_alerts:
                    schedule = await self.generate_maintenance_schedule(maintenance_alerts)
                    await self._broadcast_maintenance_schedule(schedule)
                
                # Log status
                logging.info(f"Analyzed {len(component_health)} components, "
                           f"generated {len(maintenance_alerts)} alerts")
                
                # Wait for next analysis cycle
                await asyncio.sleep(self.config['analysis_interval'])
                
            except Exception as e:
                logging.error(f"Predictive maintenance cycle failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    engine = PredictiveMaintenanceEngine()
    asyncio.run(engine.start_predictive_maintenance())