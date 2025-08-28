/**
 * Performance Monitor Component
 * Real-time system and service performance monitoring
 */

class PerformanceMonitor {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            refreshInterval: 5000,
            maxDataPoints: 50,
            showGrid: true,
            animated: true,
            ...options
        };
        
        this.charts = {};
        this.data = {
            system: {
                cpu: [],
                memory: [],
                disk: [],
                network: { in: [], out: [] }
            },
            services: {}
        };
        
        this.isRunning = false;
        this.init();
    }

    init() {
        this.createHTML();
        this.setupCharts();
        this.start();
    }

    createHTML() {
        this.container.innerHTML = `
            <div class="performance-monitor">
                <div class="monitor-header">
                    <h3>Performance Monitor</h3>
                    <div class="monitor-controls">
                        <button class="btn-control" data-action="toggle">⏸️</button>
                        <button class="btn-control" data-action="reset">🔄</button>
                        <button class="btn-control" data-action="fullscreen">⛶</button>
                    </div>
                </div>
                
                <div class="metrics-overview">
                    <div class="metric-card">
                        <div class="metric-label">CPU Usage</div>
                        <div class="metric-value" id="cpu-value">0%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="cpu-fill" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Memory</div>
                        <div class="metric-value" id="memory-value">0%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="memory-fill" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Disk I/O</div>
                        <div class="metric-value" id="disk-value">0 MB/s</div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="disk-fill" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Network</div>
                        <div class="metric-value" id="network-value">0 MB/s</div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="network-fill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="charts-container">
                    <div class="chart-section">
                        <div class="chart-header">
                            <h4>System Performance</h4>
                            <div class="chart-legend">
                                <span class="legend-item cpu">CPU</span>
                                <span class="legend-item memory">Memory</span>
                                <span class="legend-item disk">Disk</span>
                            </div>
                        </div>
                        <canvas id="system-chart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="chart-section">
                        <div class="chart-header">
                            <h4>Network Traffic</h4>
                            <div class="chart-legend">
                                <span class="legend-item network-in">In</span>
                                <span class="legend-item network-out">Out</span>
                            </div>
                        </div>
                        <canvas id="network-chart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="chart-section">
                        <div class="chart-header">
                            <h4>Service Performance</h4>
                            <select id="service-selector">
                                <option value="">Select a service...</option>
                            </select>
                        </div>
                        <canvas id="service-chart" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <div class="alerts-section">
                    <h4>Performance Alerts</h4>
                    <div id="alerts-container"></div>
                </div>
            </div>
        `;

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .performance-monitor {
                background: rgba(30, 41, 59, 0.8);
                backdrop-filter: blur(12px);
                border-radius: 12px;
                padding: 20px;
                color: #f1f5f9;
            }
            
            .monitor-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                border-bottom: 1px solid rgba(51, 65, 85, 0.5);
                padding-bottom: 10px;
            }
            
            .monitor-controls {
                display: flex;
                gap: 8px;
            }
            
            .btn-control {
                background: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 6px;
                padding: 6px 10px;
                color: #3b82f6;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .btn-control:hover {
                background: rgba(59, 130, 246, 0.3);
                transform: translateY(-1px);
            }
            
            .metrics-overview {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            
            .metric-card {
                background: rgba(51, 65, 85, 0.3);
                border-radius: 8px;
                padding: 16px;
                border: 1px solid rgba(51, 65, 85, 0.5);
            }
            
            .metric-label {
                font-size: 14px;
                color: #94a3b8;
                margin-bottom: 8px;
            }
            
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 8px;
                color: #f1f5f9;
            }
            
            .metric-bar {
                height: 4px;
                background: rgba(51, 65, 85, 0.5);
                border-radius: 2px;
                overflow: hidden;
            }
            
            .metric-fill {
                height: 100%;
                background: linear-gradient(90deg, #3b82f6, #06b6d4);
                border-radius: 2px;
                transition: width 0.5s ease;
            }
            
            .charts-container {
                display: grid;
                gap: 20px;
                margin-bottom: 24px;
            }
            
            .chart-section {
                background: rgba(51, 65, 85, 0.2);
                border-radius: 8px;
                padding: 16px;
                border: 1px solid rgba(51, 65, 85, 0.3);
            }
            
            .chart-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
            }
            
            .chart-legend {
                display: flex;
                gap: 16px;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                font-size: 12px;
                color: #94a3b8;
            }
            
            .legend-item::before {
                content: '';
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 6px;
            }
            
            .legend-item.cpu::before { background: #3b82f6; }
            .legend-item.memory::before { background: #10b981; }
            .legend-item.disk::before { background: #f59e0b; }
            .legend-item.network-in::before { background: #8b5cf6; }
            .legend-item.network-out::before { background: #ef4444; }
            
            #service-selector {
                background: rgba(51, 65, 85, 0.5);
                border: 1px solid rgba(51, 65, 85, 0.7);
                border-radius: 6px;
                padding: 4px 8px;
                color: #f1f5f9;
                font-size: 12px;
            }
            
            .alerts-section {
                border-top: 1px solid rgba(51, 65, 85, 0.5);
                padding-top: 16px;
            }
            
            #alerts-container {
                min-height: 40px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .alert {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                color: #fecaca;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .alert.warning {
                background: rgba(245, 158, 11, 0.1);
                border-color: rgba(245, 158, 11, 0.3);
                color: #fde68a;
            }
            
            .alert.info {
                background: rgba(59, 130, 246, 0.1);
                border-color: rgba(59, 130, 246, 0.3);
                color: #bfdbfe;
            }
            
            .alert-close {
                background: none;
                border: none;
                color: inherit;
                cursor: pointer;
                font-size: 16px;
                opacity: 0.7;
            }
            
            .alert-close:hover {
                opacity: 1;
            }
            
            @media (max-width: 768px) {
                .metrics-overview {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .chart-header {
                    flex-direction: column;
                    gap: 8px;
                    align-items: flex-start;
                }
            }
        `;
        document.head.appendChild(style);

        // Setup event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        const controls = this.container.querySelectorAll('.btn-control');
        controls.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                this.handleControlAction(action);
            });
        });

        const serviceSelector = this.container.querySelector('#service-selector');
        serviceSelector.addEventListener('change', (e) => {
            this.updateServiceChart(e.target.value);
        });
    }

    setupCharts() {
        // System performance chart
        const systemCanvas = this.container.querySelector('#system-chart');
        this.charts.system = new Chart(systemCanvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Memory',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Disk',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: this.getChartOptions()
        });

        // Network chart
        const networkCanvas = this.container.querySelector('#network-chart');
        this.charts.network = new Chart(networkCanvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'In',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Out',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: this.getChartOptions('MB/s')
        });

        // Service chart
        const serviceCanvas = this.container.querySelector('#service-chart');
        this.charts.service = new Chart(serviceCanvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: this.getChartOptions('%')
        });
    }

    getChartOptions(yAxisLabel = '%') {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    },
                    title: {
                        display: true,
                        text: yAxisLabel,
                        color: '#94a3b8'
                    },
                    min: 0,
                    max: yAxisLabel === '%' ? 100 : undefined
                }
            },
            elements: {
                point: {
                    radius: 2,
                    hoverRadius: 4
                }
            },
            animation: {
                duration: this.options.animated ? 300 : 0
            }
        };
    }

    updateData(newData) {
        const timestamp = new Date().toLocaleTimeString();
        
        // Update system metrics
        if (newData.system) {
            this.updateSystemMetrics(newData.system, timestamp);
        }
        
        // Update service metrics
        if (newData.services) {
            this.updateServiceMetrics(newData.services, timestamp);
        }
        
        // Check for alerts
        this.checkAlerts(newData);
    }

    updateSystemMetrics(systemData, timestamp) {
        // Update metric cards
        this.updateMetricCard('cpu', systemData.cpu?.usage || 0, '%');
        this.updateMetricCard('memory', systemData.memory?.percentage || 0, '%');
        this.updateMetricCard('disk', (systemData.disk?.readRate + systemData.disk?.writeRate) / 1024 / 1024 || 0, 'MB/s');
        this.updateMetricCard('network', (systemData.network?.bytesIn + systemData.network?.bytesOut) / 1024 / 1024 || 0, 'MB/s');

        // Update chart data
        const systemChart = this.charts.system;
        systemChart.data.labels.push(timestamp);
        systemChart.data.datasets[0].data.push(systemData.cpu?.usage || 0);
        systemChart.data.datasets[1].data.push(systemData.memory?.percentage || 0);
        systemChart.data.datasets[2].data.push(systemData.disk?.usage || 0);

        // Keep only last N data points
        if (systemChart.data.labels.length > this.options.maxDataPoints) {
            systemChart.data.labels.shift();
            systemChart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        systemChart.update('none');

        // Update network chart
        const networkChart = this.charts.network;
        networkChart.data.labels.push(timestamp);
        networkChart.data.datasets[0].data.push((systemData.network?.bytesIn || 0) / 1024 / 1024);
        networkChart.data.datasets[1].data.push((systemData.network?.bytesOut || 0) / 1024 / 1024);

        if (networkChart.data.labels.length > this.options.maxDataPoints) {
            networkChart.data.labels.shift();
            networkChart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        networkChart.update('none');
    }

    updateMetricCard(type, value, unit) {
        const valueElement = this.container.querySelector(`#${type}-value`);
        const fillElement = this.container.querySelector(`#${type}-fill`);
        
        if (valueElement) {
            valueElement.textContent = `${Math.round(value * 100) / 100}${unit}`;
        }
        
        if (fillElement) {
            const percentage = unit === '%' ? Math.min(value, 100) : Math.min((value / 100) * 100, 100);
            fillElement.style.width = `${percentage}%`;
            
            // Color based on usage
            if (percentage > 80) {
                fillElement.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
            } else if (percentage > 60) {
                fillElement.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
            } else {
                fillElement.style.background = 'linear-gradient(90deg, #10b981, #059669)';
            }
        }
    }

    updateServiceMetrics(servicesData, timestamp) {
        const serviceSelector = this.container.querySelector('#service-selector');
        const currentServices = Array.from(serviceSelector.options).map(opt => opt.value);
        
        // Add new services to selector
        Object.keys(servicesData).forEach(serviceName => {
            if (!currentServices.includes(serviceName)) {
                const option = document.createElement('option');
                option.value = serviceName;
                option.textContent = serviceName;
                serviceSelector.appendChild(option);
            }
        });

        // Store service data
        Object.keys(servicesData).forEach(serviceName => {
            if (!this.data.services[serviceName]) {
                this.data.services[serviceName] = { cpu: [], memory: [], timestamps: [] };
            }
            
            const service = this.data.services[serviceName];
            service.timestamps.push(timestamp);
            service.cpu.push(servicesData[serviceName].cpu || 0);
            service.memory.push(servicesData[serviceName].memory || 0);
            
            // Keep only last N data points
            if (service.timestamps.length > this.options.maxDataPoints) {
                service.timestamps.shift();
                service.cpu.shift();
                service.memory.shift();
            }
        });

        // Update service chart if a service is selected
        const selectedService = serviceSelector.value;
        if (selectedService && this.data.services[selectedService]) {
            this.updateServiceChart(selectedService);
        }
    }

    updateServiceChart(serviceName) {
        const serviceChart = this.charts.service;
        
        if (!serviceName || !this.data.services[serviceName]) {
            serviceChart.data.labels = [];
            serviceChart.data.datasets = [];
            serviceChart.update();
            return;
        }

        const service = this.data.services[serviceName];
        serviceChart.data.labels = service.timestamps;
        serviceChart.data.datasets = [
            {
                label: 'CPU',
                data: service.cpu,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: false,
                tension: 0.4
            },
            {
                label: 'Memory',
                data: service.memory,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: false,
                tension: 0.4
            }
        ];
        
        serviceChart.update();
    }

    checkAlerts(data) {
        const alerts = [];
        
        // System alerts
        if (data.system?.cpu?.usage > 90) {
            alerts.push({ type: 'error', message: 'CPU usage critical (>90%)', id: 'cpu-critical' });
        } else if (data.system?.cpu?.usage > 80) {
            alerts.push({ type: 'warning', message: 'CPU usage high (>80%)', id: 'cpu-high' });
        }
        
        if (data.system?.memory?.percentage > 95) {
            alerts.push({ type: 'error', message: 'Memory usage critical (>95%)', id: 'memory-critical' });
        } else if (data.system?.memory?.percentage > 85) {
            alerts.push({ type: 'warning', message: 'Memory usage high (>85%)', id: 'memory-high' });
        }
        
        if (data.system?.disk?.percentage > 95) {
            alerts.push({ type: 'error', message: 'Disk space critical (>95%)', id: 'disk-critical' });
        } else if (data.system?.disk?.percentage > 85) {
            alerts.push({ type: 'warning', message: 'Disk space high (>85%)', id: 'disk-high' });
        }
        
        // Service alerts
        if (data.services) {
            Object.keys(data.services).forEach(serviceName => {
                const service = data.services[serviceName];
                if (service.status === 'offline') {
                    alerts.push({ type: 'error', message: `${serviceName} is offline`, id: `service-${serviceName}-offline` });
                }
                if (service.cpu > 95) {
                    alerts.push({ type: 'warning', message: `${serviceName} CPU usage high (${service.cpu}%)`, id: `service-${serviceName}-cpu` });
                }
                if (service.memory > 95) {
                    alerts.push({ type: 'warning', message: `${serviceName} memory usage high (${service.memory}%)`, id: `service-${serviceName}-memory` });
                }
            });
        }
        
        this.updateAlerts(alerts);
    }

    updateAlerts(alerts) {
        const alertsContainer = this.container.querySelector('#alerts-container');
        
        // Remove existing alerts that are no longer relevant
        const existingAlerts = alertsContainer.querySelectorAll('.alert');
        const currentAlertIds = alerts.map(a => a.id);
        existingAlerts.forEach(alert => {
            if (!currentAlertIds.includes(alert.dataset.id)) {
                alert.remove();
            }
        });
        
        // Add new alerts
        alerts.forEach(alert => {
            if (!alertsContainer.querySelector(`[data-id="${alert.id}"]`)) {
                const alertElement = document.createElement('div');
                alertElement.className = `alert ${alert.type}`;
                alertElement.dataset.id = alert.id;
                alertElement.innerHTML = `
                    <span>${alert.message}</span>
                    <button class="alert-close" onclick="this.parentElement.remove()">×</button>
                `;
                alertsContainer.appendChild(alertElement);
            }
        });
        
        // Show "No alerts" message if no alerts
        if (alerts.length === 0 && alertsContainer.children.length === 0) {
            const noAlerts = document.createElement('div');
            noAlerts.className = 'text-gray-400 text-sm';
            noAlerts.textContent = 'No performance alerts';
            alertsContainer.appendChild(noAlerts);
        }
    }

    handleControlAction(action) {
        switch (action) {
            case 'toggle':
                this.isRunning ? this.pause() : this.resume();
                break;
            case 'reset':
                this.reset();
                break;
            case 'fullscreen':
                this.toggleFullscreen();
                break;
        }
    }

    start() {
        this.isRunning = true;
        this.updateControlButton('toggle', '⏸️');
    }

    pause() {
        this.isRunning = false;
        this.updateControlButton('toggle', '▶️');
    }

    resume() {
        this.isRunning = true;
        this.updateControlButton('toggle', '⏸️');
    }

    reset() {
        // Clear all chart data
        Object.values(this.charts).forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            chart.update();
        });
        
        // Clear stored data
        this.data.system = { cpu: [], memory: [], disk: [], network: { in: [], out: [] } };
        this.data.services = {};
        
        // Clear alerts
        const alertsContainer = this.container.querySelector('#alerts-container');
        alertsContainer.innerHTML = '';
        
        // Reset metric cards
        ['cpu', 'memory', 'disk', 'network'].forEach(type => {
            this.updateMetricCard(type, 0, type === 'disk' || type === 'network' ? 'MB/s' : '%');
        });
    }

    toggleFullscreen() {
        if (this.container.classList.contains('fullscreen')) {
            this.container.classList.remove('fullscreen');
            this.updateControlButton('fullscreen', '⛶');
        } else {
            this.container.classList.add('fullscreen');
            this.updateControlButton('fullscreen', '🗙');
            
            // Add fullscreen styles
            if (!document.querySelector('#fullscreen-styles')) {
                const style = document.createElement('style');
                style.id = 'fullscreen-styles';
                style.textContent = `
                    .performance-monitor.fullscreen {
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        z-index: 9999;
                        border-radius: 0;
                        background: rgba(15, 23, 42, 0.95);
                        overflow-y: auto;
                    }
                `;
                document.head.appendChild(style);
            }
        }
        
        // Resize charts after fullscreen toggle
        setTimeout(() => {
            Object.values(this.charts).forEach(chart => chart.resize());
        }, 300);
    }

    updateControlButton(action, icon) {
        const button = this.container.querySelector(`[data-action="${action}"]`);
        if (button) {
            button.textContent = icon;
        }
    }

    destroy() {
        this.isRunning = false;
        
        // Destroy charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        // Remove fullscreen styles
        const fullscreenStyles = document.querySelector('#fullscreen-styles');
        if (fullscreenStyles) {
            fullscreenStyles.remove();
        }
        
        // Clear container
        this.container.innerHTML = '';
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceMonitor;
} else if (typeof window !== 'undefined') {
    window.PerformanceMonitor = PerformanceMonitor;
}