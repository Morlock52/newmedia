// Charts and Data Visualization Module

class ChartsManager {
    constructor() {
        this.charts = new Map();
        this.chartColors = {
            primary: '#3A86FF',
            secondary: '#FF006E',
            success: '#06FFA5',
            warning: '#FFBE0B',
            danger: '#FF5722',
            purple: '#8338EC'
        };
        this.init();
    }

    init() {
        this.initActivityChart();
        this.setupRealTimeUpdates();
    }

    initActivityChart() {
        const canvas = document.getElementById('activityChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Create gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, 'rgba(255, 0, 110, 0.4)');
        gradient.addColorStop(1, 'rgba(255, 0, 110, 0)');

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.getLast7Days(),
                datasets: [{
                    label: 'Active Streams',
                    data: this.generateStreamData(),
                    fill: true,
                    backgroundColor: gradient,
                    borderColor: this.chartColors.secondary,
                    borderWidth: 2,
                    tension: 0.4,
                    pointBackgroundColor: this.chartColors.secondary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(33, 38, 45, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            font: {
                                size: 12
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            font: {
                                size: 12
                            }
                        },
                        beginAtZero: true
                    }
                },
                elements: {
                    point: {
                        hoverBackgroundColor: this.chartColors.secondary,
                        hoverBorderColor: '#fff'
                    }
                }
            }
        });

        this.charts.set('activity', chart);
    }

    initSystemChart() {
        const canvas = document.getElementById('systemChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.getLast24Hours(),
                datasets: [
                    {
                        label: 'CPU Usage (%)',
                        data: this.generateSystemData('cpu'),
                        borderColor: this.chartColors.primary,
                        backgroundColor: 'rgba(58, 134, 255, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Memory Usage (%)',
                        data: this.generateSystemData('memory'),
                        borderColor: this.chartColors.purple,
                        backgroundColor: 'rgba(131, 56, 236, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(33, 38, 45, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)'
                        },
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        this.charts.set('system', chart);
    }

    initNetworkChart() {
        const canvas = document.getElementById('networkChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.getLast24Hours(),
                datasets: [
                    {
                        label: 'Download (MB/s)',
                        data: this.generateNetworkData('download'),
                        backgroundColor: 'rgba(6, 255, 165, 0.8)',
                        borderColor: this.chartColors.success,
                        borderWidth: 1
                    },
                    {
                        label: 'Upload (MB/s)',
                        data: this.generateNetworkData('upload'),
                        backgroundColor: 'rgba(255, 190, 11, 0.8)',
                        borderColor: this.chartColors.warning,
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(33, 38, 45, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });

        this.charts.set('network', chart);
    }

    initStorageChart() {
        const canvas = document.getElementById('storageChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Movies', 'TV Shows', 'Music', 'Other', 'Free Space'],
                datasets: [{
                    data: [35, 25, 15, 10, 15],
                    backgroundColor: [
                        this.chartColors.primary,
                        this.chartColors.secondary,
                        this.chartColors.success,
                        this.chartColors.warning,
                        'rgba(255, 255, 255, 0.1)'
                    ],
                    borderColor: [
                        this.chartColors.primary,
                        this.chartColors.secondary,
                        this.chartColors.success,
                        this.chartColors.warning,
                        'rgba(255, 255, 255, 0.2)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(33, 38, 45, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                },
                cutout: '70%'
            }
        });

        this.charts.set('storage', chart);
    }

    initServiceStatusChart() {
        const canvas = document.getElementById('serviceStatusChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const stats = window.servicesManager?.getServiceStats() || { online: 8, offline: 1, warning: 1 };
        
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Online', 'Offline', 'Warning'],
                datasets: [{
                    data: [stats.online, stats.offline, stats.warning],
                    backgroundColor: [
                        this.chartColors.success,
                        this.chartColors.danger,
                        this.chartColors.warning
                    ],
                    borderColor: [
                        this.chartColors.success,
                        this.chartColors.danger,
                        this.chartColors.warning
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(33, 38, 45, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8
                    }
                },
                cutout: '60%'
            }
        });

        this.charts.set('serviceStatus', chart);
    }

    getLast7Days() {
        const days = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            days.push(date.toLocaleDateString('en-US', { weekday: 'short' }));
        }
        return days;
    }

    getLast24Hours() {
        const hours = [];
        for (let i = 23; i >= 0; i--) {
            const date = new Date();
            date.setHours(date.getHours() - i);
            hours.push(date.toLocaleTimeString('en-US', { hour: '2-digit', hour12: false }));
        }
        return hours.filter((_, index) => index % 2 === 0); // Show every 2nd hour
    }

    generateStreamData() {
        return Array.from({ length: 7 }, () => Math.floor(Math.random() * 30) + 5);
    }

    generateSystemData(type) {
        const baseValues = {
            cpu: 25,
            memory: 60
        };
        
        return Array.from({ length: 12 }, () => {
            const base = baseValues[type] || 50;
            return base + (Math.random() - 0.5) * 20;
        });
    }

    generateNetworkData(type) {
        const baseValues = {
            download: 50,
            upload: 20
        };
        
        return Array.from({ length: 12 }, () => {
            const base = baseValues[type] || 30;
            return Math.max(0, base + (Math.random() - 0.5) * 40);
        });
    }

    updateChart(chartId, newData) {
        const chart = this.charts.get(chartId);
        if (!chart) return;

        if (newData.labels) {
            chart.data.labels = newData.labels;
        }
        
        if (newData.datasets) {
            newData.datasets.forEach((dataset, index) => {
                if (chart.data.datasets[index]) {
                    chart.data.datasets[index].data = dataset.data;
                    if (dataset.label) {
                        chart.data.datasets[index].label = dataset.label;
                    }
                }
            });
        }

        chart.update('none');
    }

    setupRealTimeUpdates() {
        // Update activity chart every 5 minutes
        setInterval(() => {
            this.updateActivityChart();
        }, 5 * 60 * 1000);

        // Update system charts every 30 seconds
        setInterval(() => {
            this.updateSystemCharts();
        }, 30 * 1000);
    }

    async updateActivityChart() {
        try {
            const response = await fetch('/api/stats/activity');
            if (response.ok) {
                const data = await response.json();
                this.updateChart('activity', {
                    datasets: [{
                        data: data.streams || this.generateStreamData()
                    }]
                });
            }
        } catch (error) {
            console.error('Error updating activity chart:', error);
        }
    }

    async updateSystemCharts() {
        try {
            const response = await fetch('/api/system/metrics');
            if (response.ok) {
                const data = await response.json();
                
                // Update system chart
                if (this.charts.has('system')) {
                    this.updateChart('system', {
                        datasets: [
                            { data: data.cpu || this.generateSystemData('cpu') },
                            { data: data.memory || this.generateSystemData('memory') }
                        ]
                    });
                }
                
                // Update network chart
                if (this.charts.has('network')) {
                    this.updateChart('network', {
                        datasets: [
                            { data: data.networkDown || this.generateNetworkData('download') },
                            { data: data.networkUp || this.generateNetworkData('upload') }
                        ]
                    });
                }
            }
        } catch (error) {
            console.error('Error updating system charts:', error);
        }
    }

    createCustomChart(containerId, config) {
        const canvas = document.getElementById(containerId);
        if (!canvas) return null;

        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            ...config,
            options: {
                ...config.options,
                plugins: {
                    ...config.options?.plugins,
                    tooltip: {
                        backgroundColor: 'rgba(33, 38, 45, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        ...config.options?.plugins?.tooltip
                    }
                },
                scales: config.options?.scales ? {
                    ...Object.keys(config.options.scales).reduce((acc, key) => {
                        acc[key] = {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)',
                                drawBorder: false
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.6)'
                            },
                            ...config.options.scales[key]
                        };
                        return acc;
                    }, {})
                } : undefined
            }
        });

        return chart;
    }

    destroyChart(chartId) {
        const chart = this.charts.get(chartId);
        if (chart) {
            chart.destroy();
            this.charts.delete(chartId);
        }
    }

    destroyAllCharts() {
        this.charts.forEach((chart, id) => {
            chart.destroy();
        });
        this.charts.clear();
    }

    // Method to initialize charts based on current page
    initPageCharts(pageId) {
        switch (pageId) {
            case 'monitoring':
                setTimeout(() => {
                    this.initSystemChart();
                    this.initNetworkChart();
                }, 100);
                break;
            case 'services':
                setTimeout(() => {
                    this.initServiceStatusChart();
                }, 100);
                break;
            case 'backup':
                setTimeout(() => {
                    this.initStorageChart();
                }, 100);
                break;
        }
    }
}

// Initialize charts manager
const chartsManager = new ChartsManager();

// Export for global access
window.chartsManager = chartsManager;