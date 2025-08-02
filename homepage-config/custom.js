// Import cyberpunk functionality
function loadCyberpunkScript() {
    const script = document.createElement('script');
    script.src = './cyberpunk.js';
    script.onload = function() {
        console.log('=€ Cyberpunk dashboard loaded');
        
        // Add cyberpunk effects to existing elements
        setTimeout(() => {
            addCyberpunkEffect('.service-card', 'glow');
            addCyberpunkEffect('.widget', 'pulse');
            
            // Add data attributes for typewriter effect
            const headers = document.querySelectorAll('h1, h2');
            headers.forEach(header => {
                header.setAttribute('data-typewriter', '');
            });
            
            // Add service status monitoring
            monitorServices();
        }, 1000);
    };
    document.head.appendChild(script);
}

// Service monitoring function
function monitorServices() {
    const services = [
        { name: 'jellyfin', url: 'http://localhost:8096', element: '[data-service="jellyfin"]' },
        { name: 'sonarr', url: 'http://localhost:8989', element: '[data-service="sonarr"]' },
        { name: 'radarr', url: 'http://localhost:7878', element: '[data-service="radarr"]' },
        { name: 'prowlarr', url: 'http://localhost:9696', element: '[data-service="prowlarr"]' },
        { name: 'qbittorrent', url: 'http://localhost:8080', element: '[data-service="qbittorrent"]' },
        { name: 'overseerr', url: 'http://localhost:5055', element: '[data-service="overseerr"]' }
    ];

    services.forEach(service => {
        checkServiceStatus(service);
    });

    // Check every 30 seconds
    setInterval(() => {
        services.forEach(service => {
            checkServiceStatus(service);
        });
    }, 30000);
}

async function checkServiceStatus(service) {
    try {
        const response = await fetch(service.url, { 
            method: 'HEAD', 
            mode: 'no-cors',
            timeout: 5000 
        });
        updateServiceStatus(service.name, true);
    } catch (error) {
        updateServiceStatus(service.name, false);
    }
}

// Enhanced service card creation
function createServiceCard(serviceName, url, description, status = 'unknown') {
    return `
        <div class="service" data-service="${serviceName}">
            <div class="status-indicator status-${status}"></div>
            <h3>${serviceName.toUpperCase()}</h3>
            <p>${description}</p>
            <a href="${url}" class="btn service-link" target="_blank">
                LAUNCH ${serviceName.toUpperCase()}
            </a>
        </div>
    `;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('<® Initializing cyberpunk media dashboard...');
    loadCyberpunkScript();
    
    // Add reveal animations to elements
    const cards = document.querySelectorAll('.service, .widget, .card');
    cards.forEach((card, index) => {
        card.setAttribute('data-reveal', '');
        card.style.animationDelay = (index * 100) + 'ms';
    });
});

// Keyboard shortcuts for cyberpunk feel
document.addEventListener('keydown', function(e) {
    // Ctrl+Shift+C for console toggle
    if (e.ctrlKey && e.shiftKey && e.code === 'KeyC') {
        e.preventDefault();
        toggleCyberConsole();
    }
    
    // Ctrl+Shift+M for matrix mode
    if (e.ctrlKey && e.shiftKey && e.code === 'KeyM') {
        e.preventDefault();
        toggleMatrixMode();
    }
});

function toggleCyberConsole() {
    const console = document.getElementById('cyber-console');
    if (console) {
        console.style.display = console.style.display === 'none' ? 'block' : 'none';
    } else {
        createCyberConsole();
    }
}

function createCyberConsole() {
    const console = document.createElement('div');
    console.id = 'cyber-console';
    console.style.cssText = `
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 200px;
        background: rgba(10, 14, 39, 0.95);
        border-top: 2px solid #00ff9f;
        color: #00ff9f;
        font-family: 'Share Tech Mono', monospace;
        padding: 10px;
        z-index: 10000;
        overflow-y: auto;
    `;
    
    console.innerHTML = `
        <div style="margin-bottom: 10px;">CYBER TERMINAL v2.0.77 - READY</div>
        <div>> System status: OPERATIONAL</div>
        <div>> Media stack: INTEGRATED</div>
        <div>> Security level: MAXIMUM</div>
        <div style="color: #ff0080;">> Welcome to the Grid, User.</div>
    `;
    
    document.body.appendChild(console);
}

function toggleMatrixMode() {
    document.body.classList.toggle('matrix-mode');
    const matrixRain = document.getElementById('matrix-rain');
    if (matrixRain) {
        matrixRain.style.opacity = document.body.classList.contains('matrix-mode') ? '0.3' : '0.1';
    }
}