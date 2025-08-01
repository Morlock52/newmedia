# Accessibility Fixes - Code Examples

## 1. Skip Navigation Link

### HTML Addition (index.html after body tag):
```html
<body>
    <!-- Add skip navigation link -->
    <a href="#main-content" class="skip-link">Skip to main content</a>
    
    <!-- Existing content -->
    <div id="webgl-container"></div>
    
    <div id="hud-interface">
        <!-- ... existing header and nav ... -->
        
        <!-- Add main landmark and id -->
        <main id="main-content" class="hud-content" data-router-view role="main">
            <!-- existing content -->
        </main>
    </div>
```

### CSS Addition (main.css):
```css
/* Skip Link Styles */
.skip-link {
    position: absolute;
    top: -40px;
    left: 10px;
    background: var(--holo-cyan);
    color: var(--bg-primary);
    padding: 12px 24px;
    text-decoration: none;
    font-weight: 600;
    border-radius: 4px;
    z-index: 10000;
    transition: top 0.3s ease;
}

.skip-link:focus {
    top: 10px;
    outline: 3px solid var(--holo-magenta);
    outline-offset: 2px;
}
```

## 2. Improved Color Contrast

### CSS Updates (main.css):
```css
:root {
    /* Updated color palette for WCAG AA compliance */
    --text-primary: #FFFFFF;        /* 21:1 on dark bg ✓ */
    --text-secondary: #D0D0D0;      /* 10.5:1 on dark bg ✓ */
    --text-tertiary: #A0A0A0;       /* 5.8:1 on dark bg ✓ */
    
    /* Adjusted accent colors for better contrast */
    --holo-cyan-accessible: #00E5E5;     /* Brighter cyan */
    --holo-magenta-accessible: #FF33FF;  /* Brighter magenta */
    --holo-yellow-accessible: #FFD700;   /* Gold instead of pure yellow */
    
    /* Keep original for effects, use accessible for text */
    --holo-cyan: #00FFFF;
    --holo-magenta: #FF00FF;
    --holo-yellow: #FFFF00;
}

/* Update text colors throughout */
.status-item {
    color: var(--text-secondary); /* Changed from --text-tertiary */
}

.activity-time {
    color: var(--text-tertiary);
    font-weight: 500; /* Increase weight for better readability */
}
```

## 3. Enhanced Focus Indicators

### CSS Updates (main.css):
```css
/* Universal focus indicator */
:focus-visible {
    outline: 3px solid var(--holo-cyan-accessible);
    outline-offset: 2px;
    border-radius: 4px;
    position: relative;
    z-index: 1;
}

/* Remove default focus outline */
:focus:not(:focus-visible) {
    outline: none;
}

/* Enhanced button focus */
button:focus-visible,
.nav-btn:focus-visible,
.control-btn:focus-visible {
    outline: 3px solid var(--holo-cyan-accessible);
    outline-offset: 3px;
    box-shadow: 0 0 0 6px rgba(0, 229, 229, 0.25);
}

/* Link focus styles */
a:focus-visible {
    outline: 2px solid var(--holo-cyan-accessible);
    outline-offset: 4px;
    text-decoration: underline;
    text-decoration-thickness: 2px;
}
```

## 4. ARIA Labels and Roles

### HTML Updates (index.html):
```html
<!-- Navigation with ARIA -->
<nav class="holo-nav" role="navigation" aria-label="Main navigation">
    <div class="nav-wrapper">
        <button class="nav-btn active" 
                data-section="dashboard" 
                data-route="/dashboard"
                aria-label="Dashboard section"
                aria-current="page">
            <span class="nav-icon" aria-hidden="true">⊞</span>
            <span>Dashboard</span>
        </button>
        <button class="nav-btn" 
                data-section="movies" 
                data-route="/movies"
                aria-label="Movies section">
            <span class="nav-icon" aria-hidden="true">🎬</span>
            <span>Movies</span>
        </button>
        <!-- Continue pattern for all nav buttons -->
    </div>
</nav>

<!-- Control Panel with ARIA -->
<div class="control-panel" role="toolbar" aria-label="Display controls">
    <button class="control-btn" 
            id="toggle-effects" 
            title="Toggle Effects"
            aria-label="Toggle visual effects"
            aria-pressed="false">
        <span aria-hidden="true">🎨</span>
    </button>
    <button class="control-btn" 
            id="toggle-particles" 
            title="Toggle Particles"
            aria-label="Toggle particle effects"
            aria-pressed="false">
        <span aria-hidden="true">✨</span>
    </button>
    <!-- Continue pattern for all control buttons -->
</div>

<!-- Stats Panel with Live Regions -->
<div class="stats-panel glass-panel" role="region" aria-label="System statistics">
    <div class="stat-item">
        <div class="stat-value" 
             id="total-media" 
             aria-live="polite"
             aria-atomic="true">0</div>
        <div class="stat-label">Total Media</div>
    </div>
    <!-- Continue pattern for all stats -->
</div>
```

## 5. Keyboard Navigation Enhancement

### JavaScript Addition (keyboard-navigation.js):
```javascript
class KeyboardNavigation {
    constructor() {
        this.shortcuts = new Map([
            ['1', () => this.navigateToSection('dashboard')],
            ['2', () => this.navigateToSection('movies')],
            ['3', () => this.navigateToSection('series')],
            ['4', () => this.navigateToSection('music')],
            ['5', () => this.navigateToSection('live')],
            ['6', () => this.navigateToSection('analytics')],
            ['?', () => this.showKeyboardHelp()],
            ['Escape', () => this.closeModals()],
            ['/', () => this.focusSearch()],
        ]);
        
        this.init();
    }
    
    init() {
        document.addEventListener('keydown', this.handleKeyPress.bind(this));
        this.setupFocusTrap();
        this.addRovingTabIndex();
    }
    
    handleKeyPress(event) {
        // Skip if user is typing in an input
        if (event.target.matches('input, textarea')) return;
        
        // Check for Ctrl/Cmd + K
        if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
            event.preventDefault();
            this.showQuickNav();
            return;
        }
        
        // Check shortcuts
        const handler = this.shortcuts.get(event.key);
        if (handler) {
            event.preventDefault();
            handler();
        }
    }
    
    navigateToSection(section) {
        const button = document.querySelector(`[data-section="${section}"]`);
        if (button) {
            button.click();
            button.focus();
            this.announceNavigation(`Navigated to ${section}`);
        }
    }
    
    setupFocusTrap() {
        const modal = document.querySelector('.preview-panel');
        if (!modal) return;
        
        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        
        modal.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                if (e.shiftKey && document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                } else if (!e.shiftKey && document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        });
    }
    
    addRovingTabIndex() {
        const groups = document.querySelectorAll('[role="toolbar"], .nav-wrapper');
        
        groups.forEach(group => {
            const buttons = group.querySelectorAll('button');
            let currentIndex = 0;
            
            buttons.forEach((button, index) => {
                button.setAttribute('tabindex', index === 0 ? '0' : '-1');
                
                button.addEventListener('keydown', (e) => {
                    let newIndex = currentIndex;
                    
                    switch(e.key) {
                        case 'ArrowRight':
                        case 'ArrowDown':
                            e.preventDefault();
                            newIndex = (currentIndex + 1) % buttons.length;
                            break;
                        case 'ArrowLeft':
                        case 'ArrowUp':
                            e.preventDefault();
                            newIndex = (currentIndex - 1 + buttons.length) % buttons.length;
                            break;
                        case 'Home':
                            e.preventDefault();
                            newIndex = 0;
                            break;
                        case 'End':
                            e.preventDefault();
                            newIndex = buttons.length - 1;
                            break;
                        default:
                            return;
                    }
                    
                    buttons[currentIndex].setAttribute('tabindex', '-1');
                    buttons[newIndex].setAttribute('tabindex', '0');
                    buttons[newIndex].focus();
                    currentIndex = newIndex;
                });
            });
        });
    }
    
    announceNavigation(message) {
        const announcer = document.getElementById('announcer') || this.createAnnouncer();
        announcer.textContent = message;
        
        // Clear after announcement
        setTimeout(() => {
            announcer.textContent = '';
        }, 1000);
    }
    
    createAnnouncer() {
        const announcer = document.createElement('div');
        announcer.id = 'announcer';
        announcer.className = 'sr-only';
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        document.body.appendChild(announcer);
        return announcer;
    }
    
    showKeyboardHelp() {
        const helpContent = `
            <div class="keyboard-help-modal" role="dialog" aria-labelledby="help-title">
                <h2 id="help-title">Keyboard Shortcuts</h2>
                <dl>
                    <dt>1-6</dt>
                    <dd>Navigate to sections</dd>
                    <dt>Ctrl/Cmd + K</dt>
                    <dd>Quick navigation</dd>
                    <dt>/</dt>
                    <dd>Focus search</dd>
                    <dt>Escape</dt>
                    <dd>Close dialogs</dd>
                    <dt>?</dt>
                    <dd>Show this help</dd>
                    <dt>Arrow keys</dt>
                    <dd>Navigate within toolbars</dd>
                </dl>
                <button onclick="this.closest('.keyboard-help-modal').remove()">Close</button>
            </div>
        `;
        
        // Add to DOM and focus
        const modal = document.createElement('div');
        modal.innerHTML = helpContent;
        document.body.appendChild(modal.firstElementChild);
        modal.querySelector('button').focus();
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    new KeyboardNavigation();
});
```

## 6. Screen Reader Announcements

### JavaScript Addition (announcements.js):
```javascript
class ScreenReaderAnnouncements {
    constructor() {
        this.announcer = this.createAnnouncer();
        this.setupMutationObserver();
    }
    
    createAnnouncer() {
        const container = document.createElement('div');
        container.className = 'sr-announcements';
        container.innerHTML = `
            <div aria-live="polite" aria-atomic="true" class="sr-only" id="polite-announcer"></div>
            <div aria-live="assertive" aria-atomic="true" class="sr-only" id="assertive-announcer"></div>
        `;
        document.body.appendChild(container);
        return container;
    }
    
    announce(message, priority = 'polite') {
        const announcer = document.getElementById(`${priority}-announcer`);
        if (announcer) {
            announcer.textContent = message;
            setTimeout(() => {
                announcer.textContent = '';
            }, 1000);
        }
    }
    
    setupMutationObserver() {
        // Announce dynamic content changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach((node) => {
                        if (node.classList?.contains('activity-item')) {
                            this.announce('New activity added');
                        }
                        if (node.classList?.contains('notification')) {
                            this.announce(node.textContent, 'assertive');
                        }
                    });
                }
            });
        });
        
        // Observe activity feed and notification areas
        const activityFeed = document.getElementById('activity-feed');
        if (activityFeed) {
            observer.observe(activityFeed, { childList: true });
        }
    }
}
```

## 7. Touch Target Improvements

### CSS Updates (ui-components.css):
```css
/* Ensure minimum touch target size */
button,
.control-btn,
.nav-btn,
.action-btn,
a {
    min-width: 44px;
    min-height: 44px;
    touch-action: manipulation; /* Prevent double-tap zoom */
}

/* Adjust control panel buttons */
.control-btn {
    width: 56px;  /* Increased from 50px */
    height: 56px; /* Increased from 50px */
    margin: 4px;  /* Add spacing between targets */
}

/* Mobile-specific adjustments */
@media (pointer: coarse) {
    .nav-btn {
        padding: 12px 20px; /* Increase padding */
        margin: 4px;        /* Add spacing */
    }
    
    .stat-item {
        padding: 16px;      /* Make stats tappable */
        cursor: pointer;
    }
    
    /* Increase clickable area without changing visual size */
    .control-btn::before {
        content: '';
        position: absolute;
        top: -8px;
        right: -8px;
        bottom: -8px;
        left: -8px;
    }
}
```

## 8. Animation Controls

### JavaScript Addition (animation-controls.js):
```javascript
class AnimationControls {
    constructor() {
        this.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        this.animationsEnabled = !this.prefersReducedMotion.matches;
        this.init();
    }
    
    init() {
        // Listen for preference changes
        this.prefersReducedMotion.addEventListener('change', (e) => {
            this.animationsEnabled = !e.matches;
            this.updateAnimations();
        });
        
        // Add animation toggle button
        this.addToggleButton();
        
        // Apply initial state
        this.updateAnimations();
    }
    
    addToggleButton() {
        const controlPanel = document.querySelector('.control-panel');
        if (!controlPanel) return;
        
        const button = document.createElement('button');
        button.className = 'control-btn';
        button.id = 'toggle-animations';
        button.innerHTML = '<span aria-hidden="true">🎞️</span>';
        button.setAttribute('aria-label', 'Toggle animations');
        button.setAttribute('aria-pressed', this.animationsEnabled);
        button.title = 'Toggle Animations (M)';
        
        button.addEventListener('click', () => {
            this.animationsEnabled = !this.animationsEnabled;
            button.setAttribute('aria-pressed', this.animationsEnabled);
            this.updateAnimations();
            this.announce(`Animations ${this.animationsEnabled ? 'enabled' : 'disabled'}`);
        });
        
        controlPanel.appendChild(button);
    }
    
    updateAnimations() {
        document.documentElement.classList.toggle('reduce-motion', !this.animationsEnabled);
        
        // Pause/play CSS animations
        const animated = document.querySelectorAll('[data-animated]');
        animated.forEach(element => {
            element.style.animationPlayState = this.animationsEnabled ? 'running' : 'paused';
        });
        
        // Notify 3D scene
        if (window.dashboard?.scene) {
            window.dashboard.scene.setAnimationsEnabled(this.animationsEnabled);
        }
    }
    
    announce(message) {
        const announcer = document.getElementById('announcer');
        if (announcer) {
            announcer.textContent = message;
        }
    }
}

// CSS for reduced motion
const reducedMotionCSS = `
    .reduce-motion * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    
    .reduce-motion .glitch::before,
    .reduce-motion .glitch::after {
        display: none;
    }
    
    .reduce-motion .loading-screen {
        transition: none;
    }
`;

// Add styles
const style = document.createElement('style');
style.textContent = reducedMotionCSS;
document.head.appendChild(style);
```

## Implementation Notes

1. **Testing Required**: All fixes should be tested with:
   - Screen readers (NVDA, JAWS, VoiceOver)
   - Keyboard-only navigation
   - Mobile devices
   - Browser zoom at 200%

2. **Progressive Enhancement**: Ensure all features work without JavaScript

3. **Performance**: Monitor impact of accessibility features on rendering performance

4. **Documentation**: Update user documentation with keyboard shortcuts and accessibility features