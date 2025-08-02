// Mobile Navigation Enhancement
// Provides touch-friendly navigation with gestures and responsive design

class MobileNavigation {
    constructor(options = {}) {
        this.options = {
            swipeThreshold: 50,
            tapThreshold: 200,
            enableGestures: true,
            enableMobileMenu: true,
            breakpoint: 768,
            ...options
        };
        
        this.touchStartX = 0;
        this.touchStartY = 0;
        this.touchStartTime = 0;
        this.isMobile = window.innerWidth <= this.options.breakpoint;
        this.menuOpen = false;
        
        this.init();
    }
    
    init() {
        this.detectMobile();
        this.setupTouchHandlers();
        this.setupMobileMenu();
        this.setupResizeHandler();
        this.enhanceButtonsForTouch();
        this.setupSwipeNavigation();
    }
    
    detectMobile() {
        this.isMobile = window.innerWidth <= this.options.breakpoint || 
                      /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        // Update body class for CSS targeting
        document.body.classList.toggle('mobile-device', this.isMobile);
        
        // Adjust configuration for mobile
        if (this.isMobile && window.CONFIG) {
            // Reduce particle count for mobile
            window.CONFIG.particles.count = Math.min(window.CONFIG.particles.count, 500);
            window.CONFIG.mediaCards.rows = 2;
            window.CONFIG.mediaCards.columns = 2;
            window.CONFIG.setQuality('low');
        }
    }
    
    setupTouchHandlers() {
        if (!this.options.enableGestures) return;
        
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
        document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true });
    }
    
    handleTouchStart(e) {
        const touch = e.touches[0];
        this.touchStartX = touch.clientX;
        this.touchStartY = touch.clientY;
        this.touchStartTime = Date.now();
    }
    
    handleTouchMove(e) {
        // Prevent pull-to-refresh on iOS
        if (e.target === document.body) {
            e.preventDefault();
        }
    }
    
    handleTouchEnd(e) {
        const touch = e.changedTouches[0];
        const deltaX = touch.clientX - this.touchStartX;
        const deltaY = touch.clientY - this.touchStartY;
        const deltaTime = Date.now() - this.touchStartTime;
        
        // Check for swipe gestures
        if (Math.abs(deltaX) > this.options.swipeThreshold && 
            Math.abs(deltaY) < this.options.swipeThreshold * 2 &&
            deltaTime < 500) {
            
            if (deltaX > 0) {
                // Swipe right - go back
                this.handleSwipeRight();
            } else {
                // Swipe left - go forward or next section
                this.handleSwipeLeft();
            }
        }
        
        // Check for edge swipe to open menu
        if (this.touchStartX < 20 && deltaX > this.options.swipeThreshold) {
            this.openMobileMenu();
        }
    }
    
    handleSwipeRight() {
        // Navigate back
        if (window.navigationManager) {
            window.navigationManager.navigateBack();
        } else if (window.dashboard && window.dashboard.uiController) {
            window.dashboard.uiController.navigateBack();
        }
    }
    
    handleSwipeLeft() {
        // Navigate to next section or forward
        if (window.navigationManager) {
            window.navigationManager.navigateForward();
        } else {
            this.navigateToNextSection();
        }
    }
    
    navigateToNextSection() {
        const sections = ['dashboard', 'movies', 'series', 'music', 'live', 'analytics'];
        const currentSection = window.dashboard?.uiController?.currentSection || 'dashboard';
        const currentIndex = sections.indexOf(currentSection);
        const nextIndex = (currentIndex + 1) % sections.length;
        
        if (window.dashboard && window.dashboard.uiController) {
            window.dashboard.uiController.navigateToSection(sections[nextIndex]);
            this.showNavigationHint(`Swiped to ${sections[nextIndex]}`);
        }
    }
    
    setupMobileMenu() {
        if (!this.options.enableMobileMenu || !this.isMobile) return;
        
        // Create mobile menu button
        const menuButton = document.createElement('button');
        menuButton.className = 'mobile-menu-toggle';
        menuButton.innerHTML = `
            <span class="hamburger-line"></span>
            <span class="hamburger-line"></span>
            <span class="hamburger-line"></span>
        `;
        menuButton.style.cssText = `
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 1001;
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 3px;
        `;
        
        // Style hamburger lines
        menuButton.querySelectorAll('.hamburger-line').forEach(line => {
            line.style.cssText = `
                width: 20px;
                height: 2px;
                background: var(--neon-cyan, #00ffff);
                transition: all 0.3s ease;
            `;
        });
        
        document.body.appendChild(menuButton);
        menuButton.addEventListener('click', () => this.toggleMobileMenu());
        
        // Create mobile menu overlay
        this.createMobileMenuOverlay();
    }
    
    createMobileMenuOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'mobile-menu-overlay';
        overlay.innerHTML = `
            <div class="mobile-menu-content">
                <div class="mobile-menu-header">
                    <h3>Navigation</h3>
                    <button class="mobile-menu-close">&times;</button>
                </div>
                <nav class="mobile-menu-nav">
                    <button class="mobile-nav-item" data-section="dashboard">
                        <span class="nav-icon">⊞</span>
                        Dashboard
                    </button>
                    <button class="mobile-nav-item" data-section="movies">
                        <span class="nav-icon">🎬</span>
                        Movies
                    </button>
                    <button class="mobile-nav-item" data-section="series">
                        <span class="nav-icon">📺</span>
                        Series
                    </button>
                    <button class="mobile-nav-item" data-section="music">
                        <span class="nav-icon">🎵</span>
                        Music
                    </button>
                    <button class="mobile-nav-item" data-section="live">
                        <span class="nav-icon">📡</span>
                        Live TV
                    </button>
                    <button class="mobile-nav-item" data-section="analytics">
                        <span class="nav-icon">📊</span>
                        Analytics
                    </button>
                </nav>
                <div class="mobile-menu-footer">
                    <div class="gesture-hints">
                        <p>💡 Swipe left/right to navigate</p>
                        <p>💡 Swipe from edge to open menu</p>
                    </div>
                </div>
            </div>
        `;
        
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(10, 10, 15, 0.95);
            backdrop-filter: blur(10px);
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        `;
        
        document.body.appendChild(overlay);
        
        // Setup menu interactions
        this.setupMobileMenuInteractions(overlay);
    }
    
    setupMobileMenuInteractions(overlay) {
        // Close button
        const closeBtn = overlay.querySelector('.mobile-menu-close');
        closeBtn.addEventListener('click', () => this.closeMobileMenu());
        
        // Navigation items
        overlay.querySelectorAll('.mobile-nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                this.navigateToSection(section);
                this.closeMobileMenu();
            });
        });
        
        // Close on overlay click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.closeMobileMenu();
            }
        });
        
        // Swipe to close
        let startY = 0;
        overlay.addEventListener('touchstart', (e) => {
            startY = e.touches[0].clientY;
        }, { passive: true });
        
        overlay.addEventListener('touchend', (e) => {
            const endY = e.changedTouches[0].clientY;
            if (startY - endY > 50) { // Swipe up to close
                this.closeMobileMenu();
            }
        }, { passive: true });
    }
    
    toggleMobileMenu() {
        if (this.menuOpen) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }
    
    openMobileMenu() {
        const overlay = document.querySelector('.mobile-menu-overlay');
        const menuButton = document.querySelector('.mobile-menu-toggle');
        
        if (overlay) {
            overlay.style.opacity = '1';
            overlay.style.visibility = 'visible';
            this.menuOpen = true;
            
            // Animate hamburger to X
            if (menuButton) {
                const lines = menuButton.querySelectorAll('.hamburger-line');
                lines[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
                lines[1].style.opacity = '0';
                lines[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
            }
            
            // Update current section
            this.updateMobileMenuActive();
        }
    }
    
    closeMobileMenu() {
        const overlay = document.querySelector('.mobile-menu-overlay');
        const menuButton = document.querySelector('.mobile-menu-toggle');
        
        if (overlay) {
            overlay.style.opacity = '0';
            overlay.style.visibility = 'hidden';
            this.menuOpen = false;
            
            // Reset hamburger
            if (menuButton) {
                const lines = menuButton.querySelectorAll('.hamburger-line');
                lines.forEach(line => {
                    line.style.transform = 'none';
                    line.style.opacity = '1';
                });
            }
        }
    }
    
    updateMobileMenuActive() {
        const currentSection = window.dashboard?.uiController?.currentSection || 'dashboard';
        const menuItems = document.querySelectorAll('.mobile-nav-item');
        
        menuItems.forEach(item => {
            const isActive = item.dataset.section === currentSection;
            item.classList.toggle('active', isActive);
        });
    }
    
    navigateToSection(section) {
        if (window.dashboard && window.dashboard.uiController) {
            window.dashboard.uiController.navigateToSection(section);
        }
    }
    
    setupResizeHandler() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                const wasMobile = this.isMobile;
                this.detectMobile();
                
                // If device type changed, reinitialize
                if (wasMobile !== this.isMobile) {
                    this.reinitialize();
                }
            }, 250);
        });
    }
    
    reinitialize() {
        // Remove existing mobile elements if switching to desktop
        if (!this.isMobile) {
            const menuButton = document.querySelector('.mobile-menu-toggle');
            const menuOverlay = document.querySelector('.mobile-menu-overlay');
            
            if (menuButton) menuButton.remove();
            if (menuOverlay) menuOverlay.remove();
        } else {
            // Reinitialize mobile features
            this.setupMobileMenu();
        }
    }
    
    enhanceButtonsForTouch() {
        if (!this.isMobile) return;
        
        // Enhance all navigation buttons for touch
        const buttons = document.querySelectorAll('.nav-btn, .control-btn, .action-btn');
        buttons.forEach(button => {
            // Increase touch target size
            button.style.minHeight = '44px';
            button.style.minWidth = '44px';
            
            // Add touch feedback
            button.addEventListener('touchstart', () => {
                button.style.transform = 'scale(0.95)';
            }, { passive: true });
            
            button.addEventListener('touchend', () => {
                button.style.transform = 'scale(1)';
            }, { passive: true });
        });
    }
    
    setupSwipeNavigation() {
        if (!this.isMobile || !this.options.enableGestures) return;
        
        // Add swipe indicators
        this.createSwipeIndicators();
        
        // Show gesture hints on first visit
        if (!localStorage.getItem('mobile-gestures-shown')) {
            setTimeout(() => {
                this.showGestureHints();
                localStorage.setItem('mobile-gestures-shown', 'true');
            }, 2000);
        }
    }
    
    createSwipeIndicators() {
        const leftIndicator = document.createElement('div');
        leftIndicator.className = 'swipe-indicator left';
        leftIndicator.innerHTML = '◀';
        leftIndicator.style.cssText = `
            position: fixed;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: rgba(0, 255, 255, 0.3);
            font-size: 24px;
            z-index: 100;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        const rightIndicator = document.createElement('div');
        rightIndicator.className = 'swipe-indicator right';
        rightIndicator.innerHTML = '▶';
        rightIndicator.style.cssText = `
            position: fixed;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: rgba(0, 255, 255, 0.3);
            font-size: 24px;
            z-index: 100;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(leftIndicator);
        document.body.appendChild(rightIndicator);
        
        // Show indicators during swipe
        let showTimeout;
        document.addEventListener('touchmove', () => {
            clearTimeout(showTimeout);
            leftIndicator.style.opacity = '1';
            rightIndicator.style.opacity = '1';
            
            showTimeout = setTimeout(() => {
                leftIndicator.style.opacity = '0';
                rightIndicator.style.opacity = '0';
            }, 1000);
        });
    }
    
    showGestureHints() {
        const hints = document.createElement('div');
        hints.className = 'gesture-hints-overlay';
        hints.innerHTML = `
            <div class="gesture-hints-content">
                <h3>Touch Navigation</h3>
                <div class="hint-item">
                    <span class="hint-icon">👆</span>
                    <span>Swipe from left edge to open menu</span>
                </div>
                <div class="hint-item">
                    <span class="hint-icon">👈</span>
                    <span>Swipe right to go back</span>
                </div>
                <div class="hint-item">
                    <span class="hint-icon">👉</span>
                    <span>Swipe left to go forward</span>
                </div>
                <button class="hints-close-btn">Got it!</button>
            </div>
        `;
        
        hints.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(10, 10, 15, 0.9);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        document.body.appendChild(hints);
        
        hints.querySelector('.hints-close-btn').addEventListener('click', () => {
            hints.remove();
        });
        
        // Auto-close after 10 seconds
        setTimeout(() => {
            if (hints.parentNode) {
                hints.remove();
            }
        }, 10000);
    }
    
    showNavigationHint(message) {
        const hint = document.createElement('div');
        hint.className = 'navigation-hint';
        hint.textContent = message;
        hint.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 20px;
            padding: 10px 20px;
            color: var(--neon-cyan, #00ffff);
            font-size: 14px;
            z-index: 1000;
            opacity: 0;
            animation: fadeInOut 3s ease-in-out forwards;
        `;
        
        document.body.appendChild(hint);
        
        setTimeout(() => {
            if (hint.parentNode) {
                hint.remove();
            }
        }, 3000);
    }
    
    // Haptic feedback for supported devices
    vibrate(pattern = [100]) {
        if (navigator.vibrate) {
            navigator.vibrate(pattern);
        }
    }
    
    // Orientation change handler
    handleOrientationChange() {
        setTimeout(() => {
            this.detectMobile();
            
            // Refresh layout
            if (window.dashboard && window.dashboard.scene) {
                window.dashboard.scene.handleResize();
            }
        }, 100);
    }
}

// Add CSS animations
const mobileStyles = document.createElement('style');
mobileStyles.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateX(-50%) translateY(20px); }
        20% { opacity: 1; transform: translateX(-50%) translateY(0); }
        80% { opacity: 1; transform: translateX(-50%) translateY(0); }
        100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
    }
    
    .mobile-nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 15px 20px;
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.1);
        border-radius: 8px;
        margin-bottom: 8px;
        color: var(--neon-cyan, #00ffff);
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
    }
    
    .mobile-nav-item:hover,
    .mobile-nav-item.active {
        background: rgba(0, 255, 255, 0.1);
        border-color: rgba(0, 255, 255, 0.3);
        transform: translateX(5px);
    }
    
    .mobile-menu-content {
        background: rgba(10, 10, 15, 0.95);
        border-radius: 12px;
        padding: 20px;
        margin: 20px;
        max-height: 90vh;
        overflow-y: auto;
    }
    
    .mobile-menu-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    .mobile-menu-close {
        background: none;
        border: none;
        color: var(--neon-cyan, #00ffff);
        font-size: 24px;
        cursor: pointer;
        padding: 5px;
    }
    
    .gesture-hints {
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    .gesture-hints p {
        margin: 5px 0;
        font-size: 12px;
        color: rgba(0, 255, 255, 0.7);
    }
    
    .gesture-hints-content {
        background: rgba(10, 10, 15, 0.95);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        max-width: 90%;
    }
    
    .hint-item {
        display: flex;
        align-items: center;
        gap: 15px;
        margin: 15px 0;
        font-size: 16px;
    }
    
    .hint-icon {
        font-size: 24px;
        min-width: 30px;
    }
    
    .hints-close-btn {
        background: linear-gradient(45deg, var(--neon-cyan, #00ffff), var(--neon-purple, #8a2be2));
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        color: black;
        font-weight: bold;
        margin-top: 20px;
        cursor: pointer;
    }
    
    /* Mobile device adjustments */
    @media (max-width: 768px) {
        .holo-nav {
            display: none !important;
        }
        
        .hud-content {
            padding: 10px;
        }
        
        .stats-panel {
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        
        .activity-panel {
            max-height: 200px;
            overflow-y: auto;
        }
    }
`;

document.head.appendChild(mobileStyles);

// Initialize mobile navigation
window.addEventListener('DOMContentLoaded', () => {
    window.mobileNavigation = new MobileNavigation();
    
    // Handle orientation changes
    window.addEventListener('orientationchange', () => {
        window.mobileNavigation.handleOrientationChange();
    });
});

// Export for use in other modules
window.MobileNavigation = MobileNavigation;