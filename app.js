// SignBridge AI - Main Application Router & Controller with Protected Routes
import { renderLoginPage, initLoginPage } from './pages/LoginPage.js';
import { renderDeafPage, initDeafPage } from './pages/DeafPage.js';
import { renderAdminPage, initAdminPage } from './pages/AdminPage.js';
import { renderHistoryPage, initHistoryPage } from './pages/HistoryPage.js';
import { sessionStore } from './state/sessionStore.js';
import { cameraService } from './services/cameraService.js';

class AppRouter {
  constructor() {
    this.appEl = document.getElementById('app');
    this.currentTeardown = null;

    this.routes = {
      '#/login': { render: renderLoginPage, init: initLoginPage, title: 'Login', public: true },
      '#/deaf': { render: renderDeafPage, init: initDeafPage, title: 'Deaf Person Interface', public: false },
      '#/admin': { render: renderAdminPage, init: initAdminPage, title: 'Staff Communication Console', public: false },
      '#/history': { render: renderHistoryPage, init: initHistoryPage, title: 'Chat History & Records', public: false }
    };

    window.addEventListener('hashchange', () => this.handleRoute());
  }

  start() {
    if (!window.location.hash || window.location.hash === '#' || window.location.hash === '#/') {
      // If already authenticated, navigate to #/deaf, else #/login
      window.location.hash = sessionStore.isAuthenticated() ? '#/deaf' : '#/login';
    } else {
      this.handleRoute();
    }
  }

  handleRoute() {
    const rawHash = window.location.hash || '#/login';
    const cleanHash = rawHash.split('?')[0];

    // Map potential slash paths (e.g. /deaf -> #/deaf)
    const normalizedHash = cleanHash.startsWith('#') ? cleanHash : '#' + cleanHash;
    const route = this.routes[normalizedHash] || this.routes['#/login'];

    // ROUTE PROTECTION: If page is protected and user is not authenticated, redirect to Login
    if (!route.public && !sessionStore.isAuthenticated()) {
      window.location.hash = '#/login';
      return;
    }

    // Update document title
    document.title = `SignBridge AI - ${route.title}`;

    // Clean up previous page listeners/timers/camera tracks
    if (this.currentTeardown && typeof this.currentTeardown === 'function') {
      try {
        console.log('[CAMERA STOP TRACE] Router invoking teardown for previous page', {
          targetRoute: normalizedHash,
          timestamp: new Date().toISOString()
        });
        this.currentTeardown();
      } catch (err) {
        console.error('Teardown error:', err);
      }
      this.currentTeardown = null;
    }

    // Render HTML
    if (this.appEl) {
      this.appEl.innerHTML = route.render();
    }

    // Initialize page JS
    if (route.init && typeof route.init === 'function') {
      try {
        this.currentTeardown = route.init();
      } catch (err) {
        console.error('Page init error:', err);
      }
    }

    // Bind universal sidebar actions
    this.bindGlobalActions();
  }

  bindGlobalActions() {
    // Counter Settings button
    const settingsBtn = document.getElementById('btn-counter-settings');
    if (settingsBtn) {
      settingsBtn.addEventListener('click', () => {
        this.showToast('Desk Terminal #04 configuration: WebRTC 1080p, MediaPipe Model v2.4, Audio relay disabled');
      });
    }

    // End Session / Logout button
    const logoutBtn = document.getElementById('btn-logout');
    if (logoutBtn) {
      logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        cameraService.stopCamera('user_logged_out');
        sessionStore.logout();
        window.location.hash = '#/login';
        this.showToast('Session ended. You have been logged out.');
      });
    }
  }

  showToast(message) {
    const existing = document.getElementById('app-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'app-toast';
    toast.className = 'fixed bottom-5 right-5 z-50 bg-[#001428] text-white px-4 py-3 rounded-xl shadow-xl flex items-center gap-2.5 text-xs font-semibold animate-fadeIn border border-white/20';
    toast.innerHTML = `
      <span class="material-symbols-outlined text-secondary-container text-[18px]">info</span>
      <span>${message}</span>
    `;

    document.body.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('opacity-0', 'transition-opacity', 'duration-300');
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
}

// Start application when DOM is ready
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    const router = new AppRouter();
    router.start();
  });
}
