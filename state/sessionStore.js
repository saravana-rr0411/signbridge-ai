// User Authentication & Session Store
// Clean abstraction ready for Supabase Auth integration

class SessionStore {
  constructor() {
    this.STORAGE_KEY = 'signbridge_session_v1';
    this.listeners = new Set();
    this.session = this.loadSession();
  }

  loadSession() {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        const data = localStorage.getItem(this.STORAGE_KEY);
        if (data) {
          return JSON.parse(data);
        }
      }
    } catch (e) {
      console.warn('Unable to read session from localStorage', e);
    }
    // Default mock authenticated session for development, or null if logged out
    return {
      isAuthenticated: true,
      user: {
        email: 'staff.vance@civicdesk.org',
        mobile: '+1 (555) 234-8901',
        name: 'Officer J. Vance',
        desk: 'Counter Desk #04',
        role: 'Public Service Staff'
      }
    };
  }

  saveSession() {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        if (this.session) {
          localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.session));
        } else {
          localStorage.removeItem(this.STORAGE_KEY);
        }
      }
    } catch (e) {
      console.warn('Unable to persist session to localStorage', e);
    }
  }

  notify() {
    this.saveSession();
    this.listeners.forEach((listener) => {
      try {
        listener(this.session);
      } catch (err) {
        console.error('Session listener error:', err);
      }
    });
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.session);
    return () => this.listeners.delete(listener);
  }

  isAuthenticated() {
    return Boolean(this.session && this.session.isAuthenticated);
  }

  getUser() {
    return this.session ? this.session.user : null;
  }

  // Login action with validation
  login({ email, mobile, password }) {
    const cleanEmail = (email || '').trim();
    const cleanMobile = (mobile || '').trim();
    const cleanPassword = (password || '').trim();

    if (!cleanEmail) {
      return { success: false, error: 'Email Address is required' };
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) {
      return { success: false, error: 'Please enter a valid email address' };
    }
    if (!cleanMobile) {
      return { success: false, error: 'Mobile Number is required' };
    }
    if (!cleanPassword) {
      return { success: false, error: 'Password is required' };
    }
    if (cleanPassword.length < 4) {
      return { success: false, error: 'Password must be at least 4 characters' };
    }

    // Temporary frontend mock session (ready to be replaced with Supabase auth)
    this.session = {
      isAuthenticated: true,
      user: {
        email: cleanEmail,
        mobile: cleanMobile,
        name: 'Officer J. Vance',
        desk: 'Counter Desk #04',
        role: 'Public Service Staff'
      }
    };

    this.notify();
    return { success: true, user: this.session.user };
  }

  // Logout action
  logout() {
    this.session = {
      isAuthenticated: false,
      user: null
    };
    this.notify();
  }
}

export const sessionStore = new SessionStore();
