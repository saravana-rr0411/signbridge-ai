// SignBridge AI - Bi-directional Communication & State Relay Bridge
import { INITIAL_CONVERSATION } from './mockData.js';

class StateBridge {
  constructor() {
    this.listeners = new Set();
    this.channel = null;

    try {
      if (typeof window !== 'undefined' && 'BroadcastChannel' in window) {
        this.channel = new BroadcastChannel('signbridge_relay_channel');
        this.channel.onmessage = (event) => {
          if (event.data && event.data.type === 'STATE_UPDATE') {
            this.state = event.data.state;
            this.notify();
          }
        };
      }
    } catch (e) {
      console.warn('BroadcastChannel not available, using in-memory relay', e);
    }

    // Load initial state or cached state
    const saved = this.loadFromStorage();
    this.state = saved || {
      conversation: [...INITIAL_CONVERSATION],
      activeSignAnimation: {
        text: 'Please wait here. The doctor will examine you shortly.',
        cycle: 1,
        maxCycles: 2,
        isPlaying: true,
        completed: false
      },
      liveSignTranscript: 'I need help with my appointment',
      isCameraTurnActive: false,
      selectedPresetText: '',
      activeCategory: 'HOSPITAL',
      turnStatus: 'Observing Signer', // 'Observing Signer' or 'Your Turn to Sign'
      currentUser: {
        name: 'Officer J. Vance',
        desk: 'Counter Desk #04',
        role: 'Admin / Public Service Officer'
      }
    };
  }

  loadFromStorage() {
    try {
      const data = localStorage.getItem('signbridge_state_v1');
      return data ? JSON.parse(data) : null;
    } catch (e) {
      return null;
    }
  }

  saveToStorage() {
    try {
      localStorage.setItem('signbridge_state_v1', JSON.stringify(this.state));
    } catch (e) {
      // Ignore storage errors in private mode
    }
  }

  getState() {
    return this.state;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  notify() {
    this.saveToStorage();
    this.listeners.forEach((listener) => {
      try {
        listener(this.state);
      } catch (err) {
        console.error('State update listener error:', err);
      }
    });

    if (this.channel) {
      try {
        this.channel.postMessage({
          type: 'STATE_UPDATE',
          state: this.state
        });
      } catch (err) {
        // Channel post fail
      }
    }
  }

  // ADMIN -> DEAF Flow:
  // Admin types or selects preset -> Send -> conversation update -> send to Deaf -> Sign animation plays 2x -> Camera turns green
  sendAdminMessage(text) {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newMsg = {
      id: 'm_' + Date.now(),
      sender: 'admin',
      senderName: 'Admin (Officer Vance)',
      text: cleanText,
      time: timeStr,
      timestamp: now.getTime(),
      isActiveReply: true
    };

    // Remove active reply status from previous messages
    this.state.conversation = this.state.conversation.map((m) => ({
      ...m,
      isActiveReply: false
    }));

    this.state.conversation.push(newMsg);

    // Setup animated sign response on Deaf interface
    this.state.activeSignAnimation = {
      text: cleanText,
      cycle: 1,
      maxCycles: 2,
      isPlaying: true,
      completed: false
    };

    // Reset camera active turn until playback completes 2x
    this.state.isCameraTurnActive = false;
    this.state.turnStatus = 'Observing Signer';
    this.state.selectedPresetText = '';

    this.notify();
  }

  // Complete sign animation playback (after 2 cycles)
  finishAnimationPlayback() {
    this.state.activeSignAnimation.isPlaying = false;
    this.state.activeSignAnimation.completed = true;
    this.state.isCameraTurnActive = true;
    this.state.turnStatus = 'Your Turn to Sign';
    this.notify();
  }

  restartAnimationPlayback() {
    this.state.activeSignAnimation.isPlaying = true;
    this.state.activeSignAnimation.completed = false;
    this.state.activeSignAnimation.cycle = 1;
    this.state.isCameraTurnActive = false;
    this.state.turnStatus = 'Observing Signer';
    this.notify();
  }

  // DEAF -> ADMIN Flow:
  // Camera -> Sign recognition -> Text -> Admin live transcript -> Conversation update
  updateLiveRecognizedText(text) {
    this.state.liveSignTranscript = text;
    this.notify();
  }

  sendDeafMessage(text) {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newMsg = {
      id: 'm_' + Date.now(),
      sender: 'deaf',
      senderName: 'Deaf Person',
      text: cleanText,
      time: timeStr,
      timestamp: now.getTime()
    };

    this.state.conversation.push(newMsg);
    this.state.liveSignTranscript = cleanText;
    this.notify();
  }

  // Populates Admin composer without sending
  populateComposer(text) {
    this.state.selectedPresetText = text;
    this.notify();
  }

  clearComposerPreset() {
    this.state.selectedPresetText = '';
    this.notify();
  }

  setCategory(cat) {
    this.state.activeCategory = cat;
    this.notify();
  }

  resetSession() {
    this.state.conversation = [...INITIAL_CONVERSATION];
    this.state.activeSignAnimation = {
      text: 'Please wait here. The doctor will examine you shortly.',
      cycle: 1,
      maxCycles: 2,
      isPlaying: true,
      completed: false
    };
    this.state.liveSignTranscript = 'I need help with my appointment';
    this.state.isCameraTurnActive = false;
    this.state.selectedPresetText = '';
    this.state.turnStatus = 'Observing Signer';
    this.notify();
  }
}

export const stateBridge = new StateBridge();
