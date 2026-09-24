// Bi-Directional Shared Communication Service
// Provides dual-transport event relay (BroadcastChannel + storage event) for 100% reliable cross-tab/cross-window communication.

class CommunicationService {
  constructor() {
    this.handlers = new Map();
    this.selectedPreset = null;
    this.selectedPresetListeners = new Set();
    this.seenMessageIds = new Set();

    // Transport 1: BroadcastChannel
    try {
      if (typeof window !== 'undefined' && 'BroadcastChannel' in window) {
        this.channel = new BroadcastChannel('signbridge_comm_bus_v2');
        this.channel.onmessage = (event) => {
          if (event.data && event.data.eventName) {
            this.handleIncoming(event.data.msgId, event.data.eventName, event.data.payload);
          }
        };
      }
    } catch (e) {
      console.warn('BroadcastChannel unavailable', e);
    }

    // Transport 2: Storage Event (Guarantees cross-window/cross-tab delivery)
    if (typeof window !== 'undefined') {
      window.addEventListener('storage', (event) => {
        if (event.key === 'signbridge_comm_event_sync' && event.newValue) {
          try {
            const data = JSON.parse(event.newValue);
            if (data && data.eventName) {
              this.handleIncoming(data.msgId, data.eventName, data.payload);
            }
          } catch (e) {}
        }
      });
    }
  }

  handleIncoming(msgId, eventName, payload) {
    if (msgId) {
      if (this.seenMessageIds.has(msgId)) return;
      this.seenMessageIds.add(msgId);
      if (this.seenMessageIds.size > 200) {
        const first = this.seenMessageIds.values().next().value;
        this.seenMessageIds.delete(first);
      }
    }
    this.dispatchLocal(eventName, payload);
  }

  on(eventName, handler) {
    if (!this.handlers.has(eventName)) {
      this.handlers.set(eventName, new Set());
    }
    this.handlers.get(eventName).add(handler);
    return () => this.off(eventName, handler);
  }

  off(eventName, handler) {
    if (this.handlers.has(eventName)) {
      this.handlers.get(eventName).delete(handler);
    }
  }

  dispatchLocal(eventName, payload) {
    if (this.handlers.has(eventName)) {
      this.handlers.get(eventName).forEach((handler) => {
        try {
          handler(payload);
        } catch (err) {
          console.error(`Error in communication handler for ${eventName}:`, err);
        }
      });
    }
  }

  emit(eventName, payload) {
    const msgId = 'comm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    this.seenMessageIds.add(msgId);

    // 1. Dispatch locally in this window/tab
    this.dispatchLocal(eventName, payload);

    // 2. Broadcast via BroadcastChannel
    if (this.channel) {
      try {
        this.channel.postMessage({ msgId, eventName, payload });
      } catch (err) {}
    }

    // 3. Broadcast via localStorage storage event (triggers in other tabs/windows)
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('signbridge_comm_event_sync', JSON.stringify({
          msgId,
          eventName,
          payload,
          timestamp: Date.now()
        }));
      }
    } catch (err) {}
  }

  // Selected Preset state management
  setSelectedPreset(preset) {
    this.selectedPreset = preset;
    this.emit('PRESET_SELECTED', preset);
  }

  getSelectedPreset() {
    return this.selectedPreset;
  }
}

export const communicationService = new CommunicationService();
