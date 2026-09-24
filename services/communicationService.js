// Bi-Directional Shared Communication Service
// Abstracts local events and cross-tab/cross-window relay (ready for WebSocket/Backend ML API)

class CommunicationService {
  constructor() {
    this.handlers = new Map();
    this.selectedPreset = null;
    this.selectedPresetListeners = new Set();

    try {
      if (typeof window !== 'undefined' && 'BroadcastChannel' in window) {
        this.channel = new BroadcastChannel('signbridge_comm_bus');
        this.channel.onmessage = (event) => {
          if (event.data && event.data.eventName) {
            this.dispatchLocal(event.data.eventName, event.data.payload);
          }
        };
      }
    } catch (e) {
      console.warn('BroadcastChannel unavailable', e);
    }
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
    // 1. Dispatch locally in this window/tab
    this.dispatchLocal(eventName, payload);

    // 2. Broadcast across tabs/windows
    if (this.channel) {
      try {
        this.channel.postMessage({
          eventName,
          payload
        });
      } catch (err) {
        // Channel post error
      }
    }
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
