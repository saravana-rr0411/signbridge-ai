// services/communicationService.js - Real-Time Cross-Device & Local Communication Bus
// Combines WebSocket relay (for cross-device Vercel <-> Render communication)
// with BroadcastChannel + storage event fallback (for same-machine multi-tab testing).

import { getWsRelayUrl } from './apiConfig.js';

class CommunicationService {
  constructor() {
    this.handlers = new Map();
    this.selectedPreset = null;
    this.selectedPresetListeners = new Set();
    this.seenMessageIds = new Set();

    // WebSocket Cross-Device Transport State
    this.ws = null;
    this.roomId = 'desk_04';
    this.role = null; // 'deaf' | 'admin'
    this.isConnected = false;
    this.isConnecting = false;
    this.manualDisconnect = false;
    this.reconnectTimer = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 10000;
    this.customWs = null; // Injectable for Node unit tests

    // Transport 1: BroadcastChannel (local fallback)
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

    // Transport 2: Storage Event (local fallback)
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

  /**
   * Connect to the Render WebSocket relay endpoint for the given room and role.
   *
   * @param {string} [roomId='desk_04'] - Room or desk identifier
   * @param {string} [role='deaf'] - Client role ('deaf' or 'admin')
   */
  connect(roomId = 'desk_04', role = 'deaf') {
    this.roomId = roomId || 'desk_04';
    this.role = (role || 'deaf').toLowerCase().strip ? role.toLowerCase().strip() : String(role).toLowerCase().trim();
    this.manualDisconnect = false;

    // Check if WebSocket is supported in current environment
    const WsClass = this.customWs || (typeof WebSocket !== 'undefined' ? WebSocket : null);
    if (!WsClass) {
      return;
    }

    if (this.ws && (this.ws.readyState === 0 || this.ws.readyState === 1)) {
      if (this.isConnected && this.ws.url && this.ws.url.includes(this.role)) {
        return; // Already actively connected with correct role
      }
      try {
        this.ws.close();
      } catch (e) {}
    }

    const wsUrl = getWsRelayUrl(this.roomId, this.role);
    this.isConnecting = true;

    try {
      const socket = new WsClass(wsUrl);
      this.ws = socket;

      socket.onopen = () => {
        if (this.ws !== socket) return;
        this.isConnected = true;
        this.isConnecting = false;
        this.reconnectDelay = 1000;
        console.log(`[CommunicationService] Connected to Render WebSocket relay (${this.roomId}/${this.role})`);
        this.dispatchLocal('COMM_CONNECTED', { roomId: this.roomId, role: this.role });
      };

      socket.onmessage = (event) => {
        if (this.ws !== socket) return;
        try {
          const envelope = JSON.parse(event.data);
          if (envelope && envelope.type) {
            const payload = envelope.payload || {};
            const msgId = payload.msgId || payload.id || envelope.msgId;

            // Prevent duplicate handling of the same message
            if (msgId) {
              if (this.seenMessageIds.has(msgId)) return;
              this.seenMessageIds.add(msgId);
              if (this.seenMessageIds.size > 300) {
                const first = this.seenMessageIds.values().next().value;
                this.seenMessageIds.delete(first);
              }
            }

            // Dispatch locally to registered listeners (CRITICAL: does NOT echo back to WS)
            this.dispatchLocal(envelope.type, payload);
          }
        } catch (err) {
          console.warn('[CommunicationService] Error parsing incoming WebSocket envelope:', err);
        }
      };

      socket.onerror = (err) => {
        if (this.ws !== socket) return;
        this.isConnected = false;
        this.isConnecting = false;
        console.warn('[CommunicationService] WebSocket error on relay:', err && err.message ? err.message : err);
      };

      socket.onclose = () => {
        if (this.ws !== socket) return;
        this.isConnected = false;
        this.isConnecting = false;
        this.dispatchLocal('COMM_DISCONNECTED', { roomId: this.roomId, role: this.role });

        // Automatic reconnect with bounded exponential backoff
        if (!this.manualDisconnect) {
          if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
          this.reconnectTimer = setTimeout(() => {
            if (!this.manualDisconnect && !this.isConnected) {
              console.log(`[CommunicationService] Reconnecting to WebSocket (${this.roomId}/${this.role})...`);
              this.connect(this.roomId, this.role);
            }
          }, this.reconnectDelay);
          this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
        }
      };
    } catch (err) {
      console.warn('[CommunicationService] Failed to establish WebSocket connection:', err);
      this.isConnecting = false;
    }
  }

  /**
   * Disconnect cleanly from the WebSocket relay.
   */
  disconnect() {
    this.manualDisconnect = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close(1000, 'Normal closure');
      } catch (e) {}
      this.ws = null;
    }
    this.isConnected = false;
    this.isConnecting = false;
  }

  /**
   * Send a direct message/signaling envelope through the WebSocket to the opposite peer
   * without triggering local dispatch on the sender side.
   */
  sendWsDirect(type, payload = {}) {
    const WsOpenState = typeof WebSocket !== 'undefined' ? WebSocket.OPEN : 1;
    if (this.isConnected && this.ws && this.ws.readyState === WsOpenState) {
      try {
        const envelope = {
          roomId: this.roomId || 'desk_04',
          senderRole: this.role || 'deaf',
          targetRole: this.role === 'deaf' ? 'admin' : 'deaf',
          type: type,
          payload: payload,
          timestamp: Date.now()
        };
        this.ws.send(JSON.stringify(envelope));
        return true;
      } catch (err) {
        console.warn('[CommunicationService] Failed to send direct WS envelope:', err);
        return false;
      }
    }
    return false;
  }

  handleIncoming(msgId, eventName, payload) {
    if (msgId) {
      if (this.seenMessageIds.has(msgId)) return;
      this.seenMessageIds.add(msgId);
      if (this.seenMessageIds.size > 300) {
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

  /**
   * Emit an event: dispatches locally, broadcasts via local fallbacks (BroadcastChannel / localStorage),
   * AND relays across the internet to the remote peer via Render WebSocket if connected.
   */
  emit(eventName, payload) {
    const msgId = 'comm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    this.seenMessageIds.add(msgId);

    const safePayload = {
      ...(payload || {}),
      msgId: msgId
    };

    // 1. Dispatch locally in this window/tab
    this.dispatchLocal(eventName, safePayload);

    // 2. Broadcast via local BroadcastChannel (same-machine fallback)
    if (this.channel) {
      try {
        this.channel.postMessage({ msgId, eventName, payload: safePayload });
      } catch (err) {}
    }

    // 3. Broadcast via localStorage storage event (same-machine fallback)
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(
          'signbridge_comm_event_sync',
          JSON.stringify({
            msgId,
            eventName,
            payload: safePayload,
            timestamp: Date.now()
          })
        );
      }
    } catch (err) {}

    // 4. Relay across the internet via Render WebSocket
    const WsOpenState = typeof WebSocket !== 'undefined' ? WebSocket.OPEN : 1;
    if (this.isConnected && this.ws && this.ws.readyState === WsOpenState) {
      try {
        const envelope = {
          roomId: this.roomId || 'desk_04',
          senderRole: this.role || 'deaf',
          targetRole: this.role === 'deaf' ? 'admin' : 'deaf',
          type: eventName,
          payload: safePayload,
          timestamp: Date.now()
        };
        this.ws.send(JSON.stringify(envelope));
      } catch (err) {
        console.warn('[CommunicationService] WebSocket send error:', err);
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

  // Test helper
  setMockWebSocketClass(MockWs) {
    this.customWs = MockWs;
  }
}

export const communicationService = new CommunicationService();
export { CommunicationService };
