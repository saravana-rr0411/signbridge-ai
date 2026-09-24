// Conversation & Chat History Shared State Store
// Synchronized across tabs via localStorage and BroadcastChannel
import { INITIAL_CONVERSATION, CHAT_HISTORY_RECORDS } from '../services/mockData.js';
import { communicationService } from '../services/communicationService.js';

class ConversationStore {
  constructor() {
    this.STORAGE_KEY = 'signbridge_conversation_v2';
    this.HISTORY_KEY = 'signbridge_history_v2';
    this.CHAT_ID_KEY = 'signbridge_current_chat_id';
    this.listeners = new Set();

    this.currentChatId = this.loadChatId();
    this.conversation = this.loadConversation();
    this.historyRecords = this.loadHistory();

    // Setup Cross-tab Broadcast Channel
    try {
      if (typeof window !== 'undefined' && 'BroadcastChannel' in window) {
        this.channel = new BroadcastChannel('signbridge_conversation_channel');
        this.channel.onmessage = (event) => {
          if (event.data) {
            if (event.data.type === 'CONVERSATION_UPDATED') {
              this.conversation = event.data.conversation;
              this.historyRecords = this.loadHistory();
              this.notify(false);
            } else if (event.data.type === 'DEMO_RESET') {
              this.conversation = [...INITIAL_CONVERSATION];
              this.notify(false);
            }
          }
        };
      }
    } catch (e) {
      console.warn('BroadcastChannel not available', e);
    }
  }

  loadChatId() {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        const id = localStorage.getItem(this.CHAT_ID_KEY);
        if (id) return id;
      }
    } catch (e) {}
    return 'SB-20260924-004';
  }

  loadConversation() {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        const data = localStorage.getItem(this.STORAGE_KEY);
        if (data) {
          const parsed = JSON.parse(data);
          if (Array.isArray(parsed) && parsed.length > 0) {
            return parsed;
          }
        }
      }
    } catch (e) {
      console.warn('Failed loading conversation from localStorage', e);
    }
    return [...INITIAL_CONVERSATION];
  }

  loadHistory() {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        const data = localStorage.getItem(this.HISTORY_KEY);
        if (data) {
          const parsed = JSON.parse(data);
          if (Array.isArray(parsed) && parsed.length > 0) {
            return parsed;
          }
        }
      }
    } catch (e) {
      console.warn('Failed loading history from localStorage', e);
    }
    return [...CHAT_HISTORY_RECORDS];
  }

  save() {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.conversation));
        localStorage.setItem(this.HISTORY_KEY, JSON.stringify(this.historyRecords));
        localStorage.setItem(this.CHAT_ID_KEY, this.currentChatId);
      }
    } catch (e) {}
  }

  notify(broadcast = true) {
    this.save();
    this.listeners.forEach((listener) => {
      try {
        listener(this.conversation);
      } catch (err) {
        console.error('Conversation listener error:', err);
      }
    });

    if (broadcast && this.channel) {
      try {
        this.channel.postMessage({
          type: 'CONVERSATION_UPDATED',
          conversation: this.conversation
        });
      } catch (err) {}
    }
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.conversation);
    return () => this.listeners.delete(listener);
  }

  getConversation() {
    return this.conversation;
  }

  getHistoryRecords() {
    // Ensure newest conversations first
    return [...this.historyRecords].sort((a, b) => {
      const timeA = a.timestamp || 0;
      const timeB = b.timestamp || 0;
      if (timeA !== timeB) return timeB - timeA;
      return b.id.localeCompare(a.id);
    });
  }

  getCurrentChatId() {
    return this.currentChatId;
  }

  addMessage({ sender, senderName, text, type = 'text', isActiveReply = false }) {
    if (!text || !text.trim()) return null;
    const cleanText = text.trim();
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const message = {
      id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
      chatId: this.currentChatId,
      sender: sender === 'admin' ? 'admin' : 'deaf',
      senderName: senderName || (sender === 'admin' ? 'Admin (Officer Vance)' : 'Deaf Person'),
      text: cleanText,
      time: timeStr,
      timestamp: now.getTime(),
      type: type,
      isActiveReply: Boolean(isActiveReply)
    };

    if (sender === 'admin') {
      // Deactivate previous active replies
      this.conversation = this.conversation.map((m) => ({
        ...m,
        isActiveReply: false
      }));
    }

    this.conversation.push(message);

    // Keep history record updated for the current session
    this.syncCurrentSessionToHistory();

    this.notify(true);
    return message;
  }

  syncCurrentSessionToHistory() {
    const now = new Date();
    const dateStr = now.toISOString().split('T')[0]; // '2026-09-24'
    const displayDate = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const lastAdminMsg = [...this.conversation].reverse().find((m) => m.sender === 'admin');
    const lastDeafMsg = [...this.conversation].reverse().find((m) => m.sender === 'deaf');

    const currentRecordIndex = this.historyRecords.findIndex((r) => r.id === this.currentChatId);

    const recordData = {
      id: this.currentChatId,
      date: dateStr,
      displayDate: displayDate,
      time: timeStr,
      timestamp: now.getTime(),
      status: 'Active / Completed',
      counter: 'Desk #04',
      adminName: 'Officer J. Vance',
      adminPreview: lastAdminMsg ? lastAdminMsg.text : 'Welcome to Desk #4.',
      deafPreview: lastDeafMsg ? lastDeafMsg.text : 'I need help with my appointment.',
      messages: [...this.conversation]
    };

    if (currentRecordIndex >= 0) {
      this.historyRecords[currentRecordIndex] = recordData;
    } else {
      this.historyRecords.unshift(recordData);
    }
  }

  // Developer / Demo Reset: Clears current conversation & generates fresh state
  resetDemo() {
    this.conversation = [...INITIAL_CONVERSATION];
    this.currentChatId = 'SB-20260924-' + String(Math.floor(Math.random() * 800) + 100);
    this.syncCurrentSessionToHistory();
    this.notify(true);

    if (this.channel) {
      try {
        this.channel.postMessage({ type: 'DEMO_RESET' });
      } catch (e) {}
    }

    communicationService.emit('DEMO_RESET', { timestamp: Date.now() });
  }
}

export const conversationStore = new ConversationStore();
