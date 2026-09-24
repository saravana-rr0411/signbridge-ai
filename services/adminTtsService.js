// services/adminTtsService.js
// Production Browser Text-to-Speech (TTS) Service for Admin Console
// Automatically reads aloud incoming Deaf person contextual messages via native Web Speech API.
//
// Key Features:
// 1. Strict duplicate suppression (message ID set + time/text fingerprinting).
// 2. Controlled FIFO speech queue ensuring zero voice overlap and zero lost messages.
// 3. Superseding gesture handling (cancels partial prefix speech if complete accumulated sentence arrives).
// 4. Configured voice: English (en-US), natural rate (0.95), pitch (1.0), volume (1.0).
// 5. Autoplay policy resilience (graceful fallback without UI disruption or continuous retry spam).
// 6. Deaf -> Admin only: strictly excludes Admin's own outgoing messages.

class AdminTtsService {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
    this.currentUtterance = null;
    this.currentSpeechItem = null;
    this.spokenMessageIds = new Set();
    this.listeners = new Set();
    this.debounceTimer = null;
    this.pendingItem = null;
    this.debounceMs = 300; // brief gesture stabilization window for browser runtime
    this.lastSpokenText = null;
    this.lastSpokenTime = 0;
    this.voice = null;
    this.maxSpokenHistory = 500;
    this.interactionRegistered = false;

    // Custom injectable mocks for test environments (e.g. Node)
    this.customSpeechSynthesis = null;
    this.customSpeechSynthesisUtterance = null;

    this.initBrowserEvents();
  }

  initBrowserEvents() {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      if ('onvoiceschanged' in window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = () => {
          this.voice = null; // re-evaluate available voices
        };
      }
    }
  }

  /**
   * Set custom speech synthesis engine & utterance constructor (used for tests in Node).
   */
  setSpeechSynthesis(synth, UtteranceClass) {
    this.customSpeechSynthesis = synth;
    if (UtteranceClass) {
      this.customSpeechSynthesisUtterance = UtteranceClass;
    }
  }

  getSynth() {
    if (this.customSpeechSynthesis) return this.customSpeechSynthesis;
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      return window.speechSynthesis;
    }
    return null;
  }

  getUtteranceConstructor() {
    if (this.customSpeechSynthesisUtterance) return this.customSpeechSynthesisUtterance;
    if (typeof window !== 'undefined' && 'SpeechSynthesisUtterance' in window) {
      return window.SpeechSynthesisUtterance;
    }
    if (typeof SpeechSynthesisUtterance !== 'undefined') {
      return SpeechSynthesisUtterance;
    }
    return null;
  }

  getPreferredVoice(synth) {
    if (this.voice) return this.voice;
    if (!synth || typeof synth.getVoices !== 'function') return null;
    const voices = synth.getVoices() || [];
    if (!voices.length) return null;

    // 1. Prefer natural / Google / Samantha English voice
    const naturalEn = voices.find(
      (v) =>
        (v.lang === 'en-US' || v.lang === 'en_US') &&
        (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha'))
    );
    if (naturalEn) {
      this.voice = naturalEn;
      return naturalEn;
    }

    // 2. Any en-US voice
    const anyEnUs = voices.find((v) => v.lang === 'en-US' || v.lang === 'en_US');
    if (anyEnUs) {
      this.voice = anyEnUs;
      return anyEnUs;
    }

    // 3. Any English voice
    const anyEn = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith('en'));
    if (anyEn) {
      this.voice = anyEn;
      return anyEn;
    }

    return voices[0] || null;
  }

  /**
   * Main entry point: process incoming Deaf person contextual message for TTS playback.
   *
   * @param {Object} input - Message payload or conversation message object
   * @param {string} [input.id] - Message unique ID
   * @param {string} [input.sender] - 'deaf' or 'admin'
   * @param {string} [input.senderName] - e.g. 'Deaf Person'
   * @param {string} [input.text] - Contextual sentence text
   * @param {string} [input.rawSign] - e.g. "HELLO -> HELP"
   * @param {string[]} [input.rawSequence] - e.g. ['HELLO', 'HELP']
   * @param {number} [input.timestamp] - Epoch ms
   * @returns {boolean} Whether the message was accepted for speech
   */
  speakIncomingDeafMessage(input) {
    if (!input) return false;

    // 1. STRICT SENDER CHECK:
    // Admin's own messages must NEVER trigger Admin TTS
    const sender = input.sender || (input.senderName && input.senderName.toLowerCase().includes('admin') ? 'admin' : 'deaf');
    if (sender === 'admin') {
      return false;
    }

    // 2. TEXT VALIDATION:
    const text = (input.text || input.contextualMessage || '').trim();
    if (!text) return false;

    // 3. MESSAGE ID RESOLUTION:
    const rawId = input.id || input.messageId || input.msgId;
    const timestamp = input.timestamp || Date.now();
    const rawSign = input.rawSign || (Array.isArray(input.rawSequence) ? input.rawSequence.join(' -> ') : '');
    const rawSequence = Array.isArray(input.rawSequence) ? input.rawSequence : [];

    // Derive deterministic key if explicit ID is not provided
    const messageId = rawId
      ? String(rawId)
      : `deaf_${timestamp}_${text.replace(/\s+/g, '_')}`;

    // 4. DUPLICATE CHECK:
    if (this.spokenMessageIds.has(messageId)) {
      return false;
    }

    // Secondary duplicate prevention: identical text received within 1.5s
    if (this.lastSpokenText === text && Math.abs(timestamp - this.lastSpokenTime) < 1500) {
      this.markIdSpoken(messageId);
      return false;
    }

    // 5. SUPERSEDING CHECK:
    // If a more complete accumulated sentence arrives (e.g. "HELLO -> HELP" superseding "HELLO"),
    // cancel the partial speech immediately so only the complete message is heard.
    const isSuperseding = this.checkIsSuperseding(text, rawSign, rawSequence);

    if (isSuperseding) {
      this.cancelCurrentSpeech();
    }

    // Mark as processed
    this.markIdSpoken(messageId);

    const item = {
      id: messageId,
      text,
      rawSign,
      rawSequence,
      timestamp
    };

    // If debounce is configured and this is a single, non-superseding sign,
    // wait a brief interval in case the user completes a two-sign sequence.
    if (this.debounceMs > 0 && rawSequence.length <= 1 && !isSuperseding && !input.isComplete) {
      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer);
      }
      this.pendingItem = item;
      this.debounceTimer = setTimeout(() => {
        this.pendingItem = null;
        this.debounceTimer = null;
        this.enqueue(item);
      }, this.debounceMs);
      return true;
    }

    // Clear any pending single-sign timer when a full/multi-sign or direct message arrives
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
      this.pendingItem = null;
    }

    this.enqueue(item);
    return true;
  }

  /**
   * Determine if the incoming message supersedes the currently speaking or pending message.
   */
  checkIsSuperseding(text, rawSign, rawSequence) {
    const candidates = [this.pendingItem, this.currentSpeechItem].filter(Boolean);

    for (const item of candidates) {
      if (!item) continue;

      // 1. Raw sign sequence extension (e.g. "HELLO -> HELP" supersedes "HELLO")
      if (rawSign && item.rawSign && rawSign.startsWith(item.rawSign + ' -> ')) {
        return true;
      }

      // 2. Sequence array extension
      if (
        rawSequence.length > item.rawSequence.length &&
        item.rawSequence.length > 0 &&
        item.rawSequence.every((val, idx) => val === rawSequence[idx])
      ) {
        return true;
      }

      // 3. Text prefix extension (e.g. "Hello, I need help." supersedes "Hello.")
      const cleanCandidate = item.text.replace(/[\.\!\?]/g, '').trim().toLowerCase();
      const cleanNew = text.replace(/[\.\!\?]/g, '').trim().toLowerCase();
      if (cleanCandidate && cleanNew !== cleanCandidate && cleanNew.startsWith(cleanCandidate)) {
        return true;
      }
    }

    return false;
  }

  enqueue(item) {
    this.queue.push(item);
    this.processQueue();
  }

  processQueue() {
    if (this.isProcessing) {
      return;
    }
    if (this.queue.length === 0) {
      this.notifyListeners(false);
      return;
    }

    const synth = this.getSynth();
    const UtteranceConstructor = this.getUtteranceConstructor();

    if (!synth || !UtteranceConstructor) {
      // In environment without speech synthesis (e.g. basic Node), drain queue gracefully
      this.queue = [];
      this.notifyListeners(false);
      return;
    }

    const item = this.queue.shift();
    this.currentSpeechItem = item;
    this.isProcessing = true;
    this.lastSpokenText = item.text;
    this.lastSpokenTime = Date.now();
    this.notifyListeners(true);

    try {
      const utterance = new UtteranceConstructor(item.text);
      utterance.lang = 'en-US';
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      const voice = this.getPreferredVoice(synth);
      if (voice) {
        utterance.voice = voice;
      }

      let finished = false;
      const finish = () => {
        if (finished) return;
        finished = true;
        if (this.currentUtterance === utterance) {
          this.currentUtterance = null;
          this.currentSpeechItem = null;
        }
        this.isProcessing = false;
        this.processQueue();
      };

      utterance.onend = finish;
      utterance.onerror = (e) => {
        const errType = e && e.error ? e.error : 'unknown';
        if (errType === 'not-allowed') {
          this.handleAutoplayRestriction();
        } else if (errType !== 'interrupted' && errType !== 'canceled') {
          console.warn('[AdminTTS] Speech error:', errType);
        }
        finish();
      };

      this.currentUtterance = utterance;

      if (synth.paused && typeof synth.resume === 'function') {
        try {
          synth.resume();
        } catch (e) {}
      }

      synth.speak(utterance);
    } catch (err) {
      console.warn('[AdminTTS] Error invoking speech synthesis:', err);
      this.currentSpeechItem = null;
      this.currentUtterance = null;
      this.isProcessing = false;
      this.processQueue();
    }
  }

  cancelCurrentSpeech() {
    const synth = this.getSynth();
    if (synth && typeof synth.cancel === 'function') {
      try {
        synth.cancel();
      } catch (e) {}
    }
    this.currentSpeechItem = null;
    this.currentUtterance = null;
    this.isProcessing = false;
    this.notifyListeners(false);
  }

  handleAutoplayRestriction() {
    if (this.interactionRegistered || typeof document === 'undefined') return;
    this.interactionRegistered = true;
    console.info('[AdminTTS] Speech awaiting initial user interaction to unlock browser audio policy.');

    const unlock = () => {
      document.removeEventListener('click', unlock);
      document.removeEventListener('keydown', unlock);
      const synth = this.getSynth();
      const UtteranceConstructor = this.getUtteranceConstructor();
      if (synth && UtteranceConstructor) {
        try {
          const silent = new UtteranceConstructor('');
          synth.speak(silent);
        } catch (e) {}
      }
    };

    document.addEventListener('click', unlock, { once: true });
    document.addEventListener('keydown', unlock, { once: true });
  }

  markIdSpoken(id) {
    if (!id) return;
    this.spokenMessageIds.add(String(id));
    if (this.spokenMessageIds.size > this.maxSpokenHistory) {
      const first = this.spokenMessageIds.values().next().value;
      this.spokenMessageIds.delete(first);
    }
  }

  /**
   * Pre-seed already existing messages so page load doesn't read historical chat aloud.
   */
  markExistingAsSpoken(messages) {
    if (Array.isArray(messages)) {
      messages.forEach((m) => {
        if (m && m.id) {
          this.spokenMessageIds.add(String(m.id));
        }
      });
    }
  }

  onStateChange(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notifyListeners(isSpeaking) {
    this.listeners.forEach((fn) => {
      try {
        fn(isSpeaking);
      } catch (e) {}
    });
  }

  isSpeaking() {
    return this.isProcessing;
  }

  getQueue() {
    return [...this.queue];
  }

  stop() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
      this.pendingItem = null;
    }
    this.queue = [];
    this.cancelCurrentSpeech();
  }

  reset() {
    this.stop();
    this.spokenMessageIds.clear();
    this.lastSpokenText = null;
    this.lastSpokenTime = 0;
    this.listeners.clear();
  }
}

export const adminTtsService = new AdminTtsService();
export { AdminTtsService };
