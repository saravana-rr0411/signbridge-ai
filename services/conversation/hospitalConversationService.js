// services/conversation/hospitalConversationService.js
// Dedicated Public-Service Conversation Layer for Hospital First-Visit.
// Converts recognized ASL sign sequence events into controlled contextual hospital communication messages.
// Does NOT modify or retrain ML models; operates strictly as a separate conversation layer.

import {
  HOSPITAL_PHRASES,
  V3_ML_ALLOWED_SIGNS,
  getPhraseById,
  resolveHospitalSentence
} from './hospitalPhrases.js';
import { conversationStore } from '../../state/conversationStore.js';
import { communicationService } from '../communicationService.js';

class HospitalConversationService {
  constructor() {
    this.currentContext = 'hospital';

    // Sequence Accumulation & De-duplication State
    this.currentSignSequence = [];        // Array of distinct sequential signs: e.g. ['HELLO', 'HELP']
    this.lastAcceptedSign = null;         // Most recent sign accepted into sequence
    this.lastAcceptedTimestamp = 0;       // Epoch ms of last accepted sign
    this.lastDispatchedSentence = null;   // To prevent re-dispatching identical sentence
    this.gestureHoldCooldownMs = 3000;    // 3s cooldown for holding the same gesture
    this.sequenceTimeoutMs = 6000;        // 6s timeout: gap between distinct gestures starts new sequence

    // UI Presentation State
    this.currentSign = '—';               // Raw sequence string: e.g. "HELLO -> HELP"
    this.currentPhrase = 'Show a sign to begin'; // Contextual sentence: e.g. "Hello, I need help."
    this.currentConfidence = null;
    this.currentTimestamp = Date.now();
    this.sender = 'deaf';
    this.conversationHistory = [];

    this.listeners = new Set();

    // Listen for tab synchronization or demo reset events
    if (typeof window !== 'undefined') {
      communicationService.on('DEMO_RESET', () => {
        this.reset();
      });

      communicationService.on('HOSPITAL_SYNC', (payload) => {
        if (payload) {
          this.applyExternalState(payload);
        }
      });
    }
  }

  // --- GETTERS ---
  getContext() {
    return this.currentContext;
  }

  getRecognizedSignSequence() {
    return [...this.currentSignSequence];
  }

  getCurrentPhrase() {
    return this.currentPhrase;
  }

  getCurrentSign() {
    return this.currentSign;
  }

  getConfidence() {
    return this.currentConfidence;
  }

  getTimestamp() {
    return this.currentTimestamp;
  }

  getSender() {
    return this.sender;
  }

  getConversationHistory() {
    return [...this.conversationHistory];
  }

  getState() {
    return {
      context: this.currentContext,
      recognizedSignSequence: [...this.currentSignSequence],
      currentSign: this.currentSign,
      currentPhrase: this.currentPhrase,
      confidence: this.currentConfidence,
      timestamp: this.currentTimestamp,
      sender: this.sender,
      historyCount: this.conversationHistory.length
    };
  }

  // --- CONTEXT MANAGEMENT ---
  setContext(context = 'hospital') {
    this.currentContext = context;
    this.notify(true);
  }

  // --- OBSERVER SUBSCRIPTION ---
  subscribe(listener) {
    this.listeners.add(listener);
    try {
      listener(this.getState());
    } catch (err) {
      console.error('[HospitalConversationService] Listener initialization error:', err);
    }
    return () => this.listeners.delete(listener);
  }

  notify(broadcast = true) {
    const state = this.getState();
    this.listeners.forEach((listener) => {
      try {
        listener(state);
      } catch (err) {
        console.error('[HospitalConversationService] Listener notification error:', err);
      }
    });

    if (broadcast) {
      communicationService.emit('HOSPITAL_SYNC', {
        context: this.currentContext,
        currentSign: this.currentSign,
        currentPhrase: this.currentPhrase,
        confidence: this.currentConfidence,
        timestamp: this.currentTimestamp,
        historyCount: this.conversationHistory.length
      });
    }
  }

  applyExternalState(payload) {
    if (payload.currentSign !== undefined) this.currentSign = payload.currentSign;
    if (payload.currentPhrase !== undefined) this.currentPhrase = payload.currentPhrase;
    if (payload.confidence !== undefined) this.currentConfidence = payload.confidence;
    if (payload.timestamp !== undefined) this.currentTimestamp = payload.timestamp;
    this.notify(false);
  }

  // --- CORE RECOGNITION CONVERSION PIPELINE ---

  /**
   * Process a recognized sign event from the ML pipeline.
   * Converts the recognized sign sequence into a controlled contextual hospital sentence.
   * Suppresses duplicate continuous holds and accumulates distinct sequential signs.
   *
   * @param {Object} event
   * @param {string} event.sign - e.g. "HELP", "YES", "NO", "PLEASE", "HELLO", "THANK_YOU"
   * @param {number} [event.confidence=0.9] - ML confidence score (e.g. 0.95)
   * @param {number} [event.timestamp=Date.now()] - Event epoch ms
   * @param {boolean} [event.sendToChat=true] - Whether to append to conversationStore & notify Admin
   * @returns {Object} Resulting conversation event
   */
  processSignEvent({ sign, confidence = 0.9, timestamp = Date.now(), sendToChat = true } = {}) {
    if (!sign) return null;

    const normalizedSign = String(sign).toUpperCase().trim();
    const isAllowedV3 = V3_ML_ALLOWED_SIGNS.includes(normalizedSign);
    const now = timestamp || Date.now();

    // 1. SEQUENCE TIMEOUT CHECK:
    // If more than sequenceTimeoutMs (6s) has elapsed since the last accepted sign,
    // clear the previous sequence and start a fresh one.
    if (this.lastAcceptedTimestamp > 0 && (now - this.lastAcceptedTimestamp) > this.sequenceTimeoutMs) {
      this.currentSignSequence = [];
      this.lastAcceptedSign = null;
    }

    // 2. DE-DUPLICATION / GESTURE-HOLD SUPPRESSION:
    // If the user continues to hold the same sign (e.g. HELLO, HELLO, HELLO...),
    // treat it as the same continuous gesture, NOT multiple events.
    if (normalizedSign === this.lastAcceptedSign && (now - this.lastAcceptedTimestamp) < this.gestureHoldCooldownMs) {
      this.lastAcceptedTimestamp = now; // refresh hold window
      return {
        isDuplicate: true,
        recognizedSign: this.currentSign,
        contextualMessage: this.currentPhrase,
        currentSequence: [...this.currentSignSequence],
        confidence: this.currentConfidence
      };
    }

    // 3. NEW DISTINCT SIGN TRANSITION:
    // User changed gestures (e.g. from HELLO to HELP), or started a new sequence.
    this.currentSignSequence.push(normalizedSign);
    // Keep max 4 signs in rolling sequence
    if (this.currentSignSequence.length > 4) {
      this.currentSignSequence.shift();
    }

    this.lastAcceptedSign = normalizedSign;
    this.lastAcceptedTimestamp = now;

    // 4. FORMAT RECOGNIZED SIGN SEQUENCE STRING (CARD 1: "Recognized Signs (ML Model)")
    // e.g. "HELLO -> HELP" or "HELP"
    const rawSignSequence = this.currentSignSequence.join(' -> ');

    // 5. RESOLVE DETERMINISTIC HOSPITAL CONTEXT SENTENCE (CARD 2: "Contextual Message (Phrase Layer)")
    const { sentence, matchedSequence } = resolveHospitalSentence(this.currentSignSequence);

    // Update state
    this.currentContext = 'hospital';
    this.currentSign = rawSignSequence;
    this.currentPhrase = sentence;
    this.currentConfidence = confidence;
    this.currentTimestamp = now;
    this.sender = 'deaf';

    const historyRecord = {
      id: `hosp_${now}_${Math.random().toString(36).substr(2, 4)}`,
      context: 'hospital',
      sender: 'deaf',
      senderName: 'Deaf Person',
      recognizedSign: rawSignSequence,
      rawSequence: [...this.currentSignSequence],
      contextualMessage: sentence,
      confidence: confidence,
      timestamp: now,
      source: 'ml_recognition',
      isAllowedV3Sign: isAllowedV3
    };

    this.conversationHistory.push(historyRecord);

    // 6. DISPATCH TO CHAT STREAM & ADMIN CONSOLE:
    if (sendToChat) {
      conversationStore.addMessage({
        sender: 'deaf',
        senderName: 'Deaf Person',
        text: sentence,
        rawSign: rawSignSequence,
        type: 'hospital_sentence',
        metadata: {
          recognizedSigns: rawSignSequence,
          rawSequence: [...this.currentSignSequence],
          confidence: confidence,
          source: 'ml_recognition'
        }
      });

      communicationService.emit('DEAF_MESSAGE_SENT', {
        text: sentence,
        rawSign: rawSignSequence,
        rawSequence: [...this.currentSignSequence],
        confidence: confidence,
        timestamp: now
      });

      communicationService.emit('HOSPITAL_PHRASE_EVENT', historyRecord);
      this.lastDispatchedSentence = sentence;
    }

    this.notify(true);
    return historyRecord;
  }

  /**
   * Direct Template Selection for phrases not covered by the 6 ML signs.
   * Explicitly sets recognizedSign to null so UI distinguishes template selection from ML recognition.
   *
   * @param {string} phraseId - e.g. "hosp_07" ("I need an appointment.")
   * @param {Object} options
   * @param {boolean} [options.sendToChat=true]
   * @returns {Object} Resulting conversation event
   */
  selectPhraseById(phraseId, { sendToChat = true } = {}) {
    const phrase = getPhraseById(phraseId);
    if (!phrase) {
      console.warn(`[HospitalConversationService] Unknown phraseId: ${phraseId}`);
      return null;
    }

    const now = Date.now();

    // Reset sequence buffer when manual phrase shortcut is clicked
    this.currentSignSequence = [];
    this.lastAcceptedSign = null;
    this.lastAcceptedTimestamp = 0;

    // Explicitly set recognizedSign = "— (Controlled Shortcut)" (NO direct ML claim)
    this.currentContext = 'hospital';
    this.currentSign = '— (Controlled Shortcut)';
    this.currentPhrase = phrase.displayText;
    this.currentConfidence = 1.0;
    this.currentTimestamp = now;
    this.sender = 'deaf';

    const historyRecord = {
      id: `hosp_tpl_${now}_${Math.random().toString(36).substr(2, 4)}`,
      context: 'hospital',
      sender: 'deaf',
      senderName: 'Deaf Person',
      recognizedSign: null, // CLEARLY DISTINGUISHED: Not ML recognized
      contextualMessage: phrase.displayText,
      phraseId: phrase.id,
      confidence: 1.0,
      timestamp: now,
      source: 'template_selection',
      isAllowedV3Sign: false
    };

    this.conversationHistory.push(historyRecord);

    if (sendToChat) {
      conversationStore.addMessage({
        sender: 'deaf',
        senderName: 'Deaf Person',
        text: phrase.displayText,
        rawSign: null,
        type: 'hospital_template_phrase',
        metadata: {
          recognizedSign: null,
          phraseId: phrase.id,
          confidence: 1.0,
          source: 'template_selection'
        }
      });

      communicationService.emit('DEAF_MESSAGE_SENT', {
        text: phrase.displayText,
        rawSign: '— (Controlled Shortcut)',
        phraseId: phrase.id,
        confidence: 1.0,
        timestamp: now
      });

      communicationService.emit('HOSPITAL_PHRASE_EVENT', historyRecord);
      this.lastDispatchedSentence = phrase.displayText;
    }

    this.notify(true);
    return historyRecord;
  }

  /**
   * Reset conversation state to initial clean state
   */
  reset() {
    this.currentSignSequence = [];
    this.lastAcceptedSign = null;
    this.lastAcceptedTimestamp = 0;
    this.lastDispatchedSentence = null;
    this.currentSign = '—';
    this.currentPhrase = 'Show a sign to begin';
    this.currentConfidence = null;
    this.currentTimestamp = Date.now();
    this.sender = 'deaf';
    this.conversationHistory = [];
    this.notify(false);
  }
}

export const hospitalConversationService = new HospitalConversationService();
