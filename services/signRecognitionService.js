// services/signRecognitionService.js - Central Sign-to-Text Recognition Service
// Orchestrates active recognition adapter (Mock vs FastAPI), confirmation debounce, and message dispatch.
import { MockRecognitionAdapter, MOCK_RECOGNITION_PHRASES } from './recognition/mockRecognitionAdapter.js';
import { FastAPIRecognitionAdapter } from './recognition/fastapiRecognitionAdapter.js';
import { conversationStore } from '../state/conversationStore.js';
import { communicationService } from './communicationService.js';
import { hospitalConversationService } from './conversation/hospitalConversationService.js';

export const RECOGNIZED_PHRASES = [...MOCK_RECOGNITION_PHRASES];

class SignRecognitionService {
  constructor() {
    // Current adapter: Real FastAPI ML Recognition Adapter
    this.adapter = new FastAPIRecognitionAdapter();
    this.adapterType = 'fastapi'; // Real ML inference active

    this.currentTranscript = 'Show a sign to begin';
    this.rawSign = '—';
    this.contextualMessage = 'Hello, I need help.';
    this.statusText = 'Recognition: Ready';
    this.statusCode = 'READY'; // 'READY' | 'RECOGNIZING' | 'CONFIRMED' | 'WAITING' | 'OFFLINE'
    this.livePredictionText = 'Detecting...';
    this.isRecognizing = false;

    // Debounce & Duplicate Suppression State (Requirement 6)
    this.lastConfirmedText = null;
    this.debounceTimer = null;
    this.debounceWindowMs = 600; // 600ms stability requirement

    this.listeners = new Set();
    this.videoElement = null;

    // Sync across tabs
    communicationService.on('TRANSCRIPT_SYNC', (payload) => {
      if (payload) {
        if (payload.transcript) this.currentTranscript = payload.transcript;
        if (payload.rawSign) this.rawSign = payload.rawSign;
        if (payload.contextualMessage) this.contextualMessage = payload.contextualMessage;
        if (payload.statusText) this.statusText = payload.statusText;
        if (payload.statusCode) this.statusCode = payload.statusCode;
        if (payload.livePredictionText) this.livePredictionText = payload.livePredictionText;
        this.isRecognizing = Boolean(payload.isRecognizing);
        this.notify(false);
      }
    });

    communicationService.on('DEMO_RESET', () => {
      this.reset();
    });
  }

  notify(broadcast = true) {
    const state = this.getCurrentPrediction();
    this.listeners.forEach((listener) => {
      try {
        listener(state);
      } catch (err) {
        console.error('Sign recognition listener error:', err);
      }
    });

    if (broadcast) {
      communicationService.emit('TRANSCRIPT_SYNC', {
        transcript: this.currentTranscript,
        rawSign: this.rawSign,
        contextualMessage: this.contextualMessage,
        statusText: this.statusText,
        statusCode: this.statusCode,
        livePredictionText: this.livePredictionText,
        isRecognizing: this.isRecognizing
      });
    }
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.getCurrentPrediction());
    return () => this.listeners.delete(listener);
  }

  getCurrentPrediction() {
    return {
      transcript: this.currentTranscript,
      rawSign: this.rawSign || '—',
      contextualMessage: this.contextualMessage || this.currentTranscript,
      statusText: this.statusText,
      statusCode: this.statusCode,
      livePredictionText: this.livePredictionText,
      isRecognizing: this.isRecognizing,
      adapterType: this.adapterType
    };
  }

  getTranscript() {
    return this.currentTranscript;
  }

  getRawSign() {
    return this.rawSign;
  }

  getContextualMessage() {
    return this.contextualMessage;
  }

  getStatusText() {
    return this.statusText;
  }

  getLivePredictionText() {
    return this.livePredictionText;
  }

  // 1. START RECOGNITION (Attached to existing live video element)
  startRecognition(videoElement) {
    if (!videoElement) return;
    this.videoElement = videoElement;
    this.livePredictionText = 'Detecting...';
    this.notify(true);

    this.adapter.start(videoElement, {
      onPrediction: (pred) => this.handleAdapterPrediction(pred),
      onStatus: (text, code) => this.handleAdapterStatus(text, code)
    });
  }

  // 2. STOP RECOGNITION
  stopRecognition() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    this.adapter.stop();
    this.videoElement = null;
    this.isRecognizing = false;
    this.statusText = 'Recognition: Stopped';
    this.statusCode = 'STOPPED';
    this.livePredictionText = 'Detecting...';
    this.notify(false);
  }

  // 3. ADAPTER STATUS CALLBACK
  handleAdapterStatus(text, code) {
    this.statusText = text;
    this.statusCode = code;
    this.isRecognizing = (code === 'RECOGNIZING');
    this.notify(true);
  }

  // 4. CENTRAL DISPATCHER FOR ALL DEAF-ORIGINATED MESSAGES (Direct or Fallback)
  dispatchDeafMessage(text, isDemo = false) {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    const signName = cleanText.toUpperCase();

    // Process through Hospital Conversation Sequence Layer
    const event = hospitalConversationService.processSignEvent({
      sign: signName,
      confidence: 0.95,
      sendToChat: true
    });

    const contextualSentence = (event && event.contextualMessage) ? event.contextualMessage : cleanText;
    const rawSignSeq = (event && event.recognizedSign) ? event.recognizedSign : cleanText;

    this.currentTranscript = contextualSentence;
    this.rawSign = rawSignSeq;
    this.contextualMessage = contextualSentence;
    this.livePredictionText = `Prediction: ${cleanText}`;
    this.statusText = `Detected: ${cleanText}`;
    this.statusCode = 'CONFIRMED';
    this.isRecognizing = false;

    this.notify(true);
  }

  // 5. ADAPTER PREDICTION CALLBACK (From Real MediaPipe + FastAPI Bi-GRU)
  handleAdapterPrediction(prediction) {
    if (!prediction) return;

    if (prediction.isLive && prediction.predictionText) {
      this.livePredictionText = prediction.predictionText;
    }

    if (prediction.isUncertain) {
      this.statusText = 'Uncertain';
      this.statusCode = 'UNCERTAIN';
      this.notify(true);
      return;
    }

    if (!prediction.text) {
      // In-flight recognizing / stabilizing
      if (prediction.signName) {
        this.statusText = `Recognizing: ${prediction.signName}...`;
        this.statusCode = 'RECOGNIZING';
        this.isRecognizing = true;
      }
      this.notify(true);
      return;
    }

    const cleanText = prediction.text.trim();
    const signName = (prediction.signName || cleanText).toUpperCase();

    // Process through Hospital Conversation Sequence Layer (accumulates & de-duplicates)
    const event = hospitalConversationService.processSignEvent({
      sign: signName,
      confidence: prediction.confidence || 0.95,
      sendToChat: true
    });

    // Update UI indicators
    this.livePredictionText = `Prediction: ${signName}`;
    this.statusText = `Detected: ${signName}`;
    this.statusCode = 'CONFIRMED';
    this.isRecognizing = false;

    if (event) {
      this.currentTranscript = event.contextualMessage;
      this.rawSign = event.recognizedSign;
      this.contextualMessage = event.contextualMessage;
    }

    // Temporal suppression for rapid frame-by-frame streaming
    if (cleanText === this.lastConfirmedText) {
      this.notify(true);
      return;
    }
    this.lastConfirmedText = cleanText;
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.lastConfirmedText = null;
    }, 2500);

    this.notify(true);
  }

  // 6. DEMO SIGN BUTTON: Immediately dispatches message without requiring camera/FastAPI
  recognizePhrase(phrase) {
    if (!phrase) return;
    this.dispatchDeafMessage(phrase, true);
  }

  // Switch adapter (Mock vs future FastAPI backend)
  switchAdapter(type = 'mock', endpoint) {
    const wasRunning = this.adapter.isRunning;
    const currentVideo = this.videoElement;

    this.adapter.stop();

    if (type === 'fastapi') {
      this.adapter = new FastAPIRecognitionAdapter(endpoint);
      this.adapterType = 'fastapi';
    } else {
      this.adapter = new MockRecognitionAdapter();
      this.adapterType = 'mock';
    }

    if (wasRunning && currentVideo) {
      this.startRecognition(currentVideo);
    }
  }

  setModelMode(mode = 'v3_six_sign') {
    if (this.adapterType === 'fastapi') {
      const wasRunning = this.adapter.isRunning;
      const currentVideo = this.videoElement;
      this.adapter.stop();
      this.adapter = new FastAPIRecognitionAdapter(null, mode);
      if (wasRunning && currentVideo) {
        this.startRecognition(currentVideo);
      }
    }
  }

  reset() {
    this.currentTranscript = 'Show a sign to begin';
    this.rawSign = '—';
    this.contextualMessage = 'Hello, I need help.';
    this.statusText = 'Recognition: Ready';
    this.statusCode = 'READY';
    this.livePredictionText = 'Detecting...';
    this.isRecognizing = false;
    this.lastConfirmedText = null;
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    hospitalConversationService.reset();
    this.notify(false);
  }

  getAvailablePhrases() {
    return this.adapter.getAvailablePhrases ? this.adapter.getAvailablePhrases() : RECOGNIZED_PHRASES;
  }
}

export const signRecognitionService = new SignRecognitionService();
