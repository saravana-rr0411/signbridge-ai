// services/signRecognitionService.js - Central Sign-to-Text Recognition Service
// Orchestrates active recognition adapter (Mock vs FastAPI), confirmation debounce, and message dispatch.
import { MockRecognitionAdapter, MOCK_RECOGNITION_PHRASES } from './recognition/mockRecognitionAdapter.js';
import { FastAPIRecognitionAdapter } from './recognition/fastapiRecognitionAdapter.js';
import { conversationStore } from '../state/conversationStore.js';
import { communicationService } from './communicationService.js';

export const RECOGNIZED_PHRASES = [...MOCK_RECOGNITION_PHRASES];

class SignRecognitionService {
  constructor() {
    // Current adapter (defaults to isolated Mock adapter)
    this.adapter = new MockRecognitionAdapter();
    this.adapterType = 'mock'; // 'mock' | 'fastapi'

    this.currentTranscript = 'I need help';
    this.statusText = 'Recognition: Ready';
    this.statusCode = 'READY'; // 'READY' | 'RECOGNIZING' | 'CONFIRMED' | 'WAITING' | 'OFFLINE'
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
        if (payload.statusText) this.statusText = payload.statusText;
        if (payload.statusCode) this.statusCode = payload.statusCode;
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
        statusText: this.statusText,
        statusCode: this.statusCode,
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
      statusText: this.statusText,
      statusCode: this.statusCode,
      isRecognizing: this.isRecognizing,
      adapterType: this.adapterType
    };
  }

  getTranscript() {
    return this.currentTranscript;
  }

  getStatusText() {
    return this.statusText;
  }

  // 1. START RECOGNITION (Attached to existing live video element)
  startRecognition(videoElement) {
    if (!videoElement) return;
    this.videoElement = videoElement;

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
    this.notify(false);
  }

  // 3. ADAPTER STATUS CALLBACK
  handleAdapterStatus(text, code) {
    this.statusText = text;
    this.statusCode = code;
    this.isRecognizing = (code === 'RECOGNIZING');
    this.notify(true);
  }

  // 4. ADAPTER PREDICTION CALLBACK & CONFIRMATION DEBOUNCE (Requirement 5 & 6)
  handleAdapterPrediction(prediction) {
    if (!prediction || !prediction.text) return;
    const cleanText = prediction.text.trim();

    this.currentTranscript = cleanText;

    if (!prediction.isFinal) {
      this.statusText = 'Recognizing...';
      this.statusCode = 'RECOGNIZING';
      this.isRecognizing = true;
      this.notify(true);
      return;
    }

    // Final prediction confirmed by vision pipeline
    this.statusText = `Detected: ${cleanText}`;
    this.statusCode = 'CONFIRMED';
    this.isRecognizing = false;
    this.notify(true);

    // DEBOUNCE / DUPLICATE SUPPRESSION:
    // Avoid sending the same text repeatedly if prediction didn't change
    if (cleanText === this.lastConfirmedText) {
      console.log('[SignRecognitionService] Duplicate prediction suppressed:', cleanText);
      return;
    }

    // Set lock to prevent duplicate frames from triggering duplicate messages
    this.lastConfirmedText = cleanText;

    // 1. Add ONE confirmed Deaf Person message to shared conversationStore
    conversationStore.addMessage({
      sender: 'deaf',
      senderName: 'Deaf Person',
      text: cleanText,
      type: 'sign'
    });

    // 2. Transmit to Admin console via communication event bus
    communicationService.emit('DEAF_RECOGNITION_SENT', {
      text: cleanText,
      timestamp: Date.now()
    });

    // Reset lock after 4 seconds to allow re-signing the same phrase in a later turn if desired
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.lastConfirmedText = null;
    }, 4000);
  }

  // Trigger gesture recognition (delegated to active adapter)
  recognizePhrase(phrase) {
    if (!phrase) return;
    this.adapter.triggerPhrase(phrase);
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

  reset() {
    this.currentTranscript = 'I need help';
    this.statusText = 'Recognition: Ready';
    this.statusCode = 'READY';
    this.isRecognizing = false;
    this.lastConfirmedText = null;
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    this.notify(false);
  }

  getAvailablePhrases() {
    return this.adapter.getAvailablePhrases ? this.adapter.getAvailablePhrases() : RECOGNIZED_PHRASES;
  }
}

export const signRecognitionService = new SignRecognitionService();
