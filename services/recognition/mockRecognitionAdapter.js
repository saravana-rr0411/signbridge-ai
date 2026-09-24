// services/recognition/mockRecognitionAdapter.js - Mock Sign Recognition Pipeline Adapter
// Simulates predictions from the existing live webcam video pipeline
// Explicitly isolated from UI components and easily swappable with FastAPI backend

import { BaseRecognitionAdapter } from './recognitionAdapter.js';

export const MOCK_RECOGNITION_PHRASES = [
  'I need help',
  'I need a doctor',
  'Where is the counter?',
  'I have an appointment',
  'I want to open an account',
  'Please help me',
  'I don\'t understand',
  'Here is my ticket: #A-204.',
  'Okay, thank you.'
];

export class MockRecognitionAdapter extends BaseRecognitionAdapter {
  constructor() {
    super('MockRecognitionAdapter', 'mock');
    this.phrases = [...MOCK_RECOGNITION_PHRASES];
    this.currentIndex = 0;
    this.isProcessing = false;
    this.frameCheckTimer = null;
  }

  start(videoElement, callbacks) {
    super.start(videoElement, callbacks);
    console.log('[MockRecognitionAdapter] Attached to existing live video element without creating duplicate stream.');

    // Start video liveness watcher to verify video is receiving frames
    if (this.frameCheckTimer) clearInterval(this.frameCheckTimer);
    this.frameCheckTimer = setInterval(() => {
      if (this.isReady()) {
        if (!this.isProcessing && this.onStatus) {
          // Status: "Recognition: Ready"
          this.onStatus('Recognition: Ready', 'READY');
        }
      } else if (this.onStatus) {
        this.onStatus('Recognition: Waiting for camera...', 'WAITING');
      }
    }, 1000);

    if (this.onStatus) {
      this.onStatus('Recognition: Ready', 'READY');
    }
  }

  stop() {
    if (this.frameCheckTimer) {
      clearInterval(this.frameCheckTimer);
      this.frameCheckTimer = null;
    }
    this.isProcessing = false;
    super.stop();
    console.log('[MockRecognitionAdapter] Stopped.');
  }

  // Simulates token-by-token vision inference stream from the current live video feed
  async triggerPhrase(phrase) {
    if (!phrase || this.isProcessing) return;
    const cleanPhrase = phrase.trim();
    this.isProcessing = true;

    if (this.onStatus) {
      this.onStatus('Recognizing...', 'RECOGNIZING');
    }

    const words = cleanPhrase.split(' ');
    let progressive = '';

    for (let i = 0; i < words.length; i++) {
      progressive += (i > 0 ? ' ' : '') + words[i];

      if (this.onPrediction) {
        this.onPrediction({
          text: progressive,
          rawPhrase: cleanPhrase,
          confidence: +(0.88 + Math.random() * 0.1).toFixed(2),
          isFinal: false,
          status: 'RECOGNIZING'
        });
      }

      // 180ms delay per token to emulate computer vision token inference
      await new Promise((r) => setTimeout(r, 180));
    }

    // Final completed phrase prediction
    if (this.onPrediction) {
      this.onPrediction({
        text: cleanPhrase,
        rawPhrase: cleanPhrase,
        confidence: +(0.95 + Math.random() * 0.04).toFixed(2),
        isFinal: true,
        status: 'CONFIRMED'
      });
    }

    if (this.onStatus) {
      this.onStatus(`Detected: ${cleanPhrase}`, 'CONFIRMED');
    }

    this.isProcessing = false;

    // Reset status to "Recognition: Ready" after 2.5s display window
    setTimeout(() => {
      if (!this.isProcessing && this.isRunning && this.onStatus) {
        this.onStatus('Recognition: Ready', 'READY');
      }
    }, 2500);
  }

  getAvailablePhrases() {
    return this.phrases;
  }
}
