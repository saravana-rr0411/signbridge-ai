// services/recognition/fastapiRecognitionAdapter.js - Future FastAPI ML Recognition Adapter
// Ready for drop-in integration with Python FastAPI (OpenCV / MediaPipe / Sign ML model)
// Protocol: Video Frame (Offscreen Canvas / Blob) -> WebSocket -> FastAPI -> Prediction JSON -> Frontend

import { BaseRecognitionAdapter } from './recognitionAdapter.js';

export class FastAPIRecognitionAdapter extends BaseRecognitionAdapter {
  constructor(endpoint = 'ws://localhost:8000/api/v1/stream-recognition') {
    super('FastAPIRecognitionAdapter', 'fastapi');
    this.endpoint = endpoint;
    this.websocket = null;
    this.frameCanvas = null;
    this.frameInterval = null;
    this.fps = 15; // 15 frames per second stream to Python server
    this.reconnectAttempts = 0;
  }

  start(videoElement, callbacks) {
    super.start(videoElement, callbacks);
    console.log(`[FastAPIRecognitionAdapter] Initializing connection to ${this.endpoint}...`);

    if (this.onStatus) {
      this.onStatus('Connecting to FastAPI ML backend...', 'CONNECTING');
    }

    this.connectWebSocket();
  }

  connectWebSocket() {
    try {
      this.websocket = new WebSocket(this.endpoint);

      this.websocket.onopen = () => {
        console.log('[FastAPIRecognitionAdapter] WebSocket connected to Python ML service.');
        this.reconnectAttempts = 0;
        if (this.onStatus) {
          this.onStatus('Recognition: Ready (FastAPI ML)', 'READY');
        }
        this.startFrameStreaming();
      };

      this.websocket.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data);
          // Expected Python FastAPI payload:
          // { text: "I need help", confidence: 0.94, isFinal: true, status: "CONFIRMED" }
          if (result && result.text && this.onPrediction) {
            this.onPrediction({
              text: result.text,
              confidence: result.confidence || 0.9,
              isFinal: Boolean(result.isFinal),
              status: result.status || (result.isFinal ? 'CONFIRMED' : 'RECOGNIZING')
            });
          }

          if (this.onStatus) {
            if (result.isFinal) {
              this.onStatus(`Detected: ${result.text}`, 'CONFIRMED');
            } else {
              this.onStatus('Recognizing...', 'RECOGNIZING');
            }
          }
        } catch (e) {
          console.warn('[FastAPIRecognitionAdapter] Error parsing backend message:', e);
        }
      };

      this.websocket.onerror = () => {
        console.warn(`[FastAPIRecognitionAdapter] Python ML service not running at ${this.endpoint}.`);
        if (this.onStatus) {
          this.onStatus('FastAPI: Offline (Please start Python ML backend)', 'OFFLINE');
        }
      };

      this.websocket.onclose = () => {
        this.stopFrameStreaming();
        if (this.isRunning && this.reconnectAttempts < 3) {
          this.reconnectAttempts++;
          setTimeout(() => this.connectWebSocket(), 3000);
        }
      };
    } catch (e) {
      console.warn('[FastAPIRecognitionAdapter] WebSocket setup error:', e);
      if (this.onStatus) {
        this.onStatus('FastAPI: Offline', 'OFFLINE');
      }
    }
  }

  startFrameStreaming() {
    if (!this.frameCanvas) {
      this.frameCanvas = document.createElement('canvas');
    }

    if (this.frameInterval) clearInterval(this.frameInterval);

    this.frameInterval = setInterval(() => {
      if (!this.isReady() || !this.websocket || this.websocket.readyState !== WebSocket.OPEN) return;

      const video = this.videoElement;
      this.frameCanvas.width = 320;
      this.frameCanvas.height = 240;
      const ctx = this.frameCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 320, 240);

      // Send compressed JPEG frame to FastAPI MediaPipe inference loop
      this.frameCanvas.toBlob((blob) => {
        if (blob && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          this.websocket.send(blob);
        }
      }, 'image/jpeg', 0.6);
    }, 1000 / this.fps);
  }

  stopFrameStreaming() {
    if (this.frameInterval) {
      clearInterval(this.frameInterval);
      this.frameInterval = null;
    }
  }

  stop() {
    this.stopFrameStreaming();
    if (this.websocket) {
      try { this.websocket.close(); } catch (e) {}
      this.websocket = null;
    }
    super.stop();
  }
}
