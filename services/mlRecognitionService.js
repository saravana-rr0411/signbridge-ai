// services/mlRecognitionService.js - Real-Time FastAPI ML Recognition Interface
// Architecture:
// 1. Receives existing MediaPipe landmark sequence from live webcam.
// 2. Maintains a rolling 30-frame buffer.
// 3. Once 30 frames are available, sends landmark sequence to POST /predict/sequence.
// 4. Receives prediction and stabilizes across consecutive temporal windows.
// 5. Converts accepted prediction into communication event for Deaf & Admin interfaces.
// 6. Reuses the SAME existing camera stream without requesting a second camera stream.

import { signRecognitionService } from './signRecognitionService.js';
import { FastAPIRecognitionAdapter } from './recognition/fastapiRecognitionAdapter.js';
import { API_ENDPOINTS, API_BASE_URL } from './apiConfig.js';

export class MLRecognitionService {
  constructor() {
    this.adapter = new FastAPIRecognitionAdapter(API_ENDPOINTS.PREDICT_SEQUENCE);
  }

  start(videoElement) {
    return signRecognitionService.startRecognition(videoElement);
  }

  stop() {
    return signRecognitionService.stopRecognition();
  }

  getCurrentPrediction() {
    return signRecognitionService.getCurrentPrediction();
  }

  subscribe(listener) {
    return signRecognitionService.subscribe(listener);
  }

  triggerPhrase(phrase) {
    return signRecognitionService.recognizePhrase(phrase);
  }

  getApiBaseUrl() {
    return API_BASE_URL;
  }
}

export const mlRecognitionService = new MLRecognitionService();
