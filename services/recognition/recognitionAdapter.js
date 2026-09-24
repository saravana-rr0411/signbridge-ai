// services/recognition/recognitionAdapter.js - Base Abstract Adapter Interface
// Establishes the contract for Sign Recognition adapters (Mock vs Future FastAPI ML Backend)

export class BaseRecognitionAdapter {
  constructor(name = 'BaseRecognitionAdapter', type = 'abstract') {
    this.name = name;
    this.type = type;
    this.videoElement = null;
    this.onPrediction = null; // Callback: ({ text, confidence, isFinal, status }) => {}
    this.onStatus = null;     // Callback: (statusText, statusCode) => {}
    this.isRunning = false;
  }

  // Connects adapter to the existing live video element without creating extra camera streams
  start(videoElement, { onPrediction, onStatus }) {
    this.videoElement = videoElement;
    this.onPrediction = onPrediction;
    this.onStatus = onStatus;
    this.isRunning = true;
  }

  stop() {
    this.isRunning = false;
    this.videoElement = null;
    this.onPrediction = null;
    this.onStatus = null;
  }

  // Optional manual simulation trigger (e.g. for mock or test harness)
  triggerPhrase(phrase) {
    // Implemented by subclasses
  }

  isReady() {
    return Boolean(
      this.isRunning &&
      this.videoElement &&
      this.videoElement.videoWidth > 0 &&
      !this.videoElement.paused
    );
  }
}
