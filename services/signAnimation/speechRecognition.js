class SpeechRecognitionService {
  recognition = null;
  isListening = false;
  callbacks = {};
  constructor() {
    this.initRecognition();
  }
  /**
   * Check if speech recognition is supported in current browser
   */
  isSupported() {
    if (typeof window === "undefined") return false;
    const win = window;
    return !!(win.SpeechRecognition || win.webkitSpeechRecognition);
  }
  initRecognition() {
    if (!this.isSupported()) return;
    try {
      const win = window;
      const SpeechRecognitionClass = win.SpeechRecognition || win.webkitSpeechRecognition;
      this.recognition = new SpeechRecognitionClass();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = "en-US";
      this.recognition.maxAlternatives = 1;
      this.recognition.onstart = () => {
        this.isListening = true;
        this.callbacks.onStart?.();
      };
      this.recognition.onresult = (event) => {
        let interimTranscript = "";
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const result = event.results[i];
          const transcript = result[0].transcript;
          if (result.isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        if (finalTranscript) {
          this.callbacks.onResult?.(finalTranscript.trim(), true);
        } else if (interimTranscript) {
          this.callbacks.onResult?.(interimTranscript.trim(), false);
        }
      };
      this.recognition.onerror = (event) => {
        const errorType = event.error;
        let humanMessage = `Speech recognition error: ${errorType}`;
        switch (errorType) {
          case "not-allowed":
          case "permission-denied":
            humanMessage = "Microphone access denied. Please grant microphone permission in your browser address bar.";
            break;
          case "no-speech":
            humanMessage = "No speech detected. Please speak clearly into your microphone.";
            break;
          case "audio-capture":
            humanMessage = "No microphone was found. Ensure that a microphone is plugged in and configured.";
            break;
          case "network":
            humanMessage = "Speech recognition network error. Please verify your internet connection.";
            break;
          case "aborted":
            humanMessage = "Speech recognition was stopped.";
            break;
          default:
            humanMessage = `Speech recognition error (${errorType}). Please try again.`;
        }
        this.callbacks.onError?.(humanMessage);
      };
      this.recognition.onend = () => {
        this.isListening = false;
        this.callbacks.onEnd?.();
      };
    } catch (err) {
      console.error("Failed to initialize SpeechRecognition:", err);
    }
  }
  /**
   * Start listening for voice input
   */
  start(callbacks) {
    if (!this.isSupported()) {
      callbacks.onError?.(
        "Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari, or use text input."
      );
      return false;
    }
    if (!this.recognition) {
      this.initRecognition();
    }
    if (!this.recognition) {
      callbacks.onError?.("Failed to start speech recognition engine.");
      return false;
    }
    this.callbacks = callbacks;
    try {
      if (this.isListening) {
        this.recognition.stop();
      }
      this.recognition.start();
      return true;
    } catch (err) {
      if (err.name === "InvalidStateError") {
        this.recognition.stop();
        setTimeout(() => {
          try {
            this.recognition.start();
          } catch (e) {
            this.callbacks.onError?.("Could not activate speech recognition.");
          }
        }, 150);
        return true;
      }
      this.callbacks.onError?.(`Speech start error: ${err.message || "Unknown error"}`);
      return false;
    }
  }
  /**
   * Stop listening
   */
  stop() {
    if (this.recognition && this.isListening) {
      try {
        this.recognition.stop();
      } catch (err) {
        console.warn("Error stopping recognition:", err);
      }
      this.isListening = false;
    }
  }
  /**
   * Abort listening immediately
   */
  abort() {
    if (this.recognition) {
      try {
        this.recognition.abort();
      } catch (err) {
        console.warn("Error aborting recognition:", err);
      }
      this.isListening = false;
    }
  }
  getIsListening() {
    return this.isListening;
  }
}
const speechService = new SpeechRecognitionService();
export {
  speechService
};
