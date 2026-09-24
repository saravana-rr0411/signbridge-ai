// cameraService.js - Centralized Live Browser Camera & WebRTC Feed Service
// Automatically requests getUserMedia on mount, mirrors local preview, manages stream lifecycle,
// and delegates WebRTC peer-to-peer streaming to webrtcService.
import { webrtcService } from './webrtcService.js';

class CameraService {
  constructor() {
    this.stream = null;
    this.localVideoElement = null;
    this.status = 'IDLE'; // 'IDLE' | 'REQUESTING' | 'ACTIVE' | 'STOPPED' | 'DENIED' | 'UNAVAILABLE' | 'ERROR'
    this.errorMessage = null;
    this.listeners = new Set();
  }

  notify() {
    this.listeners.forEach((listener) => {
      try {
        listener({
          status: this.status,
          stream: this.stream,
          errorMessage: this.errorMessage
        });
      } catch (err) {
        console.error('Camera listener error:', err);
      }
    });
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener({
      status: this.status,
      stream: this.stream,
      errorMessage: this.errorMessage
    });
    return () => this.listeners.delete(listener);
  }

  // 1. DEAF PERSON CAMERA: AUTOMATICALLY REQUEST & START LIVE FEED
  async startCamera(videoElement) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.status = 'NOT_SUPPORTED';
      this.errorMessage = 'Web camera access is not supported by your browser.';
      this.notify();
      return { success: false, error: this.errorMessage };
    }

    // Do NOT request a second camera stream if one already exists and is active
    if (this.stream && this.stream.active) {
      const activeTracks = this.stream.getVideoTracks().filter(t => t.readyState === 'live');
      if (activeTracks.length > 0) {
        if (videoElement && videoElement.srcObject !== this.stream) {
          this.localVideoElement = videoElement;
          videoElement.srcObject = this.stream;
          videoElement.autoplay = true;
          videoElement.playsInline = true;
          videoElement.muted = true;
          videoElement.play().catch(() => {});
        }
        // Ensure WebRTC peer relay is published even when reusing stream
        webrtcService.publishStream(this.stream, 'cameraService.startCamera_reuse');
        return { success: true, stream: this.stream };
      }
    }

    try {
      this.status = 'REQUESTING';
      this.errorMessage = null;
      this.notify();

      // Request actual hardware webcam video
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      this.stream = stream;
      this.status = 'ACTIVE';
      this.errorMessage = null;

      if (videoElement) {
        this.localVideoElement = videoElement;
        videoElement.srcObject = stream;
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;
        // Local preview mirrored horizontally as specified
        videoElement.style.transform = 'scaleX(-1)';
        videoElement.style.objectFit = 'cover';
        await videoElement.play().catch(() => {});
      }

      // Publish outgoing stream to WebRTC service for cross-window Admin relay
      webrtcService.publishStream(stream, 'cameraService.startCamera_fresh');

      this.notify();
      return { success: true, stream };
    } catch (err) {
      console.warn('Camera access error:', err.name, err.message);

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        this.status = 'DENIED';
        this.errorMessage = 'Camera access was denied. Please allow camera permissions in your browser address bar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        this.status = 'UNAVAILABLE';
        this.errorMessage = 'No camera device found. Please connect a webcam.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        this.status = 'UNAVAILABLE';
        this.errorMessage = 'Camera is currently in use by another application.';
      } else {
        this.status = 'ERROR';
        this.errorMessage = 'Could not access the camera. Please check your browser device permissions.';
      }

      this.notify();
      return { success: false, error: this.errorMessage };
    }
  }

  // 2. STOP CAMERA & CLEANUP TRACKS (CALLED ON UNMOUNT)
  stopCamera(reason = 'cameraService.stopCamera') {
    // Unpublish from WebRTC service with exact caller and reason
    webrtcService.unpublishStream('cameraService.stopCamera', reason);

    if (this.stream) {
      this.stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {}
      });
      this.stream = null;
    }

    if (this.localVideoElement) {
      this.localVideoElement.srcObject = null;
      this.localVideoElement = null;
    }

    this.status = 'STOPPED';
    this.notify();
  }

  // 3. ADMIN PAGE LIVE STREAM CONNECTION (DELEGATED TO WEBRTC SERVICE)
  connectAdminFeed(videoElement, onStateChange) {
    if (!videoElement) return;

    // If stream is active in memory (e.g. single tab SPA test)
    if (this.stream && this.stream.active) {
      videoElement.srcObject = this.stream;
      videoElement.autoplay = true;
      videoElement.playsInline = true;
      videoElement.muted = true;
      videoElement.style.transform = 'none';
      videoElement.style.objectFit = 'cover';
      videoElement.play().catch(() => {});
      if (onStateChange) onStateChange(true);
    }

    // Subscribe to cross-window WebRTC stream
    webrtcService.subscribeStream(videoElement, (isStreaming) => {
      if (onStateChange) onStateChange(isStreaming);
    });
  }

  disconnectAdminFeed(reason = 'cameraService.disconnectAdminFeed') {
    webrtcService.unsubscribeStream(reason);
  }

  getStream() {
    return this.stream;
  }

  isStreaming() {
    return Boolean(this.status === 'ACTIVE' && this.stream && this.stream.active);
  }
}

export const cameraService = new CameraService();
