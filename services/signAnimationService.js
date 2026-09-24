// Sign Language Animation Service with Automatic 2X Playback & Turn-Taking Trigger
import { communicationService } from './communicationService.js';

class SignAnimationService {
  constructor() {
    this.listeners = new Set();
    this.singleCycleDuration = 4000; // 4s per playback pass
    this.timerId = null;
    this.startTime = 0;
    this.animationFrame = null;

    // Default active animation
    this.state = {
      text: 'Please wait here. The doctor will examine you shortly.',
      assetUrl: './assets/images/ai-avatar.jpg',
      cycle: 1,
      maxCycles: 2,
      isPlaying: false,
      completed: true,
      cameraResponseComplete: true,
      progressPercent: 100,
      timeDisplay: '00:04 / 00:04'
    };

    // Asset mappings for messages/presets
    this.assetMap = {
      default: './assets/images/ai-avatar.jpg'
    };
  }

  notify() {
    this.listeners.forEach((listener) => {
      try {
        listener(this.state);
      } catch (err) {
        console.error('Sign animation listener error:', err);
      }
    });

    communicationService.emit('ANIMATION_STATE_CHANGED', {
      cycle: this.state.cycle,
      isPlaying: this.state.isPlaying,
      completed: this.state.completed,
      cameraResponseComplete: this.state.cameraResponseComplete,
      text: this.state.text
    });
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  getState() {
    return this.state;
  }

  // Play animation for an incoming Admin message
  playAnimationForMessage(text) {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);

    this.state.text = text;
    this.state.cycle = 1;
    this.state.isPlaying = true;
    this.state.completed = false;
    this.state.cameraResponseComplete = false;
    this.state.progressPercent = 0;
    this.startTime = Date.now();

    // Camera becomes neutral during playback
    communicationService.emit('CAMERA_TURN_ACTIVE', { active: false });

    this.notify();
    this.runCycleLoop();
  }

  runCycleLoop() {
    const elapsed = Date.now() - this.startTime;
    const progress = Math.min((elapsed / this.singleCycleDuration) * 100, 100);
    const sec = Math.min(Math.floor(elapsed / 1000), 4);

    this.state.progressPercent = progress;
    this.state.timeDisplay = `00:0${sec} / 00:04`;
    this.notify();

    if (elapsed < this.singleCycleDuration) {
      this.animationFrame = requestAnimationFrame(() => this.runCycleLoop());
    } else {
      // Completed current cycle pass
      if (this.state.cycle < this.state.maxCycles) {
        this.state.cycle++;
        this.startTime = Date.now();
        this.notify();
        this.animationFrame = requestAnimationFrame(() => this.runCycleLoop());
      } else {
        // Automatic 2X Playback ENDED!
        this.finishPlayback();
      }
    }
  }

  finishPlayback() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);

    this.state.isPlaying = false;
    this.state.completed = true;
    this.state.cameraResponseComplete = true;
    this.state.progressPercent = 100;
    this.state.timeDisplay = '00:04 / 00:04';

    // Trigger green active camera state on Deaf Person interface!
    communicationService.emit('CAMERA_TURN_ACTIVE', { active: true });

    this.notify();
  }

  replay() {
    this.playAnimationForMessage(this.state.text);
  }

  stop() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.state.isPlaying = false;
    this.notify();
  }
}

export const signAnimationService = new SignAnimationService();
