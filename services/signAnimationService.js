// Sign Language Animation Service with Procedural 3D Playback & Turn-Taking Trigger
import { communicationService } from './communicationService.js';
import { getPhraseDuration } from './signAnimation/engine.js';
import { SIGNING_MODES, getSignConfig } from './signAnimation/signConfig.js';

const safeRequestAnim = typeof requestAnimationFrame === 'function'
  ? requestAnimationFrame
  : (cb) => setTimeout(cb, 16);

const safeCancelAnim = typeof cancelAnimationFrame === 'function'
  ? cancelAnimationFrame
  : (id) => clearTimeout(id);

class SignAnimationService {
  constructor() {
    this.listeners = new Set();
    this.viewer = null;
    this.animationFrame = null;
    this.startTime = 0;
    this.duration = 3000;

    // Default active animation state
    this.state = {
      text: 'Please wait here. The doctor will examine you shortly.',
      mode: SIGNING_MODES.FINGERSPELLING,
      signSequence: [],
      cycle: 1,
      maxCycles: 2,
      isPlaying: false,
      completed: true,
      cameraResponseComplete: true,
      progressPercent: 100,
      timeDisplay: '00:03 / 00:03',
      activeToken: '',
      isDirectWord: false,
      supported: true
    };
  }

  attachViewer(viewerInstance) {
    this.viewer = viewerInstance;
    if (this.viewer) {
      this.viewer.onProgressCallback = (progress) => {
        this.updateProgress(progress);
      };
      this.viewer.onTokenChangeCallback = (info) => {
        this.state.activeToken = info.token || '';
        this.state.isDirectWord = Boolean(info.isDirectWord);
        this.notify();
      };
      this.viewer.onCompletedCallback = () => {
        this.finishPlayback();
      };
    }
  }

  detachViewer() {
    this.viewer = null;
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
      text: this.state.text,
      mode: this.state.mode,
      activeToken: this.state.activeToken
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

  updateProgress(progressRatio) {
    const pct = Math.min(100, Math.max(0, Math.round(progressRatio * 100)));
    if (pct === this.state.progressPercent) return;
    const totalSec = Math.round(this.duration / 1000);
    const curSec = Math.min(totalSec, Math.round((pct / 100) * totalSec));
    this.state.progressPercent = pct;
    this.state.timeDisplay = `00:0${curSec} / 00:0${totalSec}`;
    this.notify();
  }

  // Play animation for an incoming Admin message
  playAnimationForMessage(text, mode = SIGNING_MODES.FINGERSPELLING, signSequence = null) {
    // 1. Cancel previous sequence cleanly to avoid race conditions
    if (this.animationFrame) safeCancelAnim(this.animationFrame);
    if (this.viewer) {
      this.viewer.stop();
    }

    const config = getSignConfig(text, mode);
    const estDurationSec = getPhraseDuration(text, mode);
    this.duration = Math.max(2000, Math.round(estDurationSec * 1000));

    this.state.text = text;
    this.state.mode = mode;
    this.state.signSequence = signSequence || config.signSequence || [];
    this.state.supported = config.supported;
    this.state.cycle = 1;
    this.state.maxCycles = 2;
    this.state.isPlaying = config.supported;
    this.state.completed = !config.supported;
    this.state.cameraResponseComplete = !config.supported;
    this.state.progressPercent = 0;
    this.state.activeToken = '';
    this.state.isDirectWord = false;
    this.startTime = Date.now();

    // 2. Camera becomes neutral during animation playback
    communicationService.emit('CAMERA_TURN_ACTIVE', { active: false });
    this.notify();

    // 3. Drive 3D Viewer if attached
    if (this.viewer) {
      this.viewer.play(text, mode);
    } else {
      // Fallback timer if viewer is not mounted (e.g. headless unit tests)
      this.runFallbackTimer();
    }
  }

  runFallbackTimer() {
    const elapsed = Date.now() - this.startTime;
    const progress = Math.min((elapsed / this.duration) * 100, 100);
    const totalSec = Math.round(this.duration / 1000);
    const sec = Math.min(Math.floor(elapsed / 1000), totalSec);

    this.state.progressPercent = progress;
    this.state.timeDisplay = `00:0${sec} / 00:0${totalSec}`;
    this.notify();

    if (elapsed < this.duration) {
      this.animationFrame = safeRequestAnim(() => this.runFallbackTimer());
    } else {
      this.finishPlayback();
    }
  }

  finishPlayback() {
    if (this.animationFrame) safeCancelAnim(this.animationFrame);

    this.state.isPlaying = false;
    this.state.completed = true;
    this.state.cameraResponseComplete = true;
    this.state.progressPercent = 100;
    const totalSec = Math.round(this.duration / 1000);
    this.state.timeDisplay = `00:0${totalSec} / 00:0${totalSec}`;

    // Trigger green active camera turn-taking on Deaf interface
    communicationService.emit('CAMERA_TURN_ACTIVE', { active: true });
    this.notify();
  }

  replay() {
    this.playAnimationForMessage(this.state.text, this.state.mode, this.state.signSequence);
  }

  stop() {
    if (this.animationFrame) safeCancelAnim(this.animationFrame);
    if (this.viewer) {
      this.viewer.stop();
    }
    this.state.isPlaying = false;
    this.notify();
  }
}

export const signAnimationService = new SignAnimationService();
