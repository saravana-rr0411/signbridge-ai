// Sign Animation Component — Re-exports and wraps the 3D SignAnimationViewer
import { renderSignAnimationViewer, SignAnimationViewer } from './sign/SignAnimationViewer.js';
import { stateBridge } from '../services/stateBridge.js';
import { signAnimationService } from '../services/signAnimationService.js';

export { SignAnimationViewer, renderSignAnimationViewer };

export class SignAnimationController {
  constructor(options = {}) {
    this.viewer = options.viewer || null;
    this.duration = options.duration || 3000;
  }

  start() {
    signAnimationService.replay();
  }

  restart() {
    signAnimationService.replay();
  }

  stop() {
    signAnimationService.stop();
  }

  complete() {
    signAnimationService.finishPlayback();
  }
}

/**
 * Renders the Staff Sign Translation component for the Deaf Page middle panel.
 * Uses 3D procedural xbot avatar viewer with Fingerspelling, ASL, and ISL support.
 */
export function renderSignAnimation(activeAnimation) {
  return renderSignAnimationViewer(activeAnimation);
}
