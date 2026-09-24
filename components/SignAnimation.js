// Animated Sign Language Response Component with Automatic 2x Playback Simulation
import { stateBridge } from '../services/stateBridge.js';

export class SignAnimationController {
  constructor(options = {}) {
    this.singleCycleDuration = options.duration || 4000; // 4 seconds per cycle
    this.maxCycles = 2;
    this.currentCycle = 1;
    this.startTime = Date.now();
    this.animFrame = null;
    this.isPaused = false;
    this.onFinish = options.onFinish || null;
  }

  start() {
    this.currentCycle = 1;
    this.startTime = Date.now();
    this.isPaused = false;
    this.update();
  }

  restart() {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.start();
  }

  stop() {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.isPaused = true;
  }

  update() {
    if (this.isPaused) return;

    const elapsed = Date.now() - this.startTime;
    const progressPercent = Math.min((elapsed / this.singleCycleDuration) * 100, 100);

    const progressBar = document.getElementById('playback-progress');
    const timeRemaining = document.getElementById('time-remaining');
    const cycleLabel = document.getElementById('cycle-label');
    const playbackBadge = document.getElementById('playback-badge');
    const playbackStatusText = document.getElementById('playback-status-text');

    if (progressBar) progressBar.style.width = progressPercent + '%';
    if (timeRemaining) {
      const sec = Math.min(Math.floor(elapsed / 1000), 4);
      timeRemaining.textContent = `00:0${sec} / 00:04`;
    }

    if (elapsed < this.singleCycleDuration) {
      this.animFrame = requestAnimationFrame(() => this.update());
    } else {
      // Completed current pass
      if (this.currentCycle < this.maxCycles) {
        this.currentCycle++;
        if (playbackStatusText) playbackStatusText.textContent = `Playing ${this.currentCycle} of ${this.maxCycles}`;
        if (cycleLabel) cycleLabel.textContent = `Cycle ${this.currentCycle} of ${this.maxCycles} (Playing)`;
        this.startTime = Date.now();
        this.animFrame = requestAnimationFrame(() => this.update());
      } else {
        // Both 2 playback passes finished!
        this.complete();
      }
    }
  }

  complete() {
    const progressBar = document.getElementById('playback-progress');
    const cycleLabel = document.getElementById('cycle-label');
    const playbackBadge = document.getElementById('playback-badge');

    if (progressBar) progressBar.style.width = '100%';
    if (cycleLabel) cycleLabel.textContent = 'Completed 2 of 2 iterations';
    if (playbackBadge) {
      playbackBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-900 border border-emerald-300 text-xs font-bold uppercase tracking-wide';
      playbackBadge.innerHTML = `
        <span class="material-symbols-outlined text-[16px] text-emerald-700">check_circle</span>
        <span>Complete (2 of 2)</span>
      `;
    }

    // Trigger green camera turn-taking
    stateBridge.finishAnimationPlayback();

    if (this.onFinish) {
      this.onFinish();
    }
  }
}

export function renderSignAnimation(activeAnimation) {
  const text = activeAnimation?.text || 'Please wait here. The doctor will examine you shortly.';
  const isPlaying = activeAnimation?.isPlaying !== false;

  return `
    <div class="flex flex-col h-full">
      <!-- Section Header & Playback Badge -->
      <div class="flex items-center justify-between pb-3.5 flex-shrink-0">
        <div>
          <h2 class="text-xl font-extrabold text-primary tracking-tight">Staff Sign Translation</h2>
          <p class="text-xs text-on-surface-variant mt-0.5 font-medium">Automated sign relay for officer's statement</p>
        </div>
        <div class="flex items-center gap-2">
          <div id="playback-badge" class="flex items-center gap-1.5 px-2.5 py-1 rounded-full ${activeAnimation?.completed ? 'bg-emerald-100 text-emerald-900 border border-emerald-300' : 'bg-primary-container text-secondary-container'} text-xs font-bold uppercase tracking-wide shadow-xs">
            ${activeAnimation?.completed ? `
              <span class="material-symbols-outlined text-[16px] text-emerald-700">check_circle</span>
              <span>Complete (2 of 2)</span>
            ` : `
              <span class="w-2 h-2 rounded-full bg-secondary animate-ping"></span>
              <span id="playback-status-text">Playing 1 of 2</span>
            `}
          </div>
          <button type="button" 
                  id="btn-replay-animation" 
                  class="p-1.5 rounded-lg text-on-surface-variant hover:bg-surface-container hover:text-primary transition-colors cursor-pointer"
                  title="Replay 2x Sign Cycle">
            <span class="material-symbols-outlined text-[20px]">replay</span>
          </button>
        </div>
      </div>

      <!-- AI Sign Language Avatar Player -->
      <div class="relative w-full flex-1 min-h-[240px] bg-primary-container rounded-xl overflow-hidden shadow-md flex items-center justify-center">
        <!-- AI Avatar Video Feed -->
        <img src="./assets/images/ai-avatar.jpg" 
             alt="AI Sign Language Avatar demonstrating ASL translation" 
             class="w-full h-full object-cover select-none opacity-95" />

        <!-- Overlay Minimal Playback Progress Bar -->
        <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-primary/95 via-primary/70 to-transparent p-3.5 flex flex-col gap-1.5 z-10">
          <div class="w-full bg-white/20 h-1.5 rounded-full overflow-hidden">
            <div id="playback-progress" class="bg-secondary h-full w-0 transition-all duration-100 ease-linear"></div>
          </div>
          <div class="flex items-center justify-between text-white text-xs font-mono">
            <span id="cycle-label">${activeAnimation?.completed ? 'Completed 2 of 2 iterations' : 'Cycle 1 of 2 (Playing)'}</span>
            <span id="time-remaining">00:00 / 00:04</span>
          </div>
        </div>
      </div>

      <!-- Synchronized Staff Message Text Box -->
      <div class="mt-3.5 p-4 bg-surface-container-low rounded-xl border border-outline-variant/40 flex flex-col gap-1.5 flex-shrink-0 shadow-xs">
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold uppercase tracking-wider text-on-surface-variant flex items-center gap-1.5">
            <span class="material-symbols-outlined text-primary text-[16px]">support_agent</span>
            Officer's Message
          </span>
          <span class="text-[11px] font-semibold px-2 py-0.5 rounded bg-surface-container text-on-surface-variant">
            Synchronized
          </span>
        </div>
        <p class="text-xl lg:text-2xl font-extrabold text-primary leading-snug break-words" id="sync-animation-text">
          "${text}"
        </p>
      </div>
    </div>
  `;
}
