// LiveCamera.js - Real Device Live Camera Component (Zero Templates, Fully Automated)
// Displays pure HTML5 video element with automated webcam stream, clean borders,
// no "Start Camera" buttons, and subtle emerald glow ONLY on 2x Admin response completion.

export function renderLiveCamera({
  isDeafView = true,
  title = 'Sign Camera',
  isTurnActive = false,
  turnStatus = 'Observing Signer'
}) {
  const containerId = isDeafView ? 'deaf-camera-container' : 'admin-camera-container';
  const badgeId = isDeafView ? 'deaf-turn-badge' : 'admin-turn-badge';

  const borderClass = isTurnActive
    ? 'camera-turn-active border-2 border-emerald-500 shadow-[0_0_16px_rgba(16,185,129,0.3)]'
    : 'border border-outline-variant/30';

  const badgeClass = isTurnActive
    ? 'bg-emerald-100 text-emerald-900 border border-emerald-300'
    : 'bg-surface-container text-on-surface-variant';

  const dotClass = isTurnActive
    ? 'w-2 h-2 rounded-full bg-emerald-500 animate-ping'
    : 'w-2 h-2 rounded-full bg-outline-variant';

  return `
    <div class="flex flex-col flex-1 min-h-0">
      <!-- Section Header & Status Indicator -->
      <div class="flex items-center justify-between pb-2 flex-shrink-0 gap-1.5">
        <h2 class="text-sm lg:text-base font-extrabold text-primary tracking-tight whitespace-nowrap">
          ${title}
        </h2>
        <div class="flex items-center gap-1.5 flex-shrink-0">
          ${isDeafView ? `
            <!-- CAMERA CONTROLS -->
            <button type="button" 
                    id="btn-start-camera" 
                    class="px-2 py-0.5 text-[11px] font-bold bg-emerald-600 hover:bg-emerald-700 text-white rounded-md shadow-xs transition-colors flex items-center gap-0.5 cursor-pointer">
              <span class="material-symbols-outlined text-[14px]">videocam</span>
              <span>Start</span>
            </button>
            <button type="button" 
                    id="btn-stop-camera" 
                    class="px-2 py-0.5 text-[11px] font-bold bg-slate-700 hover:bg-slate-800 text-white rounded-md shadow-xs transition-colors flex items-center gap-0.5 cursor-pointer">
              <span class="material-symbols-outlined text-[14px]">videocam_off</span>
              <span>Stop</span>
            </button>
          ` : ''}
          <div id="${badgeId}" class="transition-all duration-300 flex items-center gap-1 px-2 py-0.5 rounded-full ${badgeClass} font-bold text-[10px] uppercase tracking-wider shadow-xs">
            <span class="${dotClass}" id="${isDeafView ? 'turn-indicator-dot' : 'admin-turn-dot'}"></span>
            <span id="${isDeafView ? 'turn-status-text' : 'admin-status-text'}">
              ${isDeafView ? (isTurnActive ? 'Your Turn' : 'Ready') : 'Active'}
            </span>
          </div>
        </div>
      </div>

      <!-- Main Video Container (Clean, Minimal, Occupied Entirely by Live Video) -->
      <div id="${containerId}" class="relative w-full flex-1 min-h-[220px] lg:min-h-[280px] bg-slate-950 rounded-xl overflow-hidden transition-all duration-300 ${borderClass} shadow-xs flex items-center justify-center">
        
        ${isDeafView ? `
          <!-- REAL BROWSER WEBCAM LIVE VIDEO (Mirrored horizontally for natural self-view) -->
          <video id="deaf-camera-video" 
                 autoplay 
                 playsinline 
                 muted 
                 class="w-full h-full object-cover select-none"
                 style="transform: scaleX(-1);"></video>

          <!-- Real-Time Hand Tracking Overlay Canvas (21 Landmarks + Thin Green Bounding Box) -->
          <canvas id="hand-tracking-canvas" 
                  class="absolute inset-0 w-full h-full pointer-events-none z-10"></canvas>

          <!-- Camera Stopped Indicator Overlay (Testing Mode: initial state or stopped state) -->
          <div id="camera-stopped-overlay" class="absolute inset-0 z-15 flex flex-col items-center justify-center p-6 text-center bg-slate-950 text-slate-300">
            <span class="material-symbols-outlined text-slate-500 text-3xl mb-1.5">videocam_off</span>
            <p class="text-xs font-semibold text-slate-200">Camera stopped</p>
            <p class="text-[11px] text-slate-400 mt-1 max-w-xs">
              Click "Start Camera" above to activate the live webcam feed.
            </p>
          </div>

          <!-- Accessible Camera Permission Denied / Error Banner (Hidden by default) -->
          <div id="camera-permission-banner" class="hidden absolute inset-0 z-20 flex flex-col items-center justify-center p-6 text-center bg-slate-900/95 text-white">
            <span class="material-symbols-outlined text-amber-400 text-3xl mb-2">videocam_off</span>
            <h4 class="font-bold text-sm text-slate-100 mb-1">Camera Permission Required</h4>
            <p class="text-xs text-slate-300 max-w-xs mb-3" id="camera-error-message">
              Please allow camera access in your browser to enable live sign language recognition.
            </p>
            <button type="button" id="btn-request-permission" class="px-3.5 py-1.5 text-xs font-semibold bg-primary hover:bg-primary-container text-white rounded-lg border border-white/20 transition-colors cursor-pointer">
              Grant Permission
            </button>
          </div>

          <!-- Top Status Tag (Subtle pill, does not obscure live video) -->
          <div class="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-black/60 backdrop-blur-md rounded-full text-white text-[11px] font-bold uppercase tracking-wider z-10">
            <span class="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            <span>LIVE CAMERA</span>
          </div>

          <!-- Real-Time Live ML Prediction Overlay (Inside camera area, driven ONLY by FastAPI V2 callback) -->
          <div id="live-camera-prediction-badge" class="absolute bottom-3 left-3 flex items-center gap-2 px-3 py-1.5 bg-black/75 backdrop-blur-md border border-white/20 rounded-lg text-white text-xs font-mono font-bold z-10 shadow-md pointer-events-none transition-all duration-150">
            <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" id="live-camera-prediction-dot"></span>
            <span id="live-camera-prediction-text" class="tracking-wide">Detecting...</span>
          </div>
        ` : `
          <!-- ADMIN VIEW: Real Remote Live Stream Video Element (Un-mirrored) -->
          <video id="admin-camera-video" 
                 autoplay 
                 playsinline 
                 muted 
                 class="w-full h-full object-cover select-none"></video>

          <!-- Top Status Tags -->
          <div class="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-black/60 backdrop-blur-md rounded-full text-white text-[11px] font-bold uppercase tracking-wider z-10">
            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>CITIZEN LIVE FEED</span>
          </div>

          <!-- Neutral Waiting Indicator (Displayed only until the live stream arrives) -->
          <div id="admin-waiting-stream" class="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-slate-950 text-slate-300 z-5">
            <span class="w-2.5 h-2.5 rounded-full bg-secondary animate-ping mb-2.5"></span>
            <p class="text-xs font-semibold text-slate-200">Awaiting Citizen Live Video Feed</p>
            <p class="text-[11px] text-slate-400 mt-1 max-w-xs">
              Live webcam feed will stream automatically when the Deaf person interface is active.
            </p>
          </div>
        `}
      </div>

      <!-- Preserved Hidden Select/Button for Contract Compatibility (Zero UI Space) -->
      ${isDeafView ? `
        <div class="hidden" style="display: none;" aria-hidden="true">
          <select id="hospital-phrase-select">
            <option value="hosp_01">1. "Hello, I need help."</option>
            <option value="hosp_07">7. "I need an appointment."</option>
          </select>
          <button type="button" id="btn-trigger-hospital-phrase">Send</button>
        </div>
      ` : ''}
    </div>
  `;
}
