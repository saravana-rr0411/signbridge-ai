// LiveCamera.js - Real Device Live Camera Component (Zero Templates, Fully Automated)
// Displays pure HTML5 video element with automated webcam stream, clean borders,
// no "Start Camera" buttons, and subtle emerald glow ONLY on 2x Admin response completion.

export function renderLiveCamera({
  isDeafView = true,
  title = 'Start speaking',
  isTurnActive = false,
  turnStatus = 'Observing Signer'
}) {
  const containerId = isDeafView ? 'deaf-camera-container' : 'admin-camera-container';
  const badgeId = isDeafView ? 'deaf-turn-badge' : 'admin-turn-badge';

  const borderClass = isTurnActive
    ? 'camera-turn-active border-2 border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.35)]'
    : 'border border-outline-variant/30';

  const badgeClass = isTurnActive
    ? 'bg-emerald-100 text-emerald-900 border border-emerald-300'
    : 'bg-surface-container text-on-surface-variant';

  const dotClass = isTurnActive
    ? 'w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping'
    : 'w-2 h-2 rounded-full bg-outline-variant';

  return `
    <div class="flex flex-col h-full">
      <!-- Section Header & Status Indicator -->
      <div class="flex items-center justify-between pb-3.5 flex-shrink-0">
        <h2 class="text-2xl font-extrabold text-primary tracking-tight">
          ${title}
        </h2>
        <div id="${badgeId}" class="transition-all duration-300 flex items-center gap-1.5 px-3 py-1 rounded-full ${badgeClass} font-bold text-xs uppercase tracking-wider shadow-xs">
          <span class="${dotClass}" id="${isDeafView ? 'turn-indicator-dot' : 'admin-turn-dot'}"></span>
          <span id="${isDeafView ? 'turn-status-text' : 'admin-status-text'}">
            ${isDeafView ? (isTurnActive ? 'Your Turn to Sign' : turnStatus) : 'Citizen Stream Active'}
          </span>
        </div>
      </div>

      <!-- Main Video Container (Clean, Minimal, Occupied Entirely by Live Video) -->
      <div id="${containerId}" class="relative w-full flex-1 min-h-[260px] bg-slate-950 rounded-xl overflow-hidden transition-all duration-300 ${borderClass} shadow-xs flex items-center justify-center">
        
        ${isDeafView ? `
          <!-- REAL BROWSER WEBCAM LIVE VIDEO (Mirrored horizontally for natural self-view) -->
          <video id="deaf-camera-video" 
                 autoplay 
                 playsinline 
                 muted 
                 class="w-full h-full object-cover select-none"
                 style="transform: scaleX(-1);"></video>

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

          <div class="absolute top-3 right-3 px-2 py-0.5 rounded-full bg-black/50 backdrop-blur-md text-slate-300 text-[10px] font-mono z-10">
            FPS: 60.0
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

      <!-- Mock Sign Recognition Demo Control (DEAF -> ADMIN Flow) -->
      ${isDeafView ? `
        <div class="mt-2.5 flex flex-col gap-1.5 px-0.5">
          <div class="flex items-center justify-between gap-2 flex-wrap">
            <div class="flex items-center gap-1.5 flex-1 min-w-[240px]">
              <span class="text-xs text-on-surface-variant font-bold flex items-center gap-1 flex-shrink-0">
                <span class="material-symbols-outlined text-[16px] text-secondary">sign_language</span>
                Demo Sign:
              </span>
              <select id="demo-sign-select" 
                      class="text-xs bg-surface-container border border-outline-variant/50 rounded-lg px-2 py-1.5 text-primary font-medium focus:outline-none focus:ring-1 focus:ring-primary flex-1 truncate cursor-pointer">
                <option value="I need help">"I need help"</option>
                <option value="I have an appointment">"I have an appointment"</option>
                <option value="Where is the counter?">"Where is the counter?"</option>
                <option value="I need a doctor">"I need a doctor"</option>
                <option value="I want to open an account">"I want to open an account"</option>
                <option value="Please help me">"Please help me"</option>
                <option value="I don't understand">"I don't understand"</option>
                <option value="Here is my ticket: #A-204.">"Here is my ticket: #A-204."</option>
                <option value="Okay, thank you.">"Okay, thank you."</option>
              </select>
              <button type="button" 
                      id="btn-trigger-selected-sign"
                      class="px-3 py-1.5 text-xs font-bold bg-[#001428] hover:bg-[#0f2942] text-white rounded-lg shadow-xs transition-colors flex items-center gap-1 cursor-pointer flex-shrink-0">
                <span>Sign</span>
                <span class="material-symbols-outlined text-[14px]">send</span>
              </button>
            </div>
            <div class="flex items-center gap-1 flex-wrap">
              <button type="button" 
                      class="btn-simulate-sign px-2 py-1 text-[11px] font-semibold bg-surface-container hover:bg-surface-container-high rounded-md text-primary transition-colors cursor-pointer"
                      data-sign="I need help">
                "Need help"
              </button>
              <button type="button" 
                      class="btn-simulate-sign px-2 py-1 text-[11px] font-semibold bg-surface-container hover:bg-surface-container-high rounded-md text-primary transition-colors cursor-pointer"
                      data-sign="I have an appointment">
                "Appointment"
              </button>
              <button type="button" 
                      class="btn-simulate-sign px-2 py-1 text-[11px] font-semibold bg-surface-container hover:bg-surface-container-high rounded-md text-primary transition-colors cursor-pointer"
                      data-sign="Where is the counter?">
                "Counter?"
              </button>
            </div>
          </div>
        </div>
      ` : ''}
    </div>
  `;
}
