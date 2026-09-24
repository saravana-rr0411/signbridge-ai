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
    <div class="flex flex-col flex-1 min-h-0">
      <!-- Section Header & Status Indicator -->
      <div class="flex items-center justify-between pb-3.5 flex-shrink-0 flex-wrap gap-2">
        <h2 class="text-2xl font-extrabold text-primary tracking-tight">
          ${title}
        </h2>
        <div class="flex items-center gap-2 flex-wrap">
          ${isDeafView ? `
            <!-- CAMERA CONTROLS -->
            <button type="button" 
                    id="btn-start-camera" 
                    class="px-2.5 py-1 text-xs font-bold bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg shadow-xs transition-colors flex items-center gap-1 cursor-pointer">
              <span class="material-symbols-outlined text-[15px]">videocam</span>
              <span>Start Camera</span>
            </button>
            <button type="button" 
                    id="btn-stop-camera" 
                    class="px-2.5 py-1 text-xs font-bold bg-slate-700 hover:bg-slate-800 text-white rounded-lg shadow-xs transition-colors flex items-center gap-1 cursor-pointer">
              <span class="material-symbols-outlined text-[15px]">videocam_off</span>
              <span>Stop Camera</span>
            </button>
          ` : ''}
          <div id="${badgeId}" class="transition-all duration-300 flex items-center gap-1.5 px-3 py-1 rounded-full ${badgeClass} font-bold text-xs uppercase tracking-wider shadow-xs">
            <span class="${dotClass}" id="${isDeafView ? 'turn-indicator-dot' : 'admin-turn-dot'}"></span>
            <span id="${isDeafView ? 'turn-status-text' : 'admin-status-text'}">
              ${isDeafView ? (isTurnActive ? 'Your Turn to Sign' : turnStatus) : 'Citizen Stream Active'}
            </span>
          </div>
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


      <!-- Hospital First-Visit Controlled Phrase Layer (Template & Sign Trigger) -->
      ${isDeafView ? `
        <div class="mt-2.5 flex flex-col gap-1.5 px-0.5">
          <div class="flex items-center justify-between gap-2 flex-wrap">
            <div class="flex items-center gap-1.5 flex-1 min-w-[260px]">
              <span class="text-xs text-on-surface-variant font-bold flex items-center gap-1 flex-shrink-0" title="Hospital First-Visit Predefined Phrases">
                <span class="material-symbols-outlined text-[16px] text-rose-600">local_hospital</span>
                Hospital Phrase:
              </span>
              <select id="hospital-phrase-select" 
                      class="text-xs bg-surface-container border border-outline-variant/50 rounded-lg px-2 py-1.5 text-primary font-medium focus:outline-none focus:ring-1 focus:ring-primary flex-1 truncate cursor-pointer">
                <optgroup label="Greeting & Assistance">
                  <option value="hosp_01">1. "Hello, I need help." [ML: HELLO / HELP]</option>
                </optgroup>
                <optgroup label="Registration & Visit Purpose">
                  <option value="hosp_02">2. "I am here to see the doctor." [Template]</option>
                  <option value="hosp_07">7. "I need an appointment." [Template]</option>
                </optgroup>
                <optgroup label="Symptoms & Conditions">
                  <option value="hosp_03">3. "I am not feeling well." [Template]</option>
                  <option value="hosp_04">4. "I have been feeling sick since yesterday." [Template]</option>
                  <option value="hosp_05">5. "I have pain here." [Template]</option>
                  <option value="hosp_06">6. "I need to tell you about my problem." [Template]</option>
                </optgroup>
                <optgroup label="Communication Support">
                  <option value="hosp_08">8. "Please speak slowly." [ML: PLEASE]</option>
                  <option value="hosp_09">9. "I cannot hear you clearly." [Template]</option>
                  <option value="hosp_10">10. "I don't understand." [ML: NO]</option>
                  <option value="hosp_11">11. "Please write it down." [Template]</option>
                </optgroup>
                <optgroup label="Navigation & Urgent Requests">
                  <option value="hosp_12">12. "Where should I wait?" [Template]</option>
                  <option value="hosp_13">13. "Where is the consultation room?" [Template]</option>
                  <option value="hosp_14">14. "Please ask the doctor to come." [Template]</option>
                </optgroup>
                <optgroup label="Closing & Gratitude">
                  <option value="hosp_15">15. "Thank you for helping me." [ML: THANK_YOU]</option>
                </optgroup>
              </select>
              <button type="button" 
                      id="btn-trigger-hospital-phrase"
                      class="px-3 py-1.5 text-xs font-bold bg-rose-700 hover:bg-rose-800 text-white rounded-lg shadow-xs transition-colors flex items-center gap-1 cursor-pointer flex-shrink-0"
                      title="Send selected predefined hospital phrase">
                <span>Send</span>
                <span class="material-symbols-outlined text-[14px]">send</span>
              </button>
            </div>
            <div class="flex items-center gap-1 flex-wrap">
              <button type="button" 
                      class="btn-quick-hospital-phrase px-2 py-1 text-[11px] font-semibold bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 rounded-md transition-colors cursor-pointer"
                      data-phrase-id="hosp_07">
                Appointment
              </button>
              <button type="button" 
                      class="btn-quick-hospital-phrase px-2 py-1 text-[11px] font-semibold bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 rounded-md transition-colors cursor-pointer"
                      data-phrase-id="hosp_02">
                See Doctor
              </button>
              <button type="button" 
                      class="btn-quick-hospital-phrase px-2 py-1 text-[11px] font-semibold bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 rounded-md transition-colors cursor-pointer"
                      data-phrase-id="hosp_05">
                Pain Here
              </button>
              <button type="button" 
                      class="btn-quick-hospital-phrase px-2 py-1 text-[11px] font-semibold bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 rounded-md transition-colors cursor-pointer"
                      data-phrase-id="hosp_12">
            </div>
          </div>
        </div>
      ` : ''}
    </div>
  `;
}
