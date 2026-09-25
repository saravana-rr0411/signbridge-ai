// Staff / Admin Interface View (3-Section Balanced Viewport)
import { renderSidebar } from '../components/Sidebar.js';
import { renderSignTranscript } from '../components/SignTranscript.js';
import { renderPresetLibrary } from '../components/PresetLibrary.js';
import { renderChatPanel } from '../components/ChatPanel.js';
import { conversationStore } from '../state/conversationStore.js';
import { signRecognitionService } from '../services/signRecognitionService.js';
import { communicationService } from '../services/communicationService.js';
import { PRESET_CATEGORIES } from '../services/mockData.js';
import { hospitalConversationService } from '../services/conversation/hospitalConversationService.js';
import { SIGNING_MODES, getSignConfig } from '../services/signAnimation/signConfig.js';
import { speechService } from '../services/signAnimation/speechRecognition.js';
import { adminTtsService } from '../services/adminTtsService.js';
import { API_ENDPOINTS } from '../services/apiConfig.js';

export function renderAdminPage() {
  const conversation = conversationStore.getConversation();
  const hospState = hospitalConversationService.getState();
  const transcript = hospState.currentPhrase || signRecognitionService.getTranscript() || 'Show a sign to begin';
  const rawSign = hospState.currentSign || signRecognitionService.getRawSign() || '—';

  return `
    <div class="h-screen w-full flex bg-surface font-body text-on-surface antialiased overflow-hidden">
      <!-- Reusable Left Sidebar -->
      ${renderSidebar('#/admin')}

      <!-- Main Three-Panel Viewport (Equal ~33.3% 3-Column Desktop Grid) -->
      <main class="flex-1 h-full overflow-y-auto lg:overflow-hidden p-3.5 lg:p-4 bg-surface flex flex-col min-w-0 min-h-0">
        <!-- 3-Panel Split Layout Extending to Top -->
        <div class="w-full flex-1 flex flex-col lg:flex-row gap-3.5 lg:gap-4 min-h-0 h-full">
          <!-- SECTION 1 (LEFT): AI RECOGNITION TELEMETRY PANEL -->
          <section class="flex-1 lg:w-1/3 min-w-0 min-h-0 flex flex-col h-full bg-surface-container-lowest rounded-2xl p-3.5 lg:p-4 shadow-sm border border-outline-variant/30 overflow-hidden">
            <!-- AI RECOGNITION PANEL HEADER -->
            <div class="flex items-center justify-between pb-3 flex-shrink-0 gap-1.5 border-b border-outline-variant/20">
              <div class="flex items-center gap-2">
                <span class="material-symbols-outlined text-primary text-[20px]">psychology</span>
                <h2 class="text-sm lg:text-base font-extrabold text-primary tracking-tight whitespace-nowrap">
                  AI RECOGNITION
                </h2>
              </div>
              <div class="flex items-center gap-2 flex-shrink-0">
                <span class="text-[10px] font-mono font-bold text-slate-600 bg-surface-container px-2 py-0.5 rounded uppercase tracking-wider">
                  V6 • 10-SIGN
                </span>
                <div id="admin-connection-badge" class="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-bold text-[10px] uppercase tracking-wider border border-slate-200 transition-colors duration-200">
                  <span id="admin-connection-dot" class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                  <span id="admin-connection-text">Connecting...</span>
                </div>
              </div>
            </div>

            <!-- AI TELEMETRY CARDS (Replaces Live Camera Viewport) -->
            <div class="flex-1 min-h-0 flex flex-col gap-2.5 py-2.5 overflow-y-auto">
              <!-- PRIMARY METRICS: 2-COLUMN GRID -->
              <div class="grid grid-cols-2 gap-2.5 flex-shrink-0">
                <!-- 1. RECOGNITION STATUS -->
                <div class="p-3 bg-surface-container-low rounded-xl border border-outline-variant/30 flex flex-col justify-between">
                  <span class="text-[10px] font-extrabold uppercase tracking-wider text-slate-500 flex items-center gap-1 mb-1">
                    <span class="material-symbols-outlined text-[13px] text-slate-400">sensors</span>
                    Recognition Status
                  </span>
                  <div class="flex items-center gap-1.5 mt-0.5">
                    <span id="admin-recognition-state-dot" class="w-2 h-2 rounded-full bg-slate-400"></span>
                    <span id="admin-recognition-state-text" class="text-xs font-mono font-bold text-slate-700 truncate">
                      Checking...
                    </span>
                  </div>
                </div>

                <!-- 2. INFERENCE LATENCY -->
                <div class="p-3 bg-surface-container-low rounded-xl border border-outline-variant/30 flex flex-col justify-between">
                  <span class="text-[10px] font-extrabold uppercase tracking-wider text-slate-500 flex items-center gap-1 mb-1">
                    <span class="material-symbols-outlined text-[13px] text-slate-400">speed</span>
                    Inference Latency
                  </span>
                  <div class="flex items-baseline gap-1 mt-0.5">
                    <span id="admin-latency-text" class="text-sm font-mono font-bold text-slate-800">
                      —
                    </span>
                  </div>
                </div>

                <!-- 3. CURRENT RECOGNIZED SIGN & CONFIDENCE -->
                <div class="col-span-2 p-3 bg-gradient-to-br from-white to-surface-container-lowest rounded-xl border-2 border-primary/15 shadow-2xs flex flex-col justify-between">
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-[10px] font-extrabold uppercase tracking-wider text-primary flex items-center gap-1">
                      <span class="material-symbols-outlined text-secondary text-[14px]">sign_language</span>
                      Current Recognized Sign
                    </span>
                    <div class="flex items-center gap-1">
                      <span class="text-[10px] font-extrabold uppercase tracking-wider text-slate-400">Confidence:</span>
                      <span id="admin-confidence-text" class="text-xs font-mono font-bold text-emerald-700">
                        —
                      </span>
                    </div>
                  </div>
                  <div class="flex items-center justify-between my-1">
                    <span id="admin-current-sign-text" class="text-xl lg:text-2xl font-black font-mono text-primary tracking-wide">
                      ${rawSign}
                    </span>
                    <div id="admin-current-sign-badge" class="hidden px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                      LIVE
                    </div>
                  </div>
                  <!-- Confidence Bar Indicator -->
                  <div class="w-full bg-slate-200/70 h-1 rounded-full overflow-hidden mt-1">
                    <div id="admin-confidence-bar" class="h-full bg-emerald-600 transition-all duration-300" style="width: 0%;"></div>
                  </div>
                </div>
              </div>

              <!-- 4. RECENT RECOGNIZED SIGNS -->
              <div class="flex-1 min-h-[110px] flex flex-col p-2.5 bg-surface-container-low rounded-xl border border-outline-variant/30 overflow-hidden">
                <div class="flex items-center justify-between pb-1.5 mb-1 border-b border-outline-variant/20 flex-shrink-0">
                  <span class="text-[10px] font-extrabold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                    <span class="material-symbols-outlined text-[13px] text-slate-400">history</span>
                    Recent Recognized Signs
                  </span>
                  <span id="admin-recent-signs-count" class="text-[10px] font-mono font-bold text-slate-400">
                    0 events
                  </span>
                </div>
                <div id="admin-recent-signs-list" class="flex-1 overflow-y-auto flex flex-col gap-1.5 pr-0.5">
                  <div id="admin-recent-empty" class="text-[11px] text-slate-400 text-center py-4 italic flex flex-col items-center justify-center gap-1">
                    <span class="material-symbols-outlined text-slate-300 text-lg">history_toggle_off</span>
                    <span>No recognition events yet</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- PRESERVED SIGN TRANSCRIPT / CONTEXTUAL PHRASE -->
            ${renderSignTranscript(
              transcript,
              'admin-live-transcript',
              'Recognition Telemetry Active',
              'admin-recognition-status',
              rawSign,
              'Hospital First-Visit'
            )}
          </section>

          <!-- SECTION 2 (CENTER): PRESET RESPONSES (BANK, HOSPITAL, GOV OFFICE, ASL, ISL) -->
          <section class="flex-1 lg:w-1/3 min-w-0 min-h-0 flex flex-col h-full bg-surface-container-lowest rounded-2xl p-3.5 lg:p-4 shadow-sm border border-outline-variant/30 overflow-hidden" id="admin-preset-section">
            ${renderPresetLibrary(PRESET_CATEGORIES.HOSPITAL, SIGNING_MODES.FINGERSPELLING)}
          </section>

          <!-- SECTION 3 (RIGHT): CONVERSATION + FIXED MESSAGE COMPOSER -->
          <section class="flex-1 lg:w-1/3 min-w-0 min-h-0 flex flex-col h-full bg-surface-container-lowest rounded-2xl p-3.5 lg:p-4 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderChatPanel({
              conversation: conversation,
              isDeafView: false,
              composerText: '',
              activeMode: SIGNING_MODES.FINGERSPELLING
            })}
          </section>
        </div>
      </main>
    </div>
  `;
}

export function initAdminPage() {
  // Connect to cross-device WebSocket relay as 'admin' peer in room 'desk_04'
  communicationService.connect('desk_04', 'admin');

  const composerInput = document.getElementById('composer-input');
  const clearBtn = document.getElementById('composer-clear-btn');
  const composerForm = document.getElementById('admin-composer-form');
  let currentCategory = PRESET_CATEGORIES.HOSPITAL;
  let currentSigningMode = SIGNING_MODES.FINGERSPELLING;

  // Helper to scroll conversation timeline
  function scrollToBottom() {
    const chatTimeline = document.getElementById('chat-stream-timeline');
    if (chatTimeline) {
      chatTimeline.scrollTop = chatTimeline.scrollHeight;
    }
  }

  function updateComposerModeBadge() {
    const badge = document.getElementById('composer-mode-badge');
    if (badge) {
      badge.textContent = currentSigningMode;
    }
  }

  function reRenderPresetSection() {
    const presetSection = document.getElementById('admin-preset-section');
    if (presetSection) {
      presetSection.innerHTML = renderPresetLibrary(currentCategory, currentSigningMode);
      bindSigningModeTabs();
      bindCategoryTabs();
      bindPresetClicks();
    }
  }

  // Bind Signing Mode Tabs (FINGERSPELLING, ASL, ISL)
  function bindSigningModeTabs() {
    document.querySelectorAll('.signing-mode-tab-btn').forEach((tabBtn) => {
      tabBtn.addEventListener('click', () => {
        const mode = tabBtn.getAttribute('data-signing-mode');
        if (mode) {
          currentSigningMode = mode;
          updateComposerModeBadge();
          reRenderPresetSection();
        }
      });
    });
  }

  // Bind Preset Item Clicks -> Populates Composer WITHOUT Sending
  function bindPresetClicks() {
    document.querySelectorAll('.preset-item-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const text = btn.getAttribute('data-text');
        const mode = btn.getAttribute('data-mode');
        if (mode) {
          currentSigningMode = mode;
          updateComposerModeBadge();
        }
        if (text && composerInput) {
          composerInput.value = text;
          if (clearBtn) clearBtn.classList.remove('hidden');
          composerInput.focus();
        }
      });
    });
  }

  // Bind Category Tab Switch (BANK, HOSPITAL, GOV OFFICE)
  function bindCategoryTabs() {
    document.querySelectorAll('.preset-tab-btn').forEach((tabBtn) => {
      tabBtn.addEventListener('click', () => {
        const category = tabBtn.getAttribute('data-category');
        if (category) {
          currentCategory = category;
          reRenderPresetSection();
        }
      });
    });
  }

  // Clear button for composer input
  if (clearBtn && composerInput) {
    clearBtn.addEventListener('click', () => {
      composerInput.value = '';
      clearBtn.classList.add('hidden');
      composerInput.focus();
    });

    composerInput.addEventListener('input', () => {
      if (composerInput.value.trim().length > 0) {
        clearBtn.classList.remove('hidden');
      } else {
        clearBtn.classList.add('hidden');
      }
    });

    // Enter key sends (without Shift)
    composerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitMessage();
      }
    });
  }

  // Web Speech API Voice Input (Speech-to-Text)
  const micBtn = document.getElementById('composer-mic-btn');
  const micIcon = document.getElementById('composer-mic-icon');
  const speechStatus = document.getElementById('speech-recognition-status');

  if (micBtn) {
    micBtn.addEventListener('click', () => {
      if (speechService.getIsListening()) {
        speechService.stop();
        if (micIcon) micIcon.textContent = 'mic';
        micBtn.classList.remove('text-rose-600', 'bg-rose-100');
        if (speechStatus) speechStatus.classList.add('hidden');
      } else {
        const started = speechService.start({
          onStart: () => {
            if (micIcon) micIcon.textContent = 'mic';
            micBtn.classList.add('text-rose-600', 'bg-rose-100');
            if (speechStatus) speechStatus.classList.remove('hidden');
          },
          onResult: (transcript) => {
            if (composerInput) {
              composerInput.value = transcript;
              if (clearBtn) clearBtn.classList.remove('hidden');
            }
          },
          onEnd: () => {
            if (micIcon) micIcon.textContent = 'mic';
            micBtn.classList.remove('text-rose-600', 'bg-rose-100');
            if (speechStatus) speechStatus.classList.add('hidden');
          },
          onError: () => {
            if (micIcon) micIcon.textContent = 'mic';
            micBtn.classList.remove('text-rose-600', 'bg-rose-100');
            if (speechStatus) speechStatus.classList.add('hidden');
          }
        });
        if (!started) {
          console.warn('Speech recognition not available or denied.');
        }
      }
    });
  }

  // Send message action
  function submitMessage() {
    if (!composerInput) return;
    const messageText = composerInput.value.trim();
    if (!messageText) return;

    const signConfig = getSignConfig(messageText, currentSigningMode);

    // 1. Add Admin Message to conversationStore
    conversationStore.addMessage({
      sender: 'admin',
      senderName: 'Admin (Officer Vance)',
      text: messageText,
      type: 'text',
      isActiveReply: true,
      mode: currentSigningMode
    });

    // 2. Dispatch legacy message to communicationService
    communicationService.emit('ADMIN_MESSAGE_SENT', {
      text: messageText,
      timestamp: Date.now(),
      mode: currentSigningMode,
      hasSignResponse: true
    });

    // 3. Dispatch dedicated sign response payload for 3D sign animation
    communicationService.emit('ADMIN_SIGN_RESPONSE', {
      type: "ADMIN_SIGN_RESPONSE",
      messageId: 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
      text: messageText,
      mode: currentSigningMode,
      signSequence: signConfig.signSequence || [],
      timestamp: Date.now()
    });

    // 4. Clear composer input
    composerInput.value = '';
    if (clearBtn) clearBtn.classList.add('hidden');
    composerInput.focus();

    scrollToBottom();
  }

  // Composer Form Submit
  if (composerForm) {
    composerForm.addEventListener('submit', (e) => {
      e.preventDefault();
      submitMessage();
    });
  }

  // Initial bindings
  bindSigningModeTabs();
  bindCategoryTabs();
  bindPresetClicks();
  scrollToBottom();

  // AI RECOGNITION TELEMETRY CONTROLS & STATE
  const connectionBadge = document.getElementById('admin-connection-badge');
  const connectionDot = document.getElementById('admin-connection-dot');
  const connectionText = document.getElementById('admin-connection-text');
  const recogStateDot = document.getElementById('admin-recognition-state-dot');
  const recogStateText = document.getElementById('admin-recognition-state-text');
  const latencyText = document.getElementById('admin-latency-text');
  const currentSignText = document.getElementById('admin-current-sign-text');
  const currentSignBadge = document.getElementById('admin-current-sign-badge');
  const confidenceText = document.getElementById('admin-confidence-text');
  const confidenceBar = document.getElementById('admin-confidence-bar');
  const recentSignsList = document.getElementById('admin-recent-signs-list');
  const recentSignsCount = document.getElementById('admin-recent-signs-count');

  let currentRecognitionStatus = 'Checking...';
  let isFastApiOnline = false;
  let liveBadgeTimer = null;
  let processingStatusTimer = null;
  const recentRecognizedSigns = [];

  function updateRecognitionStatus(status) {
    if (!status) return;
    let normalized = status;
    const s = String(status).toUpperCase();
    if (s.includes('RECOGNIZ') || s.includes('PROCESS') || s.includes('INFER')) {
      normalized = 'Processing';
    } else if (s.includes('READY') || s.includes('LISTEN') || s.includes('COLLECT') || s.includes('WAIT')) {
      normalized = 'Listening';
    } else if (s.includes('OFFLINE') || s.includes('STOP') || s.includes('UNAVAIL')) {
      normalized = 'Offline';
    } else if (s.includes('CONNECT')) {
      normalized = 'Connected';
    }
    currentRecognitionStatus = normalized;

    if (recogStateText) recogStateText.textContent = normalized;
    if (recogStateDot) {
      if (normalized === 'Listening') {
        recogStateDot.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      } else if (normalized === 'Processing') {
        recogStateDot.className = 'w-2 h-2 rounded-full bg-amber-500 animate-ping';
      } else if (normalized === 'Connected') {
        recogStateDot.className = 'w-2 h-2 rounded-full bg-emerald-500';
      } else {
        recogStateDot.className = 'w-2 h-2 rounded-full bg-slate-400';
      }
    }
  }

  function updateConnectionStatus(status) {
    if (!connectionText || !connectionDot || !connectionBadge) return;
    let normalized = status;
    const s = String(status).toUpperCase();
    if (s.includes('CONNECT') || s.includes('ONLINE') || s.includes('READY')) {
      normalized = 'Connected';
    } else if (s.includes('RECONNECT') || s.includes('CHECK')) {
      normalized = 'Connecting...';
    } else {
      normalized = 'Offline';
    }

    if (normalized === 'Connected') {
      connectionBadge.className = 'flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 font-bold text-[10px] uppercase tracking-wider border border-emerald-200 transition-colors duration-200';
      connectionDot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-500';
      connectionText.textContent = 'Connected';
    } else if (normalized === 'Connecting...') {
      connectionBadge.className = 'flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-800 font-bold text-[10px] uppercase tracking-wider border border-amber-200 transition-colors duration-200';
      connectionDot.className = 'w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse';
      connectionText.textContent = 'Connecting...';
    } else {
      connectionBadge.className = 'flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-bold text-[10px] uppercase tracking-wider border border-slate-200 transition-colors duration-200';
      connectionDot.className = 'w-1.5 h-1.5 rounded-full bg-slate-400';
      connectionText.textContent = 'Offline';
    }
  }

  function updateCurrentSign(sign) {
    if (!sign || sign === '—' || String(sign).includes('Template Selected')) return;
    const cleanSign = String(sign).trim().toUpperCase();
    if (currentSignText) currentSignText.textContent = cleanSign;

    if (currentSignBadge) {
      currentSignBadge.classList.remove('hidden');
      if (liveBadgeTimer) clearTimeout(liveBadgeTimer);
      liveBadgeTimer = setTimeout(() => {
        if (currentSignBadge) currentSignBadge.classList.add('hidden');
      }, 3000);
    }

    const rawSignEl = document.getElementById('admin-live-transcript-raw-sign');
    if (rawSignEl) rawSignEl.textContent = cleanSign;
  }

  function updateConfidence(conf) {
    if (!confidenceText) return;
    if (conf === null || conf === undefined || conf === '—' || conf === '') {
      confidenceText.textContent = '—';
      if (confidenceBar) confidenceBar.style.width = '0%';
      return;
    }
    const num = typeof conf === 'number' ? conf : parseFloat(conf);
    if (!isNaN(num)) {
      const pct = num <= 1 ? (num * 100).toFixed(1) : num.toFixed(1);
      confidenceText.textContent = `${pct}%`;
      if (confidenceBar) confidenceBar.style.width = `${Math.min(100, Math.max(0, parseFloat(pct)))}%`;
    } else {
      confidenceText.textContent = String(conf);
    }
  }

  function updateLatency(lat) {
    if (!latencyText) return;
    if (lat === null || lat === undefined || lat === '—' || lat === '') {
      latencyText.textContent = '—';
      return;
    }
    const num = typeof lat === 'number' ? lat : parseFloat(lat);
    if (!isNaN(num)) {
      latencyText.textContent = `${num.toFixed(1)} ms`;
    } else {
      latencyText.textContent = String(lat);
    }
  }

  function renderRecentSigns() {
    if (!recentSignsList) return;
    if (recentRecognizedSigns.length === 0) {
      recentSignsList.innerHTML = `
        <div id="admin-recent-empty" class="text-[11px] text-slate-400 text-center py-4 italic flex flex-col items-center justify-center gap-1">
          <span class="material-symbols-outlined text-slate-300 text-lg">history_toggle_off</span>
          <span>No recognition events yet</span>
        </div>
      `;
      if (recentSignsCount) recentSignsCount.textContent = '0 events';
      return;
    }

    if (recentSignsCount) {
      recentSignsCount.textContent = `${recentRecognizedSigns.length} event${recentRecognizedSigns.length > 1 ? 's' : ''}`;
    }

    recentSignsList.innerHTML = recentRecognizedSigns.map(item => `
      <div class="flex items-center justify-between px-2.5 py-1.5 bg-white rounded-lg border border-outline-variant/30 text-xs shadow-2xs animate-fadeIn">
        <div class="flex items-center gap-1.5 min-w-0">
          <span class="material-symbols-outlined text-[14px] text-emerald-600 flex-shrink-0">check_circle</span>
          <span class="font-mono font-bold text-primary truncate">${item.sign}</span>
        </div>
        <div class="flex items-center gap-2 text-[10px] font-mono text-slate-500 flex-shrink-0">
          ${item.confidence ? `<span class="text-emerald-700 font-bold">${item.confidence}</span>` : ''}
          ${item.latency ? `<span class="text-slate-500 font-bold">${item.latency}</span>` : ''}
          <span class="text-slate-400">${item.time}</span>
        </div>
      </div>
    `).join('');
  }

  function addRecentSign(sign, conf, timestamp, latency) {
    if (!sign || sign === '—' || String(sign).includes('Template Selected')) return;
    const cleanSign = String(sign).trim().toUpperCase();
    const now = Date.now();

    // Prevent immediate duplicate spam within 2 seconds
    if (recentRecognizedSigns.length > 0) {
      const top = recentRecognizedSigns[0];
      if (top.sign === cleanSign && (now - (top._ts || 0) < 2000)) {
        return;
      }
    }

    const timeStr = timestamp
      ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    let confStr = '';
    if (conf !== null && conf !== undefined && conf !== '—' && conf !== '') {
      const num = typeof conf === 'number' ? conf : parseFloat(conf);
      if (!isNaN(num)) {
        confStr = `${(num <= 1 ? num * 100 : num).toFixed(1)}%`;
      }
    }

    let latStr = '';
    if (latency !== null && latency !== undefined && latency !== '—' && latency !== '') {
      const num = typeof latency === 'number' ? latency : parseFloat(latency);
      if (!isNaN(num)) {
        latStr = `${num.toFixed(1)} ms`;
      }
    }

    recentRecognizedSigns.unshift({
      sign: cleanSign,
      confidence: confStr,
      latency: latStr,
      time: timeStr,
      _ts: now
    });

    if (recentRecognizedSigns.length > 8) {
      recentRecognizedSigns.pop();
    }

    renderRecentSigns();
  }

  // Pre-populate recent recognized signs from existing conversation history if any
  const existingConv = conversationStore.getConversation();
  if (Array.isArray(existingConv)) {
    existingConv.forEach(m => {
      if (m.sender === 'deaf' && m.rawSign && m.rawSign !== '—' && !m.rawSign.includes('Template')) {
        const timeStr = m.time || (m.timestamp ? new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'Earlier');
        let confStr = '';
        if (m.metadata?.confidence) {
          const num = m.metadata.confidence;
          confStr = `${(num <= 1 ? num * 100 : num).toFixed(1)}%`;
        }
        recentRecognizedSigns.push({
          sign: m.rawSign.toUpperCase(),
          confidence: confStr,
          time: timeStr,
          _ts: m.timestamp || 0
        });
      }
    });
    if (recentRecognizedSigns.length > 0) {
      renderRecentSigns();
      const latest = recentRecognizedSigns[0];
      updateCurrentSign(latest.sign);
      if (latest.confidence) updateConfidence(latest.confidence);
    }
  }

  // Real backend health verification (fast non-blocking fetch)
  async function verifyBackendConnection() {
    try {
      const resp = await fetch(API_ENDPOINTS.HEALTH, { method: 'GET' });
      if (resp.ok) {
        isFastApiOnline = true;
        updateConnectionStatus('Connected');
        if (currentRecognitionStatus === 'Offline' || currentRecognitionStatus === 'Checking...') {
          updateRecognitionStatus('Listening');
        }
        return;
      }
    } catch (err) {}
    isFastApiOnline = false;
    if (communicationService.isConnected) {
      updateConnectionStatus('Connected');
      if (currentRecognitionStatus === 'Offline' || currentRecognitionStatus === 'Checking...') {
        updateRecognitionStatus('Listening');
      }
    } else {
      updateConnectionStatus('Offline');
      updateRecognitionStatus('Offline');
    }
  }
  verifyBackendConnection();

  // Subscribe to live recognized sign text from Deaf terminal
  const unsubRecog = signRecognitionService.subscribe((recog) => {
    const transcriptEl = document.getElementById('admin-live-transcript');
    const rawSignEl = document.getElementById('admin-live-transcript-raw-sign');

    if (transcriptEl) {
      transcriptEl.innerHTML = `"${recog.transcript}" <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>`;
    }
    if (rawSignEl && recog.rawSign) {
      rawSignEl.textContent = recog.rawSign;
    }

    const statusTextEl = document.getElementById('admin-recognition-status');
    const statusDotEl = document.getElementById('admin-recognition-status-dot');
    if (statusTextEl && recog.statusText) {
      statusTextEl.textContent = recog.statusText;
    }
    if (statusDotEl && recog.statusCode) {
      if (recog.statusCode === 'RECOGNIZING') {
        statusDotEl.className = 'w-2 h-2 rounded-full bg-amber-500 animate-ping';
      } else if (recog.statusCode === 'CONFIRMED') {
        statusDotEl.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      } else {
        statusDotEl.className = 'w-2 h-2 rounded-full bg-emerald-500';
      }
    }

    // Update AI Recognition Telemetry
    if (recog.statusCode) {
      if (recog.statusCode === 'RECOGNIZING') {
        updateRecognitionStatus('Processing');
      } else if (recog.statusCode === 'READY') {
        updateRecognitionStatus('Listening');
      } else if (recog.statusCode === 'OFFLINE') {
        updateRecognitionStatus('Offline');
      }
    }
    if (recog.rawSign && recog.rawSign !== '—' && !recog.rawSign.includes('Template')) {
      updateCurrentSign(recog.rawSign);
    }
  });

  // Subscribe to Hospital Conversation Layer updates
  const unsubHosp = hospitalConversationService.subscribe((hosp) => {
    const transcriptEl = document.getElementById('admin-live-transcript');
    const rawSignEl = document.getElementById('admin-live-transcript-raw-sign');
    if (transcriptEl && hosp.currentPhrase) {
      transcriptEl.innerHTML = `"${hosp.currentPhrase}" <span class="inline-block w-1.5 h-4 bg-secondary ml-1 animate-pulse align-middle"></span>`;
    }
    if (rawSignEl) {
      rawSignEl.textContent = hosp.currentSign || '— (Template Selected)';
    }
    if (hosp.currentSign && hosp.currentSign !== '—' && !hosp.currentSign.includes('Template')) {
      updateCurrentSign(hosp.currentSign);
      if (hosp.currentConfidence) updateConfidence(hosp.currentConfidence);
    }
  });

  // Helper to render chat messages
  function renderChatMessages(conversation) {
    const chatTimeline = document.getElementById('chat-stream-timeline');
    if (!chatTimeline || !Array.isArray(conversation)) return;

    chatTimeline.innerHTML = conversation.map(msg => {
      const isAdmin = msg.sender === 'admin';
      if (isAdmin) {
        const activeStyle = msg.isActiveReply
          ? 'bg-secondary/10 border-secondary/30 ring-1 ring-secondary/20'
          : 'bg-surface-container-low border-outline-variant/50';
        return `
          <div class="flex flex-col gap-1 items-start w-full animate-fadeIn">
            <div class="flex items-center gap-1.5 text-xs text-on-surface-variant pl-1">
              <span class="material-symbols-outlined text-[15px] text-secondary">support_agent</span>
              <span class="font-bold text-primary">${msg.senderName || 'Admin (Officer Vance)'}</span>
              <span class="text-[10px] text-outline font-mono">${msg.time || 'Just now'}</span>
            </div>
            <div class="max-w-[90%] ${activeStyle} border text-on-surface rounded-2xl rounded-tl-xs px-3.5 py-2.5 shadow-xs">
              <p class="text-sm font-medium leading-relaxed">${msg.text}</p>
            </div>
          </div>
        `;
      } else {
        return `
          <div class="flex flex-col gap-1 items-end w-full animate-fadeIn">
            <div class="flex items-center gap-1.5 text-xs text-on-surface-variant pr-1">
              <span class="text-[10px] text-outline font-mono">${msg.time || 'Just now'}</span>
              <span class="font-bold text-primary">${msg.senderName || 'Deaf Person'}</span>
              <span class="material-symbols-outlined text-[15px] text-primary">visibility</span>
            </div>
            <div class="max-w-[90%] bg-white border-2 border-primary/20 text-primary rounded-2xl rounded-tr-xs px-3.5 py-2.5 shadow-xs">
              ${msg.rawSign ? `
                <div class="text-[10px] font-mono font-bold text-slate-500 mb-1 pb-1 border-b border-outline-variant/30 flex items-center gap-1">
                  <span class="material-symbols-outlined text-[12px] text-slate-400">sign_language</span>
                  <span>Recognized: <span class="text-emerald-700">${msg.rawSign}</span></span>
                </div>
              ` : ''}
              <p class="text-sm font-semibold leading-relaxed">"${msg.text}"</p>
            </div>
          </div>
        `;
      }
    }).join('');
    scrollToBottom();
  }

  // Pre-seed already existing messages so page load does not read historical chat aloud
  adminTtsService.markExistingAsSpoken(conversationStore.getConversation());

  // Subscribe to conversationStore updates
  const unsubConv = conversationStore.subscribe((conversation) => {
    renderChatMessages(conversation);
    // Handle cross-tab/storage conversation updates for new incoming Deaf messages
    if (Array.isArray(conversation) && conversation.length > 0) {
      const latestMsg = conversation[conversation.length - 1];
      if (latestMsg && latestMsg.sender === 'deaf') {
        adminTtsService.speakIncomingDeafMessage(latestMsg);
      }
    }
  });

  // Listen for incoming Deaf person messages via communicationService event bus
  const unsubDeafMsg = communicationService.on('DEAF_MESSAGE_SENT', (payload) => {
    if (payload && payload.text) {
      const transcriptEl = document.getElementById('admin-live-transcript');
      const rawSignEl = document.getElementById('admin-live-transcript-raw-sign');
      if (transcriptEl) {
        transcriptEl.innerHTML = `"${payload.text}" <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>`;
      }
      if (rawSignEl && payload.rawSign) {
        rawSignEl.textContent = payload.rawSign;
      }
      const statusTextEl = document.getElementById('admin-recognition-status');
      if (statusTextEl) {
        statusTextEl.textContent = `Detected: ${payload.rawSign || payload.text}`;
      }
      const statusDotEl = document.getElementById('admin-recognition-status-dot');
      if (statusDotEl) {
        statusDotEl.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      }

      // Update AI Telemetry Panel
      const sign = payload.rawSign || (payload.metadata && payload.metadata.recognizedSigns);
      const conf = payload.confidence !== undefined ? payload.confidence : (payload.metadata && payload.metadata.confidence);
      const lat = payload.latency ?? (payload.metadata && payload.metadata.latency);
      if (sign && sign !== '—' && !sign.includes('Template')) {
        updateCurrentSign(sign);
        addRecentSign(sign, conf, payload.timestamp, lat);
        updateRecognitionStatus('Processing');
        if (processingStatusTimer) clearTimeout(processingStatusTimer);
        processingStatusTimer = setTimeout(() => updateRecognitionStatus('Listening'), 2500);
      }
      if (conf !== undefined && conf !== null) updateConfidence(conf);
      if (lat !== undefined && lat !== null) updateLatency(lat);

      // Sync incoming Deaf message to conversationStore if not already present
      const msgId = payload.messageId || payload.id || payload.msgId;
      const conv = conversationStore.getConversation();
      const exists = conv.some(
        (m) =>
          (msgId && m.id === msgId) ||
          (m.sender === 'deaf' && m.text === payload.text && Math.abs((m.timestamp || 0) - (payload.timestamp || Date.now())) < 4000)
      );
      if (!exists) {
        conversationStore.addMessage({
          id: msgId,
          sender: 'deaf',
          senderName: payload.senderName || 'Deaf Person',
          text: payload.text,
          rawSign: payload.rawSign || null,
          type: 'hospital_sentence',
          timestamp: payload.timestamp
        });
      } else {
        renderChatMessages(conversationStore.getConversation());
      }

      // AUTOMATIC TEXT-TO-SPEECH AT RUNTIME FOR INCOMING DEAF MESSAGE
      adminTtsService.speakIncomingDeafMessage(payload);
    }
  });

  // Listen for TTS speaking state to update UI indicator
  const unsubTtsState = adminTtsService.onStateChange((isSpeaking) => {
    const ttsIndicator = document.getElementById('admin-live-transcript-tts-indicator');
    if (ttsIndicator) {
      if (isSpeaking) {
        ttsIndicator.classList.remove('hidden');
      } else {
        ttsIndicator.classList.add('hidden');
      }
    }
  });

  const unsubDeafRecog = communicationService.on('DEAF_RECOGNITION_SENT', (payload) => {
    if (payload && payload.text) {
      const transcriptEl = document.getElementById('admin-live-transcript');
      if (transcriptEl) {
        transcriptEl.innerHTML = `"${payload.text}" <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>`;
      }
      const statusTextEl = document.getElementById('admin-recognition-status');
      if (statusTextEl) {
        statusTextEl.textContent = `Detected: ${payload.text}`;
      }
      const statusDotEl = document.getElementById('admin-recognition-status-dot');
      if (statusDotEl) {
        statusDotEl.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      }
      if (payload.text) {
        updateCurrentSign(payload.text);
        addRecentSign(payload.text, payload.confidence || null, Date.now());
      }
      renderChatMessages(conversationStore.getConversation());
    }
  });

  // Listen for runtime AI_RECOGNITION_EVENT
  const unsubAiRecog = communicationService.on('AI_RECOGNITION_EVENT', (payload) => {
    if (!payload) return;
    const sign = payload.sign || payload.label || payload.recognizedSign || payload.text;
    const conf = payload.confidence;
    const lat = payload.latency ?? payload.inference_latency_ms ?? payload.latencyMs;
    const status = payload.status || payload.recognitionStatus;
    const connStatus = payload.connectionStatus;

    if (connStatus) updateConnectionStatus(connStatus);
    if (status) updateRecognitionStatus(status);
    if (sign && sign !== '—' && !String(sign).includes('Template')) {
      updateCurrentSign(sign);
      addRecentSign(sign, conf, payload.timestamp, lat);
      if (!status) {
        updateRecognitionStatus('Processing');
        if (processingStatusTimer) clearTimeout(processingStatusTimer);
        processingStatusTimer = setTimeout(() => updateRecognitionStatus('Listening'), 2000);
      }
    }
    if (conf !== undefined && conf !== null) updateConfidence(conf);
    if (lat !== undefined && lat !== null) updateLatency(lat);
  });

  // Listen for HOSPITAL_PHRASE_EVENT
  const unsubHospPhrase = communicationService.on('HOSPITAL_PHRASE_EVENT', (payload) => {
    if (!payload) return;
    const sign = payload.recognizedSign;
    const conf = payload.confidence;
    if (sign && sign !== '—' && !sign.includes('Template')) {
      updateCurrentSign(sign);
      addRecentSign(sign, conf, payload.timestamp);
    }
    if (conf !== undefined && conf !== null) updateConfidence(conf);
  });

  // Listen for live transcript synchronization across tabs/windows
  const unsubTranscriptSync = communicationService.on('TRANSCRIPT_SYNC', (payload) => {
    if (payload) {
      const transcriptEl = document.getElementById('admin-live-transcript');
      if (transcriptEl && payload.transcript) {
        transcriptEl.innerHTML = `"${payload.transcript}" <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>`;
      }
      const statusTextEl = document.getElementById('admin-recognition-status');
      if (statusTextEl && payload.statusText) {
        statusTextEl.textContent = payload.statusText;
      }
      const statusDotEl = document.getElementById('admin-recognition-status-dot');
      if (statusDotEl && payload.statusCode) {
        statusDotEl.className = payload.statusCode === 'RECOGNIZING'
          ? 'w-2 h-2 rounded-full bg-amber-500 animate-ping'
          : 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      }

      // Update AI Telemetry Panel
      if (payload.statusCode) {
        if (payload.statusCode === 'RECOGNIZING') {
          updateRecognitionStatus('Processing');
        } else if (payload.statusCode === 'READY' || payload.statusCode === 'COLLECTING' || payload.statusCode === 'WAITING') {
          updateRecognitionStatus('Listening');
        } else if (payload.statusCode === 'CONFIRMED') {
          updateRecognitionStatus('Processing');
          if (processingStatusTimer) clearTimeout(processingStatusTimer);
          processingStatusTimer = setTimeout(() => updateRecognitionStatus('Listening'), 2000);
        } else if (payload.statusCode === 'OFFLINE' || payload.statusCode === 'STOPPED') {
          updateRecognitionStatus('Offline');
        }
      }
      if (payload.rawSign && payload.rawSign !== '—' && !payload.rawSign.includes('Template')) {
        updateCurrentSign(payload.rawSign);
      }
    }
  });

  // Listen for WebSocket relay connection state
  const unsubCommConnected = communicationService.on('COMM_CONNECTED', () => {
    updateConnectionStatus('Connected');
    if (currentRecognitionStatus === 'Offline' || currentRecognitionStatus === 'Checking...') {
      updateRecognitionStatus('Listening');
    }
  });

  const unsubCommDisconnected = communicationService.on('COMM_DISCONNECTED', () => {
    if (!isFastApiOnline) {
      updateConnectionStatus('Offline');
      updateRecognitionStatus('Offline');
    }
  });

  // Demo Reset Button
  const resetDemoBtn = document.getElementById('btn-reset-demo');
  if (resetDemoBtn) {
    resetDemoBtn.addEventListener('click', () => {
      conversationStore.resetDemo();
      signRecognitionService.reset();
      hospitalConversationService.reset();
      recentRecognizedSigns.length = 0;
      renderRecentSigns();
      updateCurrentSign('—');
      updateConfidence(null);
      updateLatency(null);
      if (composerInput) {
        composerInput.value = '';
        if (clearBtn) clearBtn.classList.add('hidden');
      }
    });
  }

  const unsubReset = communicationService.on('DEMO_RESET', () => {
    recentRecognizedSigns.length = 0;
    renderRecentSigns();
    updateCurrentSign('—');
    updateConfidence(null);
    updateLatency(null);
    if (composerInput) {
      composerInput.value = '';
      if (clearBtn) clearBtn.classList.add('hidden');
    }
  });

  // TEARDOWN FUNCTION: Clean up subscriptions
  return () => {
    // Disconnect cleanly from WebSocket relay
    communicationService.disconnect();

    speechService.stop();
    adminTtsService.stop();
    unsubTtsState();
    unsubRecog();
    unsubHosp();
    unsubConv();
    unsubDeafMsg();
    unsubDeafRecog();
    unsubAiRecog();
    unsubHospPhrase();
    unsubTranscriptSync();
    unsubCommConnected();
    unsubCommDisconnected();
    unsubReset();
    if (liveBadgeTimer) clearTimeout(liveBadgeTimer);
    if (processingStatusTimer) clearTimeout(processingStatusTimer);
  };
}
