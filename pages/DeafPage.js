// Deaf Person Interface View (3-Section Balanced Viewport)
import { renderSidebar } from '../components/Sidebar.js';
import { renderLiveCamera } from '../components/LiveCamera.js';
import { renderSignTranscript } from '../components/SignTranscript.js';
import { renderSignAnimation } from '../components/SignAnimation.js';
import { renderChatPanel } from '../components/ChatPanel.js';
import { cameraService } from '../services/cameraService.js';
import { signRecognitionService } from '../services/signRecognitionService.js';
import { signAnimationService } from '../services/signAnimationService.js';
import { conversationStore } from '../state/conversationStore.js';
import { communicationService } from '../services/communicationService.js';

export function renderDeafPage() {
  const conversation = conversationStore.getConversation();
  const animState = signAnimationService.getState();
  const transcript = signRecognitionService.getTranscript();

  return `
    <div class="h-screen w-full flex bg-surface font-body text-on-surface antialiased overflow-hidden">
      <!-- Reusable Left Sidebar -->
      ${renderSidebar('#/deaf')}

      <!-- Main Three-Panel Viewport (~33% / ~33% / ~34%) -->
      <main class="flex-1 h-full overflow-y-auto lg:overflow-hidden p-4 lg:p-5 bg-surface flex flex-col">
        <!-- Top Mobile Header / Bar (Only visible on small screens) -->
        <div class="lg:hidden flex items-center justify-between pb-3 mb-2 border-b border-outline-variant/30 flex-shrink-0">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-primary text-[24px]">sign_language</span>
            <span class="font-bold text-lg text-primary">Deaf Person Interface</span>
          </div>
          <span class="text-xs px-2.5 py-1 rounded-full bg-secondary-container text-on-secondary-container font-bold uppercase tracking-wider">
            Desk #04
          </span>
        </div>

        <!-- 3-Panel Split View -->
        <div class="w-full flex-1 flex flex-col lg:flex-row gap-4 lg:gap-5 min-h-0">
          <!-- SECTION 1 (LEFT ~33%): REAL LIVE CAMERA & SIGN-TO-TEXT TRANSCRIPT -->
          <section class="flex-1 lg:w-[33%] flex flex-col h-full bg-surface-container-lowest rounded-2xl p-4 lg:p-5 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderLiveCamera({
              isDeafView: true,
              title: 'Start speaking',
              isTurnActive: animState.cameraResponseComplete,
              turnStatus: animState.cameraResponseComplete ? 'Your Turn to Sign' : 'Observing Signer'
            })}
            ${renderSignTranscript(
              transcript,
              'deaf-live-transcript',
              signRecognitionService.getStatusText(),
              'deaf-recognition-status'
            )}
          </section>

          <!-- SECTION 2 (CENTER ~33%): ANIMATED SIGN REPLY (AUTOMATIC 2X PLAYBACK) -->
          <section class="flex-1 lg:w-[33%] flex flex-col h-full bg-surface-container-lowest rounded-2xl p-4 lg:p-5 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderSignAnimation(animState)}
          </section>

          <!-- SECTION 3 (RIGHT ~34%): CONVERSATION (CHAT STREAM) -->
          <section class="flex-1 lg:w-[34%] flex flex-col h-full bg-surface-container-lowest rounded-2xl p-4 lg:p-5 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderChatPanel({
              conversation: conversation,
              isDeafView: true
            })}
          </section>
        </div>
      </main>
    </div>
  `;
}

export function initDeafPage() {
  const videoEl = document.getElementById('deaf-camera-video');
  const permissionBanner = document.getElementById('camera-permission-banner');
  const cameraErrorMessage = document.getElementById('camera-error-message');
  const btnRequestPermission = document.getElementById('btn-request-permission');

  const cameraContainer = document.getElementById('deaf-camera-container');
  const turnBadge = document.getElementById('deaf-turn-badge');
  const turnIndicatorDot = document.getElementById('turn-indicator-dot');
  const turnStatusText = document.getElementById('turn-status-text');

  // Visual helper for Green Camera Turn-Taking state (Subtle border/glow ONLY on 2x response completion)
  // THE REAL LIVE CAMERA FEED REMAINS 100% VISIBLE AT ALL TIMES
  function applyGreenCameraState(active) {
    if (active) {
      if (cameraContainer) {
        cameraContainer.classList.add('camera-turn-active');
        cameraContainer.classList.remove('border-outline-variant/30');
      }
      if (turnBadge) {
        turnBadge.className = 'transition-all duration-300 flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-100 text-emerald-900 border border-emerald-300 font-bold text-xs uppercase tracking-wider shadow-sm';
      }
      if (turnIndicatorDot) {
        turnIndicatorDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping';
      }
      if (turnStatusText) {
        turnStatusText.textContent = 'Your Turn to Sign';
      }
    } else {
      if (cameraContainer) {
        cameraContainer.classList.remove('camera-turn-active');
        cameraContainer.classList.add('border-outline-variant/30');
      }
      if (turnBadge) {
        turnBadge.className = 'transition-all duration-300 flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface-container text-on-surface-variant font-bold text-xs uppercase tracking-wider';
      }
      if (turnIndicatorDot) {
        turnIndicatorDot.className = 'w-2 h-2 rounded-full bg-outline-variant';
      }
      if (turnStatusText) {
        turnStatusText.textContent = 'Observing Signer';
      }
    }
  }

  // AUTOMATIC CAMERA START: Immediately request camera permission & stream live feed
  async function startLiveCamera() {
    if (!videoEl) return;
    const res = await cameraService.startCamera(videoEl);
    if (res.success) {
      if (permissionBanner) permissionBanner.classList.add('hidden');
      // Connect Sign-to-Text Recognition to existing live webcam element (No extra camera stream!)
      signRecognitionService.startRecognition(videoEl);
    } else {
      if (permissionBanner) {
        permissionBanner.classList.remove('hidden');
        if (cameraErrorMessage && res.error) {
          cameraErrorMessage.textContent = res.error;
        }
      }
    }
  }

  // Automatically start immediately on page load
  startLiveCamera();

  // Retry permission if previously denied
  if (btnRequestPermission) {
    btnRequestPermission.addEventListener('click', () => {
      startLiveCamera();
    });
  }

  // Auto-scroll chat timeline to bottom
  function scrollToBottom() {
    const chatTimeline = document.getElementById('chat-stream-timeline');
    if (chatTimeline) {
      chatTimeline.scrollTop = chatTimeline.scrollHeight;
    }
  }

  // Replay Button listener for sign animation
  const replayBtn = document.getElementById('btn-replay-animation');
  if (replayBtn) {
    replayBtn.addEventListener('click', () => {
      applyGreenCameraState(false);
      signAnimationService.replay();
    });
  }

  // Mock Sign Selection Demo Control (Dropdown + Send Button)
  const demoSignSelect = document.getElementById('demo-sign-select');
  const btnTriggerSign = document.getElementById('btn-trigger-selected-sign');

  if (btnTriggerSign && demoSignSelect) {
    btnTriggerSign.addEventListener('click', () => {
      const phrase = demoSignSelect.value;
      if (phrase) {
        signRecognitionService.recognizePhrase(phrase);
      }
    });
  }

  // Quick 1-click Sign Simulation Pills
  document.querySelectorAll('.btn-simulate-sign').forEach((btn) => {
    btn.addEventListener('click', () => {
      const signText = btn.getAttribute('data-sign');
      if (signText) {
        if (demoSignSelect) demoSignSelect.value = signText;
        signRecognitionService.recognizePhrase(signText);
      }
    });
  });

  // Demo Reset Button
  const resetDemoBtn = document.getElementById('btn-reset-demo');
  if (resetDemoBtn) {
    resetDemoBtn.addEventListener('click', () => {
      conversationStore.resetDemo();
      signRecognitionService.reset();
      signAnimationService.stop();
      applyGreenCameraState(false);
    });
  }

  // Subscribe to Sign Animation Lifecycle (2X Automatic Playback)
  const unsubAnim = signAnimationService.subscribe((state) => {
    const progressBar = document.getElementById('playback-progress');
    const timeRemaining = document.getElementById('time-remaining');
    const cycleLabel = document.getElementById('cycle-label');
    const playbackBadge = document.getElementById('playback-badge');
    const playbackStatusText = document.getElementById('playback-status-text');
    const syncText = document.getElementById('sync-animation-text');

    if (progressBar) progressBar.style.width = state.progressPercent + '%';
    if (timeRemaining) timeRemaining.textContent = state.timeDisplay;
    if (syncText) syncText.textContent = `"${state.text}"`;

    if (state.completed) {
      if (cycleLabel) cycleLabel.textContent = 'Completed 2 of 2 iterations';
      if (playbackBadge) {
        playbackBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-900 border border-emerald-300 text-xs font-bold uppercase tracking-wide';
        playbackBadge.innerHTML = `
          <span class="material-symbols-outlined text-[16px] text-emerald-700">check_circle</span>
          <span>Complete (2 of 2)</span>
        `;
      }
      applyGreenCameraState(true);
    } else if (state.isPlaying) {
      if (cycleLabel) cycleLabel.textContent = `Cycle ${state.cycle} of ${state.maxCycles} (Playing)`;
      if (playbackBadge) {
        playbackBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-primary-container text-secondary-container text-xs font-bold uppercase tracking-wide shadow-xs';
        playbackBadge.innerHTML = `
          <span class="w-2 h-2 rounded-full bg-secondary animate-ping"></span>
          <span>Playing ${state.cycle} of ${state.maxCycles}</span>
        `;
      }
      applyGreenCameraState(false);
    }
  });

  // Subscribe to Sign Recognition updates (progressive real-time text & status)
  const unsubRecog = signRecognitionService.subscribe((recog) => {
    const transcriptEl = document.getElementById('deaf-live-transcript');
    if (transcriptEl) {
      transcriptEl.innerHTML = `"${recog.transcript}" <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>`;
    }

    const statusTextEl = document.getElementById('deaf-recognition-status');
    const statusDotEl = document.getElementById('deaf-recognition-status-dot');
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

    const liveSigningIndicator = document.getElementById('live-signing-indicator');
    const liveSigningText = document.getElementById('live-signing-text');
    if (liveSigningIndicator && liveSigningText) {
      if (recog.isRecognizing) {
        liveSigningIndicator.classList.remove('hidden');
        liveSigningText.textContent = `"${recog.transcript}"`;
        scrollToBottom();
      } else {
        liveSigningIndicator.classList.add('hidden');
      }
    }
  });

  // Subscribe to Conversation Store updates
  const unsubConv = conversationStore.subscribe((conversation) => {
    const chatTimeline = document.getElementById('chat-stream-timeline');
    if (chatTimeline) {
      const messagesHtml = conversation.map(msg => {
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
                <p class="text-sm font-semibold leading-relaxed">"${msg.text}"</p>
              </div>
            </div>
          `;
        }
      }).join('');

      // Keep live signing indicator at bottom if active
      const liveSigningIndicator = document.getElementById('live-signing-indicator');
      chatTimeline.innerHTML = messagesHtml + (liveSigningIndicator ? liveSigningIndicator.outerHTML : '');
      scrollToBottom();
    }
  });

  // Listen for incoming Admin messages via communicationService (triggers 2x sign animation!)
  const unsubComm = communicationService.on('ADMIN_MESSAGE_SENT', (payload) => {
    if (payload && payload.text) {
      applyGreenCameraState(false);
      signAnimationService.playAnimationForMessage(payload.text);
    }
  });

  // Listen for demo reset across tabs
  const unsubReset = communicationService.on('DEMO_RESET', () => {
    applyGreenCameraState(false);
    signAnimationService.stop();
    signRecognitionService.reset();
  });

  // Initial check on camera state
  const currentAnim = signAnimationService.getState();
  if (currentAnim.cameraResponseComplete) {
    applyGreenCameraState(true);
  }
  scrollToBottom();

  // TEARDOWN FUNCTION: Called when navigating away
  return () => {
    // Stop all active camera tracks so webcam light turns off
    cameraService.stopCamera();
    signRecognitionService.stopRecognition();
    signAnimationService.stop();
    unsubAnim();
    unsubRecog();
    unsubConv();
    unsubComm();
    unsubReset();
  };
}
