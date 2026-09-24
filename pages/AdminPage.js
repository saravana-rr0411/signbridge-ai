// Staff / Admin Interface View (3-Section Balanced Viewport)
import { renderSidebar } from '../components/Sidebar.js';
import { renderLiveCamera } from '../components/LiveCamera.js';
import { renderSignTranscript } from '../components/SignTranscript.js';
import { renderPresetLibrary } from '../components/PresetLibrary.js';
import { renderChatPanel } from '../components/ChatPanel.js';
import { conversationStore } from '../state/conversationStore.js';
import { signRecognitionService } from '../services/signRecognitionService.js';
import { cameraService } from '../services/cameraService.js';
import { communicationService } from '../services/communicationService.js';
import { PRESET_CATEGORIES } from '../services/mockData.js';
import { hospitalConversationService } from '../services/conversation/hospitalConversationService.js';
import { SIGNING_MODES, getSignConfig } from '../services/signAnimation/signConfig.js';
import { speechService } from '../services/signAnimation/speechRecognition.js';
import { adminTtsService } from '../services/adminTtsService.js';

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
          <!-- SECTION 1 (LEFT): DEAF PERSON LIVE FEED & REAL-TIME RECOGNIZED TEXT -->
          <section class="flex-1 lg:w-1/3 min-w-0 min-h-0 flex flex-col h-full bg-surface-container-lowest rounded-2xl p-3.5 lg:p-4 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderLiveCamera({
              isDeafView: false,
              title: 'Deaf Person',
              isTurnActive: false
            })}
            ${renderSignTranscript(
              transcript,
              'admin-live-transcript',
              'Citizen Stream Active',
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
      renderChatMessages(conversationStore.getConversation());
    }
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
    }
  });

  // Connect Admin video element to real live Deaf person's MediaStream (WebRTC / shared stream)
  const adminVideo = document.getElementById('admin-camera-video');
  const waitingOverlay = document.getElementById('admin-waiting-stream');

  if (adminVideo) {
    const handleVideoPlaying = () => {
      if (waitingOverlay) waitingOverlay.classList.add('hidden');
    };
    adminVideo.addEventListener('playing', handleVideoPlaying);
    adminVideo.addEventListener('loadeddata', handleVideoPlaying);
    adminVideo.addEventListener('loadedmetadata', handleVideoPlaying);
  }

  cameraService.connectAdminFeed(adminVideo, (isStreaming) => {
    if (waitingOverlay) {
      if (isStreaming) {
        waitingOverlay.classList.add('hidden');
      } else {
        waitingOverlay.classList.remove('hidden');
      }
    }
  });

  // Demo Reset Button
  const resetDemoBtn = document.getElementById('btn-reset-demo');
  if (resetDemoBtn) {
    resetDemoBtn.addEventListener('click', () => {
      conversationStore.resetDemo();
      signRecognitionService.reset();
      hospitalConversationService.reset();
      if (composerInput) {
        composerInput.value = '';
        if (clearBtn) clearBtn.classList.add('hidden');
      }
    });
  }

  const unsubReset = communicationService.on('DEMO_RESET', () => {
    if (composerInput) {
      composerInput.value = '';
      if (clearBtn) clearBtn.classList.add('hidden');
    }
  });

  // TEARDOWN FUNCTION: Clean up subscriptions and remote video
  return () => {
    // Disconnect cleanly from WebSocket relay
    communicationService.disconnect();

    speechService.stop();
    adminTtsService.stop();
    unsubTtsState();
    cameraService.disconnectAdminFeed();
    unsubRecog();
    unsubHosp();
    unsubConv();
    unsubDeafMsg();
    unsubDeafRecog();
    unsubTranscriptSync();
    unsubReset();
  };
}
