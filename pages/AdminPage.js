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

export function renderAdminPage() {
  const conversation = conversationStore.getConversation();
  const transcript = signRecognitionService.getTranscript();

  return `
    <div class="h-screen w-full flex bg-surface font-body text-on-surface antialiased overflow-hidden">
      <!-- Reusable Left Sidebar -->
      ${renderSidebar('#/admin')}

      <!-- Main Three-Panel Viewport (~33% / ~33% / ~34%) -->
      <main class="flex-1 h-full overflow-y-auto lg:overflow-hidden p-4 lg:p-5 bg-surface flex flex-col">
        <!-- Top Operational Bar -->
        <div class="flex items-center justify-between pb-3.5 mb-2 border-b border-outline-variant/30 flex-shrink-0">
          <div class="flex items-center gap-3">
            <span class="material-symbols-outlined text-primary text-[24px]">support_agent</span>
            <div>
              <h1 class="text-xl lg:text-2xl font-extrabold text-primary tracking-tight">Staff Communication Console</h1>
              <span class="text-xs text-on-surface-variant font-medium">Bi-directional Live Relay &amp; Presets</span>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <div class="flex items-center gap-1.5 px-3 py-1 rounded-full bg-secondary-container text-on-secondary-container text-xs font-bold uppercase tracking-wider shadow-xs">
              <span class="w-2 h-2 rounded-full bg-secondary animate-pulse"></span>
              <span>Desk #04 Active</span>
            </div>
          </div>
        </div>

        <!-- 3-Panel Split Layout -->
        <div class="w-full flex-1 flex flex-col lg:flex-row gap-4 lg:gap-5 min-h-0">
          <!-- SECTION 1 (LEFT ~33%): DEAF PERSON LIVE FEED & REAL-TIME RECOGNIZED TEXT -->
          <section class="flex-1 lg:w-[33%] flex flex-col h-full bg-surface-container-lowest rounded-2xl p-4 lg:p-5 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderLiveCamera({
              isDeafView: false,
              title: 'Deaf Person',
              isTurnActive: false
            })}
            ${renderSignTranscript(
              transcript,
              'admin-live-transcript',
              'Citizen Stream Active',
              'admin-recognition-status'
            )}
          </section>

          <!-- SECTION 2 (CENTER ~33%): PRESET RESPONSES (BANK, HOSPITAL, GOV OFFICE) -->
          <!-- Note: NO sign language animation video on Admin page as per requirement -->
          <section class="flex-1 lg:w-[33%] flex flex-col h-full bg-surface-container-lowest rounded-2xl p-4 lg:p-5 shadow-sm border border-outline-variant/30 overflow-hidden" id="admin-preset-section">
            ${renderPresetLibrary(PRESET_CATEGORIES.HOSPITAL)}
          </section>

          <!-- SECTION 3 (RIGHT ~34%): CONVERSATION + FIXED MESSAGE COMPOSER -->
          <section class="flex-1 lg:w-[34%] flex flex-col h-full bg-surface-container-lowest rounded-2xl p-4 lg:p-5 shadow-sm border border-outline-variant/30 overflow-hidden">
            ${renderChatPanel({
              conversation: conversation,
              isDeafView: false,
              composerText: ''
            })}
          </section>
        </div>
      </main>
    </div>
  `;
}

export function initAdminPage() {
  const composerInput = document.getElementById('composer-input');
  const clearBtn = document.getElementById('composer-clear-btn');
  const composerForm = document.getElementById('admin-composer-form');
  let currentCategory = PRESET_CATEGORIES.HOSPITAL;

  // Helper to scroll conversation timeline
  function scrollToBottom() {
    const chatTimeline = document.getElementById('chat-stream-timeline');
    if (chatTimeline) {
      chatTimeline.scrollTop = chatTimeline.scrollHeight;
    }
  }

  // Bind Preset Item Clicks -> Populates Composer WITHOUT Sending
  function bindPresetClicks() {
    document.querySelectorAll('.preset-item-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const text = btn.getAttribute('data-text');
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
          const presetSection = document.getElementById('admin-preset-section');
          if (presetSection) {
            presetSection.innerHTML = renderPresetLibrary(category);
            bindCategoryTabs();
            bindPresetClicks();
          }
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

  // Send message action
  function submitMessage() {
    if (!composerInput) return;
    const messageText = composerInput.value.trim();
    if (!messageText) return;

    // 1. Add Admin Message to conversationStore
    conversationStore.addMessage({
      sender: 'admin',
      senderName: 'Admin (Officer Vance)',
      text: messageText,
      type: 'text',
      isActiveReply: true
    });

    // 2. Dispatch to Deaf interface via communicationService
    communicationService.emit('ADMIN_MESSAGE_SENT', {
      text: messageText,
      timestamp: Date.now()
    });

    // 3. Clear composer input
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
  bindCategoryTabs();
  bindPresetClicks();
  scrollToBottom();

  // Subscribe to live recognized sign text from Deaf terminal
  const unsubRecog = signRecognitionService.subscribe((recog) => {
    const transcriptEl = document.getElementById('admin-live-transcript');
    if (transcriptEl) {
      transcriptEl.innerHTML = `"${recog.transcript}" <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>`;
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

  // Subscribe to conversationStore updates
  const unsubConv = conversationStore.subscribe((conversation) => {
    const chatTimeline = document.getElementById('chat-stream-timeline');
    if (chatTimeline) {
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
                <p class="text-sm font-semibold leading-relaxed">"${msg.text}"</p>
              </div>
            </div>
          `;
        }
      }).join('');
      scrollToBottom();
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
    cameraService.disconnectAdminFeed();
    unsubRecog();
    unsubConv();
    unsubReset();
  };
}
