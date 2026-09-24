import { renderSidebar } from '../components/Sidebar.js';
import { renderDateRangeFilter } from '../components/DateRangeFilter.js';
import { ParticleBackground } from '../components/ParticleBackground.js';
import { conversationStore } from '../state/conversationStore.js';

export function renderHistoryPage() {
  return `
    <div class="h-screen w-full flex bg-surface font-body text-on-surface antialiased overflow-hidden relative">
      <!-- Full-Screen Continuous Roaming Particle Field Canvas (Subtle Background) -->
      <canvas id="particleCanvas" class="fixed inset-0 pointer-events-none z-0 w-full h-full opacity-60"></canvas>

      <!-- Reusable Left Sidebar -->
      ${renderSidebar('#/history')}

      <!-- Main Chat History Interface Viewport -->
      <div class="flex-1 flex flex-col h-screen overflow-hidden relative z-10">
        <!-- Top Sticky Filter Bar -->
        ${renderDateRangeFilter({
          fromDate: '2026-09-24',
          toDate: '2026-09-30',
          activePreset: '7days',
          totalCount: conversationStore.getHistoryRecords().length
        })}

        <!-- Scrollable Conversation Records List -->
        <main class="flex-1 overflow-y-auto p-4 lg:p-6 w-full" id="history-scroll-container">
          <div class="max-w-5xl mx-auto space-y-4" id="conversation-cards-list">
            <!-- Cards rendered dynamically by initHistoryPage -->
          </div>
        </main>
      </div>

      <!-- Detail Dialog Modal (Hidden by Default) -->
      <div id="chat-detail-modal" class="hidden fixed inset-0 z-50 flex items-center justify-center p-4 bg-primary/40 backdrop-blur-xs">
        <div class="bg-surface-container-lowest border border-outline-variant/50 rounded-2xl max-w-2xl w-full p-6 shadow-2xl flex flex-col max-h-[90vh] overflow-hidden animate-fadeIn">
          <div class="flex items-center justify-between pb-4 border-b border-outline-variant/30 flex-shrink-0">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-white">
                <span class="material-symbols-outlined text-[20px]">receipt_long</span>
              </div>
              <div>
                <h3 class="text-lg font-bold text-primary" id="modal-chat-id">Chat ID</h3>
                <span class="text-xs text-on-surface-variant font-mono" id="modal-chat-meta">Timestamp • Desk</span>
              </div>
            </div>
            <button type="button" 
                    id="modal-close-btn"
                    class="p-1.5 rounded-lg text-slate-400 hover:text-primary hover:bg-surface-container transition-colors cursor-pointer">
              <span class="material-symbols-outlined text-[20px]">close</span>
            </button>
          </div>

          <!-- Modal Transcript Body -->
          <div class="flex-1 overflow-y-auto py-4 space-y-3 pr-1" id="modal-messages-container">
            <!-- Messages injected here -->
          </div>

          <!-- Modal Footer -->
          <div class="pt-4 border-t border-outline-variant/30 flex items-center justify-between text-xs text-on-surface-variant flex-shrink-0">
            <span class="flex items-center gap-1.5 text-emerald-700 font-semibold">
              <span class="material-symbols-outlined text-[16px]">verified</span>
              Cryptographically Verified Civic Log
            </span>
            <button type="button" 
                    onclick="window.print()" 
                    class="px-3 py-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary font-bold transition-colors cursor-pointer">
              Print Case File
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
}

export function initHistoryPage() {
  // Init subtle particle background
  const canvas = document.getElementById('particleCanvas');
  let particleBg = null;
  if (canvas) {
    particleBg = new ParticleBackground(canvas);
  }

  const allRecords = conversationStore.getHistoryRecords();
  let activeRecords = [...allRecords];
  const cardsList = document.getElementById('conversation-cards-list');
  const countBadge = document.getElementById('records-count-badge');
  const dateFrom = document.getElementById('date-from');
  const dateTo = document.getElementById('date-to');
  const modal = document.getElementById('chat-detail-modal');
  const modalCloseBtn = document.getElementById('modal-close-btn');

  // Render conversation card items
  function renderCards(records) {
    if (!cardsList) return;

    if (records.length === 0) {
      cardsList.innerHTML = `
        <div class="p-12 text-center bg-white rounded-xl border border-slate-200 shadow-xs">
          <span class="material-symbols-outlined text-slate-400 text-[48px] mb-2">event_busy</span>
          <h3 class="text-base font-bold text-slate-700">No conversation records found</h3>
          <p class="text-xs text-slate-500 mt-1">Try expanding your date filter or selecting "Last 30 Days".</p>
        </div>
      `;
      if (countBadge) countBadge.textContent = 'Showing 0 sessions';
      return;
    }

    cardsList.innerHTML = records.map((rec) => `
      <article class="chat-card bg-white border border-slate-200 hover:border-[#0f2942]/60 rounded-xl p-5 hover:shadow-md transition-all duration-150 cursor-pointer group shadow-xs" 
               data-id="${rec.id}">
        <!-- Top Metadata Row -->
        <div class="flex flex-wrap items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
          <div class="flex items-center gap-2.5">
            <span class="font-mono text-xs font-bold text-slate-700 bg-slate-100 px-2.5 py-1 rounded-md">
              Chat ID: ${rec.id}
            </span>
            <span class="inline-flex items-center gap-1.5 text-xs text-emerald-800 font-bold bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 rounded-full">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse"></span>
              ${rec.status || 'Completed'}
            </span>
          </div>
          <div class="text-xs font-semibold text-slate-500 flex items-center gap-1.5 font-mono">
            <span class="material-symbols-outlined text-[16px] text-slate-400">schedule</span>
            <span>${rec.displayDate} • ${rec.time}</span>
          </div>
        </div>

        <!-- Conversation Previews -->
        <div class="space-y-2 text-sm text-slate-800">
          <div class="flex items-start gap-3">
            <span class="text-xs font-bold uppercase tracking-wider text-slate-500 w-24 shrink-0 pt-0.5">
              Admin:
            </span>
            <p class="text-slate-700 font-medium leading-relaxed">
              "${rec.adminPreview}"
            </p>
          </div>
          <div class="flex items-start gap-3">
            <span class="text-xs font-bold uppercase tracking-wider text-secondary w-24 shrink-0 pt-0.5">
              Deaf Person:
            </span>
            <p class="text-slate-900 font-bold leading-relaxed">
              "${rec.deafPreview}"
            </p>
          </div>
        </div>

        <!-- Bottom Click Prompt -->
        <div class="mt-3 pt-2 border-t border-slate-50 flex items-center justify-between text-xs text-slate-400">
          <span class="font-mono text-[11px]">${rec.counter} • ${rec.adminName}</span>
          <span class="group-hover:text-primary font-semibold flex items-center gap-1 transition-colors">
            <span>View Transcript</span>
            <span class="material-symbols-outlined text-[16px]">arrow_forward</span>
          </span>
        </div>
      </article>
    `).join('');

    if (countBadge) {
      countBadge.textContent = `Showing ${records.length} session${records.length === 1 ? '' : 's'}`;
    }

    // Bind card clicks to open detail modal
    document.querySelectorAll('.chat-card').forEach((card) => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-id');
        const record = conversationStore.getHistoryRecords().find((r) => r.id === id);
        if (record) openDetailModal(record);
      });
    });
  }

  // Open Detailed Modal
  function openDetailModal(record) {
    if (!modal) return;
    const titleEl = document.getElementById('modal-chat-id');
    const metaEl = document.getElementById('modal-chat-meta');
    const messagesEl = document.getElementById('modal-messages-container');

    if (titleEl) titleEl.textContent = `Session ${record.id}`;
    if (metaEl) metaEl.textContent = `${record.displayDate} at ${record.time} • ${record.counter} • ${record.adminName}`;

    if (messagesEl && record.messages) {
      messagesEl.innerHTML = record.messages.map((m) => {
        const isAdmin = m.sender === 'admin';
        return `
          <div class="flex flex-col gap-1 ${isAdmin ? 'items-start' : 'items-end'}">
            <div class="text-[11px] font-bold text-slate-500 font-mono">
              ${isAdmin ? 'Officer Vance' : 'Deaf Citizen'} • ${m.time}
            </div>
            <div class="max-w-[85%] p-3 rounded-xl text-sm ${isAdmin ? 'bg-slate-100 text-slate-800' : 'bg-teal-50 border border-teal-200 text-teal-950 font-medium'}">
              "${m.text}"
            </div>
          </div>
        `;
      }).join('');
    }

    modal.classList.remove('hidden');
  }

  // Close Detail Modal
  if (modalCloseBtn && modal) {
    modalCloseBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
    });
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });
  }

  // Filter Logic:
  // Today = 2026-09-24
  // Last 7 Days = 2026-09-17 to 2026-09-24
  // Last 30 Days = 2026-08-25 to 2026-09-24
  function applyDateFilter(preset) {
    const todayStr = '2026-09-24';
    let from = '2026-09-17';
    let to = todayStr;

    if (preset === 'today') {
      from = todayStr;
      to = todayStr;
    } else if (preset === '7days') {
      from = '2026-09-17';
      to = todayStr;
    } else if (preset === '30days') {
      from = '2026-08-25';
      to = todayStr;
    }

    if (dateFrom) dateFrom.value = from;
    if (dateTo) dateTo.value = to;

    updateQuickButtonStyles(preset);
    filterRecordsByDates(from, to);
  }

  function updateQuickButtonStyles(activePreset) {
    document.querySelectorAll('.date-filter-preset-btn').forEach((btn) => {
      const p = btn.getAttribute('data-preset');
      if (p === activePreset) {
        btn.className = 'date-filter-preset-btn px-3 py-1.5 text-xs rounded-lg transition-all cursor-pointer bg-[#001428] text-white shadow-xs font-bold';
      } else {
        btn.className = 'date-filter-preset-btn px-3 py-1.5 text-xs rounded-lg transition-all cursor-pointer bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-semibold';
      }
    });
  }

  function filterRecordsByDates(from, to) {
    const currentRecords = conversationStore.getHistoryRecords();
    const filtered = currentRecords.filter((r) => {
      return r.date >= from && r.date <= to;
    });
    renderCards(filtered);
  }

  // Bind Quick Buttons
  document.querySelectorAll('.date-filter-preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const preset = btn.getAttribute('data-preset');
      if (preset) applyDateFilter(preset);
    });
  });

  // Bind manual date change
  if (dateFrom && dateTo) {
    const handleManual = () => {
      updateQuickButtonStyles(null);
      filterRecordsByDates(dateFrom.value, dateTo.value);
    };
    dateFrom.addEventListener('change', handleManual);
    dateTo.addEventListener('change', handleManual);
  }

  // Initial render (defaults to 7 days)
  applyDateFilter('7days');

  return () => {
    if (particleBg) particleBg.destroy();
  };
}
