// Reusable Chat Conversation Panel Component
import { renderChatMessage } from './ChatMessage.js';
import { renderMessageComposer } from './MessageComposer.js';

export function renderChatPanel({
  conversation = [],
  isDeafView = true,
  composerText = ''
}) {
  const badgeText = isDeafView ? 'Live Stream' : 'Synchronized';
  const badgeClass = isDeafView
    ? 'bg-secondary-container/50 text-secondary'
    : 'bg-emerald-50 text-emerald-800 border border-emerald-200';

  const dotClass = isDeafView
    ? 'bg-secondary animate-pulse'
    : 'bg-emerald-600 animate-pulse';

  const messagesHtml = conversation.map(renderChatMessage).join('');

  return `
    <div class="flex flex-col h-full overflow-hidden">
      <!-- Chat Header -->
      <div class="flex items-center justify-between pb-3.5 border-b border-outline-variant/30 flex-shrink-0">
        <div class="flex items-center gap-2">
          <span class="material-symbols-outlined text-secondary text-[22px]">forum</span>
          <h2 class="text-xl font-extrabold text-primary tracking-tight">Conversation</h2>
        </div>
        <div class="flex items-center gap-2">
          <button type="button" 
                  id="btn-reset-demo" 
                  class="p-1 rounded-lg text-outline hover:text-primary hover:bg-surface-container transition-colors cursor-pointer"
                  title="Reset Demo Session">
            <span class="material-symbols-outlined text-[18px]">restart_alt</span>
          </button>
          <div class="flex items-center gap-1.5 px-2.5 py-1 rounded-full ${badgeClass} text-xs font-bold uppercase tracking-wider shadow-xs">
            <span class="w-2 h-2 rounded-full ${dotClass}"></span>
            <span>${badgeText}</span>
          </div>
        </div>
      </div>

      <!-- Scrollable Message Timeline Stream -->
      <div id="chat-stream-timeline" class="flex-1 overflow-y-auto pr-1 py-3.5 flex flex-col gap-3 min-h-0">
        ${messagesHtml}
        
        <!-- Live Pending Signing indicator for Deaf View -->
        ${isDeafView ? `
          <div id="live-signing-indicator" class="hidden flex flex-col gap-1 items-end mt-1 animate-fadeIn">
            <div class="flex items-center gap-1.5 text-xs text-on-surface-variant pr-1">
              <span class="text-[10px] text-emerald-600 font-mono font-bold uppercase tracking-wider flex items-center gap-1">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping"></span>
                Signing now
              </span>
              <span class="font-bold text-primary">Deaf Person</span>
            </div>
            <div class="max-w-[90%] bg-emerald-50/90 border border-emerald-300 text-emerald-950 rounded-2xl rounded-tr-xs px-3.5 py-2.5 shadow-xs">
              <p class="text-sm font-medium leading-relaxed italic flex items-center gap-1.5">
                <span id="live-signing-text">"..."</span>
                <span class="inline-block w-1.5 h-3.5 bg-emerald-600 rounded-xs animate-pulse"></span>
              </p>
            </div>
          </div>
        ` : ''}
      </div>

      <!-- Bottom: Either Minimal Status Footer (Deaf) or Fixed Composer (Admin) -->
      ${isDeafView ? `
        <div class="pt-3 border-t border-outline-variant/30 flex items-center justify-between text-xs text-on-surface-variant flex-shrink-0">
          <span class="flex items-center gap-1.5 font-medium">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
            Encrypted Civic Relay Active
          </span>
          <span class="font-mono text-[11px] text-outline">Desk #04</span>
        </div>
      ` : `
        ${renderMessageComposer(composerText)}
      `}
    </div>
  `;
}
