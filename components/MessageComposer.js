// Message Composer Component for Admin Interface
import { stateBridge } from '../services/stateBridge.js';

export function renderMessageComposer(initialText = '', activeMode = 'FINGERSPELLING') {
  return `
    <div class="pt-3 border-t border-outline-variant/30 flex-shrink-0 flex flex-col gap-2">
      <!-- Active Signing Mode Indicator Badge -->
      <div class="flex items-center justify-between px-1">
        <span class="text-[11px] font-bold uppercase tracking-wider text-on-surface-variant flex items-center gap-1.5">
          <span class="material-symbols-outlined text-primary text-[15px]">sign_language</span>
          <span>Target Relay Mode:</span>
          <span id="composer-mode-badge" class="px-2 py-0.5 rounded-full bg-primary/10 text-primary font-mono text-[11px] font-bold">
            ${activeMode}
          </span>
        </span>
        <span id="speech-recognition-status" class="hidden text-[11px] font-semibold text-rose-600 animate-pulse flex items-center gap-1">
          <span class="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span>
          <span>Listening...</span>
        </span>
      </div>

      <!-- Composer input form -->
      <form id="admin-composer-form" class="flex items-center gap-2">
        <div class="relative flex-1">
          <input type="text" 
                 id="composer-input"
                 value="${escapeAttr(initialText)}"
                 placeholder="Type a message or select a preset..." 
                 class="w-full h-11 lg:h-12 pl-3.5 pr-20 rounded-xl bg-surface-container-low text-on-surface placeholder:text-outline border border-outline-variant/50 focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none text-sm lg:text-base transition-colors"
                 autocomplete="off" />

          <!-- Action buttons inside right of input -->
          <div class="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
            <button type="button" 
                    id="composer-clear-btn" 
                    class="${initialText ? 'flex' : 'hidden'} text-on-surface-variant hover:text-on-surface p-1.5 rounded-md transition-colors cursor-pointer"
                    title="Clear text">
              <span class="material-symbols-outlined text-[18px]">close</span>
            </button>
            <button type="button" 
                    id="composer-mic-btn" 
                    class="text-on-surface-variant hover:text-primary p-1.5 rounded-md hover:bg-surface-container transition-colors cursor-pointer"
                    title="Voice Input (Speech-to-Text)">
              <span class="material-symbols-outlined text-[19px]" id="composer-mic-icon">mic</span>
            </button>
          </div>
        </div>

        <button type="submit" 
                id="composer-send-btn"
                class="flex items-center justify-center gap-1.5 px-4 lg:px-5 h-11 lg:h-12 rounded-xl bg-[#001428] hover:bg-[#0f2942] active:bg-[#000d1a] text-white font-bold text-sm transition-colors cursor-pointer shadow-xs min-w-[85px] lg:min-w-[95px] flex-shrink-0">
          <span>Send</span>
          <span class="material-symbols-outlined text-[18px]">send</span>
        </button>
      </form>
    </div>
  `;
}

function escapeAttr(str) {
  if (!str) return '';
  return String(str).replace(/"/g, '&quot;');
}
