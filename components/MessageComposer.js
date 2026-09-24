// Message Composer Component for Admin Interface
import { stateBridge } from '../services/stateBridge.js';

export function renderMessageComposer(initialText = '') {
  return `
    <div class="pt-3 border-t border-outline-variant/30 flex-shrink-0 flex flex-col gap-2">
      <!-- Composer input form -->
      <form id="admin-composer-form" class="flex items-center gap-2">
        <div class="relative flex-1">
          <input type="text" 
                 id="composer-input"
                 value="${escapeAttr(initialText)}"
                 placeholder="Type a message or select a preset..." 
                 class="w-full h-12 pl-4 pr-10 rounded-xl bg-surface-container-low text-on-surface placeholder:text-outline border border-outline-variant/50 focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none text-base transition-colors"
                 autocomplete="off" />
          <button type="button" 
                  id="composer-clear-btn" 
                  class="${initialText ? 'flex' : 'hidden'} absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface p-1 rounded-md transition-colors"
                  title="Clear text">
            <span class="material-symbols-outlined text-[18px]">close</span>
          </button>
        </div>

        <button type="submit" 
                id="composer-send-btn"
                class="flex items-center justify-center gap-1.5 px-5 h-12 rounded-xl bg-[#001428] hover:bg-[#0f2942] active:bg-[#000d1a] text-white font-bold text-sm transition-colors cursor-pointer shadow-xs min-w-[95px] flex-shrink-0">
          <span>Send</span>
          <span class="material-symbols-outlined text-[18px]">send</span>
        </button>
      </form>

      <!-- Subtext caption -->
      <p class="text-[11px] text-on-surface-variant flex items-center gap-1.5 leading-tight px-1">
        <span class="material-symbols-outlined text-secondary text-[14px]">sync_alt</span>
        <span>Pressing Send transmits text &amp; triggers synchronized sign animation on citizen's terminal.</span>
      </p>
    </div>
  `;
}

function escapeAttr(str) {
  if (!str) return '';
  return String(str).replace(/"/g, '&quot;');
}
