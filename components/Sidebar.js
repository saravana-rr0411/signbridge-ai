// Reusable Sidebar Navigation Component matching Stitch Design

export function renderSidebar(currentPath = '#/deaf') {
  const isDeaf = currentPath === '#/deaf' || currentPath === 'deaf';
  const isAdmin = currentPath === '#/admin' || currentPath === 'admin';
  const isHistory = currentPath === '#/history' || currentPath === 'history';

  const deafActiveClass = isDeaf
    ? 'bg-primary-container text-on-primary font-bold shadow-xs'
    : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface font-semibold';

  const adminActiveClass = isAdmin
    ? 'bg-primary-container text-on-primary font-bold shadow-xs'
    : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface font-semibold';

  const historyActiveClass = isHistory
    ? 'bg-primary-container text-on-primary font-bold shadow-xs'
    : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface font-semibold';

  return `
    <aside id="main-sidebar" class="w-72 h-full bg-surface-container-lowest shadow-[0_1px_8px_rgba(0,0,0,0.04)] z-50 flex flex-col justify-between select-none flex-shrink-0 transition-transform duration-200">
      <div class="flex flex-col">
        <!-- Logo / Title Header -->
        <a href="#/deaf" class="h-20 px-4 flex items-center gap-3 bg-surface-container-low border-b border-outline-variant/20 hover:opacity-95 transition-opacity">
          <div class="w-10 h-10 rounded-xl bg-primary flex items-center justify-center flex-shrink-0 shadow-xs">
            <span class="material-symbols-outlined text-on-primary text-[24px]">sign_language</span>
          </div>
          <div class="flex flex-col">
            <span class="text-xl font-bold text-primary leading-tight tracking-tight">SignBridge AI</span>
            <span class="text-xs text-on-surface-variant font-medium">Accessible Civic Relay</span>
          </div>
        </a>

        <!-- Operational Interfaces Navigation -->
        <div class="p-3">
          <div class="px-2 py-1.5 text-xs uppercase text-on-surface-variant tracking-wider font-bold">
            Operational Interfaces
          </div>
          <nav class="flex flex-col gap-1.5 mt-1.5">
            <a href="#/deaf" 
               class="flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all min-h-[48px] ${deafActiveClass}"
               aria-current="${isDeaf ? 'page' : 'false'}">
              <span class="material-symbols-outlined text-[22px]">visibility</span>
              <span class="text-sm">Deaf Person Interface</span>
            </a>

            <a href="#/admin" 
               class="flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all min-h-[48px] ${adminActiveClass}"
               aria-current="${isAdmin ? 'page' : 'false'}">
              <span class="material-symbols-outlined text-[22px]">support_agent</span>
              <span class="text-sm">Staff Interface</span>
            </a>

            <a href="#/history" 
               class="flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all min-h-[48px] ${historyActiveClass}"
               aria-current="${isHistory ? 'page' : 'false'}">
              <span class="material-symbols-outlined text-[22px]">analytics</span>
              <span class="text-sm">Chat History & Records</span>
            </a>
          </nav>
        </div>
      </div>

      <!-- Counter Administration & Profile -->
      <div class="flex flex-col bg-surface-container-low p-3 gap-2 border-t border-outline-variant/30">
        <div class="px-2 py-1 text-xs uppercase text-on-surface-variant tracking-wider font-bold">
          Counter Administration
        </div>
        
        <nav class="flex flex-col gap-1">
          <button type="button" 
                  id="btn-counter-settings"
                  class="flex items-center gap-3 px-4 py-2.5 rounded-xl text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors min-h-[44px] text-left w-full cursor-pointer">
            <span class="material-symbols-outlined text-[20px]">tune</span>
            <span class="text-sm font-semibold">Counter Settings</span>
          </button>
        </nav>

        <!-- Officer Profile Chip -->
        <div class="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-surface-container mt-0.5">
          <div class="w-9 h-9 rounded-full bg-primary flex items-center justify-center flex-shrink-0 text-white shadow-xs">
            <span class="material-symbols-outlined text-[20px]">person</span>
          </div>
          <div class="flex flex-col min-w-0 flex-1">
            <span class="text-sm font-bold text-on-surface truncate">Officer J. Vance</span>
            <span class="text-xs text-on-surface-variant truncate font-mono">Counter Desk #04</span>
          </div>
        </div>

        <!-- End Session / Logout Button -->
        <a href="#/login" 
           id="btn-logout"
           class="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-error-container text-on-error-container hover:bg-error hover:text-on-error transition-all min-h-[44px] mt-1 font-semibold text-sm cursor-pointer shadow-xs">
          <span class="material-symbols-outlined text-[20px]">power_settings_new</span>
          <span>End Session / Logout</span>
        </a>
      </div>
    </aside>
  `;
}
