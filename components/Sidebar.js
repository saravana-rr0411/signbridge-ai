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
    <aside id="main-sidebar" class="w-64 h-full bg-surface-container-lowest shadow-[0_1px_6px_rgba(0,0,0,0.03)] z-50 flex flex-col justify-between select-none flex-shrink-0">
      <div class="flex flex-col">
        <!-- Logo / Title Header -->
        <a href="#/deaf" class="h-16 px-4 flex items-center gap-3 bg-surface-container-low border-b border-outline-variant/20 hover:opacity-90 transition-opacity">
          <div class="w-9 h-9 rounded-xl bg-primary flex items-center justify-center flex-shrink-0 shadow-xs">
            <span class="material-symbols-outlined text-on-primary text-[22px]">sign_language</span>
          </div>
          <div class="flex flex-col min-w-0">
            <span class="text-base font-bold text-primary leading-tight tracking-tight truncate">SignBridge AI</span>
            <span class="text-[11px] text-on-surface-variant font-medium truncate">Accessible Civic Relay</span>
          </div>
        </a>

        <!-- Operational Interfaces Navigation -->
        <div class="p-3">
          <div class="px-2 py-1 text-[11px] uppercase text-on-surface-variant/80 tracking-wider font-bold">
            Operational Interfaces
          </div>
          <nav class="flex flex-col gap-1 mt-1">
            <a href="#/deaf" 
               class="flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all min-h-[44px] ${deafActiveClass}"
               aria-current="${isDeaf ? 'page' : 'false'}">
              <span class="material-symbols-outlined text-[20px]">visibility</span>
              <span class="text-sm">Deaf Person Interface</span>
            </a>

            <a href="#/admin" 
               class="flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all min-h-[44px] ${adminActiveClass}"
               aria-current="${isAdmin ? 'page' : 'false'}">
              <span class="material-symbols-outlined text-[20px]">support_agent</span>
              <span class="text-sm">Staff Interface</span>
            </a>

            <a href="#/history" 
               class="flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all min-h-[44px] ${historyActiveClass}"
               aria-current="${isHistory ? 'page' : 'false'}">
              <span class="material-symbols-outlined text-[20px]">analytics</span>
              <span class="text-sm">Chat History & Records</span>
            </a>
          </nav>
        </div>
      </div>

      <!-- Bottom Section: Logout Only -->
      <div class="p-3 border-t border-outline-variant/30 flex-shrink-0">
        <a href="#/login" 
           id="btn-logout"
           class="flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl bg-error-container text-on-error-container hover:bg-error hover:text-on-error transition-all min-h-[42px] font-semibold text-xs cursor-pointer shadow-xs">
          <span class="material-symbols-outlined text-[18px]">logout</span>
          <span>Logout</span>
        </a>
      </div>
    </aside>
  `;
}
