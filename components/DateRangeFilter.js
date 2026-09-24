// Date Range Filter Component for Chat History Screen

export function renderDateRangeFilter({
  fromDate = '2026-09-24',
  toDate = '2026-09-30',
  activePreset = '7days',
  totalCount = 7
}) {
  const quickBtnClass = (preset) => activePreset === preset
    ? 'bg-[#001428] text-white shadow-xs font-bold'
    : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-semibold';

  return `
    <header class="sticky top-0 bg-white/95 backdrop-blur-md border-b border-slate-200/80 p-4 z-20 flex flex-wrap items-center justify-between gap-4 shadow-xs">
      <div class="flex flex-wrap items-center gap-3">
        <!-- Calendar Date Range Inputs -->
        <div class="relative flex items-center bg-white border border-slate-300 rounded-xl shadow-xs px-3.5 py-2 text-sm focus-within:ring-2 focus-within:ring-primary focus-within:border-primary">
          <span class="material-symbols-outlined text-slate-500 mr-2 text-[18px]">calendar_month</span>
          
          <label for="date-from" class="text-xs font-semibold text-slate-500 mr-1.5">From:</label>
          <input type="date" 
                 id="date-from" 
                 value="${fromDate}" 
                 class="border-0 p-0 text-xs font-bold text-slate-800 focus:ring-0 focus:outline-none bg-transparent cursor-pointer" />
          
          <span class="text-slate-400 mx-2 font-medium">to</span>
          
          <label for="date-to" class="text-xs font-semibold text-slate-500 mr-1.5">To:</label>
          <input type="date" 
                 id="date-to" 
                 value="${toDate}" 
                 class="border-0 p-0 text-xs font-bold text-slate-800 focus:ring-0 focus:outline-none bg-transparent cursor-pointer" />
        </div>

        <!-- Quick Filter Buttons: Today / Last 7 Days / Last 30 Days -->
        <div class="inline-flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl border border-slate-200" role="group">
          <button type="button" 
                  id="btn-filter-today" 
                  data-preset="today"
                  class="date-filter-preset-btn px-3 py-1.5 text-xs rounded-lg transition-all cursor-pointer ${quickBtnClass('today')}">
            Today
          </button>
          <button type="button" 
                  id="btn-filter-7days" 
                  data-preset="7days"
                  class="date-filter-preset-btn px-3 py-1.5 text-xs rounded-lg transition-all cursor-pointer ${quickBtnClass('7days')}">
            Last 7 Days
          </button>
          <button type="button" 
                  id="btn-filter-30days" 
                  data-preset="30days"
                  class="date-filter-preset-btn px-3 py-1.5 text-xs rounded-lg transition-all cursor-pointer ${quickBtnClass('30days')}">
            Last 30 Days
          </button>
        </div>
      </div>

      <!-- Desk Records Summary & Export Control -->
      <div class="flex items-center gap-3">
        <div class="text-right">
          <span class="text-xs text-slate-500 block">Civic Desk Records</span>
          <span id="records-count-badge" class="text-xs font-bold text-slate-800 font-mono">
            Showing ${totalCount} sessions
          </span>
        </div>
        <button type="button" 
                id="btn-export-records"
                onclick="window.print()" 
                class="flex items-center gap-1.5 px-3.5 py-2 text-xs font-bold rounded-xl bg-white border border-slate-300 text-slate-800 hover:bg-slate-50 shadow-xs transition-colors cursor-pointer" 
                title="Print or Export Records">
          <span class="material-symbols-outlined text-[16px]">print</span>
          <span>Export</span>
        </button>
      </div>
    </header>
  `;
}
