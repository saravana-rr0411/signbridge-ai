// Real-Time Recognized Sign Transcript Component with Dynamic Recognition Status
// Displays Live Camera -> Recognition Status -> Recognized Text

export function renderSignTranscript(
  transcriptText = 'I need help',
  id = 'transcript-display',
  statusText = 'Recognition: Ready',
  statusId = 'recognition-status-text'
) {
  const dotId = statusId ? `${statusId}-dot` : 'recognition-status-dot';

  return `
    <div class="mt-3 p-3.5 bg-surface-container-low rounded-xl border border-outline-variant/40 flex flex-col gap-2 flex-shrink-0 shadow-xs">
      <!-- Recognition Status Indicator (Requirement 4) -->
      <div class="flex items-center justify-between pb-1.5 border-b border-outline-variant/25">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" id="${dotId}"></span>
          <span class="text-xs font-mono font-bold text-slate-700" id="${statusId}">
            ${statusText}
          </span>
        </div>
        <span class="text-[10px] font-mono font-bold text-slate-500 bg-surface-container px-2 py-0.5 rounded uppercase tracking-wider">
          ML Adapter
        </span>
      </div>

      <!-- Recognized Text Output Display -->
      <div class="flex flex-col gap-1">
        <span class="text-[11px] font-bold uppercase tracking-wider text-on-surface-variant flex items-center gap-1.5">
          <span class="material-symbols-outlined text-secondary text-[15px]">translate</span>
          Recognized Phrase
        </span>
        <p class="text-xl lg:text-2xl font-extrabold text-primary leading-snug break-words" id="${id}">
          "${transcriptText}"
          <span class="inline-block w-2 h-5 bg-secondary ml-1 animate-pulse align-middle"></span>
        </p>
      </div>
    </div>
  `;
}
