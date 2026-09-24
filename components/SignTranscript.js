// components/SignTranscript.js - Real-Time Recognized Sign Transcript Component
// Displays Live Recognition Status with Clear Separation Between:
// A) ML Recognized Sign (Raw 6-class ASL prediction)
// B) Contextual Hospital Message (Predefined public-service phrase layer)

export function renderSignTranscript(
  transcriptText = 'Show a sign to begin',
  id = 'transcript-display',
  statusText = 'Recognition: Ready',
  statusId = 'recognition-status-text',
  rawSign = '—',
  contextName = 'Hospital First-Visit',
  modelBadge = 'V6 • 10-SIGN'
) {
  const dotId = statusId ? `${statusId}-dot` : 'recognition-status-dot';
  const rawSignId = `${id}-raw-sign`;

  return `
    <div class="mt-3 p-3.5 bg-surface-container-low rounded-xl border border-outline-variant/40 flex flex-col gap-2.5 flex-shrink-0 shadow-xs">
      <!-- Recognition Status Indicator & Public Service Context Badge -->
      <div class="flex items-center justify-between pb-1.5 border-b border-outline-variant/25 flex-wrap gap-1.5">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" id="${dotId}"></span>
          <span class="text-xs font-mono font-bold text-slate-700" id="${statusId}">
            ${statusText}
          </span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-[10px] font-mono font-bold text-rose-700 bg-rose-50 border border-rose-200/80 px-2 py-0.5 rounded flex items-center gap-1 uppercase tracking-wider">
            <span class="material-symbols-outlined text-[13px] text-rose-600">local_hospital</span>
            <span>${contextName}</span>
          </span>
          <span class="text-[10px] font-mono font-bold text-slate-600 bg-surface-container px-2 py-0.5 rounded uppercase tracking-wider" id="${id}-model-badge">
            ${modelBadge}
          </span>
        </div>
      </div>

      <!-- DUAL DISPLAY ARCHITECTURE: Clear separation between ML model & Phrase layer -->
      <div class="grid grid-cols-1 sm:grid-cols-12 gap-2">
        <!-- (A) ML RECOGNIZED SIGNS -->
        <div class="sm:col-span-5 p-2.5 bg-surface-container-lowest rounded-lg border border-outline-variant/40 flex flex-col justify-between">
          <div class="flex items-center justify-between mb-1">
            <span class="text-[10px] font-extrabold uppercase tracking-wider text-slate-600 flex items-center gap-1">
              <span class="material-symbols-outlined text-primary text-[14px]">sign_language</span>
              Recognized Signs (ML Model)
            </span>
            <span class="text-[9px] font-mono px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-bold">
              ML Model
            </span>
          </div>
          <div class="flex items-center my-0.5">
            <span class="text-sm lg:text-base font-black font-mono text-emerald-700 tracking-wide break-words" id="${rawSignId}">
              ${rawSign}
            </span>
          </div>
          <span class="text-[9px] text-slate-400 font-medium">Actual ML-recognized sign sequence</span>
        </div>

        <!-- (B) CONTEXTUAL MESSAGE (PHRASE LAYER) -->
        <div class="sm:col-span-7 p-2.5 bg-white rounded-lg border-2 border-primary/20 flex flex-col justify-between shadow-2xs">
          <div class="flex items-center justify-between mb-1">
            <span class="text-[10px] font-extrabold uppercase tracking-wider text-primary flex items-center gap-1">
              <span class="material-symbols-outlined text-secondary text-[14px]">translate</span>
              Contextual Message (Phrase Layer)
            </span>
            <span class="text-[9px] font-mono px-1.5 py-0.2 rounded bg-primary/10 text-primary font-bold">
              Phrase Layer
            </span>
          </div>
          <p class="text-sm lg:text-base font-extrabold text-primary leading-snug break-words" id="${id}">
            "${transcriptText}"
            <span class="inline-block w-1.5 h-4 bg-secondary ml-1 animate-pulse align-middle"></span>
          </p>
          <span class="text-[9px] text-slate-500 font-medium">Deterministic hospital-context sentence</span>
        </div>
      </div>
    </div>
  `;
}
