// Preset Responses Library Component for Staff/Admin Interface
import { PRESETS, PRESET_CATEGORIES } from '../services/mockData.js';
import { stateBridge } from '../services/stateBridge.js';

export function renderPresetLibrary(activeCategory = PRESET_CATEGORIES.HOSPITAL) {
  const currentPresets = PRESETS[activeCategory] || PRESETS[PRESET_CATEGORIES.HOSPITAL];

  const isHospital = activeCategory === PRESET_CATEGORIES.HOSPITAL;
  const isBank = activeCategory === PRESET_CATEGORIES.BANK;
  const isGov = activeCategory === PRESET_CATEGORIES.GOV;

  const tabBtnClass = (isActive) => isActive
    ? 'bg-[#001428] text-white font-bold shadow-xs'
    : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface font-semibold';

  const itemsHtml = currentPresets.map((preset) => {
    const isPriority = preset.isPriority;
    const itemCardClass = isPriority
      ? 'bg-red-50/90 hover:bg-red-100/90 border border-red-200 text-red-950 shadow-xs'
      : 'bg-surface-container-low hover:bg-surface-container-high border border-outline-variant/30 text-on-surface shadow-xs';

    const iconColor = isPriority ? 'text-red-700' : 'text-secondary';
    const tag = isPriority
      ? '<span class="text-[10px] font-mono font-bold bg-red-200 text-red-800 px-1.5 py-0.5 rounded">PRIORITY</span>'
      : `<span class="material-symbols-outlined text-[18px] text-outline group-hover:text-primary transition-colors">add_circle_outline</span>`;

    return `
      <button type="button" 
              class="preset-item-btn w-full p-3 rounded-xl ${itemCardClass} transition-all duration-150 text-left flex items-center justify-between gap-3 group cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary"
              data-text="${escapeAttr(preset.text)}"
              title="Click to populate composer">
        <div class="flex items-center gap-2.5 min-w-0 flex-1">
          <span class="material-symbols-outlined ${iconColor} text-[20px] flex-shrink-0">${preset.icon || 'chat_bubble'}</span>
          <span class="text-sm font-semibold truncate group-hover:text-primary leading-tight">${escapeHtml(preset.text)}</span>
        </div>
        <div class="flex-shrink-0">
          ${tag}
        </div>
      </button>
    `;
  }).join('');

  return `
    <div class="flex flex-col h-full overflow-hidden">
      <!-- Section Header -->
      <div class="pb-3 flex-shrink-0">
        <div class="flex items-center justify-between">
          <h2 class="text-xl font-extrabold text-primary tracking-tight">Preset Responses</h2>
          <span class="text-xs font-mono font-semibold px-2 py-0.5 rounded-full bg-surface-container text-on-surface-variant">
            15 Presets
          </span>
        </div>
        <p class="text-xs text-on-surface-variant mt-0.5 font-medium">Instant civic phrases to populate composer</p>
      </div>

      <!-- Category Filter Tabs: BANK / HOSPITAL / GOV OFFICE -->
      <div class="flex items-center gap-1.5 p-1 bg-surface-container-low rounded-xl border border-outline-variant/30 mb-3 flex-shrink-0" role="tablist">
        <button type="button" 
                class="preset-tab-btn flex-1 py-1.5 px-2 rounded-lg text-xs transition-all text-center cursor-pointer ${tabBtnClass(isBank)}"
                data-category="${PRESET_CATEGORIES.BANK}">
          BANK
        </button>
        <button type="button" 
                class="preset-tab-btn flex-1 py-1.5 px-2 rounded-lg text-xs transition-all text-center cursor-pointer ${tabBtnClass(isHospital)}"
                data-category="${PRESET_CATEGORIES.HOSPITAL}">
          HOSPITAL
        </button>
        <button type="button" 
                class="preset-tab-btn flex-1 py-1.5 px-2 rounded-lg text-xs transition-all text-center cursor-pointer ${tabBtnClass(isGov)}"
                data-category="${PRESET_CATEGORIES.GOV}">
          GOV OFFICE
        </button>
      </div>

      <!-- Scrollable List of 15 Presets -->
      <div class="flex-1 overflow-y-auto pr-1 flex flex-col gap-2 min-h-0" id="preset-list-container">
        ${itemsHtml}
      </div>
    </div>
  `;
}

function escapeHtml(string) {
  return String(string)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeAttr(str) {
  if (!str) return '';
  return String(str).replace(/"/g, '&quot;');
}
