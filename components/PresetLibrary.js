// Preset Responses Library Component for Staff/Admin Interface
import { PRESETS, PRESET_CATEGORIES } from '../services/mockData.js';
import { SIGNING_MODES, MODE_LABELS } from '../services/signAnimation/signConfig.js';

// The strictly verified 5 signs for ASL & ISL
const VERIFIED_ASL_PRESETS = [
  { text: 'Hello', slug: 'hello', desc: 'Salute B-hand forward from temple', icon: 'waving_hand' },
  { text: 'Good morning', slug: 'good-morning', desc: 'Compound: Good (chin to palm) + Morning (sunrise)', icon: 'wb_sunny' },
  { text: 'Thank you', slug: 'thank-you', desc: 'Flat B-hand from lips/chin forward to addressee', icon: 'favorite' },
  { text: 'Yes', slug: 'yes', desc: 'S-fist vertical nodding from wrist', icon: 'check_circle' },
  { text: 'No', slug: 'no', desc: 'Two fingers (index+middle) snapping down to thumb', icon: 'cancel' }
];

const VERIFIED_ISL_PRESETS = [
  { text: 'Hello', slug: 'hello', desc: 'Open 5-hand greeting wave at temple with polite nod', icon: 'waving_hand' },
  { text: 'Good morning', slug: 'good-morning', desc: 'Compound: Good (thumb forward) + Morning (sunrise)', icon: 'wb_sunny' },
  { text: 'Thank you', slug: 'thank-you', desc: 'Flat dominant hand mouth to forward arc with slight bow', icon: 'favorite' },
  { text: 'Yes', slug: 'yes', desc: 'S-fist affirmative wrist nod with synchronized head nod', icon: 'check_circle' },
  { text: 'No', slug: 'no', desc: 'Open palm horizontal side-to-side wave with headshake', icon: 'cancel' }
];

export function renderPresetLibrary(
  activeCategory = PRESET_CATEGORIES.HOSPITAL,
  activeMode = SIGNING_MODES.FINGERSPELLING
) {
  const isFS = activeMode === SIGNING_MODES.FINGERSPELLING;
  const isASL = activeMode === SIGNING_MODES.ASL;
  const isISL = activeMode === SIGNING_MODES.ISL;

  const modeBtnClass = (isActive) => isActive
    ? 'bg-primary text-white font-bold shadow-xs'
    : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface font-semibold';

  let itemsHtml = '';
  let countLabel = '';
  let categoryTabsHtml = '';

  if (isASL) {
    countLabel = '5 ASL Signs';
    itemsHtml = VERIFIED_ASL_PRESETS.map((preset) => `
      <button type="button" 
              class="preset-item-btn w-full p-3 rounded-xl bg-surface-container-low hover:bg-surface-container-high border border-outline-variant/30 text-on-surface transition-all duration-150 text-left flex items-center justify-between gap-3 group cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary shadow-xs"
              data-text="${escapeAttr(preset.text)}"
              data-mode="ASL"
              title="Click to populate composer in ASL mode">
        <div class="flex items-center gap-2.5 min-w-0 flex-1">
          <span class="material-symbols-outlined text-secondary text-[20px] flex-shrink-0">${preset.icon}</span>
          <div class="min-w-0 flex-1">
            <span class="text-sm font-bold text-primary group-hover:text-primary leading-tight block">${escapeHtml(preset.text)}</span>
            <span class="text-[11px] text-on-surface-variant block truncate">${preset.desc}</span>
          </div>
        </div>
        <span class="text-[10px] font-mono font-bold bg-primary/10 text-primary px-2 py-0.5 rounded-full flex-shrink-0">
          ASL 3D
        </span>
      </button>
    `).join('');
  } else if (isISL) {
    countLabel = '5 ISL Signs';
    itemsHtml = VERIFIED_ISL_PRESETS.map((preset) => `
      <button type="button" 
              class="preset-item-btn w-full p-3 rounded-xl bg-surface-container-low hover:bg-surface-container-high border border-outline-variant/30 text-on-surface transition-all duration-150 text-left flex items-center justify-between gap-3 group cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary shadow-xs"
              data-text="${escapeAttr(preset.text)}"
              data-mode="ISL"
              title="Click to populate composer in ISL mode">
        <div class="flex items-center gap-2.5 min-w-0 flex-1">
          <span class="material-symbols-outlined text-secondary text-[20px] flex-shrink-0">${preset.icon}</span>
          <div class="min-w-0 flex-1">
            <span class="text-sm font-bold text-primary group-hover:text-primary leading-tight block">${escapeHtml(preset.text)}</span>
            <span class="text-[11px] text-on-surface-variant block truncate">${preset.desc}</span>
          </div>
        </div>
        <span class="text-[10px] font-mono font-bold bg-secondary/15 text-secondary-container px-2 py-0.5 rounded-full flex-shrink-0">
          ISL 3D
        </span>
      </button>
    `).join('');
  } else {
    // FINGERSPELLING: Displays civic preset categories
    countLabel = '15 Presets • A-Z';
    const currentPresets = PRESETS[activeCategory] || PRESETS[PRESET_CATEGORIES.HOSPITAL];
    const isHospital = activeCategory === PRESET_CATEGORIES.HOSPITAL;
    const isBank = activeCategory === PRESET_CATEGORIES.BANK;
    const isGov = activeCategory === PRESET_CATEGORIES.GOV;

    const tabBtnClass = (isActive) => isActive
      ? 'bg-[#001428] text-white font-bold shadow-xs'
      : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface font-semibold';

    categoryTabsHtml = `
      <div class="flex items-center gap-1.5 p-1 bg-surface-container-low rounded-xl border border-outline-variant/30 mb-2.5 flex-shrink-0" role="tablist">
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
    `;

    itemsHtml = currentPresets.map((preset) => {
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
                class="preset-item-btn w-full p-2.5 lg:p-3 rounded-xl ${itemCardClass} transition-all duration-150 text-left flex items-center justify-between gap-3 group cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary"
                data-text="${escapeAttr(preset.text)}"
                data-mode="FINGERSPELLING"
                title="Click to populate composer for fingerspelling">
          <div class="flex items-center gap-2.5 min-w-0 flex-1">
            <span class="material-symbols-outlined ${iconColor} text-[20px] flex-shrink-0">${preset.icon || 'chat_bubble'}</span>
            <span class="text-xs lg:text-sm font-semibold group-hover:text-primary leading-snug line-clamp-2 break-words">${escapeHtml(preset.text)}</span>
          </div>
          <div class="flex-shrink-0">
            ${tag}
          </div>
        </button>
      `;
    }).join('');
  }

  return `
    <div class="flex flex-col h-full overflow-hidden select-none">
      <!-- Section Header -->
      <div class="pb-2 flex-shrink-0">
        <div class="flex items-center justify-between gap-1.5">
          <h2 class="text-base lg:text-lg font-extrabold text-primary tracking-tight truncate">Staff Responses</h2>
          <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-surface-container text-on-surface-variant flex-shrink-0" id="preset-count-badge">
            ${countLabel}
          </span>
        </div>
        <p class="text-[11px] text-on-surface-variant mt-0.5 font-medium">Select a response to populate composer</p>
      </div>

      <!-- Signing Mode Selector Tabs (FINGERSPELLING / ASL / ISL) -->
      <div class="flex items-center gap-1 p-1 bg-surface-container-low rounded-xl border border-outline-variant/40 mb-2 flex-shrink-0">
        <button type="button"
                class="signing-mode-tab-btn flex-1 py-1 px-1.5 rounded-lg text-[11px] transition-all text-center cursor-pointer ${modeBtnClass(isFS)}"
                data-signing-mode="${SIGNING_MODES.FINGERSPELLING}">
          FINGERSPELLING
        </button>
        <button type="button"
                class="signing-mode-tab-btn flex-1 py-1 px-1.5 rounded-lg text-[11px] transition-all text-center cursor-pointer ${modeBtnClass(isASL)}"
                data-signing-mode="${SIGNING_MODES.ASL}">
          ASL (5)
        </button>
        <button type="button"
                class="signing-mode-tab-btn flex-1 py-1 px-1.5 rounded-lg text-[11px] transition-all text-center cursor-pointer ${modeBtnClass(isISL)}"
                data-signing-mode="${SIGNING_MODES.ISL}">
          ISL (5)
        </button>
      </div>

      <!-- Optional Category Filter Tabs (visible in FINGERSPELLING mode) -->
      ${categoryTabsHtml}

      <!-- Scrollable List of Presets -->
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
