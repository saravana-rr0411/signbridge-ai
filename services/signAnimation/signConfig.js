/**
 * Sign Language Animation Configuration Layer
 *
 * Source Authorities:
 * - ASL: Lifeprint / ASL University (Dr. Bill Vicars)
 * - ISL: Indian Sign Language Research and Training Centre (ISLRTC)
 * - FINGERSPELLING: Sign-Kit Procedural A-Z Engine
 *
 * Strict Honesty Rule:
 * Only 5 signs have verified 3D procedural animations in ASL and ISL:
 * 1. HELLO
 * 2. GOOD MORNING
 * 3. THANK YOU
 * 4. YES
 * 5. NO
 *
 * All other phrases return null for ASL and ISL.
 * Arbitrary text is supported in FINGERSPELLING mode only.
 */

export const SIGNING_MODES = {
  FINGERSPELLING: 'FINGERSPELLING',
  ASL: 'ASL',
  ISL: 'ISL'
};

export const MODE_LABELS = {
  FINGERSPELLING: 'Fingerspelling',
  ASL: 'American Sign Language',
  ISL: 'Indian Sign Language'
};

export const VERIFIED_SIGNS = {
  'hello': {
    id: 'sign-hello',
    phrase: 'Hello',
    displayText: 'Hello',
    slug: 'hello',
    fingerspelling: true,
    asl: 'hello',
    isl: 'hello',
    aslGloss: 'HELLO (Temple Salute)',
    islGloss: 'HELLO / GREETING (Open Wave)',
    category: 'Greetings'
  },
  'good morning': {
    id: 'sign-good-morning',
    phrase: 'Good morning',
    displayText: 'Good morning',
    slug: 'good-morning',
    fingerspelling: true,
    asl: 'good morning',
    isl: 'good morning',
    aslGloss: 'GOOD + MORNING (Compound)',
    islGloss: 'GOOD + MORNING (Compound)',
    category: 'Greetings'
  },
  'thank you': {
    id: 'sign-thank-you',
    phrase: 'Thank you',
    displayText: 'Thank you',
    slug: 'thank-you',
    fingerspelling: true,
    asl: 'thank you',
    isl: 'thank you',
    aslGloss: 'THANK-YOU (Chin Forward)',
    islGloss: 'THANK-YOU (Forward Arc)',
    category: 'Greetings'
  },
  'yes': {
    id: 'sign-yes',
    phrase: 'Yes',
    displayText: 'Yes',
    slug: 'yes',
    fingerspelling: true,
    asl: 'yes',
    isl: 'yes',
    aslGloss: 'YES (S-fist Nod)',
    islGloss: 'YES (S-fist Affirmative Nod)',
    category: 'Responses'
  },
  'no': {
    id: 'sign-no',
    phrase: 'No',
    displayText: 'No',
    slug: 'no',
    fingerspelling: true,
    asl: 'no',
    isl: 'no',
    aslGloss: 'NO (Two-Finger Snap)',
    islGloss: 'NO (Horizontal Negation Wave)',
    category: 'Responses'
  }
};

/**
 * Normalizes text for matching sign keys:
 * Lowercases, removes punctuation, trims, collapses whitespaces.
 */
export function normalizeSignText(text) {
  if (!text) return '';
  return text
    .toLowerCase()
    .trim()
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Checks if a phrase is supported in the requested mode.
 * - FINGERSPELLING: returns true if text contains at least one alphabetic character.
 * - ASL / ISL: returns true ONLY if the normalized phrase is one of the 5 verified signs.
 */
export function isSupportedSign(text, mode = SIGNING_MODES.FINGERSPELLING) {
  if (!text) return false;
  const norm = normalizeSignText(text);

  if (mode === SIGNING_MODES.FINGERSPELLING) {
    return /[a-z]/i.test(norm);
  }

  if (mode === SIGNING_MODES.ASL) {
    return Boolean(VERIFIED_SIGNS[norm]?.asl);
  }

  if (mode === SIGNING_MODES.ISL) {
    return Boolean(VERIFIED_SIGNS[norm]?.isl);
  }

  return false;
}

/**
 * Returns configuration metadata for a phrase in the given mode.
 */
export function getSignConfig(text, mode = SIGNING_MODES.FINGERSPELLING) {
  const norm = normalizeSignText(text);
  const verified = VERIFIED_SIGNS[norm] || null;

  if (mode === SIGNING_MODES.FINGERSPELLING) {
    const hasAlpha = /[a-z]/i.test(norm);
    return {
      phrase: text,
      displayText: text,
      mode: SIGNING_MODES.FINGERSPELLING,
      modeLabel: MODE_LABELS.FINGERSPELLING,
      supported: hasAlpha,
      signKey: null,
      gloss: hasAlpha ? `Fingerspelling: ${text.toUpperCase()}` : 'No letters to spell',
      fingerspelling: hasAlpha,
      asl: verified?.asl || null,
      isl: verified?.isl || null,
      signSequence: hasAlpha ? text.replace(/[^a-zA-Z]/g, '').toUpperCase().split('') : []
    };
  }

  if (mode === SIGNING_MODES.ASL) {
    const isSupported = Boolean(verified?.asl);
    return {
      phrase: verified?.phrase || text,
      displayText: verified?.displayText || text,
      mode: SIGNING_MODES.ASL,
      modeLabel: MODE_LABELS.ASL,
      supported: isSupported,
      signKey: isSupported ? verified.asl : null,
      gloss: isSupported ? verified.aslGloss : 'ASL animation unavailable',
      fingerspelling: true,
      asl: verified?.asl || null,
      isl: verified?.isl || null,
      signSequence: isSupported ? [verified.asl.toUpperCase()] : []
    };
  }

  if (mode === SIGNING_MODES.ISL) {
    const isSupported = Boolean(verified?.isl);
    return {
      phrase: verified?.phrase || text,
      displayText: verified?.displayText || text,
      mode: SIGNING_MODES.ISL,
      modeLabel: MODE_LABELS.ISL,
      supported: isSupported,
      signKey: isSupported ? verified.isl : null,
      gloss: isSupported ? verified.islGloss : 'ISL animation unavailable',
      fingerspelling: true,
      asl: verified?.asl || null,
      isl: verified?.isl || null,
      signSequence: isSupported ? [verified.isl.toUpperCase()] : []
    };
  }

  return {
    phrase: text,
    displayText: text,
    mode,
    modeLabel: mode,
    supported: false,
    asl: null,
    isl: null,
    fingerspelling: false
  };
}

/**
 * Returns preset sign options for Admin controls.
 * For ASL and ISL, strictly returns only the 5 working signs.
 */
export function getAvailablePresetsForMode(mode = SIGNING_MODES.FINGERSPELLING) {
  if (mode === SIGNING_MODES.ASL || mode === SIGNING_MODES.ISL) {
    return [
      { text: 'Hello', slug: 'hello', icon: 'waving_hand' },
      { text: 'Good morning', slug: 'good-morning', icon: 'wb_sunny' },
      { text: 'Thank you', slug: 'thank-you', icon: 'favorite' },
      { text: 'Yes', slug: 'yes', icon: 'check_circle' },
      { text: 'No', slug: 'no', icon: 'cancel' }
    ];
  }

  // Fingerspelling presets or common quick phrases
  return [
    { text: 'Hello', slug: 'hello', icon: 'waving_hand' },
    { text: 'Help', slug: 'help', icon: 'help' },
    { text: 'Doctor', slug: 'doctor', icon: 'medical_services' },
    { text: 'Please wait', slug: 'please-wait', icon: 'hourglass_empty' },
    { text: 'Thank you', slug: 'thank-you', icon: 'favorite' },
    { text: 'Yes', slug: 'yes', icon: 'check_circle' },
    { text: 'No', slug: 'no', icon: 'cancel' }
  ];
}
