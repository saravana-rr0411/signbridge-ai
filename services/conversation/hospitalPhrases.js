// services/conversation/hospitalPhrases.js
// Structured Hospital First-Visit phrase configuration for SignBridge AI.
// Controlled, predefined public-service phrase layer for a Deaf user visiting a hospital.

/**
 * Strictly allowed ML vocabulary for the focused 6-sign V3 model.
 * No signs outside this list may be declared as ML-recognized classes.
 */
export const V3_ML_ALLOWED_SIGNS = Object.freeze([
  'HELP',
  'YES',
  'NO',
  'PLEASE',
  'HELLO',
  'THANK_YOU'
]);

/**
 * 15 Predefined Hospital First-Visit phrases.
 * Each phrase defines:
 * - unique ID (hosp_01 to hosp_15)
 * - displayText: exact required public-service wording
 * - context: "hospital"
 * - intendedSpeaker: "deaf"
 * - associatedSignSequence: sequence of valid ML signs where applicable (or null)
 * - singleSignTriggers: single valid ML signs where applicable (or empty array)
 * - requiresMlRecognition: boolean indicating if this phrase is directly triggered by ML signs
 * - fallbackBehavior: behavior when sign is uncertain or manual selection is used
 * - category: conversational categorization
 */
export const HOSPITAL_PHRASES = Object.freeze([
  {
    id: 'hosp_01',
    displayText: 'Hello, I need help.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: ['HELLO', 'HELP'],
    singleSignTriggers: ['HELP', 'HELLO'],
    requiresMlRecognition: true,
    fallbackBehavior: 'display_as_greeting_help',
    category: 'greeting_assistance',
    icon: 'front_hand',
    description: 'Initial greeting and request for hospital counter assistance'
  },
  {
    id: 'hosp_02',
    displayText: 'I am here to see the doctor.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'registration',
    icon: 'medical_services',
    description: 'Stating the purpose of visit at registration desk'
  },
  {
    id: 'hosp_03',
    displayText: 'I am not feeling well.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'symptoms',
    icon: 'sick',
    description: 'General symptom disclosure to triage or counter staff'
  },
  {
    id: 'hosp_04',
    displayText: 'I have been feeling sick since yesterday.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'symptoms',
    icon: 'history_toggle_off',
    description: 'Timeline of symptom onset for medical staff'
  },
  {
    id: 'hosp_05',
    displayText: 'I have pain here.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'symptoms',
    icon: 'personal_injury',
    description: 'Indicating localized physical pain'
  },
  {
    id: 'hosp_06',
    displayText: 'I need to tell you about my problem.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'consultation',
    icon: 'chat_bubble',
    description: 'Requesting attention to describe medical issue'
  },
  {
    id: 'hosp_07',
    displayText: 'I need an appointment.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'registration',
    icon: 'calendar_today',
    description: 'Requesting a medical appointment slot (controlled phrase template)'
  },
  {
    id: 'hosp_08',
    displayText: 'Please speak slowly.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: ['PLEASE'],
    singleSignTriggers: ['PLEASE'],
    requiresMlRecognition: true,
    fallbackBehavior: 'display_polite_request',
    category: 'communication',
    icon: 'slow_motion_video',
    description: 'Polite pacing request for lip-reading or comprehension'
  },
  {
    id: 'hosp_09',
    displayText: 'I cannot hear you clearly.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'communication',
    icon: 'hearing_disabled',
    description: 'Informing hospital staff of auditory barrier'
  },
  {
    id: 'hosp_10',
    displayText: "I don't understand.",
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: ['NO'],
    singleSignTriggers: ['NO'],
    requiresMlRecognition: true,
    fallbackBehavior: 'display_clarification_request',
    category: 'communication',
    icon: 'help_center',
    description: 'Requesting clarification or alternative explanation'
  },
  {
    id: 'hosp_11',
    displayText: 'Please write it down.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'communication',
    icon: 'edit_note',
    description: 'Requesting written communication from counter staff'
  },
  {
    id: 'hosp_12',
    displayText: 'Where should I wait?',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'navigation',
    icon: 'chair',
    description: 'Asking for designated waiting area location'
  },
  {
    id: 'hosp_13',
    displayText: 'Where is the consultation room?',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'navigation',
    icon: 'meeting_room',
    description: 'Asking for doctor consultation room directions'
  },
  {
    id: 'hosp_14',
    displayText: 'Please ask the doctor to come.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: null,
    singleSignTriggers: [],
    requiresMlRecognition: false,
    fallbackBehavior: 'template_selection_only',
    category: 'urgent_request',
    icon: 'emergency',
    description: 'Requesting urgent doctor assistance at counter'
  },
  {
    id: 'hosp_15',
    displayText: 'Thank you for helping me.',
    context: 'hospital',
    intendedSpeaker: 'deaf',
    associatedSignSequence: ['THANK_YOU'],
    singleSignTriggers: ['THANK_YOU'],
    requiresMlRecognition: true,
    fallbackBehavior: 'display_gratitude',
    category: 'closing',
    icon: 'sentiment_very_satisfied',
    description: 'Expressing gratitude for completed hospital service'
  }
]);

/**
 * Look up phrase by unique ID
 */
export function getPhraseById(id) {
  if (!id) return null;
  return HOSPITAL_PHRASES.find((p) => p.id === id) || null;
}

/**
 * Match a recognized sign sequence (e.g. ['HELLO', 'HELP']) to a phrase
 */
export function findPhraseBySignSequence(sequence) {
  if (!Array.isArray(sequence) || sequence.length === 0) return null;
  const seqNormalized = sequence.map((s) => String(s).toUpperCase().trim());

  // Check multi-sign sequences first (e.g. ['HELLO', 'HELP'])
  for (const phrase of HOSPITAL_PHRASES) {
    if (phrase.associatedSignSequence && phrase.associatedSignSequence.length > 1) {
      const targetSeq = phrase.associatedSignSequence;
      if (seqNormalized.length >= targetSeq.length) {
        const tail = seqNormalized.slice(-targetSeq.length);
        const matches = targetSeq.every((sign, idx) => sign === tail[idx]);
        if (matches) return phrase;
      }
    }
  }

  // Fallback to single sign match on the latest sign
  const lastSign = seqNormalized[seqNormalized.length - 1];
  return findPhraseBySingleSign(lastSign);
}

/**
 * Match a single recognized sign to a phrase
 */
export function findPhraseBySingleSign(sign) {
  if (!sign) return null;
  const norm = String(sign).toUpperCase().trim();
  for (const phrase of HOSPITAL_PHRASES) {
    if (phrase.singleSignTriggers && phrase.singleSignTriggers.includes(norm)) {
      return phrase;
    }
  }
  return null;
}

/**
 * Categorized hospital phrases for template selector
 */
export function getPhrasesByCategory() {
  const groups = {};
  for (const phrase of HOSPITAL_PHRASES) {
    if (!groups[phrase.category]) {
      groups[phrase.category] = [];
    }
    groups[phrase.category].push(phrase);
  }
  return groups;
}

/**
 * Deterministic SIGN-SEQUENCE -> HOSPITAL SENTENCE Mapping Layer.
 * Strictly non-generative, controlled finite-state phrase system.
 */
export const HOSPITAL_SEQUENCE_MAPPINGS = Object.freeze([
  // 3-Sign Sequences (Checked first)
  {
    sequence: ['HELLO', 'HELP', 'PLEASE'],
    sentence: 'Hello, I need help, please.'
  },
  {
    sequence: ['HELLO', 'PLEASE', 'HELP'],
    sentence: 'Hello, please help me.'
  },

  // 2-Sign Sequences
  {
    sequence: ['HELLO', 'HELP'],
    sentence: 'Hello, I need help.'
  },
  {
    sequence: ['HELP', 'PLEASE'],
    sentence: 'I need help, please.'
  },
  {
    sequence: ['PLEASE', 'HELP'],
    sentence: 'Please help me.'
  },

  // 1-Sign Sequences (Single-sign exact mappings)
  {
    sequence: ['HELP'],
    sentence: 'I need help.'
  },
  {
    sequence: ['PLEASE'],
    sentence: 'Please.'
  },
  {
    sequence: ['THANK_YOU'],
    sentence: 'Thank you for helping me.'
  },
  {
    sequence: ['NO'],
    sentence: 'No.'
  },
  {
    sequence: ['HELLO'],
    sentence: 'Hello.'
  },
  {
    sequence: ['YES'],
    sentence: 'Yes.'
  }
]);

/**
 * Resolves a sequence of recognized signs into a deterministic hospital sentence.
 *
 * @param {string[]} sequence - e.g. ['HELLO', 'HELP']
 * @returns {{ sentence: string, matchedSequence: string[] }}
 */
export function resolveHospitalSentence(sequence) {
  if (!Array.isArray(sequence) || sequence.length === 0) {
    return { sentence: 'Hello.', matchedSequence: [] };
  }

  const normalized = sequence.map((s) => String(s).toUpperCase().trim());

  // Check from longest mapping (3 signs) to shortest (1 sign)
  for (const mapping of HOSPITAL_SEQUENCE_MAPPINGS) {
    const target = mapping.sequence;
    if (normalized.length >= target.length) {
      const tail = normalized.slice(-target.length);
      const isMatch = target.every((sign, idx) => sign === tail[idx]);
      if (isMatch) {
        return {
          sentence: mapping.sentence,
          matchedSequence: target
        };
      }
    }
  }

  // Fallback for single sign
  const last = normalized[normalized.length - 1];
  return {
    sentence: `Recognized: ${last}`,
    matchedSequence: [last]
  };
}

