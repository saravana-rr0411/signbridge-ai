// test_hospital_conversation_layer.mjs
// Automated Test Suite for Hospital First-Visit Conversation Layer
import assert from 'assert';
import fs from 'fs';
import crypto from 'crypto';
import {
  HOSPITAL_PHRASES,
  V3_ML_ALLOWED_SIGNS,
  HOSPITAL_SEQUENCE_MAPPINGS,
  getPhraseById,
  resolveHospitalSentence,
  getPhrasesByCategory
} from './services/conversation/hospitalPhrases.js';
import { hospitalConversationService } from './services/conversation/hospitalConversationService.js';
import { communicationService } from './services/communicationService.js';
import { conversationStore } from './state/conversationStore.js';
import { renderLiveCamera } from './components/LiveCamera.js';
import { webrtcService } from './services/webrtcService.js';
import { FastAPIRecognitionAdapter } from './services/recognition/fastapiRecognitionAdapter.js';

console.log('====================================================');
console.log('HOSPITAL FIRST-VISIT CONVERSATION LAYER TEST SUITE');
console.log('====================================================\n');

let passCount = 0;
let failCount = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`✅ [PASS] ${name}`);
    passCount++;
  } catch (err) {
    console.error(`❌ [FAIL] ${name}: ${err.message}`);
    failCount++;
  }
}

// ============================================================================
// SECTION 1: PHRASE CONFIGURATION & 15 PREDEFINED PHRASES INTEGRITY
// ============================================================================
console.log('--- SECTION 1: PHRASE CONFIGURATION & 15 PREDEFINED PHRASES ---');

const REQUIRED_15_PHRASES = [
  'Hello, I need help.',
  'I am here to see the doctor.',
  'I am not feeling well.',
  'I have been feeling sick since yesterday.',
  'I have pain here.',
  'I need to tell you about my problem.',
  'I need an appointment.',
  'Please speak slowly.',
  'I cannot hear you clearly.',
  "I don't understand.",
  'Please write it down.',
  'Where should I wait?',
  'Where is the consultation room?',
  'Please ask the doctor to come.',
  'Thank you for helping me.'
];

test('Hospital phrase configuration exports HOSPITAL_PHRASES array', () => {
  assert(Array.isArray(HOSPITAL_PHRASES), 'HOSPITAL_PHRASES is an array');
  assert.strictEqual(HOSPITAL_PHRASES.length, 15, `Expected 15 phrases, found ${HOSPITAL_PHRASES.length}`);
});

REQUIRED_15_PHRASES.forEach((expectedText, index) => {
  test(`Phrase #${index + 1} exists: "${expectedText}"`, () => {
    const found = HOSPITAL_PHRASES.some((p) => p.displayText === expectedText);
    assert(found, `Missing expected phrase: "${expectedText}"`);
  });
});

test('All 15 phrases have unique IDs following "hosp_XX"', () => {
  const ids = HOSPITAL_PHRASES.map((p) => p.id);
  const uniqueIds = new Set(ids);
  assert.strictEqual(ids.length, 15);
  assert.strictEqual(uniqueIds.size, 15, 'Duplicate phrase IDs detected');
  HOSPITAL_PHRASES.forEach((p) => {
    assert(/^hosp_\d{2}$/.test(p.id), `ID "${p.id}" does not follow hosp_XX convention`);
    assert.strictEqual(p.context, 'hospital');
    assert.strictEqual(p.intendedSpeaker, 'deaf');
  });
});

test('V3 ML allowed signs vocabulary contains strictly 6 signs', () => {
  assert.strictEqual(V3_ML_ALLOWED_SIGNS.length, 6);
  const expectedV3 = ['HELP', 'YES', 'NO', 'PLEASE', 'HELLO', 'THANK_YOU'];
  expectedV3.forEach((sign) => {
    assert(V3_ML_ALLOWED_SIGNS.includes(sign), `Missing sign: ${sign}`);
  });
});

test('"I need an appointment" is NOT claimed as direct ML recognition', () => {
  const appointmentPhrase = HOSPITAL_PHRASES.find((p) => p.displayText === 'I need an appointment.');
  assert(appointmentPhrase, 'Appointment phrase exists');
  assert.strictEqual(appointmentPhrase.requiresMlRecognition, false);
  assert.strictEqual(appointmentPhrase.associatedSignSequence, null);
});

// ============================================================================
// SECTION 2: REQUIRED SENTENCE MAPPINGS (Exact Requirement 11.1 - 11.8)
// ============================================================================
console.log('\n--- SECTION 2: DETERMINISTIC SENTENCE MAPPINGS ---');

test('1. HELP -> "I need help."', () => {
  const res = resolveHospitalSentence(['HELP']);
  assert.strictEqual(res.sentence, 'I need help.');
});

test('2. PLEASE -> "Please."', () => {
  const res = resolveHospitalSentence(['PLEASE']);
  assert.strictEqual(res.sentence, 'Please.');
});

test('3. NO -> "No."', () => {
  const res = resolveHospitalSentence(['NO']);
  assert.strictEqual(res.sentence, 'No.');
});

test('4. THANK_YOU -> "Thank you for helping me."', () => {
  const res = resolveHospitalSentence(['THANK_YOU']);
  assert.strictEqual(res.sentence, 'Thank you for helping me.');
});

test('5. HELLO + HELP -> "Hello, I need help."', () => {
  const res = resolveHospitalSentence(['HELLO', 'HELP']);
  assert.strictEqual(res.sentence, 'Hello, I need help.');
});

test('6. HELLO + HELP + PLEASE -> "Hello, I need help, please."', () => {
  const res = resolveHospitalSentence(['HELLO', 'HELP', 'PLEASE']);
  assert.strictEqual(res.sentence, 'Hello, I need help, please.');
});

test('7. HELLO + PLEASE + HELP -> "Hello, please help me."', () => {
  const res = resolveHospitalSentence(['HELLO', 'PLEASE', 'HELP']);
  assert.strictEqual(res.sentence, 'Hello, please help me.');
});

test('8. HELLO alone must NOT produce "Hello, I need help."', () => {
  const res = resolveHospitalSentence(['HELLO']);
  assert.strictEqual(res.sentence, 'Hello.');
  assert.notStrictEqual(res.sentence, 'Hello, I need help.');
});

test('Additional single sign: YES -> "Yes."', () => {
  const res = resolveHospitalSentence(['YES']);
  assert.strictEqual(res.sentence, 'Yes.');
});

// ============================================================================
// SECTION 3: SEQUENCE HANDLING, DE-DUPLICATION & GESTURE HOLDING (Req 11.9 - 11.11)
// ============================================================================
console.log('\n--- SECTION 3: DE-DUPLICATION & GESTURE-HOLD SUPPRESSION ---');

test('9. Duplicate HELLO events are suppressed during continuous hold', () => {
  hospitalConversationService.reset();
  const t0 = 1000000;

  // First HELLO event
  const e1 = hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0, sendToChat: false });
  assert.strictEqual(e1.recognizedSign, 'HELLO');
  assert.strictEqual(e1.contextualMessage, 'Hello.');
  assert.strictEqual(e1.isDuplicate, undefined);

  // Second HELLO event 400ms later (holding gesture)
  const e2 = hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0 + 400, sendToChat: false });
  assert.strictEqual(e2.isDuplicate, true, 'Second HELLO event must be flagged as duplicate');
  assert.strictEqual(e2.recognizedSign, 'HELLO');
  assert.strictEqual(e2.contextualMessage, 'Hello.');

  // Third HELLO event 800ms later (still holding gesture)
  const e3 = hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0 + 800, sendToChat: false });
  assert.strictEqual(e3.isDuplicate, true, 'Third HELLO event must be flagged as duplicate');

  // Verify internal sequence has only 1 sign, not 3
  const seq = hospitalConversationService.getRecognizedSignSequence();
  assert.deepStrictEqual(seq, ['HELLO'], `Sequence should be ['HELLO'], got: ${JSON.stringify(seq)}`);
});

test('10. Holding the same sign continuously does NOT create repeated conversation messages', () => {
  hospitalConversationService.reset();
  const initialConvCount = conversationStore.getConversation().length;
  const t0 = 2000000;

  // Sign HELP 5 times in rapid succession (simulating 30fps webcam holds)
  for (let i = 0; i < 5; i++) {
    hospitalConversationService.processSignEvent({ sign: 'HELP', timestamp: t0 + (i * 300), sendToChat: true });
  }

  const newConvCount = conversationStore.getConversation().length;
  // Should have dispatched exactly ONE message, not 5
  assert.strictEqual(newConvCount, initialConvCount + 1,
    `Expected exactly 1 new message in conversationStore, but found ${newConvCount - initialConvCount}`);

  const lastMsg = conversationStore.getConversation().slice(-1)[0];
  assert.strictEqual(lastMsg.text, 'I need help.');
  assert.strictEqual(lastMsg.rawSign, 'HELP');
});

test('11. HELLO -> HELP produces exactly "HELLO -> HELP" and NOT "HELLO -> HELLO -> HELP"', () => {
  hospitalConversationService.reset();
  const t0 = 3000000;

  // HELLO x 3
  hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0, sendToChat: false });
  hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0 + 300, sendToChat: false });
  hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0 + 600, sendToChat: false });

  // Transition to HELP x 3
  hospitalConversationService.processSignEvent({ sign: 'HELP', timestamp: t0 + 1200, sendToChat: false });
  hospitalConversationService.processSignEvent({ sign: 'HELP', timestamp: t0 + 1500, sendToChat: false });
  hospitalConversationService.processSignEvent({ sign: 'HELP', timestamp: t0 + 1800, sendToChat: false });

  const seq = hospitalConversationService.getRecognizedSignSequence();
  assert.deepStrictEqual(seq, ['HELLO', 'HELP'],
    `Expected sequence ['HELLO', 'HELP'], but got: ${JSON.stringify(seq)}`);

  const rawSignStr = hospitalConversationService.getCurrentSign();
  assert.strictEqual(rawSignStr, 'HELLO -> HELP');

  const sentenceStr = hospitalConversationService.getCurrentPhrase();
  assert.strictEqual(sentenceStr, 'Hello, I need help.');
});

// ============================================================================
// SECTION 4: ADMIN INTERFACE INTEGRATION (Exact Requirement 11.12)
// ============================================================================
console.log('\n--- SECTION 4: ADMIN RECEIVES RAW SIGN SEQUENCE & CONTEXTUAL SENTENCE ---');

test('12. Admin receives raw recognized sign sequence and contextual hospital sentence', () => {
  hospitalConversationService.reset();
  const t0 = 4000000;

  let capturedAdminPayload = null;
  const unsub = communicationService.on('DEAF_MESSAGE_SENT', (payload) => {
    capturedAdminPayload = payload;
  });

  // User signs HELLO then HELP
  hospitalConversationService.processSignEvent({ sign: 'HELLO', timestamp: t0, sendToChat: true });
  hospitalConversationService.processSignEvent({ sign: 'HELP', timestamp: t0 + 1000, sendToChat: true });

  unsub();

  assert(capturedAdminPayload !== null, 'Admin received DEAF_MESSAGE_SENT event');
  assert.strictEqual(capturedAdminPayload.text, 'Hello, I need help.',
    'Admin patient message is the full contextual sentence');
  assert.strictEqual(capturedAdminPayload.rawSign, 'HELLO -> HELP',
    'Admin receives the raw ML recognized sign sequence');
});

// ============================================================================
// SECTION 5: DEMO SIGN CONTROLS REMOVAL (Exact Requirement 11.13)
// ============================================================================
console.log('\n--- SECTION 5: DEMO SIGN SIMULATION UI REMOVAL ---');

test('13. Demo Sign simulation UI controls are completely removed from production UI', () => {
  const renderedCameraHtml = renderLiveCamera({ isDeafView: true });

  // Verify that Demo Sign elements do NOT exist anywhere in rendered markup
  assert(!renderedCameraHtml.includes('btn-simulate-sign'), 'btn-simulate-sign class must not exist');
  assert(!renderedCameraHtml.includes('demo-sign-select'), 'demo-sign-select must not exist');
  assert(!renderedCameraHtml.includes('Simulate ML Sign'), 'Simulate ML Sign text must not exist');
  assert(!renderedCameraHtml.includes('btn-trigger-selected-sign'), 'btn-trigger-selected-sign must not exist');
  assert(!renderedCameraHtml.includes('Demo Sign:'), 'Demo Sign label must not exist');

  // Verify that legitimate production elements ARE preserved
  assert(renderedCameraHtml.includes('hospital-phrase-select'), 'hospital-phrase-select must be present');
  assert(renderedCameraHtml.includes('btn-trigger-hospital-phrase'), 'btn-trigger-hospital-phrase must be present');
  assert(renderedCameraHtml.includes('btn-start-camera'), 'btn-start-camera must be present');
  assert(renderedCameraHtml.includes('btn-stop-camera'), 'btn-stop-camera must be present');
  assert(renderedCameraHtml.includes('deaf-camera-video'), 'deaf-camera-video must be present');
});

// ============================================================================
// SECTION 6: CONTROLLED TEMPLATE PHRASES (Requirement 9)
// ============================================================================
console.log('\n--- SECTION 6: CONTROLLED TEMPLATE SHORTCUTS ---');

test('Template selection for hosp_07 sets rawSign = "— (Controlled Shortcut)"', () => {
  hospitalConversationService.reset();
  const res = hospitalConversationService.selectPhraseById('hosp_07', { sendToChat: false });

  assert.strictEqual(res.recognizedSign, null, 'Template phrase recognizedSign is null');
  assert.strictEqual(res.contextualMessage, 'I need an appointment.');
  assert.strictEqual(hospitalConversationService.getCurrentSign(), '— (Controlled Shortcut)');
  assert.strictEqual(hospitalConversationService.getCurrentPhrase(), 'I need an appointment.');
});

// ============================================================================
// SECTION 7: MODEL INTEGRITY & WEBRTC PRESERVATION (Requirements 14, 15, 16)
// ============================================================================
console.log('\n--- SECTION 7: MODEL INTEGRITY & WEBRTC PRESERVATION ---');

test('14. Existing V3 six-sign inference & checkpoint SHA-256 unchanged', () => {
  const v3ModelPath = './ml/models/dynamic_bigru_v3_six_sign.pt';
  const v3LabelsPath = './ml/models/dynamic_label_mapping_v3_six_sign.json';
  assert(fs.existsSync(v3ModelPath), 'V3 model file exists');
  assert(fs.existsSync(v3LabelsPath), 'V3 label mapping exists');

  const buf = fs.readFileSync(v3ModelPath);
  const hash = crypto.createHash('sha256').update(buf).digest('hex');
  assert.strictEqual(hash, '1ecce3db8c41d40c6e3a8b7c061ee9708ad6cafe982a22d882021a9c21128469',
    'V3 model checkpoint SHA-256 must match original trained weights exactly');

  const labels = JSON.parse(fs.readFileSync(v3LabelsPath, 'utf8'));
  assert.strictEqual(Object.keys(labels).length, 6, 'V3 label mapping has strictly 6 classes');
});

test('15. Existing V2 model checkpoint SHA-256 unchanged & 70% threshold preserved', () => {
  const v2ModelPath = './ml/models/dynamic_bigru_v2.pt';
  const v2LabelsPath = './ml/models/dynamic_label_mapping_v2.json';
  assert(fs.existsSync(v2ModelPath), 'V2 model file exists');
  assert(fs.existsSync(v2LabelsPath), 'V2 label mapping exists');

  const buf = fs.readFileSync(v2ModelPath);
  const hash = crypto.createHash('sha256').update(buf).digest('hex');
  assert.strictEqual(hash, '24917cdfb4f6835beb6405463f29e0ef34172596aea02495c70f00d93149b8ec',
    'V2 model checkpoint SHA-256 must match original trained weights exactly');

  const labels = JSON.parse(fs.readFileSync(v2LabelsPath, 'utf8'));
  assert.strictEqual(Object.keys(labels).length, 17, 'V2 dynamic label mapping has strictly 17 classes');

  // Verify 70% confidence threshold
  const adapter = new FastAPIRecognitionAdapter();
  assert.strictEqual(adapter.confidenceThreshold, 0.70,
    'FastAPI recognition adapter confidence threshold must remain exactly 0.70');
});

test('16. Existing WebRTC functionality interface preserved and intact', () => {
  assert(webrtcService, 'webrtcService instance exists');
  assert.strictEqual(typeof webrtcService.publishStream, 'function');
  assert.strictEqual(typeof webrtcService.unpublishStream, 'function');
  assert.strictEqual(typeof webrtcService.subscribeStream, 'function');
  assert.strictEqual(typeof webrtcService.unsubscribeStream, 'function');
  assert.strictEqual(typeof webrtcService.initSignaling, 'function');
});

// ============================================================================
// SUMMARY
// ============================================================================
console.log('\n====================================================');
console.log(`TEST SUITE RESULTS: ${passCount} PASSED | ${failCount} FAILED`);
console.log('====================================================\n');

if (failCount > 0) {
  process.exit(1);
}

