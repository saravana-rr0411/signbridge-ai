// test_v6_10_sign_website_integration.mjs
// Verification suite for V6 10-Sign Model Website Integration

import assert from 'assert';
import { API_ENDPOINTS } from './services/apiConfig.js';
import { FastAPIRecognitionAdapter, RECOGNITION_MODEL } from './services/recognition/fastapiRecognitionAdapter.js';
import {
  V6_ML_ALLOWED_SIGNS,
  V3_ML_ALLOWED_SIGNS,
  resolveHospitalSentence
} from './services/conversation/hospitalPhrases.js';
import { hospitalConversationService } from './services/conversation/hospitalConversationService.js';
import { conversationStore } from './state/conversationStore.js';
import { communicationService } from './services/communicationService.js';
import { renderSignTranscript } from './components/SignTranscript.js';

console.log('====================================================');
console.log('TESTING V6 10-SIGN WEBSITE INTEGRATION & FALLBACKS');
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

async function runAsyncTests() {
  // --- SECTION 1: LABELS ENDPOINT VERIFICATION ---
  console.log('--- SECTION 1: LABELS ENDPOINTS VERIFICATION ---');

  await (async () => {
    try {
      const respV6 = await fetch(API_ENDPOINTS.LABELS_V6_10_SIGN);
      assert.strictEqual(respV6.status, 200, 'V6 labels endpoint returned 200');
      const dataV6 = await respV6.json();
      assert.strictEqual(dataV6.vocabulary_size, 10, 'V6 vocabulary size is strictly 10');
      assert.strictEqual(dataV6.model_version, 'v6_10_sign');
      const expectedV6 = ['hello', 'help', 'yes', 'no', 'please', 'thank_you', 'doctor', 'pain', 'sick', 'where'];
      expectedV6.forEach(cls => assert(dataV6.classes.includes(cls), `V6 missing class ${cls}`));
      console.log('✅ [PASS] V6 /labels/v6-10-sign returns strictly 10 classes');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V6 labels check failed:', err.message);
      failCount++;
    }
  })();

  await (async () => {
    try {
      const respV3 = await fetch(API_ENDPOINTS.LABELS_V3_SIX_SIGN);
      assert.strictEqual(respV3.status, 200);
      const dataV3 = await respV3.json();
      assert.strictEqual(dataV3.vocabulary_size, 6, 'V3 fallback vocabulary size is strictly 6');
      console.log('✅ [PASS] V3 /labels/v3-six-sign returns strictly 6 classes (Fallback intact)');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V3 labels fallback check failed:', err.message);
      failCount++;
    }
  })();

  await (async () => {
    try {
      const respV2 = await fetch(API_ENDPOINTS.LABELS);
      assert.strictEqual(respV2.status, 200);
      const dataV2 = await respV2.json();
      assert.strictEqual(dataV2.vocabulary_size, 18, 'V2 fallback vocabulary size is strictly 18');
      console.log('✅ [PASS] V2 /labels returns strictly 18 classes (Fallback intact)');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V2 labels fallback check failed:', err.message);
      failCount++;
    }
  })();

  // --- SECTION 2: V6 INFERENCE CONTRACT (30x168) ---
  console.log('\n--- SECTION 2: V6 INFERENCE CONTRACT (30x168) ---');

  await (async () => {
    try {
      const synthetic168 = Array(30).fill(Array(168).fill(0.05));
      const resp = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE_V6_10_SIGN, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: synthetic168, confidence_threshold: 0.50 })
      });
      assert.strictEqual(resp.status, 200);
      const data = await resp.json();
      assert(data.confidence !== undefined, 'Confidence present');
      assert.strictEqual(data.top_k.length, 10, 'Top-K contains all 10 V6 classes');
      assert.deepStrictEqual(data.tensor_shapes.model_output, [1, 10]);
      console.log('✅ [PASS] V6 endpoint accepts 30x168 sequence and produces 10-class prediction');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V6 prediction failed:', err.message);
      failCount++;
    }
  })();

  // --- SECTION 3: SHAPE REJECTION CONTRACTS ---
  console.log('\n--- SECTION 3: SHAPE REJECTION CONTRACTS ---');

  await (async () => {
    try {
      const bad150 = Array(30).fill(Array(150).fill(0.05));
      const respV6 = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE_V6_10_SIGN, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: bad150 })
      });
      assert.strictEqual(respV6.status, 422, 'V6 rejects 30x150 with HTTP 422');
      console.log('✅ [PASS] V6 endpoint rejects 30x150 with HTTP 422');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V6 shape rejection test failed:', err.message);
      failCount++;
    }
  })();

  await (async () => {
    try {
      const bad150 = Array(30).fill(Array(150).fill(0.05));
      const respV3 = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE_V3_SIX_SIGN, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: bad150 })
      });
      assert.strictEqual(respV3.status, 422, 'V3 rejects 30x150 with HTTP 422');
      console.log('✅ [PASS] V3 endpoint rejects 30x150 with HTTP 422 (Fallback intact)');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V3 shape rejection test failed:', err.message);
      failCount++;
    }
  })();

  await (async () => {
    try {
      const bad168 = Array(30).fill(Array(168).fill(0.05));
      const respV2 = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: bad168 })
      });
      assert.strictEqual(respV2.status, 422, 'V2 rejects 30x168 with HTTP 422');
      console.log('✅ [PASS] V2 endpoint rejects 30x168 with HTTP 422 (Fallback intact)');
      passCount++;
    } catch (err) {
      console.error('❌ [FAIL] V2 shape rejection test failed:', err.message);
      failCount++;
    }
  })();

  // --- SECTION 4: ADAPTER MODE VERIFICATION ---
  console.log('\n--- SECTION 4: ADAPTER MODE VERIFICATION ---');

  test('FastAPIRecognitionAdapter defaults to v6_10_sign mode as primary', () => {
    assert.strictEqual(RECOGNITION_MODEL, 'v6_10_sign');
    const adapter = new FastAPIRecognitionAdapter();
    assert.strictEqual(adapter.modelMode, 'v6_10_sign');
    assert.strictEqual(adapter.endpoint, API_ENDPOINTS.PREDICT_SEQUENCE_V6_10_SIGN);
  });

  test('FastAPIRecognitionAdapter can initialize in v3_six_sign fallback mode', () => {
    const adapter = new FastAPIRecognitionAdapter(null, 'v3_six_sign');
    assert.strictEqual(adapter.modelMode, 'v3_six_sign');
    assert.strictEqual(adapter.endpoint, API_ENDPOINTS.PREDICT_SEQUENCE_V3_SIX_SIGN);
  });

  test('FastAPIRecognitionAdapter can initialize in v2 fallback mode', () => {
    const adapter = new FastAPIRecognitionAdapter(null, 'v2');
    assert.strictEqual(adapter.modelMode, 'v2');
    assert.strictEqual(adapter.endpoint, API_ENDPOINTS.PREDICT_SEQUENCE);
  });

  // --- SECTION 5: HOSPITAL PHRASE LAYER RESOLUTIONS (NEW & EXISTING) ---
  console.log('\n--- SECTION 5: HOSPITAL PHRASE LAYER RESOLUTIONS ---');

  test('DOCTOR -> "I need the doctor."', () => {
    const res = resolveHospitalSentence(['DOCTOR']);
    assert.strictEqual(res.sentence, 'I need the doctor.');
  });

  test('PAIN -> "I have pain."', () => {
    const res = resolveHospitalSentence(['PAIN']);
    assert.strictEqual(res.sentence, 'I have pain.');
  });

  test('SICK -> "I am sick."', () => {
    const res = resolveHospitalSentence(['SICK']);
    assert.strictEqual(res.sentence, 'I am sick.');
  });

  test('WHERE -> "Where?"', () => {
    const res = resolveHospitalSentence(['WHERE']);
    assert.strictEqual(res.sentence, 'Where?');
  });

  test('HELLO + DOCTOR -> "Hello, I need the doctor."', () => {
    const res = resolveHospitalSentence(['HELLO', 'DOCTOR']);
    assert.strictEqual(res.sentence, 'Hello, I need the doctor.');
  });

  test('SICK + DOCTOR -> "I am sick. I need the doctor."', () => {
    const res = resolveHospitalSentence(['SICK', 'DOCTOR']);
    assert.strictEqual(res.sentence, 'I am sick. I need the doctor.');
  });

  test('HELP + PAIN -> "I need help. I have pain."', () => {
    const res = resolveHospitalSentence(['HELP', 'PAIN']);
    assert.strictEqual(res.sentence, 'I need help. I have pain.');
  });

  test('WHERE + DOCTOR -> "Where is the doctor?"', () => {
    const res = resolveHospitalSentence(['WHERE', 'DOCTOR']);
    assert.strictEqual(res.sentence, 'Where is the doctor?');
  });

  test('DOCTOR + PLEASE + HELP -> "Doctor, please help me."', () => {
    const res = resolveHospitalSentence(['DOCTOR', 'PLEASE', 'HELP']);
    assert.strictEqual(res.sentence, 'Doctor, please help me.');
  });

  test('Existing 6-sign combination HELLO + HELP -> "Hello, I need help."', () => {
    const res = resolveHospitalSentence(['HELLO', 'HELP']);
    assert.strictEqual(res.sentence, 'Hello, I need help.');
  });

  test('Existing 6-sign combination HELLO + HELP + PLEASE -> "Hello, I need help, please."', () => {
    const res = resolveHospitalSentence(['HELLO', 'HELP', 'PLEASE']);
    assert.strictEqual(res.sentence, 'Hello, I need help, please.');
  });

  test('Existing 6-sign combination HELLO + PLEASE + HELP -> "Hello, please help me."', () => {
    const res = resolveHospitalSentence(['HELLO', 'PLEASE', 'HELP']);
    assert.strictEqual(res.sentence, 'Hello, please help me.');
  });

  // --- SECTION 6: GESTURE-HOLD SUPPRESSION & SEQUENCE ACCUMULATION ---
  console.log('\n--- SECTION 6: GESTURE-HOLD SUPPRESSION ON NEW SIGNS ---');

  test('Holding DOCTOR continuously suppresses duplicate messages', () => {
    hospitalConversationService.reset();
    const t0 = 100000;
    const ev1 = hospitalConversationService.processSignEvent({ sign: 'DOCTOR', timestamp: t0, sendToChat: false });
    assert.strictEqual(ev1.recognizedSign, 'DOCTOR');
    assert.strictEqual(ev1.contextualMessage, 'I need the doctor.');

    const ev2 = hospitalConversationService.processSignEvent({ sign: 'DOCTOR', timestamp: t0 + 500, sendToChat: false });
    assert.strictEqual(ev2.isDuplicate, true, 'Continuous hold of DOCTOR was suppressed');
  });

  test('Sequence accumulation: SICK then DOCTOR produces "SICK -> DOCTOR"', () => {
    hospitalConversationService.reset();
    const t0 = 200000;
    hospitalConversationService.processSignEvent({ sign: 'SICK', timestamp: t0, sendToChat: false });
    const ev2 = hospitalConversationService.processSignEvent({ sign: 'DOCTOR', timestamp: t0 + 1000, sendToChat: false });
    assert.strictEqual(ev2.recognizedSign, 'SICK -> DOCTOR');
    assert.strictEqual(ev2.contextualMessage, 'I am sick. I need the doctor.');
  });

  // --- SECTION 7: UI BADGE & DUAL DISPLAY PRESENTATION ---
  console.log('\n--- SECTION 7: UI BADGE & DUAL DISPLAY PRESENTATION ---');

  test('renderSignTranscript displays "V6 • 10-SIGN" badge and preserved cards', () => {
    const html = renderSignTranscript('I need the doctor.', 'test-transcript', 'Recognition: Ready', 'test-status', 'DOCTOR');
    assert(html.includes('V6 • 10-SIGN'), 'Contains V6 • 10-SIGN model identification badge');
    assert(html.includes('Recognized Signs (ML Model)'), 'Contains Card A: Recognized Signs (ML Model)');
    assert(html.includes('Contextual Message (Phrase Layer)'), 'Contains Card B: Contextual Message (Phrase Layer)');
    assert(html.includes('DOCTOR'), 'Displays raw recognized sign');
    assert(html.includes('I need the doctor.'), 'Displays contextual sentence');
  });

  // --- SECTION 8: ADMIN RECEPTION OF NEW SIGNS ---
  console.log('\n--- SECTION 8: ADMIN RECEIVES NEW SIGNS ---');

  test('Admin receives new signs via conversationStore and communicationService', () => {
    hospitalConversationService.reset();
    let receivedDeafMsg = null;
    const unsub = communicationService.on('DEAF_MESSAGE_SENT', (payload) => {
      receivedDeafMsg = payload;
    });

    hospitalConversationService.processSignEvent({ sign: 'PAIN', timestamp: Date.now(), sendToChat: true });
    if (typeof unsub === 'function') unsub();
    assert(receivedDeafMsg !== null, 'Admin received DEAF_MESSAGE_SENT');

    assert.strictEqual(receivedDeafMsg.rawSign, 'PAIN');
    assert.strictEqual(receivedDeafMsg.text, 'I have pain.');

    const messages = conversationStore.getConversation();
    const lastMsg = messages[messages.length - 1];
    assert.strictEqual(lastMsg.sender, 'deaf');
    assert.strictEqual(lastMsg.text, 'I have pain.');
    assert.strictEqual(lastMsg.rawSign, 'PAIN');
  });

  console.log('\n====================================================');
  console.log(`TEST SUITE RESULTS: ${passCount} PASSED | ${failCount} FAILED`);
  console.log('====================================================');
  if (failCount > 0) process.exit(1);
}

runAsyncTests();
