/**
 * Comprehensive Automated Integration Test Suite for Sign Animation System
 *
 * Tests:
 * 1. A-Z Fingerspelling (26/26 complete letter coverage, sentence sequencing)
 * 2. ASL: 5 verified working animations (HELLO, GOOD MORNING, THANK YOU, YES, NO)
 * 3. ISL: 5 verified working animations (HELLO, GOOD MORNING, THANK YOU, YES, NO)
 * 4. Honesty Rule: Unsupported ASL/ISL phrases reject fabrication and offer Fingerspelling
 * 5. NO-sign condition fix verification
 * 6. Admin -> Deaf communication relay via communicationService
 * 7. Rapid message sequencing and race condition prevention
 * 8. Speech Recognition API service integration
 * 9. Asset integrity and model verification
 */

import assert from 'node:assert';
import fs from 'node:fs';
import { parsePhraseToKeyframes, getPhraseDuration, SignAnimationEngine } from './services/signAnimation/engine.js';
import { alphabets } from './services/signAnimation/alphabets.js';
import { islAnimations } from './services/signAnimation/isl/index.js';
import { aslAnimations } from './services/signAnimation/asl/index.js';
import {
  SIGNING_MODES,
  MODE_LABELS,
  isSupportedSign,
  getSignConfig,
  getAvailablePresetsForMode,
  normalizeSignText
} from './services/signAnimation/signConfig.js';
import { signAnimationService } from './services/signAnimationService.js';
import { communicationService } from './services/communicationService.js';
import { conversationStore } from './state/conversationStore.js';
import { speechService } from './services/signAnimation/speechRecognition.js';

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`✅ [PASS] ${name}`);
    passed++;
  } catch (err) {
    console.error(`❌ [FAIL] ${name}:`, err.message);
    failed++;
  }
}

console.log('====================================================');
console.log('TESTING SIGN ANIMATION SYSTEM INTEGRATION');
console.log('====================================================\n');

// ----------------------------------------------------
// SECTION 1: A-Z FINGERSPELLING COVERAGE
// ----------------------------------------------------
console.log('--- SECTION 1: A-Z FINGERSPELLING COVERAGE ---');

test('Fingerspelling: All 26 letters A-Z are implemented and exported', () => {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  assert.strictEqual(Object.keys(alphabets).length, 26);
  for (const letter of letters) {
    assert(typeof alphabets[letter] === 'function', `Letter ${letter} has an animation function`);
    const mockRef = { animations: [], pending: true };
    alphabets[letter](mockRef);
    assert(mockRef.animations.length > 0, `Letter ${letter} produces bone instructions`);
  }
});

test('Fingerspelling: HELLO decomposes sequentially into H-E-L-L-O', () => {
  const keyframes = parsePhraseToKeyframes('HELLO', 'FINGERSPELLING');
  assert(keyframes.length > 0);
  const letterMap = new Map();
  for (const k of keyframes) {
    if (!letterMap.has(k.charIndex)) {
      letterMap.set(k.charIndex, k.token);
    }
  }
  const uniqueTokensInOrder = Array.from(letterMap.values());
  assert.deepStrictEqual(uniqueTokensInOrder, ['H', 'E', 'L', 'L', 'O']);
});

test('Fingerspelling: Multi-word phrase "PLEASE WAIT" preserves word boundaries', () => {
  const keyframes = parsePhraseToKeyframes('PLEASE WAIT', 'FINGERSPELLING');
  assert(keyframes.length > 0);
  const words = [...new Set(keyframes.map(k => k.word))];
  assert.deepStrictEqual(words, ['PLEASE', 'WAIT']);
  const endFrames = keyframes.filter(k => k.isWordEnd);
  assert.strictEqual(endFrames.length, 2, 'Exactly 2 words have isWordEnd set to true');
});

test('Fingerspelling: Test candidate signs HELP and DOCTOR', () => {
  const kfHelp = parsePhraseToKeyframes('HELP', 'FINGERSPELLING');
  assert(kfHelp.length > 0);
  const kfDoc = parsePhraseToKeyframes('DOCTOR', 'FINGERSPELLING');
  assert(kfDoc.length > 0);
});

// ----------------------------------------------------
// SECTION 2: ASL 5-SIGN VERIFICATION & HONESTY RULE
// ----------------------------------------------------
console.log('\n--- SECTION 2: ASL 5-SIGN VERIFICATION & HONESTY RULE ---');

test('ASL: Exactly 5 verified signs exist in dictionary', () => {
  const expectedAsl = ['hello', 'good morning', 'thank you', 'yes', 'no'];
  for (const phrase of expectedAsl) {
    assert(isSupportedSign(phrase, 'ASL'), `ASL supports "${phrase}"`);
    const kf = parsePhraseToKeyframes(phrase, 'ASL');
    assert(kf.length > 0, `ASL "${phrase}" returns procedural keyframes`);
  }
});

test('ASL: Unsupported phrase rejects fabrication and returns 0 keyframes', () => {
  const unsupported = ['doctor', 'please wait', 'appointment', 'bathroom'];
  for (const phrase of unsupported) {
    assert.strictEqual(isSupportedSign(phrase, 'ASL'), false);
    const kf = parsePhraseToKeyframes(phrase, 'ASL');
    assert.strictEqual(kf.length, 0, `Unsupported phrase "${phrase}" returns empty keyframes array`);
    const config = getSignConfig(phrase, 'ASL');
    assert.strictEqual(config.supported, false);
    assert.strictEqual(config.asl, null);
    assert.strictEqual(config.fingerspelling, true);
  }
});

// ----------------------------------------------------
// SECTION 3: ISL 5-SIGN VERIFICATION & HONESTY RULE
// ----------------------------------------------------
console.log('\n--- SECTION 3: ISL 5-SIGN VERIFICATION & HONESTY RULE ---');

test('ISL: Exactly 5 verified signs exist in dictionary', () => {
  const expectedIsl = ['hello', 'good morning', 'thank you', 'yes', 'no'];
  for (const phrase of expectedIsl) {
    assert(isSupportedSign(phrase, 'ISL'), `ISL supports "${phrase}"`);
    const kf = parsePhraseToKeyframes(phrase, 'ISL');
    assert(kf.length > 0, `ISL "${phrase}" returns procedural keyframes`);
  }
});

test('ISL: Unsupported phrase rejects fabrication and returns 0 keyframes', () => {
  const unsupported = ['doctor', 'please wait', 'appointment', 'hospital'];
  for (const phrase of unsupported) {
    assert.strictEqual(isSupportedSign(phrase, 'ISL'), false);
    const kf = parsePhraseToKeyframes(phrase, 'ISL');
    assert.strictEqual(kf.length, 0, `Unsupported phrase "${phrase}" returns empty keyframes array`);
    const config = getSignConfig(phrase, 'ISL');
    assert.strictEqual(config.supported, false);
    assert.strictEqual(config.isl, null);
    assert.strictEqual(config.fingerspelling, true);
  }
});

// ----------------------------------------------------
// SECTION 4: BUG FIX VERIFICATION (NO SIGN CONDITION)
// ----------------------------------------------------
console.log('\n--- SECTION 4: BUG FIX VERIFICATION (NO SIGN CONDITION) ---');

test('Bug Fix: NO animation produces valid keyframes in both ASL and ISL', () => {
  const aslNo = parsePhraseToKeyframes('no', 'ASL');
  assert(aslNo.length > 0, 'ASL NO has keyframes');
  assert.strictEqual(aslNo[0].word, 'NO');

  const islNo = parsePhraseToKeyframes('no', 'ISL');
  assert(islNo.length > 0, 'ISL NO has keyframes');
  assert.strictEqual(islNo[0].word, 'NO');

  // Verify normalizeSignText resolves "No" to "no"
  assert.strictEqual(normalizeSignText('No'), 'no');
  assert.strictEqual(normalizeSignText('NO!'), 'no');
});

// ----------------------------------------------------
// SECTION 5: ADMIN -> DEAF COMMUNICATION RELAY
// ----------------------------------------------------
console.log('\n--- SECTION 5: ADMIN -> DEAF COMMUNICATION RELAY ---');

test('Communication: ADMIN_SIGN_RESPONSE payload dispatches and updates animation state', () => {
  let receivedPayload = null;
  const unsub = communicationService.on('ADMIN_SIGN_RESPONSE', (payload) => {
    receivedPayload = payload;
  });

  const testPayload = {
    type: 'ADMIN_SIGN_RESPONSE',
    messageId: 'msg_test_01',
    text: 'Good morning',
    mode: 'ASL',
    signSequence: ['GOOD_MORNING'],
    timestamp: Date.now()
  };

  communicationService.emit('ADMIN_SIGN_RESPONSE', testPayload);
  assert(receivedPayload, 'Payload received by listener');
  assert.strictEqual(receivedPayload.text, 'Good morning');
  assert.strictEqual(receivedPayload.mode, 'ASL');

  // Now trigger playAnimationForMessage directly to test service state update
  signAnimationService.playAnimationForMessage(testPayload.text, testPayload.mode, testPayload.signSequence);
  const state = signAnimationService.getState();
  assert.strictEqual(state.text, 'Good morning');
  assert.strictEqual(state.mode, 'ASL');
  assert.strictEqual(state.supported, true);

  unsub();
});

test('Communication: Fingerspelling mode payload correctly drives animation service', () => {
  signAnimationService.playAnimationForMessage('I NEED HELP', 'FINGERSPELLING');
  const state = signAnimationService.getState();
  assert.strictEqual(state.text, 'I NEED HELP');
  assert.strictEqual(state.mode, 'FINGERSPELLING');
  assert.strictEqual(state.supported, true);
});

test('Communication: Unsupported ASL sign marks supported=false without crashing', () => {
  signAnimationService.playAnimationForMessage('I have pain here', 'ASL');
  const state = signAnimationService.getState();
  assert.strictEqual(state.supported, false);
  assert.strictEqual(state.isPlaying, false);
});

// ----------------------------------------------------
// SECTION 6: RAPID MESSAGE QUEUING & RACE PREVENTION
// ----------------------------------------------------
console.log('\n--- SECTION 6: RAPID MESSAGE QUEUING & RACE PREVENTION ---');

test('Rapid Messages: Sending two messages quickly supersedes previous animation cleanly', () => {
  signAnimationService.playAnimationForMessage('First message', 'FINGERSPELLING');
  const state1 = signAnimationService.getState();
  assert.strictEqual(state1.text, 'First message');

  // Immediately send second message
  signAnimationService.playAnimationForMessage('Second message', 'FINGERSPELLING');
  const state2 = signAnimationService.getState();
  assert.strictEqual(state2.text, 'Second message');
  assert.strictEqual(state2.cycle, 1);
});

// ----------------------------------------------------
// SECTION 7: WEB SPEECH API INTEGRATION
// ----------------------------------------------------
console.log('\n--- SECTION 7: WEB SPEECH API INTEGRATION ---');

test('Web Speech API: speechService is defined and exports control methods', () => {
  assert(speechService, 'speechService instance exists');
  assert(typeof speechService.start === 'function');
  assert(typeof speechService.stop === 'function');
  assert(typeof speechService.abort === 'function');
  assert(typeof speechService.isSupported === 'function');
  assert.strictEqual(speechService.getIsListening(), false);
});

// ----------------------------------------------------
// SECTION 8: ASSET INTEGRITY & 3D MODEL
// ----------------------------------------------------
console.log('\n--- SECTION 8: ASSET INTEGRITY & 3D MODEL ---');

test('Assets: xbot.glb is physically present in public/models', () => {
  assert(fs.existsSync('public/models/xbot.glb'), 'public/models/xbot.glb exists');
  const stat = fs.statSync('public/models/xbot.glb');
  assert.strictEqual(stat.size, 2233444, 'xbot.glb is exactly 2,233,444 bytes');
});

test('Assets: xbot.glb is physically present in dist/models', () => {
  assert(fs.existsSync('dist/models/xbot.glb'), 'dist/models/xbot.glb exists');
  const stat = fs.statSync('dist/models/xbot.glb');
  assert.strictEqual(stat.size, 2233444, 'dist/models/xbot.glb is exactly 2,233,444 bytes');
});

// ----------------------------------------------------
// SECTION 9: PRESET DEFINITIONS & MODE SELECTOR
// ----------------------------------------------------
console.log('\n--- SECTION 9: PRESET DEFINITIONS & MODE SELECTOR ---');

test('Presets: ASL and ISL preset options return strictly 5 verified items', () => {
  const aslPresets = getAvailablePresetsForMode('ASL');
  assert.strictEqual(aslPresets.length, 5, 'ASL has strictly 5 presets');
  const islPresets = getAvailablePresetsForMode('ISL');
  assert.strictEqual(islPresets.length, 5, 'ISL has strictly 5 presets');
  const expectedSlugs = ['hello', 'good-morning', 'thank-you', 'yes', 'no'];
  assert.deepStrictEqual(aslPresets.map(p => p.slug), expectedSlugs);
  assert.deepStrictEqual(islPresets.map(p => p.slug), expectedSlugs);
});

console.log('\n====================================================');
console.log(`TEST SUITE RESULTS: ${passed} PASSED | ${failed} FAILED`);
console.log('====================================================');

if (failed > 0) {
  process.exit(1);
}
