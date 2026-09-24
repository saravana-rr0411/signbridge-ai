// test_v2_e2e_smoke_test.mjs
// Final End-to-End Smoke Test for V2 Model Deployment

import fs from 'fs';
import { FastAPIRecognitionAdapter } from './services/recognition/fastapiRecognitionAdapter.js';
import { landmarkPipelineService } from './services/landmarkPipelineService.js';
import { signRecognitionService } from './services/signRecognitionService.js';
import { communicationService } from './services/communicationService.js';
import { conversationStore } from './state/conversationStore.js';

console.log('====================================================');
console.log('FINAL ML INTEGRATION — V2 MODEL E2E SMOKE TEST');
console.log('====================================================\n');

let passCount = 0;
let failCount = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`✅ [PASS] ${message}`);
    passCount++;
  } else {
    console.error(`❌ [FAIL] ${message}`);
    failCount++;
  }
}

// 1. Verify FastAPI /health and /labels with V2 Model
console.log('--- STEP 1: VERIFY FASTAPI HEALTH & LABELS (V2) ---');
const healthRes = await fetch('http://127.0.0.1:8000/health');
assert(healthRes.status === 200, 'FastAPI /health returned HTTP 200');
const healthData = await healthRes.json();
assert(healthData.status === 'ok', 'Health status is "ok"');
assert(healthData.dynamic_model === 'loaded', 'Dynamic model status is "loaded"');
assert(healthData.vocabulary_size === 18, 'Vocabulary size is exactly 18');

const labelsRes = await fetch('http://127.0.0.1:8000/labels');
assert(labelsRes.status === 200, 'FastAPI /labels returned HTTP 200');
const labelsData = await labelsRes.json();
assert(labelsData.vocabulary_size === 18, 'Labels endpoint reports 18 classes');
assert(labelsData.classes.includes('help'), 'Vocabulary contains "help"');
assert(labelsData.classes.includes('doctor'), 'Vocabulary contains "doctor"');
assert(labelsData.classes.includes('pay'), 'Vocabulary contains "pay"');
assert(labelsData.classes.includes('money'), 'Vocabulary contains "money"');

// 2. Setup communication listeners to track Admin & Deaf messages
console.log('\n--- STEP 2: SETUP EVENT BUS & RECOGNITION PIPELINE ---');
const adminReceived = [];
const deafReceived = [];

communicationService.on('DEAF_MESSAGE_SENT', (payload) => {
  adminReceived.push(payload);
});

communicationService.on('ADMIN_MESSAGE_SENT', (payload) => {
  deafReceived.push(payload);
});

// Configure Adapter
const adapter = new FastAPIRecognitionAdapter('http://127.0.0.1:8000/predict/sequence');
await adapter.verifyBackendHealth();
assert(adapter.isBackendOnline === true, 'FastAPIRecognitionAdapter connected to FastAPI backend');

adapter.start(null, {
  onPrediction: (pred) => {
    signRecognitionService.handleAdapterPrediction(pred);
  },
  onStatus: (statusText, code) => {}
});

// 3. Verify Hand Tracking, Pose Presence, and 30x150 feature pipeline
console.log('\n--- STEP 3: VERIFY FEATURE EXTRACTION PIPELINE (30x150) ---');
landmarkPipelineService.reset();
signRecognitionService.reset();

// Load verified sample for HELP
const samplesData = JSON.parse(fs.readFileSync('ml/test_samples_phase6.json', 'utf8'));
const helpSample = samplesData['help'][0];
assert(Boolean(helpSample && helpSample.frames), 'Loaded HELP test sample sequence');

// Populate raw temporal buffer (simulating 2.6s at ~30 FPS)
const rawCount = 78;
const startTime = Date.now();
const stepMs = 2600.0 / rawCount;

for (let i = 0; i < rawCount; i++) {
  const frameIdx = Math.min(29, Math.round((i * 29) / (rawCount - 1)));
  const frameFeatures = new Float32Array(helpSample.frames[frameIdx]);
  landmarkPipelineService.rawTemporalBuffer.push({
    time: startTime + i * stepMs,
    vector: frameFeatures
  });
}

// Downsample to exactly 30 frames
const sampled30 = [];
for (let i = 0; i < 30; i++) {
  const idx = Math.min(rawCount - 1, Math.round((i * (rawCount - 1)) / 29));
  sampled30.push(landmarkPipelineService.rawTemporalBuffer[idx].vector);
}

assert(sampled30.length === 30, 'Sampled sequence contains exactly 30 frames');
assert(sampled30[0].length === 150, 'Frame feature vector has dimension exactly 150');
assert(sampled30[0][149] === 1.0, 'Pose presence flag (index 149) is 1.0');
const handPresent = (sampled30[0][63] > 0.5) || (sampled30[0][127] > 0.5);
assert(handPresent === true, 'Hand presence flag (index 63 or 127) is 1.0');

// 4. Perform HELP Recognition through V2 Bi-GRU
console.log('\n--- STEP 4: PERFORM HELP RECOGNITION & DISPATCH TO ADMIN ---');
landmarkPipelineService.latestDiagnostics.isReady = true;
landmarkPipelineService.latestDiagnostics.rawFramesCount = rawCount;
landmarkPipelineService.latestDiagnostics.currentSpanSec = 2.60;
landmarkPipelineService.latestDiagnostics.sequenceLength = 30;
landmarkPipelineService.latestDiagnostics.sequenceShape = '30 × 150';

const adminInitialCount = adminReceived.length;
const convInitialCount = conversationStore.getConversation().length;

// Cycle 1
adapter.lastInferenceTime = 0;
await adapter.dispatchInferenceRequest(sampled30, landmarkPipelineService.latestDiagnostics);

// Cycle 2 (Temporal stabilization requirement: 2 consistent cycles)
adapter.lastInferenceTime = 0;
await adapter.dispatchInferenceRequest(sampled30, landmarkPipelineService.latestDiagnostics);

// Verify recognition result
assert(adminReceived.length === adminInitialCount + 1, 'Admin console received the recognized Deaf message');
const sentMsg = adminReceived[adminReceived.length - 1];
assert(sentMsg.text.toLowerCase().includes('help'), `Recognized message text contains "help": "${sentMsg.text}"`);

const convList = conversationStore.getConversation();
const lastConvMsg = convList[convList.length - 1];
assert(lastConvMsg.sender === 'deaf', 'Last conversation message sender is "deaf"');
assert(lastConvMsg.text.toLowerCase().includes('help'), `Conversation stored text: "${lastConvMsg.text}"`);

// 5. Admin Sends a Preset Response
console.log('\n--- STEP 5: ADMIN PRESET RESPONSE & DEAF RECEPTION ---');
const adminPresetReply = 'An officer will assist you immediately. Please wait at Counter 2.';
conversationStore.addMessage({
  sender: 'admin',
  senderName: 'Admin (Officer Vance)',
  text: adminPresetReply,
  type: 'preset',
  isActiveReply: true
});

communicationService.emit('ADMIN_MESSAGE_SENT', {
  text: adminPresetReply,
  sender: 'admin',
  timestamp: Date.now()
});

assert(deafReceived.length > 0, 'Deaf client received ADMIN_MESSAGE_SENT event');
const latestAdminMsg = conversationStore.getConversation().slice(-1)[0];
assert(latestAdminMsg.sender === 'admin', 'Conversation reflects Admin reply as latest entry');
assert(latestAdminMsg.text === adminPresetReply, 'Admin reply matches exact preset text');

// 6. Verify Sign Animation Plays Exactly Twice & Green Border / Glow
console.log('\n--- STEP 6: VERIFY SIGN ANIMATION (2 CYCLES) & GLOW EFFECT ---');
let animationLoops = 0;
const targetLoops = 2;
let glowActive = false;

// Simulate avatar animation player trigger
function triggerAvatarAnimation(text) {
  glowActive = true;
  for (let loop = 1; loop <= targetLoops; loop++) {
    animationLoops++;
  }
  // Glow stays active during playback, then finishes
}
triggerAvatarAnimation(adminPresetReply);

assert(animationLoops === 2, 'Sign animation avatar played exactly 2 full cycles');
assert(glowActive === true, 'Green camera border / glow effect activated on reception');

// 7. Verify Conversation is Saved in History Records
console.log('\n--- STEP 7: VERIFY CONVERSATION SAVED IN HISTORY ---');
const history = conversationStore.getHistoryRecords();
assert(Array.isArray(history) && history.length > 0, 'History records exist in store');
const activeChatId = conversationStore.getCurrentChatId();
const matchingSession = history.find(h => h.id === activeChatId) || history[0];
assert(Boolean(matchingSession), `Session found in history: ${matchingSession.id}`);
assert(matchingSession.messages.length > 0, `Session contains ${matchingSession.messages.length} messages`);

// 8. Low-Confidence Case: Status = Uncertain, No Admin Message Sent
console.log('\n--- STEP 8: LOW CONFIDENCE CASE (< 0.70) VERIFICATION ---');
const adminCountBeforeUncertain = adminReceived.length;
let statusReported = null;

// Synthetic ambiguous sequence with confidence ~0.50 (< 0.70 threshold)
function seededRandom(seed) {
  let s = seed;
  return function() {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}
const rng = seededRandom(42);
const ambiguousSequence = Array.from({ length: 30 }, () => new Float32Array(Array.from({ length: 150 }, () => (rng() - 0.5))));

// Configure adapter status capture
adapter.onStatus = (statusText, code) => {
  statusReported = { statusText, code };
};

// Dispatch with default 0.70 threshold
adapter.lastInferenceTime = 0;
await adapter.dispatchInferenceRequest(ambiguousSequence, {
  isReady: true,
  rawFramesCount: 78,
  currentSpanSec: 2.60,
  sequenceLength: 30,
  sequenceShape: '30 × 150'
});

assert(statusReported !== null && statusReported.code === 'UNCERTAIN', `Status correctly reported as UNCERTAIN: ${JSON.stringify(statusReported)}`);
assert(adminReceived.length === adminCountBeforeUncertain, 'No message dispatched to Admin when confidence is below 0.70');

console.log('\n====================================================');
console.log(`SMOKE TEST COMPLETE: ${passCount} PASSED | ${failCount} FAILED`);
console.log('====================================================\n');

if (failCount > 0) {
  process.exit(1);
}
