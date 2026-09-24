// test_phase5_verification.mjs - Phase 5 Complete Automated Verification
import {
  normalizeHandLandmarks,
  normalizePoseLandmarks,
  construct150FeatureVector,
  landmarkPipelineService
} from './services/landmarkPipelineService.js';

console.log('====================================================');
console.log('PHASE 5 AUTOMATED VERIFICATION SUITE');
console.log('====================================================\n');

let allPassed = true;
function assert(condition, message) {
  if (!condition) {
    console.error(`❌ FAIL: ${message}`);
    allPassed = false;
  } else {
    console.log(`✅ PASS: ${message}`);
  }
}

// ---------------------------------------------------------------------------
// TEST 1: HANDEDNESS MAPPING & FEATURE ROUTING (FIX 1)
// ---------------------------------------------------------------------------
console.log('--- TEST 1: FIX 1 - Handedness Mapping & Feature Routing ---');

// Mock 21 hand landmarks for testing
const mockHandLandmarksA = Array.from({ length: 21 }, (_, i) => ({
  x: 0.5 + i * 0.01,
  y: 0.6 + i * 0.01,
  z: -0.05 + i * 0.005
}));

const mockHandLandmarksB = Array.from({ length: 21 }, (_, i) => ({
  x: 0.2 + i * 0.01,
  y: 0.4 + i * 0.01,
  z: -0.02 + i * 0.005
}));

// Mock 33 pose landmarks
const mockPoseLandmarks = Array.from({ length: 33 }, (_, i) => ({
  x: 0.5 + (i % 5) * 0.05,
  y: 0.3 + Math.floor(i / 5) * 0.08,
  z: -0.1 + (i % 3) * 0.02,
  visibility: 0.95
}));
// Set distinct shoulder landmarks
mockPoseLandmarks[11] = { x: 0.40, y: 0.45, z: -0.05, visibility: 0.99 }; // L shoulder
mockPoseLandmarks[12] = { x: 0.60, y: 0.45, z: -0.05, visibility: 0.99 }; // R shoulder

// Case 1A: Physical Right Hand (MediaPipe raw label "Left" on unmirrored webcam)
landmarkPipelineService.reset();
const res1A = landmarkPipelineService.processFrame(
  [mockHandLandmarksA],
  [{ label: 'Left', score: 0.98 }], // raw "Left"
  mockPoseLandmarks,
  1000
);

assert(res1A.rightPresent === true, 'Physical Right hand detected when raw MediaPipe label is "Left"');
assert(res1A.leftPresent === false, 'Physical Left hand is NOT present (correctly masked to 0)');
assert(res1A.rawRightCount === 21, 'Raw right landmark count is 21');
assert(res1A.rawLeftCount === 0, 'Raw left landmark count is 0');

// Verify feature vector placement: [0..63] should be all zeros, [64..127] populated with [127] = 1.0
const vec1A = landmarkPipelineService.rawTemporalBuffer[0].vector;
let leftSliceNonZero = 0;
for (let i = 0; i <= 63; i++) {
  if (vec1A[i] !== 0) leftSliceNonZero++;
}
assert(leftSliceNonZero === 0, 'Physical Left features [0..63] are strictly 0.0 (masked)');
assert(vec1A[127] === 1.0, 'Physical Right presence flag [127] is 1.0');
let rightSliceNonZero = 0;
for (let i = 64; i < 127; i++) {
  if (vec1A[i] !== 0) rightSliceNonZero++;
}
assert(rightSliceNonZero > 20, 'Physical Right features [64..126] contain non-zero normalized coords');

// Case 1B: Physical Left Hand (MediaPipe raw label "Right" on unmirrored webcam)
landmarkPipelineService.reset();
const res1B = landmarkPipelineService.processFrame(
  [mockHandLandmarksB],
  [{ label: 'Right', score: 0.97 }], // raw "Right"
  mockPoseLandmarks,
  1000
);

assert(res1B.leftPresent === true, 'Physical Left hand detected when raw MediaPipe label is "Right"');
assert(res1B.rightPresent === false, 'Physical Right hand is NOT present (correctly masked to 0)');
const vec1B = landmarkPipelineService.rawTemporalBuffer[0].vector;
assert(vec1B[63] === 1.0, 'Physical Left presence flag [63] is 1.0');
let rightSliceBNonZero = 0;
for (let i = 64; i <= 127; i++) {
  if (vec1B[i] !== 0) rightSliceBNonZero++;
}
assert(rightSliceBNonZero === 0, 'Physical Right features [64..127] are strictly 0.0 (masked)');

// Case 1C: Two Hands Simultaneously
landmarkPipelineService.reset();
const res1C = landmarkPipelineService.processFrame(
  [mockHandLandmarksB, mockHandLandmarksA],
  [{ label: 'Right', score: 0.96 }, { label: 'Left', score: 0.95 }], // raw Right (phys L), raw Left (phys R)
  mockPoseLandmarks,
  1000
);
assert(res1C.leftPresent === true && res1C.rightPresent === true, 'Both hands active in two-hand case');
const vec1C = landmarkPipelineService.rawTemporalBuffer[0].vector;
assert(vec1C[63] === 1.0, 'Two-hand: Left presence [63] = 1.0');
assert(vec1C[127] === 1.0, 'Two-hand: Right presence [127] = 1.0');

// ---------------------------------------------------------------------------
// TEST 2: POSE LANDMARKS NORMALIZATION & PRESENCE (FIX 2)
// ---------------------------------------------------------------------------
console.log('\n--- TEST 2: FIX 2 - Upper Body Pose Integration ---');

// Test normalizePoseLandmarks directly
const poseNormActive = normalizePoseLandmarks(mockPoseLandmarks);
assert(poseNormActive.present === true, 'Pose normalized successfully when landmarks present');
assert(poseNormActive.feat.length === 22, 'Pose feature slice length is exactly 22 (21 coords + 1 flag)');
assert(poseNormActive.feat[21] === 1.0, 'Pose presence flag (index 21 of pose slice) is 1.0');
assert(poseNormActive.rawCount === 7, 'Extracted exactly 7 upper-body anchor keypoints');

// Verify nose centered relative to mid-shoulder:
// L shoulder x=0.40, R shoulder x=0.60 => midX = 0.50, shoulder width = 0.20
// Nose x=0.50 => (0.50 - 0.50)/0.20 = 0.0
const noseNormX = poseNormActive.feat[0];
assert(Math.abs(noseNormX - 0.0) < 1e-4, `Nose normalized X correctly centered at torso origin (actual: ${noseNormX})`);

// Test missing pose fallback
const poseNormMissing = normalizePoseLandmarks(null);
assert(poseNormMissing.present === false, 'Pose marked not present when null');
assert(poseNormMissing.feat[21] === 0.0, 'Pose presence flag is 0.0 when absent');
let missingNonZero = 0;
for (let i = 0; i < 22; i++) {
  if (poseNormMissing.feat[i] !== 0) missingNonZero++;
}
assert(missingNonZero === 0, 'All 22 pose features are 0.0 when pose is absent');

// Check pose features inside 150-dimensional vector
const vec150 = construct150FeatureVector(null, mockHandLandmarksA, mockPoseLandmarks).vector;
assert(vec150.length === 150, 'Full vector length is strictly 150');
assert(vec150[149] === 1.0, 'Feature [149] (Pose Presence) is 1.0 in 150-dim vector');
let poseSliceActiveCount = 0;
for (let i = 128; i < 149; i++) {
  if (vec150[i] !== 0) poseSliceActiveCount++;
}
assert(poseSliceActiveCount > 10, `Pose feature range [128..148] contains non-zero normalized coordinates (got ${poseSliceActiveCount} non-zero values)`);

// ---------------------------------------------------------------------------
// TEST 3: TEMPORAL SAMPLING BUFFER & DOWNSAMPLING (FIX 3)
// ---------------------------------------------------------------------------
console.log('\n--- TEST 3: FIX 3 - Temporal Sampling Engine ---');

landmarkPipelineService.reset();
assert(landmarkPipelineService.captureWindowSec === 2.6, `Capture window is configured to 2.6 sec`);

// Step 3A: Simulate first 15 frames (~0.5s) - Under threshold
const fps = 30;
const frameIntervalMs = 1000 / fps;
let simTime = 10000;

for (let i = 0; i < 15; i++) {
  simTime += frameIntervalMs;
  landmarkPipelineService.processFrame(
    [mockHandLandmarksA],
    [{ label: 'Left', score: 0.98 }],
    mockPoseLandmarks,
    simTime
  );
}

const diagPartial = landmarkPipelineService.latestDiagnostics;
assert(diagPartial.isReady === false, 'Window is NOT ready at 0.5s (< 2.21s threshold)');
assert(landmarkPipelineService.frameBuffer.length === 0, 'No partial sequence emitted (frameBuffer is empty)');
assert(diagPartial.sequenceShape === '0 × 150', `Sequence shape reports "0 × 150" during collection (got ${diagPartial.sequenceShape})`);

// Step 3B: Continue simulating up to 80 frames (~2.67s)
for (let i = 15; i < 80; i++) {
  simTime += frameIntervalMs;
  landmarkPipelineService.processFrame(
    [mockHandLandmarksA],
    [{ label: 'Left', score: 0.98 }],
    mockPoseLandmarks,
    simTime
  );
}

const diagFull = landmarkPipelineService.latestDiagnostics;
assert(diagFull.isReady === true, `Window is READY when span reaches window duration (current: ${diagFull.currentSpanSec}s)`);
assert(diagFull.rawFramesCount >= 70, `Raw frames collected across ~2.6s: ${diagFull.rawFramesCount}`);
assert(landmarkPipelineService.frameBuffer.length === 30, `Sampled frames is strictly 30 (got ${landmarkPipelineService.frameBuffer.length})`);
assert(diagFull.sequenceShape === '30 × 150', `Final sequence shape is strictly 30 × 150`);

// Verify properties of the sampled sequence
const finalSeq = landmarkPipelineService.frameBuffer;
assert(finalSeq.length === 30, 'Final sequence has exactly 30 frames');
let allDim150 = true;
let hasAnyNaN = false;
for (let t = 0; t < 30; t++) {
  if (finalSeq[t].length !== 150) allDim150 = false;
  for (let f = 0; f < 150; f++) {
    const val = finalSeq[t][f];
    if (Number.isNaN(val) || !Number.isFinite(val)) hasAnyNaN = true;
  }
}
assert(allDim150, 'Every sampled frame has exactly 150 features');
assert(!hasAnyNaN, 'No NaN or Infinity in sampled 30 × 150 sequence');

// Verify pose presence across all 30 frames
let poseFlagsCount = 0;
for (let t = 0; t < 30; t++) {
  if (finalSeq[t][149] === 1.0) poseFlagsCount++;
}
assert(poseFlagsCount === 30, `Pose presence [149] is 1.0 across all 30 sampled frames (30/30)`);

// Verify right hand presence across all 30 frames
let rightFlagsCount = 0;
for (let t = 0; t < 30; t++) {
  if (finalSeq[t][127] === 1.0) rightFlagsCount++;
}
assert(rightFlagsCount === 30, `Physical Right presence [127] is 1.0 across all 30 sampled frames (30/30)`);

// ---------------------------------------------------------------------------
// TEST 4: FASTAPI MODEL INFERENCE WITH 30x150 SEQUENCE
// ---------------------------------------------------------------------------
console.log('\n--- TEST 4: Real FastAPI /predict/sequence Inference ---');

try {
  const payload = {
    frames: Array.from(finalSeq).map(f => Array.from(f))
  };

  const response = await fetch('http://127.0.0.1:8000/predict/sequence', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (response.ok) {
    const data = await response.json();
    assert(typeof data.confidence === 'number', `FastAPI /predict/sequence returned confidence score: ${(data.confidence * 100).toFixed(2)}%`);
    assert(typeof data.raw_prediction === 'string' || typeof data.label === 'string', `Prediction label received: "${data.raw_prediction || data.label}"`);
    assert(data.tensor_shapes && data.tensor_shapes.pytorch_tensor, `PyTorch tensor shape: ${JSON.stringify(data.tensor_shapes?.pytorch_tensor)}`);
    assert(typeof data.inference_latency_ms === 'number', `Inference latency: ${data.inference_latency_ms.toFixed(2)}ms`);
    console.log(`Inference result: label=${data.raw_prediction || data.label}, conf=${(data.confidence * 100).toFixed(2)}%, latency=${data.inference_latency_ms?.toFixed(2)}ms`);
  } else {
    console.error(`FastAPI inference returned status: ${response.status}`);
    allPassed = false;
  }
} catch (err) {
  console.error(`Inference error:`, err.message);
  allPassed = false;
}

console.log('\n====================================================');
if (allPassed) {
  console.log('🎉 ALL PHASE 5 VERIFICATION TESTS PASSED!');
} else {
  console.error('❌ SOME PHASE 5 TESTS FAILED!');
}
console.log('====================================================');
