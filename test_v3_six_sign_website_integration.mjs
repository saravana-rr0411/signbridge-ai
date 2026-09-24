// test_v3_six_sign_website_integration.mjs
// Verification suite for the isolated V3 Six-Sign website inference path.
import assert from 'assert';
import { FastAPIRecognitionAdapter, RECOGNITION_MODEL } from './services/recognition/fastapiRecognitionAdapter.js';
import { API_ENDPOINTS } from './services/apiConfig.js';
import samplePrototypes from './services/recognition/sampleSequences.json' with { type: 'json' };

const V3_VOCABULARY = ["help", "yes", "no", "thank_you", "please", "hello"];
const V2_VOCABULARY_18 = [
  "help", "doctor", "hospital", "sick", "appointment", "where",
  "bathroom", "yes", "no", "please", "thank_you", "wait",
  "understand", "problem", "money", "pay", "document", "letter_a"
];

console.log("====================================================");
console.log("TESTING V3 SIX-SIGN WEBSITE INFERENCE PATH");
console.log("====================================================\n");

async function runTests() {
  let passed = 0;
  let failed = 0;

  function test(name, fn) {
    try {
      fn();
      console.log(`✅ [PASS] ${name}`);
      passed++;
    } catch (err) {
      console.error(`❌ [FAIL] ${name}: ${err.message}`);
      failed++;
    }
  }

  // --- STEP 1: VERIFY PRODUCTION HEALTH STILL REPORTS V2 ---
  console.log("--- STEP 1: PRODUCTION HEALTH CHECK (V2 UNTOUCHED) ---");
  const healthResp = await fetch(API_ENDPOINTS.HEALTH);
  assert.strictEqual(healthResp.status, 200, "Health endpoint returned 200");
  const healthData = await healthResp.json();

  test("Production status is 'ok'", () => {
    assert.strictEqual(healthData.status, "ok");
  });
  test("Dynamic model reports 'loaded'", () => {
    assert.strictEqual(healthData.dynamic_model, "loaded");
  });
  test("Static model reports 'loaded'", () => {
    assert.strictEqual(healthData.static_model, "loaded");
  });
  test("Production vocabulary size is strictly 18 (V2 contract preserved)", () => {
    assert.strictEqual(healthData.vocabulary_size, 18);
  });

  // --- STEP 2: VERIFY V2 LABELS VS V3 LABELS ---
  console.log("\n--- STEP 2: LABELS ENDPOINTS VERIFICATION ---");
  const labelsResp = await fetch(API_ENDPOINTS.LABELS);
  assert.strictEqual(labelsResp.status, 200);
  const labelsData = await labelsResp.json();

  test("V2 /labels returns strictly 18 classes", () => {
    assert.strictEqual(labelsData.vocabulary_size, 18);
    assert.strictEqual(labelsData.classes.length, 18);
    for (const c of V2_VOCABULARY_18) {
      assert(labelsData.classes.includes(c), `Missing V2 class: ${c}`);
    }
  });

  const labelsV3Resp = await fetch(API_ENDPOINTS.LABELS_V3_SIX_SIGN);
  assert.strictEqual(labelsV3Resp.status, 200);
  const labelsV3Data = await labelsV3Resp.json();

  test("V3 /labels/v3-six-sign returns strictly 6 classes", () => {
    assert.strictEqual(labelsV3Data.vocabulary_size, 6);
    assert.strictEqual(labelsV3Data.classes.length, 6);
    for (const c of V3_VOCABULARY) {
      assert(labelsV3Data.classes.includes(c), `Missing V3 class: ${c}`);
    }
  });

  // --- STEP 3: VERIFY V2 POST /predict/sequence CONTINUES WORKING (30x150) ---
  console.log("\n--- STEP 3: V2 INFERENCE CONTRACT (30x150) ---");
  const v2SampleFrames = samplePrototypes["help"].frames;
  assert(v2SampleFrames && v2SampleFrames.length === 30, "V2 prototype HELP exists");

  const v2InferenceResp = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames: v2SampleFrames,
      confidence_threshold: 0.50
    })
  });
  assert.strictEqual(v2InferenceResp.status, 200);
  const v2Result = await v2InferenceResp.json();

  test("V2 /predict/sequence predicts HELP from 30x150", () => {
    assert.strictEqual(v2Result.label, "help");
    assert(v2Result.confidence > 0.90, `High confidence expected, got ${v2Result.confidence}`);
    assert.strictEqual(v2Result.accepted, true);
    assert.strictEqual(v2Result.tensor_shapes.frontend[1], 150);
  });

  // --- STEP 4: VERIFY V3 POST /predict/sequence/v3-six-sign (30x168) ---
  console.log("\n--- STEP 4: V3 INFERENCE CONTRACT (30x168) ---");
  // Construct synthetic 30x168 sequence based on HELP prototype padded with 18 body-relative coordinates
  const v3SampleFrames = v2SampleFrames.map(frame150 => {
    const frame168 = new Array(168).fill(0.0);
    for (let i = 0; i < 150; i++) frame168[i] = frame150[i];
    // Set 18 body-relative coordinates (indices 150..167)
    // Left wrist relative to shoulder center, nose, chest
    frame168[150] = -0.2; frame168[151] = 0.5; frame168[152] = 0.0;
    frame168[153] = 0.2;  frame168[154] = 0.5; frame168[155] = 0.0;
    return frame168;
  });

  const v3InferenceResp = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE_V3_SIX_SIGN, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames: v3SampleFrames,
      confidence_threshold: 0.50
    })
  });
  assert.strictEqual(v3InferenceResp.status, 200, `V3 endpoint returned status ${v3InferenceResp.status}`);
  const v3Result = await v3InferenceResp.json();

  test("V3 endpoint accepts 30x168 sequence", () => {
    assert(v3Result.confidence !== undefined);
    assert(v3Result.top_k && v3Result.top_k.length > 0);
    assert.strictEqual(v3Result.tensor_shapes.frontend[1], 168);
  });

  test("V3 top prediction is strictly one of the 6 allowed classes", () => {
    const topClass = v3Result.top_k[0].label;
    assert(V3_VOCABULARY.includes(topClass), `Prediction '${topClass}' must be in ${V3_VOCABULARY}`);
  });

  test("V3 top_k contains strictly 6 classes or fewer (no 17/18 classes)", () => {
    assert(v3Result.top_k.length <= 6, `Expected at most 6 classes in top_k, got ${v3Result.top_k.length}`);
    for (const item of v3Result.top_k) {
      assert(V3_VOCABULARY.includes(item.label), `Unexpected class '${item.label}' in V3 top_k`);
    }
  });

  // --- STEP 5: VERIFY SHAPE REJECTIONS ---
  console.log("\n--- STEP 5: SHAPE REJECTION CONTRACTS ---");
  // Sending 168 to V2 endpoint must fail with 422
  const v2RejectResp = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames: v3SampleFrames })
  });
  test("V2 endpoint rejects 30x168 with HTTP 422", () => {
    assert.strictEqual(v2RejectResp.status, 422);
  });

  // Sending 150 to V3 endpoint must fail with 422
  const v3RejectResp = await fetch(API_ENDPOINTS.PREDICT_SEQUENCE_V3_SIX_SIGN, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames: v2SampleFrames })
  });
  test("V3 endpoint rejects 30x150 with HTTP 422", () => {
    assert.strictEqual(v3RejectResp.status, 422);
  });

  // --- STEP 6: VERIFY FASTAPI RECOGNITION ADAPTER IN V3 MODE ---
  console.log("\n--- STEP 6: ADAPTER MODE VERIFICATION ---");
  const adapterV3 = new FastAPIRecognitionAdapter(null, 'v3_six_sign');
  test("FastAPIRecognitionAdapter initializes in v3_six_sign mode", () => {
    assert.strictEqual(adapterV3.modelMode, 'v3_six_sign');
    assert(adapterV3.endpoint.includes('/predict/sequence/v3-six-sign'));
  });

  const adapterV2 = new FastAPIRecognitionAdapter(null, 'v2');
  test("FastAPIRecognitionAdapter initializes in v2 mode", () => {
    assert.strictEqual(adapterV2.modelMode, 'v2');
    assert(adapterV2.endpoint.includes('/predict/sequence') && !adapterV2.endpoint.includes('v3-six-sign'));
  });

  console.log("\n====================================================");
  console.log(`INTEGRATION TESTS COMPLETE: ${passed} PASSED | ${failed} FAILED`);
  console.log("====================================================");

  if (failed > 0) process.exit(1);
}

runTests().catch(err => {
  console.error("Test execution failed:", err);
  process.exit(1);
});
