// services/modelInferenceDebugService.js - Model Inference Pipeline Debugger
// Phase 3 Debugging: Verifies 30x150 sequence -> FastAPI /predict/sequence -> PyTorch Bi-GRU -> Softmax -> Predictions.
// Strict rules:
// - NO automatic continuous recognition (Manual inference button only)
// - NO predictions sent to Admin / NO chat dispatch / NO sign animation
// - Deterministic Known Sample Test (Test A) and Live 30x150 Webcam Test (Test B)

import { API_ENDPOINTS } from './apiConfig.js';
import samplePrototypes from './recognition/sampleSequences.json' with { type: 'json' };

// Exact 17 dynamic classes expected by DynamicSignBiGRU (input_dim=150, num_classes=17)
// Note: letter_a is strictly in the static model and NOT in dynamic classes.
export const DYNAMIC_17_CLASSES = [
  "help",        // 0
  "doctor",      // 1
  "hospital",    // 2
  "sick",        // 3
  "appointment", // 4
  "where",       // 5
  "bathroom",    // 6
  "yes",         // 7
  "no",          // 8
  "please",      // 9
  "thank_you",   // 10
  "wait",        // 11
  "understand",  // 12
  "problem",     // 13
  "money",       // 14
  "pay",         // 15
  "document"     // 16
];

class ModelInferenceDebugService {
  constructor() {
    this.endpoint = API_ENDPOINTS.PREDICT_SEQUENCE;

    // Current live 30x150 sequence buffer from landmarkPipelineService
    this.currentSequence30 = null;
    this.isSequenceReady = false;

    // Model Input Diagnostics (Step 1)
    this.inputStats = {
      shape: '0 × 150',
      expected: '30 × 150',
      validation: 'WAITING (Need 30 frames)',
      isPass: false,
      min: 0.0,
      max: 0.0,
      mean: 0.0,
      std: 0.0,
      nanInf: 'NONE'
    };

    // Test A: Known Sample Result (Step 8 & 10)
    this.knownSampleResult = null;

    // Test B: Live Sequence Result (Step 4 & 10)
    this.liveSequenceResult = null;

    this.isRequestInFlight = false;
  }

  /**
   * Called by landmarkPipelineService when frames are updated.
   * Computes Step 1 Model Input Diagnostics on the 30x150 sequence.
   */
  updateLiveSequence(frameBuffer) {
    if (!frameBuffer || frameBuffer.length === 0) {
      this.currentSequence30 = null;
      this.isSequenceReady = false;
      this.inputStats = {
        shape: '0 × 150',
        expected: '30 × 150',
        validation: 'WAITING (Need 30 frames)',
        isPass: false,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
        nanInf: 'NONE'
      };
      this.updateUI();
      return;
    }

    const currentLen = frameBuffer.length;
    this.isSequenceReady = (currentLen === 30);

    if (this.isSequenceReady) {
      this.currentSequence30 = frameBuffer;

      // Compute statistics across all 30 x 150 = 4500 values
      let minVal = Infinity;
      let maxVal = -Infinity;
      let sum = 0.0;
      let hasNaNOrInf = false;
      let count = 0;

      for (let t = 0; t < 30; t++) {
        const frame = frameBuffer[t];
        for (let i = 0; i < 150; i++) {
          const v = frame[i];
          if (Number.isNaN(v) || !Number.isFinite(v)) {
            hasNaNOrInf = true;
          }
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
          sum += v;
          count++;
        }
      }

      const meanVal = count > 0 ? (sum / count) : 0.0;

      let varianceSum = 0.0;
      for (let t = 0; t < 30; t++) {
        const frame = frameBuffer[t];
        for (let i = 0; i < 150; i++) {
          const diff = frame[i] - meanVal;
          varianceSum += diff * diff;
        }
      }
      const stdVal = count > 0 ? Math.sqrt(varianceSum / count) : 0.0;

      const isShapeValid = (currentLen === 30) && !hasNaNOrInf;

      this.inputStats = {
        shape: '30 × 150',
        expected: '30 × 150',
        validation: isShapeValid ? 'PASS' : 'FAIL',
        isPass: isShapeValid,
        min: Number(minVal.toFixed(3)),
        max: Number(maxVal.toFixed(3)),
        mean: Number(meanVal.toFixed(3)),
        std: Number(stdVal.toFixed(3)),
        nanInf: hasNaNOrInf ? 'DETECTED (ERROR)' : 'NONE'
      };
    } else {
      this.currentSequence30 = null;
      this.inputStats = {
        shape: `${currentLen} × 150`,
        expected: '30 × 150',
        validation: `COLLECTING (${currentLen}/30)`,
        isPass: false,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
        nanInf: 'NONE'
      };
    }

    this.updateUI();
  }

  /**
   * TEST A: Known Deterministic Sample Test (Step 8 & 10)
   * Runs the pre-extracted "help" sequence from sampleSequences.json
   */
  async testKnownSample(sampleKey = 'help') {
    if (this.isRequestInFlight) return;
    this.isRequestInFlight = true;
    this.updateStatusBanner('Testing known sample (' + sampleKey + ')...');

    const sample = samplePrototypes[sampleKey];
    if (!sample || !sample.frames) {
      console.error(`Sample ${sampleKey} not found in sampleSequences.json`);
      this.isRequestInFlight = false;
      return;
    }

    const t0 = performance.now();
    try {
      const resp = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames: sample.frames,
          confidence_threshold: 0.0 // Raw model output without threshold rejection
        })
      });

      const latencyMs = Math.round(performance.now() - t0);
      const httpStatus = resp.status;

      if (!resp.ok) {
        throw new Error(`HTTP ${httpStatus}`);
      }

      const data = await resp.json();
      const topPred = data.top_k && data.top_k.length > 0 ? data.top_k[0] : { label: data.label, confidence: data.confidence };
      const actualClass = topPred.label;
      const confidence = topPred.confidence;

      // Tensor shape verification (Step 6)
      const shapes = data.tensor_shapes || {
        frontend: [30, 150],
        backend_received: [30, 150],
        pytorch_tensor: [1, 30, 150],
        model_output: [1, 17]
      };

      const shapePass = (
        shapes.frontend[0] === 30 && shapes.frontend[1] === 150 &&
        shapes.backend_received[0] === 30 && shapes.backend_received[1] === 150 &&
        shapes.pytorch_tensor[0] === 1 && shapes.pytorch_tensor[1] === 30 && shapes.pytorch_tensor[2] === 150 &&
        shapes.model_output[0] === 1 && shapes.model_output[1] === 17
      );

      // Model output validation (Step 7)
      const probSum = data.probability_sum !== undefined ? data.probability_sum : 1.0;
      const isProbSumValid = Math.abs(probSum - 1.0) < 0.02;

      this.knownSampleResult = {
        expected: sampleKey,
        actual: actualClass,
        confidence: confidence,
        confidencePercent: `${(confidence * 100).toFixed(2)}%`,
        isPass: (actualClass === sampleKey) && shapePass && isProbSumValid,
        httpStatus,
        latencyMs: data.inference_latency_ms || latencyMs,
        topK: (data.top_k || []).slice(0, 3),
        shapes,
        shapePass,
        probSum: probSum.toFixed(4),
        outputValidation: (shapePass && isProbSumValid) ? 'PASS' : 'FAIL'
      };

      console.log('[ModelInferenceDebugService] Known Sample Test A Result:', this.knownSampleResult);
    } catch (err) {
      console.error('[ModelInferenceDebugService] Known Sample Test failed:', err);
      this.knownSampleResult = {
        expected: sampleKey,
        actual: 'ERROR',
        confidence: 0,
        confidencePercent: '0%',
        isPass: false,
        httpStatus: 'ERR',
        latencyMs: 0,
        topK: [],
        shapes: null,
        shapePass: false,
        probSum: '0.0000',
        outputValidation: 'FAIL'
      };
    } finally {
      this.isRequestInFlight = false;
      this.updateUI();
    }
  }

  /**
   * TEST B: Live Camera 30x150 Sequence Test (Step 2, 3, 4, 10)
   * Triggered ONLY on user button click when 30 valid frames are ready
   */
  async testLiveSequence() {
    if (!this.isSequenceReady || !this.currentSequence30 || this.isRequestInFlight) {
      return;
    }

    this.isRequestInFlight = true;
    this.updateStatusBanner('Running model inference on live 30 × 150 sequence...');

    // Convert Float32Array frames to standard nested array of 150 floats
    const formattedFrames = this.currentSequence30.map((f) => Array.from(f));

    const t0 = performance.now();
    try {
      const resp = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames: formattedFrames,
          confidence_threshold: 0.0 // Raw model output without threshold rejection
        })
      });

      const latencyMs = Math.round(performance.now() - t0);
      const httpStatus = resp.status;

      if (!resp.ok) {
        throw new Error(`HTTP ${httpStatus}`);
      }

      const data = await resp.json();
      const topPred = data.top_k && data.top_k.length > 0 ? data.top_k[0] : { label: data.label, confidence: data.confidence };
      const prediction = topPred.label;
      const confidence = topPred.confidence;

      // Tensor shape verification (Step 6)
      const shapes = data.tensor_shapes || {
        frontend: [30, 150],
        backend_received: [30, 150],
        pytorch_tensor: [1, 30, 150],
        model_output: [1, 17]
      };

      const shapePass = (
        shapes.frontend[0] === 30 && shapes.frontend[1] === 150 &&
        shapes.backend_received[0] === 30 && shapes.backend_received[1] === 150 &&
        shapes.pytorch_tensor[0] === 1 && shapes.pytorch_tensor[1] === 30 && shapes.pytorch_tensor[2] === 150 &&
        shapes.model_output[0] === 1 && shapes.model_output[1] === 17
      );

      // Model output validation (Step 7)
      const probSum = data.probability_sum !== undefined ? data.probability_sum : 1.0;
      const isProbSumValid = Math.abs(probSum - 1.0) < 0.02;

      this.liveSequenceResult = {
        prediction,
        classId: data.class_id,
        confidence: confidence,
        confidencePercent: `${(confidence * 100).toFixed(2)}%`,
        accepted: data.accepted ? 'YES' : 'NO',
        httpStatus,
        latencyMs: data.inference_latency_ms || latencyMs,
        topK: (data.top_k || []).slice(0, 3),
        shapes,
        shapePass,
        probSum: probSum.toFixed(4),
        outputValidation: (shapePass && isProbSumValid) ? 'PASS' : 'FAIL',
        result: 'RECEIVED'
      };

      console.log('[ModelInferenceDebugService] Live Camera Test B Result:', this.liveSequenceResult);
    } catch (err) {
      console.error('[ModelInferenceDebugService] Live inference failed:', err);
      this.liveSequenceResult = {
        prediction: 'ERROR',
        classId: null,
        confidence: 0,
        confidencePercent: '0%',
        accepted: 'NO',
        httpStatus: 'ERR',
        latencyMs: 0,
        topK: [],
        shapes: null,
        shapePass: false,
        probSum: '0.0000',
        outputValidation: 'FAIL',
        result: 'ERROR'
      };
    } finally {
      this.isRequestInFlight = false;
      this.updateUI();
    }
  }

  updateStatusBanner(msg) {
    if (typeof document === 'undefined') return;
    const banner = document.getElementById('inference-status-msg');
    if (banner) banner.textContent = msg;
  }

  updateUI() {
    if (typeof document === 'undefined') return;

    // 1. Model Input Debug Display (Step 1)
    const inpShapeEl = document.getElementById('model-input-shape');
    const inpExpEl = document.getElementById('model-input-expected');
    const inpValEl = document.getElementById('model-input-validation');
    const inpMinEl = document.getElementById('model-input-min');
    const inpMaxEl = document.getElementById('model-input-max');
    const inpMeanEl = document.getElementById('model-input-mean');
    const inpStdEl = document.getElementById('model-input-std');
    const inpNanEl = document.getElementById('model-input-nan-inf');

    if (inpShapeEl) inpShapeEl.textContent = this.inputStats.shape;
    if (inpExpEl) inpExpEl.textContent = this.inputStats.expected;
    if (inpValEl) {
      inpValEl.textContent = this.inputStats.validation;
      inpValEl.className = this.inputStats.isPass
        ? 'px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
        : 'px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700';
    }
    if (inpMinEl) inpMinEl.textContent = this.inputStats.min;
    if (inpMaxEl) inpMaxEl.textContent = this.inputStats.max;
    if (inpMeanEl) inpMeanEl.textContent = this.inputStats.mean;
    if (inpStdEl) inpStdEl.textContent = this.inputStats.std;
    if (inpNanEl) {
      inpNanEl.textContent = this.inputStats.nanInf;
      inpNanEl.className = this.inputStats.nanInf === 'NONE'
        ? 'text-emerald-400 font-bold'
        : 'text-red-400 font-bold animate-pulse';
    }

    // 2. Button State Management (Step 2)
    const btnTestLive = document.getElementById('btn-test-live-inference');
    if (btnTestLive) {
      if (this.isSequenceReady && !this.isRequestInFlight) {
        btnTestLive.disabled = false;
        btnTestLive.className = 'w-full py-2 px-3 text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg shadow-md transition-all flex items-center justify-center gap-1.5 cursor-pointer';
      } else {
        btnTestLive.disabled = true;
        btnTestLive.className = 'w-full py-2 px-3 text-xs font-bold bg-slate-800 text-slate-500 rounded-lg border border-slate-700/60 transition-all flex items-center justify-center gap-1.5 cursor-not-allowed opacity-60';
      }
    }

    // 3. Known Sample Test A Display (Step 8 & 10)
    if (this.knownSampleResult) {
      const res = this.knownSampleResult;
      const kExpEl = document.getElementById('known-sample-expected');
      const kActEl = document.getElementById('known-sample-actual');
      const kConfEl = document.getElementById('known-sample-conf');
      const kResEl = document.getElementById('known-sample-result');
      const kLatEl = document.getElementById('known-sample-latency');
      const kTop3El = document.getElementById('known-sample-top3');

      if (kExpEl) kExpEl.textContent = res.expected;
      if (kActEl) kActEl.textContent = res.actual;
      if (kConfEl) kConfEl.textContent = res.confidencePercent;
      if (kLatEl) kLatEl.textContent = `${res.latencyMs} ms (HTTP ${res.httpStatus})`;
      if (kResEl) {
        kResEl.textContent = res.isPass ? 'PASS' : 'FAIL';
        kResEl.className = res.isPass
          ? 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/50'
          : 'px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/50';
      }
      if (kTop3El && res.topK) {
        kTop3El.innerHTML = res.topK.map((item, idx) => `
          <div class="flex justify-between text-[10px] font-mono py-0.5 border-b border-slate-800/60 last:border-0">
            <span class="text-slate-300">${idx + 1}. <strong class="text-white">${item.label}</strong></span>
            <span class="text-emerald-400 font-bold">${(item.confidence * 100).toFixed(2)}%</span>
          </div>
        `).join('');
      }
    }

    // 4. Live Sequence Test B Display (Step 4, 6, 7, 10)
    if (this.liveSequenceResult) {
      const res = this.liveSequenceResult;
      const livePredEl = document.getElementById('live-inference-pred');
      const liveConfEl = document.getElementById('live-inference-conf');
      const liveAccEl = document.getElementById('live-inference-accepted');
      const liveHttpEl = document.getElementById('live-inference-http');
      const liveLatEl = document.getElementById('live-inference-latency');
      const liveTop3El = document.getElementById('live-inference-top3');
      const liveProbSumEl = document.getElementById('live-inference-probsum');
      const liveValEl = document.getElementById('live-inference-validation');
      const liveShapesEl = document.getElementById('live-tensor-shapes');

      if (livePredEl) livePredEl.textContent = res.prediction;
      if (liveConfEl) liveConfEl.textContent = res.confidencePercent;
      if (liveAccEl) {
        liveAccEl.textContent = res.accepted;
        liveAccEl.className = res.accepted === 'YES' ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold';
      }
      if (liveHttpEl) liveHttpEl.textContent = `HTTP ${res.httpStatus}`;
      if (liveLatEl) liveLatEl.textContent = `${res.latencyMs} ms`;
      if (liveProbSumEl) liveProbSumEl.textContent = res.probSum;
      if (liveValEl) {
        liveValEl.textContent = res.outputValidation;
        liveValEl.className = res.outputValidation === 'PASS'
          ? 'px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
          : 'px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/40 animate-pulse';
      }

      if (liveTop3El && res.topK) {
        liveTop3El.innerHTML = res.topK.map((item, idx) => `
          <div class="flex justify-between text-[10px] font-mono py-0.5 border-b border-slate-800/60 last:border-0">
            <span class="text-slate-300">${idx + 1}. <strong class="text-white">${item.label}</strong></span>
            <span class="text-emerald-400 font-bold">${(item.confidence * 100).toFixed(2)}%</span>
          </div>
        `).join('');
      }

      if (liveShapesEl && res.shapes) {
        const fOk = res.shapes.frontend[0] === 30 && res.shapes.frontend[1] === 150;
        const bOk = res.shapes.backend_received[0] === 30 && res.shapes.backend_received[1] === 150;
        const tOk = res.shapes.pytorch_tensor[0] === 1 && res.shapes.pytorch_tensor[1] === 30 && res.shapes.pytorch_tensor[2] === 150;
        const oOk = res.shapes.model_output[0] === 1 && res.shapes.model_output[1] === 17;

        liveShapesEl.innerHTML = `
          <div class="flex justify-between"><span>Frontend:</span><strong class="${fOk ? 'text-white' : 'text-red-400 font-bold'}">${res.shapes.frontend.join(' × ')}</strong></div>
          <div class="flex justify-between"><span>Backend received:</span><strong class="${bOk ? 'text-white' : 'text-red-400 font-bold'}">${res.shapes.backend_received.join(' × ')}</strong></div>
          <div class="flex justify-between"><span>PyTorch tensor:</span><strong class="${tOk ? 'text-white' : 'text-red-400 font-bold'}">[${res.shapes.pytorch_tensor.join(', ')}]</strong></div>
          <div class="flex justify-between"><span>Model output:</span><strong class="${oOk ? 'text-white' : 'text-red-400 font-bold'}">[${res.shapes.model_output.join(', ')}]</strong></div>
          ${!res.shapePass ? '<div class="text-[10px] text-red-400 font-bold bg-red-950/80 p-1 rounded mt-1 border border-red-500">TENSOR SHAPE MISMATCH ERROR</div>' : ''}
        `;
      }
    }
  }
}

export const modelInferenceDebugService = new ModelInferenceDebugService();
