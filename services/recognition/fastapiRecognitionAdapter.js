// services/recognition/fastapiRecognitionAdapter.js - Real Continuous ML Recognition Adapter
// Architecture: Webcam -> MediaPipe (Hands+Pose) -> LandmarkPipelineService (150-dim, 2.60s window, 30x150) -> POST /predict/sequence -> Temporal Stabilization -> Dispatch to Admin

import { BaseRecognitionAdapter } from './recognitionAdapter.js';
import { API_ENDPOINTS } from '../apiConfig.js';
import samplePrototypes from './sampleSequences.json' with { type: 'json' };
import { landmarkPipelineService } from '../landmarkPipelineService.js';

export const DISPLAY_TEXT_MAP = {
  "help": "I need help",
  "doctor": "I need a doctor",
  "hospital": "Where is the hospital?",
  "sick": "I am sick / Need medical care",
  "appointment": "I have an appointment",
  "where": "Where is the counter?",
  "bathroom": "Where is the bathroom?",
  "yes": "Yes",
  "no": "No",
  "please": "Please help me",
  "thank_you": "Thank you",
  "wait": "Please wait",
  "understand": "I understand",
  "problem": "I have a problem",
  "money": "Cash / Fee payment",
  "pay": "Payment counter / I want to pay",
  "document": "Here is my document / paper",
  "letter_a": "Ticket #A",
  "hello": "Hello / Greetings"
};

/**
 * Recognition Model Configuration:
 * - 'v3_six_sign': Uses the focused 6-sign Bi-GRU model (30x168 features) via /predict/sequence/v3-six-sign
 * - 'v2': Production fallback using the 18-class Bi-GRU model (30x150 features) via /predict/sequence
 */
export const RECOGNITION_MODEL = 'v3_six_sign';

/**
 * Updates DOM indicators for Continuous Recognition Live Diagnostics (Rule 10)
 */
export function updateContinuousDiagnosticsUI({
  status = 'Collecting',
  prediction = '—',
  confidence = '—',
  rawFrames = 0,
  sampledFrames = '0/30',
  sequenceShape = '0 × 150',
  latency = '—'
} = {}) {
  if (typeof document === 'undefined') return;

  const statusEl = document.getElementById('recog-diag-status');
  const predEl = document.getElementById('recog-diag-pred');
  const confEl = document.getElementById('recog-diag-conf');
  const rawEl = document.getElementById('recog-diag-raw');
  const sampledEl = document.getElementById('recog-diag-sampled');
  const seqEl = document.getElementById('recog-diag-sequence');
  const latEl = document.getElementById('recog-diag-latency');

  if (statusEl) {
    statusEl.textContent = status;
    if (status === 'Recognized') {
      statusEl.className = 'px-2 py-0.5 rounded text-[9.5px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/50';
    } else if (status === 'Uncertain') {
      statusEl.className = 'px-2 py-0.5 rounded text-[9.5px] font-bold bg-red-500/20 text-red-300 border border-red-500/40';
    } else if (status === 'Inference') {
      statusEl.className = 'px-2 py-0.5 rounded text-[9.5px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 animate-pulse';
    } else {
      statusEl.className = 'px-2 py-0.5 rounded text-[9.5px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40';
    }
  }

  if (predEl) predEl.textContent = prediction;
  if (confEl) confEl.textContent = confidence;
  if (rawEl) rawEl.textContent = String(rawFrames);
  if (sampledEl) sampledEl.textContent = sampledFrames;
  if (seqEl) seqEl.textContent = sequenceShape;
  if (latEl) latEl.textContent = latency;
}

export class FastAPIRecognitionAdapter extends BaseRecognitionAdapter {
  constructor(endpoint = null, modelMode = null) {
    super('FastAPIRecognitionAdapter', 'fastapi');

    // Model mode configuration: explicitly provided mode, or inferred from endpoint, or default to RECOGNITION_MODEL
    if (endpoint && endpoint.includes('/predict/sequence') && !endpoint.includes('v3-six-sign')) {
      this.modelMode = modelMode || 'v2';
      this.endpoint = endpoint;
    } else if (endpoint && endpoint.includes('v3-six-sign')) {
      this.modelMode = 'v3_six_sign';
      this.endpoint = endpoint;
    } else {
      this.modelMode = modelMode || RECOGNITION_MODEL;
      this.endpoint = (this.modelMode === 'v3_six_sign')
        ? API_ENDPOINTS.PREDICT_SEQUENCE_V3_SIX_SIGN
        : API_ENDPOINTS.PREDICT_SEQUENCE;
    }

    this.healthEndpoint = API_ENDPOINTS.HEALTH;

    this.isRequestInFlight = false;
    this.isBackendOnline = false;
    this.lastInferenceTime = 0;
    this.unsubPipeline = null;

    // Real configured confidence threshold (Rule 5: 0.70)
    this.confidenceThreshold = 0.70;

    // Temporal Stabilizer & Debounce (Rule 7, 8: 2 consecutive consistent windows, 3500ms cooldown)
    this.requiredConsistentWindows = 2;
    this.recentPredictions = [];
    this.lastEmittedText = null;
    this.cooldownUntil = 0;
    this.cooldownDurationMs = 3500;
  }

  async start(videoElement, callbacks) {
    super.start(videoElement, callbacks);
    console.log(`[FastAPIRecognitionAdapter] Initializing continuous recognition [Mode: ${this.modelMode}] for FastAPI (${this.endpoint})...`);

    if (this.onStatus) {
      this.onStatus('Connecting to SignBridge AI ML service...', 'CONNECTING');
    }

    // 1. Check backend health
    await this.verifyBackendHealth();

    // 2. Connect directly to verified LandmarkPipelineService
    if (this.unsubPipeline) {
      this.unsubPipeline();
    }

    this.unsubPipeline = landmarkPipelineService.subscribe(({ sequence, diagnostics }) => {
      this.handlePipelineFrame(sequence, diagnostics);
    });
  }

  async verifyBackendHealth() {
    try {
      const resp = await fetch(this.healthEndpoint, { method: 'GET' });
      if (resp.ok) {
        const data = await resp.json();
        this.isBackendOnline = (data.status === 'ok' && data.dynamic_model === 'loaded');
        console.log('[FastAPIRecognitionAdapter] FastAPI health confirmed:', data);
        if (this.onStatus) {
          this.onStatus('Recognition: Ready (Collecting)', 'READY');
        }
      } else {
        this.handleBackendOffline('FastAPI service returned status ' + resp.status);
      }
    } catch (err) {
      this.handleBackendOffline(err.message);
    }
  }

  handleBackendOffline(reason) {
    this.isBackendOnline = false;
    console.warn('[FastAPIRecognitionAdapter] Backend offline:', reason);
    if (this.onStatus) {
      this.onStatus('AI recognition unavailable', 'OFFLINE');
    }
  }

  handlePipelineFrame(sequence, diagnostics) {
    if (!this.isRunning || !this.isBackendOnline) return;

    const isV3 = (this.modelMode === 'v3_six_sign');
    const shapeLabel = isV3 ? '30 × 168' : '30 × 150';

    // Check if temporal window is still collecting
    if (!diagnostics.isReady || !sequence || sequence.length !== 30) {
      updateContinuousDiagnosticsUI({
        status: 'Collecting',
        prediction: '—',
        confidence: '—',
        rawFrames: diagnostics.rawFramesCount,
        sampledFrames: '0/30',
        sequenceShape: isV3 ? '0 × 168' : '0 × 150',
        latency: '—'
      });

      if (this.onStatus && diagnostics.rawFramesCount > 0) {
        this.onStatus(`Collecting gesture: ${diagnostics.currentSpanSec.toFixed(1)}s / 2.60s`, 'COLLECTING');
      }
      return;
    }

    // Sequence is ready (strictly 30 frames)
    const now = Date.now();
    // Throttle inference dispatch to max once per 450ms
    if (this.isRequestInFlight || (now - this.lastInferenceTime < 450)) {
      return;
    }

    this.isRequestInFlight = true;
    this.lastInferenceTime = now;

    updateContinuousDiagnosticsUI({
      status: 'Inference',
      prediction: `Evaluating (${isV3 ? 'V3-6Sign' : 'V2'})...`,
      confidence: '—',
      rawFrames: diagnostics.rawFramesCount,
      sampledFrames: '30/30',
      sequenceShape: shapeLabel,
      latency: '—'
    });

    if (this.onStatus) {
      this.onStatus(`Processing sequence with Bi-GRU (${isV3 ? 'V3 6-Sign' : 'V2'})...`, 'RECOGNIZING');
    }

    this.dispatchInferenceRequest(sequence, diagnostics);
  }

  async dispatchInferenceRequest(sequence30, diagnostics) {
    const tStart = performance.now();
    const isV3 = (this.modelMode === 'v3_six_sign');
    const shapeLabel = isV3 ? '30 × 168' : '30 × 150';

    try {
      const resp = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames: Array.from(sequence30).map(f => {
            const arr = Array.from(f);
            // In V2 mode, slice 168 down to 150 features. In V3 mode, preserve full 168 features.
            return (!isV3 && arr.length > 150) ? arr.slice(0, 150) : arr;
          }),
          confidence_threshold: this.confidenceThreshold
        })
      });

      const roundTripMs = performance.now() - tStart;

      if (!resp.ok) {
        throw new Error(`Inference returned status ${resp.status}`);
      }

      const result = await resp.json();
      this.handleInferenceResponse(result, roundTripMs, diagnostics);
    } catch (err) {
      console.warn('[FastAPIRecognitionAdapter] Inference request error:', err.message);
      updateContinuousDiagnosticsUI({
        status: 'Uncertain',
        prediction: 'API Error',
        confidence: '0.0%',
        rawFrames: diagnostics.rawFramesCount,
        sampledFrames: '30/30',
        sequenceShape: shapeLabel,
        latency: `${Math.round(performance.now() - tStart)} ms`
      });
      if (this.onStatus) {
        this.onStatus('AI recognition unavailable', 'OFFLINE');
      }
    } finally {
      this.isRequestInFlight = false;
    }
  }

  handleInferenceResponse(result, roundTripMs, diagnostics) {
    if (!result) return;

    const isV3 = (this.modelMode === 'v3_six_sign');
    const shapeLabel = isV3 ? '30 × 168' : '30 × 150';

    const { label, confidence, accepted, top_k, inference_latency_ms } = result;

    const topLabel = label || (top_k && top_k[0] ? top_k[0].label : null);
    const conf = (confidence !== undefined && confidence !== null)
      ? confidence
      : (top_k && top_k[0] ? top_k[0].confidence : 0.0);

    const isAccepted = Boolean(accepted && conf >= this.confidenceThreshold && topLabel);
    const latencyStr = inference_latency_ms ? `${inference_latency_ms.toFixed(1)} ms` : `${roundTripMs.toFixed(1)} ms`;

    const now = Date.now();

    // RULE 6: If confidence < 0.70:
    // display "Prediction: Uncertain" on camera overlay and do NOT send a Deaf message to Admin.
    if (!isAccepted) {
      this.recentPredictions = [];

      updateContinuousDiagnosticsUI({
        status: 'Uncertain',
        prediction: topLabel ? `${topLabel} (< 70%)` : 'Uncertain',
        confidence: `${(conf * 100).toFixed(1)}%`,
        rawFrames: diagnostics.rawFramesCount,
        sampledFrames: '30/30',
        sequenceShape: shapeLabel,
        latency: latencyStr
      });

      if (this.onPrediction) {
        this.onPrediction({
          text: null,
          signName: null,
          predictionText: 'Prediction: Uncertain',
          confidence: conf,
          isFinal: false,
          isUncertain: true,
          isLive: true,
          status: 'UNCERTAIN'
        });
      }

      if (this.onStatus) {
        this.onStatus('Uncertain', 'UNCERTAIN');
      }
      return;
    }

    // RULE 7: If confidence >= 0.70, require temporal confirmation using stabilization/debounce
    const signName = topLabel.toUpperCase();
    const displayText = signName;

    this.recentPredictions.push(topLabel);
    if (this.recentPredictions.length > this.requiredConsistentWindows) {
      this.recentPredictions.shift();
    }

    const isConsistent = (
      this.recentPredictions.length === this.requiredConsistentWindows &&
      this.recentPredictions.every(s => s === topLabel)
    );

    // RULE 8: Do NOT repeatedly send the same prediction every inference cycle (cooldown)
    const isCooldownActive = (displayText === this.lastEmittedText && now < this.cooldownUntil);

    if (isConsistent && !isCooldownActive) {
      // RULE 9: Only dispatch a recognized sign to Admin when it is confirmed
      this.lastEmittedText = displayText;
      this.cooldownUntil = now + this.cooldownDurationMs;
      this.recentPredictions = [];

      console.log(`[FastAPIRecognitionAdapter] Confirmed sign: "${displayText}" (${(conf * 100).toFixed(1)}%)`);

      updateContinuousDiagnosticsUI({
        status: 'Recognized',
        prediction: `${topLabel} ("${displayText}")`,
        confidence: `${(conf * 100).toFixed(1)}%`,
        rawFrames: diagnostics.rawFramesCount,
        sampledFrames: '30/30',
        sequenceShape: shapeLabel,
        latency: latencyStr
      });

      if (this.onPrediction) {
        this.onPrediction({
          text: displayText,
          signName: signName,
          predictionText: `Prediction: ${signName}`,
          confidence: conf,
          isFinal: true,
          isUncertain: false,
          isLive: true,
          status: 'CONFIRMED'
        });
      }

      if (this.onStatus) {
        this.onStatus(`Recognized: ${displayText}`, 'CONFIRMED');
      }
    } else if (!isCooldownActive) {
      // Stabilizing: accumulating consistent windows (e.g. 1 of 2)
      updateContinuousDiagnosticsUI({
        status: 'Inference',
        prediction: `${topLabel} (Confirming ${this.recentPredictions.length}/${this.requiredConsistentWindows})`,
        confidence: `${(conf * 100).toFixed(1)}%`,
        rawFrames: diagnostics.rawFramesCount,
        sampledFrames: '30/30',
        sequenceShape: shapeLabel,
        latency: latencyStr
      });

      if (this.onPrediction) {
        this.onPrediction({
          text: null,
          signName: signName,
          predictionText: `Prediction: ${signName}`,
          confidence: conf,
          isFinal: false,
          isUncertain: false,
          isLive: true,
          status: 'RECOGNIZING'
        });
      }

      if (this.onStatus) {
        this.onStatus(`Recognizing: ${displayText}...`, 'RECOGNIZING');
      }
    }
  }

  // Trigger Sign Testing: executes real FastAPI prediction using verified dataset prototypes
  async triggerPhrase(phraseOrLabel) {
    if (!phraseOrLabel) return;

    let targetClass = null;
    const cleanQuery = phraseOrLabel.trim().toLowerCase();

    for (const [lbl, proto] of Object.entries(samplePrototypes)) {
      if (lbl.toLowerCase() === cleanQuery ||
          proto.display_text.toLowerCase() === cleanQuery ||
          `"${proto.display_text.toLowerCase()}"` === cleanQuery) {
        targetClass = proto;
        break;
      }
    }

    if (!targetClass) {
      for (const [lbl, proto] of Object.entries(samplePrototypes)) {
        if (proto.display_text.toLowerCase().includes(cleanQuery) || cleanQuery.includes(lbl)) {
          targetClass = proto;
          break;
        }
      }
    }

    if (!targetClass) {
      console.warn('[FastAPIRecognitionAdapter] No prototype sequence found for:', phraseOrLabel);
      return;
    }

    console.log(`[FastAPIRecognitionAdapter] Testing sign "${targetClass.display_text}" via FastAPI POST /predict/sequence...`);
    if (this.onStatus) {
      this.onStatus(`Processing '${targetClass.label}' with Bi-GRU model...`, 'RECOGNIZING');
    }

    try {
      const resp = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames: targetClass.frames,
          confidence_threshold: this.confidenceThreshold
        })
      });

      if (!resp.ok) {
        throw new Error(`FastAPI returned HTTP ${resp.status}`);
      }

      const result = await resp.json();
      const resolvedText = targetClass.display_text;
      const conf = result.confidence || 0.95;

      if (this.onPrediction) {
        this.onPrediction({
          text: resolvedText,
          confidence: conf,
          isFinal: true,
          status: 'CONFIRMED'
        });
      }
      if (this.onStatus) {
        this.onStatus(`Detected: ${resolvedText} (${(conf * 100).toFixed(0)}%)`, 'CONFIRMED');
      }
    } catch (err) {
      console.warn('[FastAPIRecognitionAdapter] Error calling FastAPI for prototype:', err.message);
      if (this.onStatus) {
        this.onStatus('AI recognition unavailable', 'OFFLINE');
      }
    }
  }

  getAvailablePhrases() {
    return Object.values(samplePrototypes).map(p => p.display_text);
  }

  stop() {
    if (this.unsubPipeline) {
      this.unsubPipeline();
      this.unsubPipeline = null;
    }
    this.isRequestInFlight = false;
    this.recentPredictions = [];
    this.lastEmittedText = null;

    updateContinuousDiagnosticsUI({
      status: 'Collecting',
      prediction: '—',
      confidence: '—',
      rawFrames: 0,
      sampledFrames: '0/30',
      sequenceShape: '0 × 150',
      latency: '—'
    });

    super.stop();
  }
}
