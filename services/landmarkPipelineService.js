// services/landmarkPipelineService.js - Landmark Feature Pipeline & Temporal Sampling Engine
// Phase 5 Verified Fixes:
// 1. Handedness Correction: Maps unmirrored raw camera feed so physical Left -> [0..63], physical Right -> [64..127].
// 2. Pose Feature Integration: Extracts 7 upper body keypoints normalized to mid-shoulder -> [128..149].
// 3. Temporal Sampling Engine: Buffers a 2.6s window of continuous gestures and uniformly downsamples to exactly 30x150.
// Strict rule: Final model input is strictly 30x150, no partial sequences sent to model.

import { modelInferenceDebugService } from './modelInferenceDebugService.js';

/**
 * Normalizes 21 3D hand landmarks into 64 features:
 * - Translation: Centered at wrist (landmark 0) -> (0.0, 0.0, 0.0).
 * - Scale: Divided by Euclidean distance from wrist (0) to middle finger MCP (9).
 * - Layout: [0..62] = 63 normalized coords (x, y, z), [63] = 1.0 (presence flag).
 * - Missing hand: 64 zeros (presence flag = 0.0).
 */
export function normalizeHandLandmarks(landmarks) {
  const feat = new Float32Array(64);
  if (!landmarks || landmarks.length < 21) {
    return { feat, present: false, rawCount: 0 };
  }

  const wrist = landmarks[0];
  const p9 = landmarks[9];

  const dx = p9.x - wrist.x;
  const dy = p9.y - wrist.y;
  const dz = (p9.z || 0) - (wrist.z || 0);
  let scale = Math.hypot(dx, dy, dz);
  if (scale < 1e-4) scale = 1.0;

  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    feat[i * 3] = (lm.x - wrist.x) / scale;
    feat[i * 3 + 1] = (lm.y - wrist.y) / scale;
    feat[i * 3 + 2] = ((lm.z || 0) - (wrist.z || 0)) / scale;
  }
  feat[63] = 1.0; // Presence flag
  return { feat, present: true, rawCount: 21 };
}

/**
 * Upper body pose features (FIX 2):
 * - 7 keypoints x 3 coords = 21 coords + 1 presence flag = 22 features.
 * - Keypoints: 0 (nose), 11 (L shoulder), 12 (R shoulder), 13 (L elbow), 14 (R elbow), 15 (L wrist), 16 (R wrist).
 * - Translation: Centered relative to mid-shoulder: (p11 + p12) / 2.
 * - Scale: Normalized by shoulder distance ||p12 - p11||.
 * - Missing pose: 22 zeros (presence flag = 0.0).
 */
export function normalizePoseLandmarks(poseLandmarks = null) {
  const feat = new Float32Array(22);
  if (!poseLandmarks || poseLandmarks.length < 17) {
    return { feat, present: false, rawCount: 0 };
  }

  const POSE_KEYPOINTS = [0, 11, 12, 13, 14, 15, 16];

  const lShoulder = poseLandmarks[11];
  const rShoulder = poseLandmarks[12];

  // Mid-shoulder anchor
  const midX = (lShoulder.x + rShoulder.x) / 2.0;
  const midY = (lShoulder.y + rShoulder.y) / 2.0;
  const midZ = ((lShoulder.z || 0) + (rShoulder.z || 0)) / 2.0;

  // Scale: shoulder width
  const dx = rShoulder.x - lShoulder.x;
  const dy = rShoulder.y - lShoulder.y;
  const dz = (rShoulder.z || 0) - (lShoulder.z || 0);
  let scale = Math.hypot(dx, dy, dz);
  if (scale < 1e-4) scale = 1.0;

  for (let i = 0; i < POSE_KEYPOINTS.length; i++) {
    const pt = poseLandmarks[POSE_KEYPOINTS[i]];
    feat[i * 3] = (pt.x - midX) / scale;
    feat[i * 3 + 1] = (pt.y - midY) / scale;
    feat[i * 3 + 2] = ((pt.z || 0) - midZ) / scale;
  }
  feat[21] = 1.0; // Presence flag

  return { feat, present: true, rawCount: 7 };
}

/**
 * Computes 18 body-relative spatial features normalized by shoulder width:
 * A) Left wrist relative to shoulder-center: 3 values (indices 150..152)
 * B) Right wrist relative to shoulder-center: 3 values (indices 153..155)
 * C) Left wrist relative to nose: 3 values (indices 156..158)
 * D) Right wrist relative to nose: 3 values (indices 159..161)
 * E) Left wrist relative to chest-center: 3 values (indices 162..164)
 * F) Right wrist relative to chest-center: 3 values (indices 165..167)
 * Total: 18 features.
 */
export function computeBodyRelativeFeatures(leftHandLms, rightHandLms, poseLms = null) {
  const relFeat = new Float32Array(18);
  if (!poseLms || poseLms.length < 17) {
    return relFeat;
  }

  const nose = poseLms[0];
  const lShoulder = poseLms[11];
  const rShoulder = poseLms[12];

  const midX = (lShoulder.x + rShoulder.x) / 2.0;
  const midY = (lShoulder.y + rShoulder.y) / 2.0;
  const midZ = ((lShoulder.z || 0) + (rShoulder.z || 0)) / 2.0;

  const dxS = rShoulder.x - lShoulder.x;
  const dyS = rShoulder.y - lShoulder.y;
  const dzS = (rShoulder.z || 0) - (lShoulder.z || 0);
  let shoulderWidth = Math.hypot(dxS, dyS, dzS);
  if (shoulderWidth < 1e-4) shoulderWidth = 1.0;

  let downX = midX - nose.x;
  let downY = midY - nose.y;
  let downZ = midZ - (nose.z || 0);
  let downDist = Math.hypot(downX, downY, downZ);
  let uDownX = 0.0, uDownY = 1.0, uDownZ = 0.0;
  if (downDist > 1e-4) {
    uDownX = downX / downDist;
    uDownY = downY / downDist;
    uDownZ = downZ / downDist;
  }

  const chestX = midX + uDownX * (0.5 * shoulderWidth);
  const chestY = midY + uDownY * (0.5 * shoulderWidth);
  const chestZ = midZ + uDownZ * (0.5 * shoulderWidth);

  // A, C, E: Left Wrist (if present)
  if (leftHandLms && leftHandLms.length >= 21) {
    const lw = leftHandLms[0];
    const lwZ = lw.z || 0;
    // A) Left wrist relative to shoulder-center [0..2] -> indices 150..152
    relFeat[0] = (lw.x - midX) / shoulderWidth;
    relFeat[1] = (lw.y - midY) / shoulderWidth;
    relFeat[2] = (lwZ - midZ) / shoulderWidth;
    // C) Left wrist relative to nose [6..8] -> indices 156..158
    relFeat[6] = (lw.x - nose.x) / shoulderWidth;
    relFeat[7] = (lw.y - nose.y) / shoulderWidth;
    relFeat[8] = (lwZ - (nose.z || 0)) / shoulderWidth;
    // E) Left wrist relative to chest-center [12..14] -> indices 162..164
    relFeat[12] = (lw.x - chestX) / shoulderWidth;
    relFeat[13] = (lw.y - chestY) / shoulderWidth;
    relFeat[14] = (lwZ - chestZ) / shoulderWidth;
  }

  // B, D, F: Right Wrist (if present)
  if (rightHandLms && rightHandLms.length >= 21) {
    const rw = rightHandLms[0];
    const rwZ = rw.z || 0;
    // B) Right wrist relative to shoulder-center [3..5] -> indices 153..155
    relFeat[3] = (rw.x - midX) / shoulderWidth;
    relFeat[4] = (rw.y - midY) / shoulderWidth;
    relFeat[5] = (rwZ - midZ) / shoulderWidth;
    // D) Right wrist relative to nose [9..11] -> indices 159..161
    relFeat[9] = (rw.x - nose.x) / shoulderWidth;
    relFeat[10] = (rw.y - nose.y) / shoulderWidth;
    relFeat[11] = (rwZ - (nose.z || 0)) / shoulderWidth;
    // F) Right wrist relative to chest-center [15..17] -> indices 165..167
    relFeat[15] = (rw.x - chestX) / shoulderWidth;
    relFeat[16] = (rw.y - chestY) / shoulderWidth;
    relFeat[17] = (rwZ - chestZ) / shoulderWidth;
  }

  return relFeat;
}

/**
 * Builds the exact 168-dimensional feature vector:
 * - [0..63]   : Physical Left Hand (63 coords + presence)
 * - [64..127] : Physical Right Hand (63 coords + presence)
 * - [128..149]: Upper Body Pose (21 coords + presence)
 * - [150..167]: Body-Relative Spatial Features (18 coords)
 * Total: strictly 168 features.
 */
export function construct168FeatureVector(leftHandLms, rightHandLms, poseLms = null) {
  const lRes = normalizeHandLandmarks(leftHandLms);
  const rRes = normalizeHandLandmarks(rightHandLms);
  const pRes = normalizePoseLandmarks(poseLms);
  const relFeat = computeBodyRelativeFeatures(leftHandLms, rightHandLms, poseLms);

  const vector = new Float32Array(168);
  vector.set(lRes.feat, 0);     // [0..63]
  vector.set(rRes.feat, 64);    // [64..127]
  vector.set(pRes.feat, 128);   // [128..149]
  vector.set(relFeat, 150);     // [150..167]

  return {
    vector,
    dimension: vector.length,
    leftPresent: lRes.present,
    rightPresent: rRes.present,
    posePresent: pRes.present,
    rawLeftCount: lRes.rawCount,
    rawRightCount: rRes.rawCount,
    rawPoseCount: pRes.rawCount
  };
}

export function construct150FeatureVector(leftHandLms, rightHandLms, poseLms = null) {
  return construct168FeatureVector(leftHandLms, rightHandLms, poseLms);
}

class LandmarkPipelineService {
  constructor() {
    this.bufferSize = 30; // 30 frames expected by trained Bi-GRU
    this.captureWindowSec = 2.6; // Configurable capture window (2.5–2.8s)
    this.rawTemporalBuffer = []; // Array of { time: number, vector: Float32Array(168) }
    this.frameBuffer = []; // Strictly sampled 30x168 sequence
    this.lastHandSeenTime = 0;
    this.isInspectorOpen = false;
    this.listeners = new Set();

    // Latest frame diagnostics
    this.latestDiagnostics = {
      rawLandmarkCount: 0,
      rawCoordCount: 0,
      normalizedCount: 0,
      featureDimension: 168,
      sequenceLength: 0,
      sequenceShape: '0 × 168',
      leftPresent: false,
      rightPresent: false,
      posePresent: false,
      handMask: '[L: 0, R: 0]',
      hasNaNOrInf: false,
      minValue: 0.0,
      maxValue: 0.0,
      sampleValues: [],
      statusText: 'Collecting window...',
      isReady: false,
      captureWindowSec: this.captureWindowSec,
      rawFramesCount: 0,
      currentSpanSec: 0.0,
      error: null
    };
  }

  reset() {
    this.rawTemporalBuffer = [];
    this.frameBuffer = [];
    this.lastHandSeenTime = 0;
    this.latestDiagnostics = {
      rawLandmarkCount: 0,
      rawCoordCount: 0,
      normalizedCount: 0,
      featureDimension: 168,
      sequenceLength: 0,
      sequenceShape: '0 × 168',
      leftPresent: false,
      rightPresent: false,
      posePresent: false,
      handMask: '[L: 0, R: 0]',
      hasNaNOrInf: false,
      minValue: 0.0,
      maxValue: 0.0,
      sampleValues: [],
      statusText: 'Collecting window...',
      isReady: false,
      captureWindowSec: this.captureWindowSec,
      rawFramesCount: 0,
      currentSpanSec: 0.0,
      error: null
    };
    this.updateUI();
    modelInferenceDebugService.updateLiveSequence([]);
    this.notifyListeners();
  }

  processFrame(multiHandLandmarks = [], multiHandedness = [], poseLandmarks = null, customNow = null) {
    const now = (customNow !== null && customNow !== undefined) ? customNow : performance.now();

    let leftHandLms = null;
    let rightHandLms = null;

    const detectedHandsCount = multiHandLandmarks.length;

    // Robust Physical Handedness Mapping:
    // 1. If PoseLandmarker is active, verify geometric proximity to Pose Left wrist (15) vs Pose Right wrist (16).
    // 2. Otherwise fall back to unmirrored feed correction: raw 'Left' -> physical Right, raw 'Right' -> physical Left.
    if (detectedHandsCount > 0) {
      const hasPoseWrists = poseLandmarks && poseLandmarks.length >= 17 &&
                            poseLandmarks[15] && poseLandmarks[16];

      for (let i = 0; i < detectedHandsCount; i++) {
        const lms = multiHandLandmarks[i];
        let physicalLabel = null;

        if (hasPoseWrists && lms && lms[0]) {
          const hw = lms[0];
          const distToPoseL = Math.hypot(hw.x - poseLandmarks[15].x, hw.y - poseLandmarks[15].y);
          const distToPoseR = Math.hypot(hw.x - poseLandmarks[16].x, hw.y - poseLandmarks[16].y);
          if (Math.abs(distToPoseL - distToPoseR) > 0.05) {
            physicalLabel = (distToPoseL < distToPoseR) ? 'Left' : 'Right';
          }
        }

        if (!physicalLabel) {
          const handednessObj = multiHandedness[i];
          const rawLabel = handednessObj ? handednessObj.label : (i === 0 ? 'Left' : 'Right');
          physicalLabel = (rawLabel === 'Left') ? 'Right' : 'Left';
        }

        if (physicalLabel === 'Left' && !leftHandLms) {
          leftHandLms = lms;
        } else if (physicalLabel === 'Right' && !rightHandLms) {
          rightHandLms = lms;
        } else if (!rightHandLms) {
          rightHandLms = lms;
        } else if (!leftHandLms) {
          leftHandLms = lms;
        }
      }
    }

    // Construct exact 168-dimensional feature vector with Hands, Pose, and Body-Relative landmarks
    const featureInfo = construct168FeatureVector(leftHandLms, rightHandLms, poseLandmarks);
    const vector = featureInfo.vector;
    const featDim = vector.length;

    // 1. Validation: Expected 168 features
    let dimensionError = null;
    if (featDim !== 168) {
      dimensionError = `ERROR: Feature dimension ${featDim} != 168`;
    }

    // 2. Validation: Check NaN / Infinity
    let hasNaNOrInf = false;
    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let i = 0; i < featDim; i++) {
      const v = vector[i];
      if (Number.isNaN(v) || !Number.isFinite(v)) {
        hasNaNOrInf = true;
      }
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    if (minVal === Infinity) minVal = 0.0;
    if (maxVal === -Infinity) maxVal = 0.0;

    const hasAnyHand = featureInfo.leftPresent || featureInfo.rightPresent;

    // Temporal Sampling Buffer Management
    if (hasAnyHand && !dimensionError && !hasNaNOrInf) {
      this.lastHandSeenTime = now;
      this.rawTemporalBuffer.push({ time: now, vector });

      // Evict raw frames older than capture window
      const cutoff = now - (this.captureWindowSec * 1000);
      while (this.rawTemporalBuffer.length > 0 && this.rawTemporalBuffer[0].time < cutoff) {
        this.rawTemporalBuffer.shift();
      }
    } else if (!hasAnyHand) {
      // If no hands detected for > 1500ms, reset temporal buffer
      if (this.lastHandSeenTime > 0 && now - this.lastHandSeenTime > 1500) {
        this.rawTemporalBuffer = [];
        this.frameBuffer = [];
      }
    }

    const rawCount = this.rawTemporalBuffer.length;
    const spanSec = rawCount > 1 ? Math.max(0, (this.rawTemporalBuffer[rawCount - 1].time - this.rawTemporalBuffer[0].time) / 1000.0) : 0.0;
    // Window is ready when span >= 85% of target window and at least 30 raw frames exist
    const isWindowReady = (spanSec >= (this.captureWindowSec * 0.85)) && (rawCount >= 30);

    if (isWindowReady) {
      // Uniformly downsample raw frames across the full 2.6s window into exactly 30 frames
      const sampled = [];
      for (let i = 0; i < this.bufferSize; i++) {
        const sampleIdx = Math.min(rawCount - 1, Math.round((i * (rawCount - 1)) / (this.bufferSize - 1)));
        const frameVec = this.rawTemporalBuffer[sampleIdx].vector;

        // HARD VALIDATION: Reject any frame that is not 168 features
        if (!frameVec || frameVec.length !== 168) {
          throw new Error(`[LandmarkPipelineService] Frame validation failed: expected 168 features, got ${frameVec ? frameVec.length : 0}`);
        }
        sampled.push(frameVec);
      }

      // HARD VALIDATION: Reject any sequence that is not strictly 30 frames
      if (sampled.length !== 30) {
        throw new Error(`[LandmarkPipelineService] Sequence validation failed: expected 30 frames, got ${sampled.length}`);
      }

      this.frameBuffer = sampled;
    } else {
      // IMPORTANT: Do NOT send partial sequences to the model!
      this.frameBuffer = [];
    }

    const currentFrames = this.frameBuffer.length;
    const isReady = (currentFrames === this.bufferSize);

    // Status determination
    let statusText = 'Collecting window...';
    if (dimensionError) {
      statusText = dimensionError;
    } else if (hasNaNOrInf) {
      statusText = 'ERROR: NaN / Infinity detected in landmarks';
    } else if (isReady) {
      statusText = `Window READY (${spanSec.toFixed(2)}s sampled) — 30 × 168`;
    } else if (rawCount > 0) {
      statusText = `Collecting window: ${spanSec.toFixed(2)}s / ${this.captureWindowSec.toFixed(2)}s (${rawCount} raw frames)`;
    } else {
      statusText = 'Waiting for hand...';
    }

    const totalRawLandmarks = (featureInfo.rawLeftCount + featureInfo.rawRightCount + featureInfo.rawPoseCount);
    const totalRawCoords = totalRawLandmarks * 3;

    // Store diagnostics
    this.latestDiagnostics = {
      rawLandmarkCount: totalRawLandmarks,
      rawCoordCount: totalRawCoords,
      rawLeftCount: featureInfo.rawLeftCount,
      rawRightCount: featureInfo.rawRightCount,
      rawPoseCount: featureInfo.rawPoseCount,
      normalizedCount: (featureInfo.leftPresent ? 63 : 0) + (featureInfo.rightPresent ? 63 : 0) + (featureInfo.posePresent ? 21 : 0) + 18,
      featureDimension: featDim,
      sequenceLength: currentFrames,
      sequenceShape: isReady ? '30 × 168' : '0 × 168',
      leftPresent: featureInfo.leftPresent,
      rightPresent: featureInfo.rightPresent,
      posePresent: featureInfo.posePresent,
      handMask: `[L: ${featureInfo.leftPresent ? 'ACTIVE' : '0'}, R: ${featureInfo.rightPresent ? 'ACTIVE' : '0'}]`,
      hasNaNOrInf,
      minValue: Number(minVal.toFixed(3)),
      maxValue: Number(maxVal.toFixed(3)),
      sampleValues: Array.from(vector.slice(0, 6)).map(v => Number(v.toFixed(3))),
      statusText,
      isReady,
      captureWindowSec: this.captureWindowSec,
      rawFramesCount: rawCount,
      currentSpanSec: Number(spanSec.toFixed(2)),
      error: dimensionError
    };

    // Update UI
    this.updateUI();

    // Update model inference debug sequence and input stats with strictly sampled 30x150
    modelInferenceDebugService.updateLiveSequence(this.frameBuffer);

    // Notify continuous recognition listeners with latest sequence and diagnostics
    this.notifyListeners();

    return this.latestDiagnostics;
  }

  updateUI() {
    if (typeof document === 'undefined') return;
    const d = this.latestDiagnostics;

    const framesEl = document.getElementById('pipeline-frames-count');
    const dimEl = document.getElementById('pipeline-feat-dim');
    const seqShapeEl = document.getElementById('pipeline-seq-shape');
    const shapeBadgeEl = document.getElementById('pipeline-shape-badge');
    const statusBannerEl = document.getElementById('pipeline-status-banner');
    const captureWinEl = document.getElementById('pipeline-capture-window');
    const rawCountEl = document.getElementById('pipeline-raw-count');

    if (captureWinEl) {
      captureWinEl.textContent = `${d.captureWindowSec.toFixed(2)} sec`;
    }

    if (rawCountEl) {
      rawCountEl.textContent = `${d.rawFramesCount}`;
      rawCountEl.className = d.rawFramesCount >= 30 ? 'font-bold text-emerald-400' : 'font-bold text-slate-200';
    }

    if (framesEl) {
      framesEl.textContent = d.isReady ? `30/30 (sampled)` : `0/30 (${d.currentSpanSec}s)`;
      framesEl.className = d.isReady ? 'font-bold text-emerald-400' : 'font-bold text-amber-300';
    }

    if (dimEl) {
      dimEl.textContent = d.featureDimension;
      dimEl.className = d.featureDimension === 150 ? 'font-bold text-emerald-400' : 'font-bold text-red-500 animate-pulse';
    }

    if (seqShapeEl) {
      seqShapeEl.textContent = d.sequenceShape;
      seqShapeEl.className = d.isReady ? 'font-bold text-emerald-400' : 'font-bold text-slate-200';
    }

    if (shapeBadgeEl) {
      shapeBadgeEl.textContent = d.sequenceShape;
      if (d.isReady) {
        shapeBadgeEl.className = 'font-mono text-[9.5px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 font-bold border border-emerald-500/50';
      } else {
        shapeBadgeEl.className = 'font-mono text-[9.5px] px-1.5 py-0.5 rounded bg-slate-800 text-teal-300 font-bold border border-teal-500/30';
      }
    }

    if (statusBannerEl) {
      if (d.error) {
        statusBannerEl.textContent = d.error;
        statusBannerEl.className = 'px-2 py-1 rounded text-[10px] font-bold bg-red-600/30 text-red-300 border border-red-500/60 animate-pulse';
      } else if (d.isReady) {
        statusBannerEl.innerHTML = `
          <div class="flex items-center justify-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Window READY (${d.currentSpanSec}s sampled) — 30 × 150</span>
          </div>
          <div class="text-[9px] text-emerald-400 font-mono mt-0.5 font-bold uppercase tracking-wider">
            Status: READY FOR MODEL
          </div>
        `;
        statusBannerEl.className = 'px-2 py-1 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/50';
      } else {
        statusBannerEl.textContent = d.statusText;
        statusBannerEl.className = 'px-2 py-1 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40';
      }
    }

    // Expandable Inspector Elements
    const rawLmEl = document.getElementById('inspect-raw-landmarks');
    const normLmEl = document.getElementById('inspect-norm-landmarks');
    const featLenEl = document.getElementById('inspect-feat-length');
    const frameIdxEl = document.getElementById('inspect-frame-idx');
    const leftHandEl = document.getElementById('inspect-left-hand');
    const rightHandEl = document.getElementById('inspect-right-hand');
    const handMaskEl = document.getElementById('inspect-hand-mask');
    const nanInfEl = document.getElementById('inspect-nan-inf');
    const minMaxEl = document.getElementById('inspect-min-max');
    const sampleEl = document.getElementById('inspect-sample-values');

    if (rawLmEl) {
      rawLmEl.textContent = `${d.rawLandmarkCount} pts (${d.rawCoordCount} coords)`;
    }
    if (normLmEl) {
      normLmEl.textContent = `${d.normalizedCount} coords + flags`;
    }
    if (featLenEl) {
      featLenEl.textContent = `${d.featureDimension} values`;
      featLenEl.className = d.featureDimension === 150 ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold';
    }
    if (frameIdxEl) {
      frameIdxEl.textContent = d.isReady ? `Sampled 30 from ${d.rawFramesCount} frames (${d.currentSpanSec}s)` : `Collecting: ${d.currentSpanSec}s / ${d.captureWindowSec}s (${d.rawFramesCount} raw)`;
    }
    if (leftHandEl) {
      leftHandEl.textContent = d.leftPresent ? 'Phys L: YES ([0..63])' : 'NO (masked: 0)';
      leftHandEl.className = d.leftPresent ? 'text-emerald-400 font-bold' : 'text-slate-400';
    }
    if (rightHandEl) {
      rightHandEl.textContent = d.rightPresent ? 'Phys R: YES ([64..127])' : 'NO (masked: 0)';
      rightHandEl.className = d.rightPresent ? 'text-emerald-400 font-bold' : 'text-slate-400';
    }
    if (handMaskEl) {
      handMaskEl.textContent = `${d.handMask} | Pose: ${d.posePresent ? 'ACTIVE ([128..149])' : '0'}`;
    }
    if (nanInfEl) {
      nanInfEl.textContent = d.hasNaNOrInf ? 'DETECTED (ERROR)' : 'NONE (Clean)';
      nanInfEl.className = d.hasNaNOrInf ? 'text-red-400 font-bold animate-pulse' : 'text-emerald-400 font-bold';
    }
    if (minMaxEl) {
      minMaxEl.textContent = `min: ${d.minValue} | max: ${d.maxValue}`;
    }
    if (sampleEl && d.sampleValues.length > 0) {
      sampleEl.textContent = `[${d.sampleValues.join(', ')}]`;
    }

    // Phase 5 Debug indicators
    const routingEl = document.getElementById('debug-routing-indicator');
    if (routingEl) {
      const leftStatus = d.leftPresent ? 'ACTIVE' : 'IDLE';
      const rightStatus = d.rightPresent ? 'ACTIVE' : 'IDLE';
      routingEl.innerHTML = `<span>Physical Left → [0..63] <b class="${d.leftPresent ? 'text-emerald-400' : 'text-slate-400'}">(${leftStatus})</b></span> &bull; <span>Physical Right → [64..127] <b class="${d.rightPresent ? 'text-emerald-400' : 'text-slate-400'}">(${rightStatus})</b></span>`;
    }

    const temporalEl = document.getElementById('debug-temporal-indicator');
    if (temporalEl) {
      temporalEl.textContent = `Capture Window: ${d.captureWindowSec}s | Raw: ${d.rawFramesCount} | Sampled: ${d.sequenceLength}/30`;
    }
  }

  toggleInspector() {
    if (typeof document === 'undefined') return false;
    this.isInspectorOpen = !this.isInspectorOpen;
    const panel = document.getElementById('pipeline-inspector-panel');
    const arrow = document.getElementById('pipeline-inspector-arrow');
    const btnText = document.getElementById('btn-toggle-pipeline-inspector-text');

    if (panel) {
      if (this.isInspectorOpen) {
        panel.classList.remove('hidden');
        if (arrow) arrow.style.transform = 'rotate(180deg)';
        if (btnText) btnText.textContent = 'Hide Feature Inspector';
      } else {
        panel.classList.add('hidden');
        if (arrow) arrow.style.transform = 'rotate(0deg)';
        if (btnText) btnText.textContent = 'Inspect Feature Pipeline';
      }
    }
    return this.isInspectorOpen;
  }

  subscribe(callback) {
    if (typeof callback === 'function') {
      this.listeners.add(callback);
    }
    return () => {
      this.listeners.delete(callback);
    };
  }

  notifyListeners() {
    if (!this.listeners || this.listeners.size === 0) return;
    const payload = {
      sequence: this.frameBuffer,
      diagnostics: this.latestDiagnostics
    };
    for (const listener of this.listeners) {
      try {
        listener(payload);
      } catch (err) {
        console.error('[LandmarkPipelineService] Listener error:', err);
      }
    }
  }
}

export const landmarkPipelineService = new LandmarkPipelineService();
