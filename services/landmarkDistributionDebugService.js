// services/landmarkDistributionDebugService.js - Live vs Training Landmark Distribution Debugger
// Phase 4 Debugging: Diagnoses representation alignment between live browser landmarks and training data.
// Features:
// 1. Capture current rolling 30x150 sequence
// 2. Compute Global statistics (min, max, mean, std, median, abs mean)
// 3. Compute Per-group statistics: Left hand [0..63], Right hand [64..127], Pose [128..149]
// 4. Verify normalization line-by-line (wrist, middle MCP, scale, normalized wrist ~0)
// 5. Compare with deterministic known sample (help)
// 6. Export captured 30x150 sequence as JSON ({ "frames": [ ... ] })

import samplePrototypes from './recognition/sampleSequences.json' with { type: 'json' };
import { landmarkPipelineService } from './landmarkPipelineService.js';

class LandmarkDistributionDebugService {
  constructor() {
    this.capturedSequence = null; // Array of 30 Float32Array(150) or nested arrays
    this.capturedStats = null;
    this.knownHelpStats = this.computeSequenceStats(samplePrototypes['help'].frames);
    this.normVerificationFrames = [];
  }

  /**
   * Computes comprehensive statistics for a 30 x 150 sequence.
   */
  computeSequenceStats(frames) {
    if (!frames || frames.length === 0) return null;

    const seqLen = frames.length;
    let minVal = Infinity;
    let maxVal = -Infinity;
    let sum = 0.0;
    let absSum = 0.0;
    let hasNaNOrInf = false;

    const allValues = [];
    let leftPresentFrames = 0;
    let rightPresentFrames = 0;
    let posePresentFrames = 0;

    for (let t = 0; t < seqLen; t++) {
      const f = frames[t];
      if (f[63] > 0.5) leftPresentFrames++;
      if (f[127] > 0.5) rightPresentFrames++;
      if (f[149] > 0.5) posePresentFrames++;

      for (let i = 0; i < 150; i++) {
        const v = f[i];
        if (Number.isNaN(v) || !Number.isFinite(v)) {
          hasNaNOrInf = true;
        }
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        sum += v;
        absSum += Math.abs(v);
        allValues.push(v);
      }
    }

    const totalCount = allValues.length;
    const mean = totalCount > 0 ? (sum / totalCount) : 0.0;
    const absMean = totalCount > 0 ? (absSum / totalCount) : 0.0;

    let varSum = 0.0;
    for (let i = 0; i < totalCount; i++) {
      const d = allValues[i] - mean;
      varSum += d * d;
    }
    const std = totalCount > 0 ? Math.sqrt(varSum / totalCount) : 0.0;

    // Median
    allValues.sort((a, b) => a - b);
    const mid = Math.floor(totalCount / 2);
    const median = totalCount % 2 !== 0 ? allValues[mid] : (allValues[mid - 1] + allValues[mid]) / 2.0;

    return {
      shape: `${seqLen} × 150`,
      frameCount: seqLen,
      min: Number(minVal.toFixed(3)),
      max: Number(maxVal.toFixed(3)),
      mean: Number(mean.toFixed(3)),
      std: Number(std.toFixed(3)),
      median: Number(median.toFixed(3)),
      absMean: Number(absMean.toFixed(3)),
      leftPresence: leftPresentFrames,
      rightPresence: rightPresentFrames,
      posePresence: posePresentFrames,
      hasNaNOrInf,
      nanInfText: hasNaNOrInf ? 'DETECTED' : 'NONE'
    };
  }

  /**
   * Captures the current rolling 30x150 buffer from landmarkPipelineService.
   */
  captureCurrentSequence() {
    const buffer = landmarkPipelineService.frameBuffer;
    if (!buffer || buffer.length === 0) {
      this.updateStatusBanner('No frames in sequence buffer. Please start camera & show hands.');
      return null;
    }

    // Clone current frames to snapshot
    this.capturedSequence = buffer.map((frame) => Array.from(frame));
    this.capturedStats = this.computeSequenceStats(this.capturedSequence);

    // Selected frames for normalization check (frame 0, 14, 29 if 30 frames)
    this.normVerificationFrames = [];
    const checkIndices = [0, Math.floor(this.capturedSequence.length / 2), this.capturedSequence.length - 1];
    
    for (const idx of checkIndices) {
      if (idx < this.capturedSequence.length) {
        const f = this.capturedSequence[idx];
        const isLeft = f[63] > 0.5;
        const isRight = f[127] > 0.5;
        const activeOffset = isRight ? 64 : (isLeft ? 0 : null);
        
        if (activeOffset !== null) {
          // Wrist is index 0 in hand features [0, 1, 2]
          const wristNormX = f[activeOffset + 0];
          const wristNormY = f[activeOffset + 1];
          const wristNormZ = f[activeOffset + 2];
          
          // Middle finger MCP is index 9 [9*3, 9*3+1, 9*3+2] = [27, 28, 29]
          const mcpNormX = f[activeOffset + 27];
          const mcpNormY = f[activeOffset + 28];
          const mcpNormZ = f[activeOffset + 29];

          const dist = Math.hypot(mcpNormX - wristNormX, mcpNormY - wristNormY, mcpNormZ - wristNormZ);

          this.normVerificationFrames.push({
            frameIdx: idx,
            hand: isRight ? 'Right' : 'Left',
            wristNorm: `(${wristNormX.toFixed(3)}, ${wristNormY.toFixed(3)}, ${wristNormZ.toFixed(3)})`,
            mcpNorm: `(${mcpNormX.toFixed(3)}, ${mcpNormY.toFixed(3)}, ${mcpNormZ.toFixed(3)})`,
            unitDistance: dist.toFixed(3), // Should be exactly 1.000 in scale normalized space
            wristIsZero: (Math.abs(wristNormX) < 1e-4 && Math.abs(wristNormY) < 1e-4 && Math.abs(wristNormZ) < 1e-4)
          });
        }
      }
    }

    this.updateStatusBanner(`Captured sequence: ${this.capturedSequence.length} × 150 frames.`);
    this.updateUI();
    return this.capturedStats;
  }

  /**
   * Exports the captured sequence as downloadable JSON.
   */
  exportSequenceJSON() {
    if (!this.capturedSequence || this.capturedSequence.length === 0) {
      this.updateStatusBanner('No captured sequence to export. Click [CAPTURE] first.');
      return;
    }

    const payload = {
      description: "SignBridge AI Live 30x150 Landmark Sequence",
      captured_at: new Date().toISOString(),
      shape: [this.capturedSequence.length, 150],
      frames: this.capturedSequence,
      statistics: this.capturedStats
    };

    const jsonStr = JSON.stringify(payload, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `captured_live_sequence_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.updateStatusBanner('Exported sequence JSON successfully.');
  }

  updateStatusBanner(msg) {
    if (typeof document === 'undefined') return;
    const banner = document.getElementById('phase4-status-msg');
    if (banner) banner.textContent = msg;
  }

  updateUI() {
    if (typeof document === 'undefined') return;
    if (!this.capturedStats) return;

    const s = this.capturedStats;
    const k = this.knownHelpStats;

    // Elements
    const shapeEl = document.getElementById('p4-live-shape');
    const lPresEl = document.getElementById('p4-live-l-pres');
    const rPresEl = document.getElementById('p4-live-r-pres');
    const pPresEl = document.getElementById('p4-live-p-pres');
    const minEl = document.getElementById('p4-live-min');
    const maxEl = document.getElementById('p4-live-max');
    const meanEl = document.getElementById('p4-live-mean');
    const stdEl = document.getElementById('p4-live-std');
    const medEl = document.getElementById('p4-live-median');
    const absMeanEl = document.getElementById('p4-live-absmean');
    const nanInfEl = document.getElementById('p4-live-nan-inf');

    if (shapeEl) shapeEl.textContent = s.shape;
    if (lPresEl) lPresEl.textContent = `${s.leftPresence} / ${s.frameCount} frames`;
    if (rPresEl) rPresEl.textContent = `${s.rightPresence} / ${s.frameCount} frames`;
    if (pPresEl) pPresEl.textContent = `${s.posePresence} / ${s.frameCount} frames`;
    if (minEl) minEl.textContent = s.min;
    if (maxEl) maxEl.textContent = s.max;
    if (meanEl) meanEl.textContent = s.mean;
    if (stdEl) stdEl.textContent = s.std;
    if (medEl) medEl.textContent = s.median;
    if (absMeanEl) absMeanEl.textContent = s.absMean;
    if (nanInfEl) nanInfEl.textContent = s.nanInfText;

    // Comparison Table
    const tableBody = document.getElementById('p4-comparison-table-body');
    if (tableBody && k) {
      tableBody.innerHTML = `
        <tr class="border-b border-slate-800">
          <td class="py-1 text-slate-400 font-semibold">Min</td>
          <td class="py-1 text-indigo-300 font-mono">${k.min}</td>
          <td class="py-1 text-teal-300 font-mono">${s.min}</td>
        </tr>
        <tr class="border-b border-slate-800">
          <td class="py-1 text-slate-400 font-semibold">Max</td>
          <td class="py-1 text-indigo-300 font-mono">${k.max}</td>
          <td class="py-1 text-teal-300 font-mono">${s.max}</td>
        </tr>
        <tr class="border-b border-slate-800">
          <td class="py-1 text-slate-400 font-semibold">Mean</td>
          <td class="py-1 text-indigo-300 font-mono">${k.mean}</td>
          <td class="py-1 text-teal-300 font-mono">${s.mean}</td>
        </tr>
        <tr class="border-b border-slate-800">
          <td class="py-1 text-slate-400 font-semibold">Std</td>
          <td class="py-1 text-indigo-300 font-mono">${k.std}</td>
          <td class="py-1 text-teal-300 font-mono">${s.std}</td>
        </tr>
        <tr class="border-b border-slate-800">
          <td class="py-1 text-slate-400 font-semibold">Left presence</td>
          <td class="py-1 text-indigo-300 font-mono">${k.leftPresence} / 30</td>
          <td class="py-1 text-teal-300 font-mono">${s.leftPresence} / ${s.frameCount}</td>
        </tr>
        <tr class="border-b border-slate-800">
          <td class="py-1 text-slate-400 font-semibold">Right presence</td>
          <td class="py-1 text-indigo-300 font-mono">${k.rightPresence} / 30</td>
          <td class="py-1 text-teal-300 font-mono">${s.rightPresence} / ${s.frameCount}</td>
        </tr>
        <tr>
          <td class="py-1 text-slate-400 font-semibold">Pose presence</td>
          <td class="py-1 text-indigo-300 font-mono">${k.posePresence} / 30</td>
          <td class="py-1 ${s.posePresence === 0 ? 'text-amber-400 font-bold' : 'text-teal-300'} font-mono">${s.posePresence} / ${s.frameCount} ${s.posePresence === 0 ? '(MISSING)' : ''}</td>
        </tr>
      `;
    }

    // Normalization check display
    const normCheckEl = document.getElementById('p4-norm-check-container');
    if (normCheckEl && this.normVerificationFrames.length > 0) {
      normCheckEl.innerHTML = this.normVerificationFrames.map((vf) => `
        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60 last:border-0 text-[9.5px]">
          <span class="text-slate-400">Frame #${vf.frameIdx} (${vf.hand}):</span>
          <span class="text-slate-300 font-mono">Wrist: ${vf.wristNorm}</span>
          <span class="font-mono ${vf.wristIsZero ? 'text-emerald-400 font-bold' : 'text-red-400'}">${vf.wristIsZero ? 'PASS (0,0,0)' : 'FAIL'}</span>
        </div>
      `).join('');
    }
  }
}

export const landmarkDistributionDebugService = new LandmarkDistributionDebugService();
