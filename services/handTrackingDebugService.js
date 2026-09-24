// services/handTrackingDebugService.js - Real-Time MediaPipe Hand Tracking Debugger
// Phase 1 Debugging: Verifies MediaPipe hand landmark extraction & tracking over live webcam stream.
// Features: 21 landmarks, skeleton connections, thin green bounding box, real-time FPS, handedness & confidence.
// Coordinate Mirroring: Maps un-mirrored camera coordinates (0..1) to mirrored display coordinates (1 - x).

import { landmarkPipelineService } from './landmarkPipelineService.js';

export const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  // Palm Base
  [0, 17], [0, 5]
];

class HandTrackingDebugService {
  constructor() {
    this.isEnabled = true; // Default ON per Requirement 11
    this.videoElement = null;
    this.canvasElement = null;
    this.ctx = null;

    this.handsDetector = null;
    this.poseDetector = null;
    this.latestPoseLandmarks = null;
    this.isDetecting = false;
    this.isPoseDetecting = false;
    this.animFrameId = null;

    // FPS Counter
    this.frameCount = 0;
    this.lastFpsUpdate = performance.now();
    this.currentFps = 0;
    this.lastFrameTime = performance.now();

    // Latest Results
    this.latestResults = null;
    this.listeners = new Set();
  }

  init() {
    if (typeof window === 'undefined') return;

    if (window.Hands && !this.handsDetector) {
      try {
        console.log('[HandTrackingDebugService] Initializing MediaPipe Hands detector...');
        this.handsDetector = new window.Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        this.handsDetector.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        this.handsDetector.onResults((results) => {
          this.handleResults(results);
        });

        console.log('[HandTrackingDebugService] MediaPipe Hands detector ready.');
      } catch (err) {
        console.warn('[HandTrackingDebugService] Hands initialization error:', err);
      }
    }

    // FIX 2: Initialize MediaPipe Pose detector on same stream
    if (window.Pose && !this.poseDetector) {
      try {
        console.log('[HandTrackingDebugService] Initializing MediaPipe Pose detector...');
        this.poseDetector = new window.Pose({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });

        this.poseDetector.setOptions({
          modelComplexity: 0, // Lite model for real-time efficiency
          smoothLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        this.poseDetector.onResults((poseResults) => {
          this.latestPoseLandmarks = poseResults.poseLandmarks || null;
        });

        console.log('[HandTrackingDebugService] MediaPipe Pose detector ready.');
      } catch (err) {
        console.warn('[HandTrackingDebugService] Pose initialization error:', err);
      }
    }

    if (this.handsDetector && this.isEnabled && this.videoElement && !this.animFrameId) {
      this.startLoop();
    } else if (!this.handsDetector) {
      console.warn('[HandTrackingDebugService] window.Hands not yet available; retrying in 800ms...');
      setTimeout(() => this.init(), 800);
    }
  }

  attach(videoElement, canvasElement) {
    this.videoElement = videoElement;
    this.canvasElement = canvasElement;
    if (canvasElement) {
      this.ctx = canvasElement.getContext('2d');
    }

    if (!this.handsDetector) {
      this.init();
    }

    if (this.isEnabled && this.videoElement) {
      this.startLoop();
    }
  }

  detach() {
    this.stopLoop();
    this.clearCanvas();
    this.videoElement = null;
    this.canvasElement = null;
    this.ctx = null;
    this.latestPoseLandmarks = null;
    this.updateDebugUI({
      detected: false,
      count: 0,
      handedness: '—',
      confidence: '—',
      fps: 0
    });
    landmarkPipelineService.reset();
  }

  startLoop() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    const processFrame = async () => {
      if (!this.isEnabled || !this.videoElement) return;

      const video = this.videoElement;
      if (video.readyState >= 2 && !video.paused && !this.isDetecting) {
        try {
          this.isDetecting = true;
          // Send video to hands detector
          if (this.handsDetector) {
            await this.handsDetector.send({ image: video });
          }
          // Concurrent pose detection (Lite model, efficient)
          if (this.poseDetector && !this.isPoseDetecting) {
            this.isPoseDetecting = true;
            this.poseDetector.send({ image: video })
              .catch(() => {})
              .finally(() => { this.isPoseDetecting = false; });
          }
        } catch (e) {
          // Frame skip error
        } finally {
          this.isDetecting = false;
        }
      }

      // Calculate real-time FPS
      this.frameCount++;
      const now = performance.now();
      if (now - this.lastFpsUpdate >= 500) {
        this.currentFps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
        this.frameCount = 0;
        this.lastFpsUpdate = now;
      }

      this.animFrameId = requestAnimationFrame(processFrame);
    };

    this.animFrameId = requestAnimationFrame(processFrame);
  }

  stopLoop() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    this.isDetecting = false;
    this.clearCanvas();
  }

  clearCanvas() {
    if (this.ctx && this.canvasElement) {
      this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
  }

  handleResults(results) {
    if (!this.isEnabled || !this.canvasElement || !this.ctx || !this.videoElement) return;

    const canvas = this.canvasElement;
    const ctx = this.ctx;
    const video = this.videoElement;

    // Match canvas pixel buffer to actual displayed video container resolution
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const vidW = video.videoWidth || W;
    const vidH = video.videoHeight || H;

    // Calculate object-cover scale & letterbox/crop offsets so landmarks align 1:1 with video
    const scale = Math.max(W / vidW, H / vidH);
    const renderW = vidW * scale;
    const renderH = vidH * scale;
    const offsetX = (W - renderW) / 2;
    const offsetY = (H - renderH) / 2;

    const hands = results.multiHandLandmarks || [];
    const handednessList = results.multiHandedness || [];

    const hasHands = hands.length > 0;

    let handednessLabels = [];
    let avgConf = 0;

    if (hasHands) {
      for (let hIdx = 0; hIdx < hands.length; hIdx++) {
        const landmarks = hands[hIdx];
        const handedness = handednessList[hIdx];

        // FIX 1: MediaPipe JS @mediapipe/hands assumes a mirrored camera image.
        // In the unmirrored raw camera feed, MediaPipe "Left" is the physical Right hand, and "Right" is the physical Left hand.
        const rawLabel = handedness ? handedness.label : (hIdx === 0 ? 'Left' : 'Right');
        const physicalLabel = (rawLabel === 'Left') ? 'Right' : 'Left';
        const score = handedness ? Math.round(handedness.score * 100) : 95;
        handednessLabels.push(physicalLabel);
        avgConf += score;

        // 1. Convert normalized landmarks to Mirrored Screen Coordinates:
        // Video has transform: scaleX(-1), so x on screen = W - (offsetX + lm.x * renderW)
        const screenPoints = landmarks.map((lm) => {
          const unmirroredX = offsetX + lm.x * renderW;
          const mirroredX = W - unmirroredX;
          const screenY = offsetY + lm.y * renderH;
          return {
            x: mirroredX,
            y: screenY,
            z: lm.z
          };
        });

        // 2. Draw Skeleton Connections
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2.0;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        for (const [idxA, idxB] of HAND_CONNECTIONS) {
          const ptA = screenPoints[idxA];
          const ptB = screenPoints[idxB];
          if (ptA && ptB) {
            ctx.beginPath();
            ctx.moveTo(ptA.x, ptA.y);
            ctx.lineTo(ptB.x, ptB.y);
            ctx.stroke();
          }
        }

        // 3. Draw All 21 Hand Landmarks as visible circular points
        for (let i = 0; i < screenPoints.length; i++) {
          const pt = screenPoints[i];
          const isFingertip = [4, 8, 12, 16, 20].includes(i);
          const isWrist = (i === 0);

          ctx.beginPath();
          ctx.arc(pt.x, pt.y, isFingertip ? 5.5 : (isWrist ? 6.5 : 4.0), 0, Math.PI * 2);
          
          if (isFingertip) {
            ctx.fillStyle = '#f59e0b'; // Amber tip
          } else if (isWrist) {
            ctx.fillStyle = '#3b82f6'; // Blue wrist
          } else {
            ctx.fillStyle = '#00ffcc'; // Cyan joints
          }
          ctx.fill();

          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }

        // 4. Calculate Dynamic Bounding Box from actual landmarks
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (const pt of screenPoints) {
          if (pt.x < minX) minX = pt.x;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.y > maxY) maxY = pt.y;
        }

        // Add 12% padding around the hand bounds
        const padX = Math.max((maxX - minX) * 0.12, 16);
        const padY = Math.max((maxY - minY) * 0.12, 16);

        const boxX = Math.max(2, minX - padX);
        const boxY = Math.max(2, minY - padY);
        const boxW = Math.min(W - boxX - 2, (maxX - minX) + padX * 2);
        const boxH = Math.min(H - boxY - 2, (maxY - minY) + padY * 2);

        // 5. Draw THIN GREEN bounding box (Follows hand in real time)
        ctx.strokeStyle = '#00ff66';
        ctx.lineWidth = 2.0;
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        // Subtle corner accents
        const cornerLen = 14;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3.0;

        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(boxX, boxY + cornerLen);
        ctx.lineTo(boxX, boxY);
        ctx.lineTo(boxX + cornerLen, boxY);
        ctx.stroke();

        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(boxX + boxW - cornerLen, boxY);
        ctx.lineTo(boxX + boxW, boxY);
        ctx.lineTo(boxX + boxW, boxY + cornerLen);
        ctx.stroke();

        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(boxX, boxY + boxH - cornerLen);
        ctx.lineTo(boxX, boxY + boxH);
        ctx.lineTo(boxX + cornerLen, boxY + boxH);
        ctx.stroke();

        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(boxX + boxW - cornerLen, boxY + boxH);
        ctx.lineTo(boxX + boxW, boxY + boxH);
        ctx.lineTo(boxX + boxW, boxY + boxH - cornerLen);
        ctx.stroke();

        // 6. Draw Small Handedness Tag attached to Bounding Box
        const tagText = physicalLabel.toUpperCase();
        ctx.font = 'bold 11px ui-monospace, SFMono-Regular, monospace';
        const tagW = ctx.measureText(tagText).width + 14;
        const tagH = 20;
        const tagY = Math.max(2, boxY - tagH - 4);

        ctx.fillStyle = 'rgba(0, 20, 40, 0.85)';
        ctx.fillRect(boxX, tagY, tagW, tagH);
        ctx.strokeStyle = '#00ff66';
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX, tagY, tagW, tagH);

        ctx.fillStyle = '#00ff88';
        ctx.fillText(tagText, boxX + 7, tagY + 14);
      }
    }

    avgConf = hasHands ? Math.round(avgConf / hands.length) : 0;

    // Update Debug Stats in Overlay
    this.updateDebugUI({
      detected: hasHands,
      count: hands.length,
      handedness: hasHands ? handednessLabels.join(', ') : '—',
      confidence: hasHands ? `${avgConf}%` : '—',
      fps: this.currentFps
    });

    // Phase 2 & 5: Process frame in Landmark Feature Pipeline (150 features & 30-frame rolling sequence)
    // FIX 2: Pass latestPoseLandmarks
    landmarkPipelineService.processFrame(hands, handednessList, this.latestPoseLandmarks);
  }

  updateDebugUI({ detected, count, handedness, confidence, fps }) {
    const handDetectedEl = document.getElementById('debug-hand-detected');
    const handsCountEl = document.getElementById('debug-hands-count');
    const handednessEl = document.getElementById('debug-handedness');
    const confidenceEl = document.getElementById('debug-confidence');
    const fpsEl = document.getElementById('debug-fps');
    const statusTextEl = document.getElementById('debug-tracking-status');
    const statusDotEl = document.getElementById('debug-status-indicator');

    if (handDetectedEl) {
      handDetectedEl.textContent = detected ? 'YES' : 'NO';
      handDetectedEl.className = detected ? 'font-bold text-emerald-400' : 'font-bold text-red-400';
    }
    if (handsCountEl) handsCountEl.textContent = count;
    if (handednessEl) handednessEl.textContent = handedness;
    if (confidenceEl) confidenceEl.textContent = confidence;
    if (fpsEl) fpsEl.textContent = fps;

    if (statusTextEl && statusDotEl) {
      if (detected) {
        statusTextEl.textContent = 'Hand Tracking: ACTIVE';
        statusTextEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
        statusDotEl.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-ping';
      } else {
        statusTextEl.textContent = 'Hand Tracking: NOT DETECTED';
        statusTextEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/30';
        statusDotEl.className = 'w-2 h-2 rounded-full bg-red-500';
      }
    }
  }

  setDebugMode(enabled) {
    this.isEnabled = Boolean(enabled);
    console.log(`[HandTrackingDebugService] Debug mode: ${this.isEnabled ? 'ON' : 'OFF'}`);

    const card = document.getElementById('hand-tracking-debug-panel') || document.getElementById('hand-tracking-debug-card');
    const canvas = this.canvasElement;

    if (card) {
      if (this.isEnabled) {
        card.classList.remove('hidden');
      } else {
        card.classList.add('hidden');
      }
    }

    if (!this.isEnabled) {
      this.clearCanvas();
      this.stopLoop();
    } else if (this.videoElement) {
      this.startLoop();
    }
  }

  toggleDebugMode() {
    this.setDebugMode(!this.isEnabled);
    return this.isEnabled;
  }
}

export const handTrackingDebugService = new HandTrackingDebugService();
