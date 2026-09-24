/**
 * SignAnimationViewer Component
 *
 * Reusable Three.js 3D avatar viewport for SignBridge AI.
 * Renders xbot.glb avatar and drives procedural skeletal sign-language animations:
 * - A-Z Fingerspelling (26/26 complete coverage for arbitrary text)
 * - American Sign Language (ASL): Verified working animations for HELLO, GOOD MORNING, THANK YOU, YES, NO
 * - Indian Sign Language (ISL): Verified working animations for HELLO, GOOD MORNING, THANK YOU, YES, NO
 *
 * Strict Honesty Rule:
 * - If a phrase is unsupported in ASL/ISL, it displays "ASL/ISL animation unavailable" and offers
 *   an explicit "[Use Fingerspelling]" fallback. No fake animations are generated.
 * - Fixes the reference SignViewer condition where 'no' was omitted from active 3D playback.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';
import { SignAnimationEngine, getPhraseDuration } from '../../services/signAnimation/engine.js';
import {
  SIGNING_MODES,
  MODE_LABELS,
  isSupportedSign,
  getSignConfig,
  normalizeSignText
} from '../../services/signAnimation/signConfig.js';

// Cache for loaded xbot.glb GLTF scene to prevent reloading per message/letter
let cachedXbotGltf = null;
let gltfLoadingPromise = null;

export function loadAvatarGltf(modelUrl = './models/xbot.glb') {
  if (cachedXbotGltf) {
    return Promise.resolve(cachedXbotGltf);
  }
  if (gltfLoadingPromise) {
    return gltfLoadingPromise;
  }

  gltfLoadingPromise = new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    // Try both relative and root-based URLs
    const tryLoad = (url) => {
      loader.load(
        url,
        (gltf) => {
          cachedXbotGltf = gltf;
          gltfLoadingPromise = null;
          resolve(gltf);
        },
        undefined,
        (err) => {
          if (url.startsWith('.')) {
            // Retry with absolute path
            loader.load(
              '/models/xbot.glb',
              (gltf2) => {
                cachedXbotGltf = gltf2;
                gltfLoadingPromise = null;
                resolve(gltf2);
              },
              undefined,
              (err2) => {
                gltfLoadingPromise = null;
                reject(err2);
              }
            );
          } else {
            gltfLoadingPromise = null;
            reject(err);
          }
        }
      );
    };

    tryLoad(modelUrl);
  });

  return gltfLoadingPromise;
}

export class SignAnimationViewer {
  constructor(options = {}) {
    this.container = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.avatar = null;
    this.placeholderMannequin = null;
    this.engine = null;
    this.reqAnimId = null;
    this.resizeObserver = null;

    this.currentText = options.text || 'Hello';
    this.currentMode = options.mode || SIGNING_MODES.FINGERSPELLING;
    this.isPlaying = options.autoPlay !== false;
    this.isLooping = options.isLooping || false;
    this.playbackRate = 1.0;

    this.activeTokenInfo = { word: '', token: '', isDirectWord: false };
    this.isModelLoaded = false;
    this.isDestroyed = false;

    // Callbacks
    this.onTokenChangeCallback = options.onTokenChange || null;
    this.onProgressCallback = options.onProgress || null;
    this.onCompletedCallback = options.onCompleted || null;

    // Orbit controls state
    this.isDragging = false;
    this.prevMousePos = { x: 0, y: 0 };
    this.cameraAngle = { theta: 0, phi: 0 };

    this.initEngine();
  }

  initEngine() {
    this.engine = new SignAnimationEngine(
      (info) => {
        this.activeTokenInfo = info;
        this.onTokenChangeCallback?.(info);
        this.updateTokenBadge(info);
      },
      () => {
        this.isPlaying = false;
        this.activeTokenInfo = { word: '', token: '', isDirectWord: false };
        this.updateTokenBadge(this.activeTokenInfo);
        this.onCompletedCallback?.();
      }
    );
  }

  mount(containerEl) {
    if (!containerEl) return;
    this.container = containerEl;
    const width = containerEl.clientWidth || 320;
    const height = containerEl.clientHeight || 260;

    // 1. Three.js Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0f1d);
    this.scene.fog = new THREE.FogExp2(0x0a0f1d, 0.12);

    // 2. Camera (perspective framing the signer's upper body and chest)
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 1.25, 2.7);
    this.camera.lookAt(0, 1.15, 0);

    // 3. Renderer with high DPR and shadows
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.renderer.domElement.style.display = 'block';

    containerEl.innerHTML = '';
    containerEl.appendChild(this.renderer.domElement);

    // 4. Studio Lighting
    const ambientLight = new THREE.AmbientLight(0x1e293b, 1.2);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xf8fafc, 1.8);
    keyLight.position.set(2, 4, 3);
    keyLight.castShadow = true;
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.6);
    rimLight.position.set(-2.5, 3, -2);
    this.scene.add(rimLight);

    const fillLight = new THREE.DirectionalLight(0x818cf8, 0.8);
    fillLight.position.set(0, -1, 2);
    this.scene.add(fillLight);

    // 5. Studio Floor Grid & Pedestal
    const gridHelper = new THREE.GridHelper(8, 24, 0x38bdf8, 0x1e293b);
    this.scene.add(gridHelper);

    const pedestalGeo = new THREE.CylinderGeometry(1.2, 1.3, 0.04, 32);
    const pedestalMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.8,
      metalness: 0.3
    });
    const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat);
    pedestal.position.y = 0.02;
    pedestal.receiveShadow = true;
    this.scene.add(pedestal);

    // 6. Anatomical Neutral Mannequin Fallback
    this.createPlaceholderMannequin();

    // 7. Load or attach cached xbot.glb avatar
    this.loadAvatar();

    // 8. Mouse interaction (Orbit dragging)
    this.bindMouseControls();

    // 9. ResizeObserver
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: nw, height: nh } = entry.contentRect;
        if (nw > 0 && nh > 0 && this.camera && this.renderer) {
          this.camera.aspect = nw / nh;
          this.camera.updateProjectionMatrix();
          this.renderer.setSize(nw, nh);
        }
      }
    });
    this.resizeObserver.observe(containerEl);

    // 10. Start Animation Render Loop
    this.startRenderLoop();
  }

  createPlaceholderMannequin() {
    const group = new THREE.Group();
    const skinMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.35, metalness: 0.2 });
    const accentMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.2, metalness: 0.8, emissive: 0x0284c7, emissiveIntensity: 0.2 });
    const torsoMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6, metalness: 0.1 });

    // Head & Neck
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 24), skinMat);
    head.scale.set(1, 1.25, 1.05);
    head.position.set(0, 1.62, 0);
    group.add(head);

    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 0.08, 16), skinMat);
    neck.position.set(0, 1.48, 0);
    group.add(neck);

    // Torso
    const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.14, 0.42, 16), torsoMat);
    torso.scale.set(1.2, 1, 0.75);
    torso.position.set(0, 1.25, 0);
    group.add(torso);

    // Shoulders
    const leftShoulder = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16), accentMat);
    leftShoulder.position.set(0.24, 1.4, 0);
    group.add(leftShoulder);

    const rightShoulder = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16), accentMat);
    rightShoulder.position.set(-0.24, 1.4, 0);
    group.add(rightShoulder);

    // Stand
    const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.88, 16), torsoMat);
    stand.position.set(0, 0.46, 0);
    group.add(stand);

    this.placeholderMannequin = group;
    this.scene.add(group);
  }

  async loadAvatar() {
    try {
      const gltf = await loadAvatarGltf('./models/xbot.glb');
      if (this.isDestroyed) return;

      const avatarScene = SkeletonUtils.clone(gltf.scene);
      avatarScene.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
        if (child.type === 'SkinnedMesh') {
          child.frustumCulled = false;
        }
      });

      avatarScene.position.set(0, 0.04, 0);
      this.avatar = avatarScene;
      this.scene.add(avatarScene);

      // Hide primitive placeholder
      if (this.placeholderMannequin) {
        this.placeholderMannequin.visible = false;
      }

      this.engine.setAvatar(avatarScene);
      this.isModelLoaded = true;

      // Count resolved bones in the active avatar skeleton
      let boneCount = 0;
      avatarScene.traverse((child) => {
        if (child.isBone || child.type === 'Bone') boneCount++;
      });
      console.log(`[SignAnimationViewer] xbot.glb avatar attached. Resolved bones in skeleton: ${boneCount}`);

      // Load initial phrase
      this.loadPhraseIntoEngine(this.currentText, this.currentMode, this.isPlaying);
    } catch (err) {
      console.warn('xbot.glb load error, keeping neutral mannequin fallback:', err);
    }
  }

  bindMouseControls() {
    if (!this.container) return;
    const onMouseDown = (e) => {
      this.isDragging = true;
      this.prevMousePos = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!this.isDragging || !this.camera) return;
      const deltaX = e.clientX - this.prevMousePos.x;
      const deltaY = e.clientY - this.prevMousePos.y;
      this.prevMousePos = { x: e.clientX, y: e.clientY };

      this.cameraAngle.theta += deltaX * 0.008;
      this.cameraAngle.phi = Math.max(-0.4, Math.min(0.6, this.cameraAngle.phi + deltaY * 0.008));

      const radius = 2.7;
      const targetY = 1.15;
      const phi = this.cameraAngle.phi;
      const theta = this.cameraAngle.theta;

      this.camera.position.x = radius * Math.sin(theta) * Math.cos(phi);
      this.camera.position.y = targetY + radius * Math.sin(phi);
      this.camera.position.z = radius * Math.cos(theta) * Math.cos(phi);
      this.camera.lookAt(0, targetY, 0);
    };

    const onMouseUp = () => {
      this.isDragging = false;
    };

    this.container.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    this._mouseCleanup = () => {
      this.container?.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }

  startRenderLoop() {
    let lastTime = performance.now();
    let clock = new THREE.Clock();
    let frameCount = 0;

    const render = () => {
      if (this.isDestroyed) return;
      this.reqAnimId = requestAnimationFrame(render);

      const now = performance.now();
      const delta = Math.min(0.1, (now - lastTime) / 1000);
      lastTime = now;
      frameCount++;

      // FIXED CONDITION (PART 7):
      // Includes 'no' so all 5 working ASL/ISL gestures (hello, good-morning, thank-you, yes, no)
      // and complete A-Z fingerspelling animate in real-time.
      const normSlug = normalizeSignText(this.currentText);
      const is5Sign =
        normSlug === 'hello' ||
        normSlug === 'good morning' ||
        normSlug === 'thank you' ||
        normSlug === 'yes' ||
        normSlug === 'no';

      const mode = (this.currentMode || '').toUpperCase();
      const is3DActive =
        this.avatar &&
        (mode === SIGNING_MODES.FINGERSPELLING ||
          (is5Sign && (mode === SIGNING_MODES.ISL || mode === SIGNING_MODES.ASL)));

      if (is3DActive) {
        if (this.isPlaying) {
          this.engine.step(delta, this.isPlaying, this.playbackRate, this.isLooping);
          const progress = this.engine.getProgress();
          this.onProgressCallback?.(progress);

          // Diagnostic log every ~60 frames while playing
          if (frameCount % 60 === 0) {
            const sampleBone = this.engine.getBone('mixamorigRightHandIndex1') || this.engine.getBone('mixamorigLeftHandIndex1');
            const rotZ = sampleBone ? sampleBone.rotation.z.toFixed(4) : 'N/A';
            const rotY = sampleBone ? sampleBone.rotation.y.toFixed(4) : 'N/A';
            console.log(`[SignAnimationViewer:Tick] frame: ${frameCount} | token: "${this.activeTokenInfo.token}" | progress: ${(progress * 100).toFixed(1)}% | sampleBone rot: (y=${rotY}, z=${rotZ})`);
          }
        }
      } else if (this.placeholderMannequin && this.placeholderMannequin.visible) {
        // Idle breathing oscillation for mannequin
        const breathe = Math.sin(clock.getElapsedTime() * 1.8) * 0.005;
        this.placeholderMannequin.position.y = breathe;
      }

      if (this.renderer && this.scene && this.camera) {
        this.renderer.render(this.scene, this.camera);
      }
    };

    render();
  }

  loadPhraseIntoEngine(text, mode, playNow = true) {
    this.currentText = text;
    this.currentMode = mode;

    const config = getSignConfig(text, mode);
    this.updateFallbackUI(config);

    if (config.supported) {
      this.engine.loadPhrase(text, mode);
      this.isPlaying = playNow;
    } else {
      // Unsupported in ASL or ISL: do not fabricate animations
      this.engine.resetToDefaultPose();
      this.isPlaying = false;
      this.activeTokenInfo = { word: '', token: '', isDirectWord: false };
      this.updateTokenBadge(this.activeTokenInfo);
      this.onCompletedCallback?.();
    }
  }

  play(text, mode = this.currentMode) {
    this.currentText = text;
    this.currentMode = mode;
    this.isPlaying = true;

    console.log(`[SignAnimationViewer] Play triggered -> text: "${text}", mode: "${mode}", modelLoaded: ${this.isModelLoaded}`);

    if (this.isModelLoaded) {
      this.loadPhraseIntoEngine(text, mode, true);
    }
  }

  stop() {
    this.isPlaying = false;
    this.engine?.resetToDefaultPose();
  }

  replay() {
    this.isPlaying = true;
    this.engine?.restart();
  }

  switchToFingerspelling() {
    this.play(this.currentText, SIGNING_MODES.FINGERSPELLING);
  }

  updateTokenBadge(info) {
    const badge = document.getElementById('active-token-badge');
    const badgeText = document.getElementById('active-token-text');
    if (!badge || !badgeText) return;

    if (info && info.token) {
      badge.classList.remove('hidden');
      if (info.isDirectWord) {
        badgeText.innerHTML = `<span class="text-black">Current Sign:</span> <span class="text-black font-bold">${info.token}</span>`;
      } else {
        badgeText.innerHTML = `<span class="text-black">Current Letter:</span> <span class="text-black font-bold">${info.token}</span>`;
      }
    } else {
      badge.classList.add('hidden');
    }
  }

  updateFallbackUI(config) {
    const banner = document.getElementById('sign-unsupported-banner');
    const bannerMsg = document.getElementById('sign-unsupported-message');
    const btnFallbackFS = document.getElementById('btn-fallback-fingerspell');

    if (!banner) return;

    if (!config.supported && (config.mode === SIGNING_MODES.ASL || config.mode === SIGNING_MODES.ISL)) {
      banner.classList.remove('hidden');
      if (bannerMsg) {
        bannerMsg.textContent = `${config.modeLabel} animation unavailable for "${config.phrase}".`;
      }
      if (btnFallbackFS) {
        btnFallbackFS.onclick = () => {
          this.switchToFingerspelling();
        };
      }
    } else {
      banner.classList.add('hidden');
    }
  }

  destroy() {
    this.isDestroyed = true;
    if (this.reqAnimId) cancelAnimationFrame(this.reqAnimId);
    this.resizeObserver?.disconnect();
    this._mouseCleanup?.();

    if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
    this.renderer?.dispose();
    this.scene = null;
    this.camera = null;
    this.avatar = null;
    this.placeholderMannequin = null;
    this.engine = null;
  }
}

/**
 * Renders the HTML markup for the Reusable Sign Animation Viewer container.
 */
export function renderSignAnimationViewer(props = {}) {
  const text = props.text || 'Hello';
  const mode = props.mode || SIGNING_MODES.FINGERSPELLING;
  const config = getSignConfig(text, mode);
  const isPlaying = props.isPlaying !== false;
  const completed = Boolean(props.completed);

  const modeBadgeText = MODE_LABELS[mode] || mode;

  return `
    <div class="flex flex-col h-full select-none" id="sign-viewer-component-root">
      <!-- Section Header & Playback Badge -->
      <div class="flex items-center justify-between pb-2 flex-shrink-0 gap-1.5">
        <h2 class="text-sm lg:text-base font-extrabold text-primary tracking-tight whitespace-nowrap">Sign Avatar</h2>
        <div class="flex items-center gap-1.5 flex-shrink-0">
          <!-- Active Mode Tag -->
          <span id="sign-active-mode-badge" class="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-surface-container text-on-surface-variant border border-outline-variant/30">
            ${modeBadgeText}
          </span>

          <div id="playback-badge" class="flex items-center gap-1 px-2 py-0.5 rounded-full ${completed ? 'bg-emerald-100 text-emerald-900 border border-emerald-300' : 'bg-primary-container text-secondary-container'} text-[10px] font-bold uppercase tracking-wide shadow-xs">
            ${completed ? `
              <span class="material-symbols-outlined text-[13px] text-emerald-700">check_circle</span>
              <span>Done</span>
            ` : `
              <span class="w-1.5 h-1.5 rounded-full bg-secondary animate-ping"></span>
              <span id="playback-status-text">Playing</span>
            `}
          </div>

          <button type="button" 
                  id="btn-replay-animation" 
                  class="p-1 rounded-md text-on-surface-variant hover:bg-surface-container hover:text-primary transition-colors cursor-pointer"
                  title="Replay Sign Animation">
            <span class="material-symbols-outlined text-[18px]">replay</span>
          </button>
        </div>
      </div>

      <!-- 3D Viewport Mount Container -->
      <div class="relative w-full flex-1 min-h-[160px] lg:min-h-[195px] bg-[#0a0f1d] rounded-xl overflow-hidden shadow-md flex items-center justify-center border border-outline-variant/30">
        <!-- Three.js Canvas Mount Point -->
        <div id="sign-viewer-canvas-mount" class="w-full h-full"></div>

        <!-- Real-time Active Token Pill Overlay (Letter / Sign) -->
        <div id="active-token-badge" class="hidden absolute top-3 left-3 z-10 px-2.5 py-1 rounded-lg bg-white/95 backdrop-blur-md border border-slate-300 text-black font-mono text-[11px] font-bold shadow-md flex items-center gap-1.5 animate-fadeIn">
          <span class="w-2 h-2 rounded-full bg-black animate-pulse"></span>
          <span id="active-token-text" class="text-black"><span class="text-black">Current Letter:</span> <span class="text-black font-bold">A</span></span>
        </div>

        <!-- Unsupported Natural Sign Fallback Banner -->
        <div id="sign-unsupported-banner" class="${!config.supported && (mode === SIGNING_MODES.ASL || mode === SIGNING_MODES.ISL) ? 'flex' : 'hidden'} absolute inset-0 z-20 bg-slate-950/85 backdrop-blur-sm p-4 flex-col items-center justify-center text-center gap-3 animate-fadeIn">
          <span class="material-symbols-outlined text-amber-400 text-3xl">info</span>
          <div>
            <h4 class="text-sm font-bold text-white mb-1" id="sign-unsupported-message">
              ${modeBadgeText} animation unavailable for this phrase.
            </h4>
            <p class="text-xs text-slate-300 max-w-xs">
              Natural ${mode} 3D signs are verified only for: Hello, Good Morning, Thank You, Yes, No.
            </p>
          </div>
          <button type="button"
                  id="btn-fallback-fingerspell"
                  class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-white hover:bg-primary-container text-xs font-bold transition-all shadow-sm cursor-pointer">
            <span class="material-symbols-outlined text-[16px]">spellcheck</span>
            <span>Use Fingerspelling Instead</span>
          </button>
        </div>

        <!-- Overlay Playback Progress Bar -->
        <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-primary/95 via-primary/70 to-transparent p-2.5 flex flex-col gap-1 z-10 pointer-events-none">
          <div class="w-full bg-white/20 h-1.5 rounded-full overflow-hidden">
            <div id="playback-progress" class="bg-secondary h-full w-0 transition-all duration-100 ease-linear"></div>
          </div>
          <div class="flex items-center justify-between text-white text-[11px] font-mono">
            <span id="cycle-label">${completed ? 'Completed' : 'Playing'}</span>
            <span id="time-remaining">00:00 / 00:03</span>
          </div>
        </div>
      </div>

      <!-- Synchronized Staff Message Text Box -->
      <div class="mt-2.5 p-3 bg-surface-container-low rounded-xl border border-outline-variant/40 flex flex-col gap-1 flex-shrink-0 shadow-xs">
        <div class="flex items-center justify-between gap-1.5">
          <span class="text-[11px] font-bold uppercase tracking-wider text-on-surface-variant flex items-center gap-1">
            <span class="material-symbols-outlined text-primary text-[15px]">support_agent</span>
            Officer's Statement
          </span>
          <span id="sync-mode-label" class="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-surface-container text-primary">
            ${modeBadgeText}
          </span>
        </div>
        <p class="text-sm lg:text-base font-bold text-primary leading-snug break-words line-clamp-2" id="sync-animation-text">
          "${text}"
        </p>
      </div>
    </div>
  `;
}
