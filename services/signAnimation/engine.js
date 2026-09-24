import { alphabets } from './alphabets.js';
import { words } from './words.js';
import { islAnimations } from './isl/index.js';
import { aslAnimations } from './asl/index.js';
function parsePhraseToKeyframes(text, mode = "FINGERSPELLING", _options) {
  const normalized = text.replace(/[^a-zA-Z\s]/g, "").trim().toLowerCase();
  const keyframes = [];
  if (mode === "ISL") {
    if (normalized === "hello") {
      const mockRef = { animations: [], pending: true };
      islAnimations.HELLO(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "HELLO",
            token: "HELLO",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    if (normalized === "good morning") {
      const mockRef = { animations: [], compoundSteps: [], pending: true };
      islAnimations.GOOD_MORNING(mockRef);
      if (mockRef.compoundSteps && mockRef.compoundSteps.length > 0) {
        for (let sIdx = 0; sIdx < mockRef.compoundSteps.length; sIdx++) {
          const cStep = mockRef.compoundSteps[sIdx];
          const isFinal = sIdx === mockRef.compoundSteps.length - 1;
          keyframes.push({
            word: cStep.word,
            token: cStep.token,
            isDirectWord: true,
            isReturnToRest: isFinal || cStep.isReturnToRest,
            isWordEnd: cStep.isWordEnd,
            bones: cStep.bones.map((b) => [...b])
          });
        }
      } else {
        const totalSteps = mockRef.animations.length;
        for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
          const step = mockRef.animations[sIdx];
          if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
            const isEnd = sIdx === totalSteps - 1;
            keyframes.push({
              word: "GOOD MORNING",
              token: "GOOD MORNING",
              isDirectWord: true,
              isReturnToRest: isEnd,
              isWordEnd: isEnd,
              bones: step.map((b) => [...b])
            });
          }
        }
      }
      return keyframes;
    }
    if (normalized === "thank you") {
      const mockRef = { animations: [], pending: true };
      islAnimations.THANK_YOU(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "THANK YOU",
            token: "THANK YOU",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    if (normalized === "yes") {
      const mockRef = { animations: [], pending: true };
      islAnimations.YES(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "YES",
            token: "YES",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    if (normalized === "no") {
      const mockRef = { animations: [], pending: true };
      islAnimations.NO(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "NO",
            token: "NO",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    return [];
  }
  if (mode === "ASL") {
    if (normalized === "hello") {
      const mockRef = { animations: [], pending: true };
      aslAnimations.HELLO(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "HELLO",
            token: "HELLO",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    if (normalized === "good morning") {
      const mockRef = { animations: [], compoundSteps: [], pending: true };
      aslAnimations.GOOD_MORNING(mockRef);
      if (mockRef.compoundSteps && mockRef.compoundSteps.length > 0) {
        for (let sIdx = 0; sIdx < mockRef.compoundSteps.length; sIdx++) {
          const cStep = mockRef.compoundSteps[sIdx];
          const isFinal = sIdx === mockRef.compoundSteps.length - 1;
          keyframes.push({
            word: cStep.word,
            token: cStep.token,
            isDirectWord: true,
            isReturnToRest: isFinal || cStep.isReturnToRest,
            isWordEnd: cStep.isWordEnd,
            bones: cStep.bones.map((b) => [...b])
          });
        }
      } else {
        const totalSteps = mockRef.animations.length;
        for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
          const step = mockRef.animations[sIdx];
          if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
            const isEnd = sIdx === totalSteps - 1;
            keyframes.push({
              word: "GOOD MORNING",
              token: "GOOD MORNING",
              isDirectWord: true,
              isReturnToRest: isEnd,
              isWordEnd: isEnd,
              bones: step.map((b) => [...b])
            });
          }
        }
      }
      return keyframes;
    }
    if (normalized === "thank you") {
      const mockRef = { animations: [], pending: true };
      aslAnimations.THANK_YOU(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "THANK YOU",
            token: "THANK YOU",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    if (normalized === "yes") {
      const mockRef = { animations: [], pending: true };
      aslAnimations.YES(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "YES",
            token: "YES",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    if (normalized === "no") {
      const mockRef = { animations: [], pending: true };
      aslAnimations.NO(mockRef);
      const totalSteps = mockRef.animations.length;
      for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
        const step = mockRef.animations[sIdx];
        if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
          const isEnd = sIdx === totalSteps - 1;
          keyframes.push({
            word: "NO",
            token: "NO",
            isDirectWord: true,
            isReturnToRest: isEnd,
            isWordEnd: isEnd,
            bones: step.map((b) => [...b])
          });
        }
      }
      return keyframes;
    }
    return [];
  }
  const sanitized = text.replace(/['’]/g, "").replace(/[^a-zA-Z\s]/g, " ");
  const rawWords = sanitized.trim().toUpperCase().split(/\s+/).filter(Boolean);
  let globalCharIdx = 0;
  for (let wIdx = 0; wIdx < rawWords.length; wIdx++) {
    const word = rawWords[wIdx];
    const chars = word.split("");
    for (let cIdx = 0; cIdx < chars.length; cIdx++) {
      const ch = chars[cIdx];
      const letterIndex = globalCharIdx++;
      if (alphabets[ch]) {
        const mockRef = { animations: [], pending: true };
        alphabets[ch](mockRef);
        const totalSteps = mockRef.animations.length;
        for (let sIdx = 0; sIdx < totalSteps; sIdx++) {
          const step = mockRef.animations[sIdx];
          if (Array.isArray(step) && step.length > 0 && typeof step[0] !== "string") {
            const isLastStepOfLetter = sIdx === totalSteps - 1;
            const isLastLetterOfWord = cIdx === chars.length - 1;
            keyframes.push({
              word,
              token: ch,
              charIndex: letterIndex,
              isDirectWord: false,
              isReturnToRest: sIdx > 0,
              isWordEnd: isLastStepOfLetter && isLastLetterOfWord,
              bones: step.map((b) => [...b])
            });
          }
        }
      }
    }
  }
  return keyframes;
}
function getPhraseDuration(text, mode = "FINGERSPELLING", options) {
  const normalized = text.trim().toLowerCase().replace(/[^a-z\s]/g, "").trim();
  if (normalized === "hello") {
    if (mode === "ISL") return 3.2;
    if (mode === "ASL") return 3;
    return 4.7;
  }
  if (normalized === "good morning") {
    if (mode === "ISL") return 4.8;
    if (mode === "ASL") return 4.6;
    return 15.2;
  }
  if (normalized === "thank you") {
    if (mode === "ISL") return 2.8;
    if (mode === "ASL") return 2.6;
  }
  if (normalized === "yes") {
    if (mode === "ISL") return 2.4;
    if (mode === "ASL") return 2.4;
  }
  if (normalized === "no") {
    if (mode === "ISL") return 2.4;
    if (mode === "ASL") return 2.4;
  }
  const keyframes = parsePhraseToKeyframes(text, mode, options);
  if (keyframes.length === 0) return 3;
  let totalSeconds = 0;
  for (const kf of keyframes) {
    totalSeconds += 0.22;
    if (kf.isWordEnd) {
      totalSeconds += 0.5;
    } else if (kf.isReturnToRest) {
      totalSeconds += 0.25;
    } else {
      totalSeconds += 0.2;
    }
  }
  return Math.max(3, Math.round(totalSeconds * 10) / 10);
}
function getPhraseClassification(text) {
  const sanitized = text.replace(/['’]/g, "").replace(/[^a-zA-Z\s]/g, " ");
  const rawWords = sanitized.trim().toUpperCase().split(/\s+/).filter(Boolean);
  const directWords = [];
  const fingerspellWords = [];
  for (const w of rawWords) {
    if (words[w]) {
      directWords.push(w);
    } else {
      fingerspellWords.push(w);
    }
  }
  return { directWords, fingerspellWords };
}
class SignAnimationEngine {
  avatar = null;
  keyframes = [];
  currentKeyframeIndex = 0;
  currentActiveBones = [];
  pauseRemaining = 0;
  wordPauseDuration = 0.5;
  // seconds to pause between words
  letterPauseDuration = 0.25;
  // seconds to pause between letters
  poseHoldDuration = 0.2;
  // seconds to hold keyframe pose
  baseSpeed = 0.08;
  // radians per frame tick at 60fps
  currentTokenInfo = { word: "", token: "", isDirectWord: false };
  onTokenChangeCallback;
  onCompletedCallback;
  constructor(onTokenChange, onCompleted) {
    this.onTokenChangeCallback = onTokenChange;
    this.onCompletedCallback = onCompleted;
  }
  setAvatar(avatar) {
    this.avatar = avatar;
    this.resetToDefaultPose();
  }
  getAvatar() {
    return this.avatar;
  }
  getCurrentTokenInfo() {
    return this.currentTokenInfo;
  }
  /**
   * Safe bone lookup supporting both standard Mixamo names (mixamorigRightArm)
   * and GLTF-native colon-prefixed names (mixamorig:RightArm).
   */
  getBone(name) {
    if (!this.avatar) return null;
    let bone = this.avatar.getObjectByName(name);
    if (!bone) {
      if (name.startsWith("mixamorig:")) {
        bone = this.avatar.getObjectByName(name.replace("mixamorig:", "mixamorig"));
      } else if (name.startsWith("mixamorig")) {
        bone = this.avatar.getObjectByName(name.replace("mixamorig", "mixamorig:"));
      }
    }
    return bone || null;
  }
  /**
   * Resets avatar skeletal bones to the standard neutral rest pose
   */
  resetToDefaultPose() {
    if (!this.avatar) return;
    this.avatar.traverse((child) => {
      if (child.name.startsWith("mixamorig") && (child.name.includes("Hand") || child.name.includes("Finger") || child.name.includes("Thumb") || child.name.includes("Index") || child.name.includes("Middle") || child.name.includes("Ring") || child.name.includes("Pinky"))) {
        child.rotation.set(0, 0, 0);
      }
    });
    const neck = this.getBone("mixamorigNeck");
    if (neck) neck.rotation.set(Math.PI / 12, 0, 0);
    const leftArm = this.getBone("mixamorigLeftArm");
    if (leftArm) leftArm.rotation.set(0, 0, -Math.PI / 3);
    const leftForeArm = this.getBone("mixamorigLeftForeArm");
    if (leftForeArm) leftForeArm.rotation.set(0, -Math.PI / 1.5, 0);
    const rightArm = this.getBone("mixamorigRightArm");
    if (rightArm) rightArm.rotation.set(0, 0, Math.PI / 3);
    const rightForeArm = this.getBone("mixamorigRightForeArm");
    if (rightForeArm) rightForeArm.rotation.set(0, Math.PI / 1.5, 0);
  }
  /**
   * Loads any phrase into the animation queue based on active signing mode.
   */
  loadPhrase(text, mode = "FINGERSPELLING", options) {
    this.keyframes = parsePhraseToKeyframes(text, mode, options);
    this.restart();
  }
  /**
   * Restarts animation from step 0
   */
  restart() {
    this.currentKeyframeIndex = 0;
    this.pauseRemaining = 0;
    this.resetToDefaultPose();
    if (this.keyframes.length > 0) {
      const first = this.keyframes[0];
      this.currentActiveBones = this.cloneBones(first.bones);
      this.currentTokenInfo = {
        word: first.word,
        token: first.token,
        charIndex: first.charIndex,
        isDirectWord: first.isDirectWord
      };
      this.onTokenChangeCallback?.(this.currentTokenInfo);
    } else {
      this.currentActiveBones = [];
      this.currentTokenInfo = { word: "", token: "", charIndex: void 0, isDirectWord: false };
      this.onTokenChangeCallback?.(this.currentTokenInfo);
    }
  }
  /**
   * Advances the animation by delta time (seconds)
   */
  step(delta, isPlaying, playbackRate, isLooping) {
    if (!isPlaying || !this.avatar || this.keyframes.length === 0) return;
    const rate = Math.max(0.1, playbackRate);
    const speed = this.baseSpeed * rate * (delta / 0.016);
    if (this.pauseRemaining > 0) {
      this.pauseRemaining -= delta * rate;
      return;
    }
    if (this.currentActiveBones.length > 0) {
      let i = 0;
      while (i < this.currentActiveBones.length) {
        const [boneName, action, axis, limit] = this.currentActiveBones[i];
        const bone = this.getBone(boneName);
        if (!bone || !bone[action]) {
          this.currentActiveBones.splice(i, 1);
          continue;
        }
        const currentVal = bone[action][axis];
        const diff = limit - currentVal;
        if (Math.abs(diff) < 1e-3) {
          bone[action][axis] = limit;
          this.currentActiveBones.splice(i, 1);
        } else {
          const move = Math.sign(diff) * Math.min(speed, Math.abs(diff));
          bone[action][axis] += move;
          i++;
        }
      }
    } else {
      const currentKf = this.keyframes[this.currentKeyframeIndex];
      if (currentKf?.isWordEnd) {
        this.pauseRemaining = this.wordPauseDuration;
      } else if (currentKf?.isReturnToRest) {
        this.pauseRemaining = this.letterPauseDuration;
      } else {
        this.pauseRemaining = this.poseHoldDuration;
      }
      this.currentKeyframeIndex++;
      if (this.currentKeyframeIndex < this.keyframes.length) {
        const nextStep = this.keyframes[this.currentKeyframeIndex];
        this.currentActiveBones = this.cloneBones(nextStep.bones);
        this.currentTokenInfo = {
          word: nextStep.word,
          token: nextStep.token,
          charIndex: nextStep.charIndex,
          isDirectWord: nextStep.isDirectWord
        };
        this.onTokenChangeCallback?.(this.currentTokenInfo);
      } else {
        if (isLooping) {
          this.restart();
        } else {
          this.resetToDefaultPose();
          this.currentTokenInfo = { word: "", token: "", charIndex: void 0, isDirectWord: false };
          this.onTokenChangeCallback?.(this.currentTokenInfo);
          this.onCompletedCallback?.();
        }
      }
    }
  }
  /**
   * Returns normalized completion progress between 0 and 1
   */
  getProgress() {
    if (this.keyframes.length === 0) return 0;
    const stepWeight = 1 / this.keyframes.length;
    const completedSteps = this.currentKeyframeIndex * stepWeight;
    const currentStepDone = this.currentActiveBones.length === 0 ? stepWeight : (1 - this.currentActiveBones.length / (this.keyframes[this.currentKeyframeIndex]?.bones.length || 1)) * stepWeight;
    return Math.min(1, completedSteps + currentStepDone);
  }
  /**
   * Seeks to a proportional point in the animation
   */
  seek(progress) {
    if (this.keyframes.length === 0) return;
    const targetIdx = Math.min(
      this.keyframes.length - 1,
      Math.max(0, Math.floor(progress * this.keyframes.length))
    );
    this.currentKeyframeIndex = targetIdx;
    const step = this.keyframes[targetIdx];
    this.currentActiveBones = this.cloneBones(step.bones);
    this.currentTokenInfo = {
      word: step.word,
      token: step.token,
      charIndex: step.charIndex,
      isDirectWord: step.isDirectWord
    };
    this.pauseRemaining = 0;
    this.onTokenChangeCallback?.(this.currentTokenInfo);
  }
  cloneBones(bones) {
    return bones.map((b) => [...b]);
  }
}
export {
  SignAnimationEngine,
  getPhraseClassification,
  getPhraseDuration,
  parsePhraseToKeyframes
};
