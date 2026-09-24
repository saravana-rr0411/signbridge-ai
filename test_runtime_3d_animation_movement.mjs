/**
 * Runtime 3D Animation Movement Test Suite
 *
 * Verifies that procedural bone rotations are actually applied to xbot.glb's
 * SkinnedMesh skeleton and boneMatrices over multiple frames.
 *
 * This directly prevents the "static robot" bug where bones were cloned
 * without SkeletonUtils, leaving SkinnedMesh bound to unmoving original bones.
 */

import assert from 'node:assert';
import fs from 'node:fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';
import { SignAnimationEngine } from './services/signAnimation/engine.js';
import { SIGNING_MODES } from './services/signAnimation/signConfig.js';

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`✅ [PASS] ${name}`);
    passed++;
  } catch (err) {
    console.error(`❌ [FAIL] ${name}:`, err.message);
    failed++;
  }
}

console.log('====================================================');
console.log('TESTING 3D RUNTIME PROCEDURAL ANIMATION MOVEMENT');
console.log('====================================================\n');

// 1. Load xbot.glb into memory
const glbBuffer = fs.readFileSync('public/models/xbot.glb');
const arrayBuffer = glbBuffer.buffer.slice(glbBuffer.byteOffset, glbBuffer.byteOffset + glbBuffer.byteLength);

const loader = new GLTFLoader();
loader.parse(arrayBuffer, '', (gltf) => {
  // Test 1: SkeletonUtils clones both SkinnedMesh and Skeleton with matching bone references
  test('Skeleton Rigging: SkeletonUtils rebinds SkinnedMesh to cloned bones', () => {
    const avatar = SkeletonUtils.clone(gltf.scene);
    let skinnedMesh = null;
    avatar.traverse((child) => {
      if (child.isSkinnedMesh) skinnedMesh = child;
    });

    assert(skinnedMesh, 'Avatar contains a SkinnedMesh');
    assert(skinnedMesh.skeleton, 'SkinnedMesh has a Skeleton');
    assert.strictEqual(skinnedMesh.skeleton.bones.length, 67, 'Skeleton has 67 bones');

    // Pick a test bone
    const sampleBoneName = 'mixamorigRightHandIndex1';
    const boneFromAvatar = avatar.getObjectByName(sampleBoneName);
    assert(boneFromAvatar, `Bone ${sampleBoneName} found in avatar scene`);

    const boneIndexInSkeleton = skinnedMesh.skeleton.bones.indexOf(boneFromAvatar);
    assert(boneIndexInSkeleton >= 0, `Bone ${sampleBoneName} is properly referenced inside SkinnedMesh.skeleton.bones`);
    assert.strictEqual(skinnedMesh.skeleton.bones[boneIndexInSkeleton], boneFromAvatar, 'Skeleton bone reference is identical to avatar bone');
  });

  // Test suite helper to run an animation and verify frame-by-frame bone movement
  function verifySignAnimationMovement(mode, phrase, expectedBones) {
    const avatar = SkeletonUtils.clone(gltf.scene);
    let skinnedMesh = null;
    avatar.traverse((child) => {
      if (child.isSkinnedMesh) skinnedMesh = child;
    });

    const engine = new SignAnimationEngine();
    engine.setAvatar(avatar);
    engine.loadPhrase(phrase, mode);

    assert(engine.keyframes.length > 0, `${mode} "${phrase}" produced keyframes`);

    // Capture rest pose matrix
    avatar.updateMatrixWorld(true);
    skinnedMesh.skeleton.update();
    const restMatrices = new Float32Array(skinnedMesh.skeleton.boneMatrices);

    const boneSamples = [];
    for (const bName of expectedBones) {
      const bone = engine.getBone(bName);
      assert(bone, `Expected bone ${bName} exists in avatar`);
      boneSamples.push({
        name: bName,
        bone,
        initRot: { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z },
        history: []
      });
    }

    // Step through 20 frames at 60fps (delta = 0.016s)
    let movedBonesCount = 0;
    for (let frame = 1; frame <= 20; frame++) {
      engine.step(0.016, true, 1.0, false);
      avatar.updateMatrixWorld(true);
      skinnedMesh.skeleton.update();

      for (const sample of boneSamples) {
        sample.history.push({
          frame,
          x: sample.bone.rotation.x,
          y: sample.bone.rotation.y,
          z: sample.bone.rotation.z
        });
      }
    }

    // Check that at least one of expected bones changed significantly over 20 frames
    let maxChange = 0;
    for (const sample of boneSamples) {
      const last = sample.history[sample.history.length - 1];
      const diff = Math.abs(last.x - sample.initRot.x) +
                   Math.abs(last.y - sample.initRot.y) +
                   Math.abs(last.z - sample.initRot.z);
      if (diff > 0.05) {
        movedBonesCount++;
      }
      if (diff > maxChange) maxChange = diff;
    }

    assert(movedBonesCount > 0, `${mode} "${phrase}": At least one target bone moved (max diff = ${maxChange.toFixed(4)})`);

    // Verify SkinnedMesh GPU boneMatrices changed from rest pose
    let matrixChanged = false;
    for (let i = 0; i < skinnedMesh.skeleton.boneMatrices.length; i++) {
      if (Math.abs(skinnedMesh.skeleton.boneMatrices[i] - restMatrices[i]) > 0.001) {
        matrixChanged = true;
        break;
      }
    }
    assert(matrixChanged, `${mode} "${phrase}": SkinnedMesh boneMatrices were updated for GPU vertex shader`);
  }

  // Test 2: FINGERSPELLING -> A
  test('Runtime Movement: FINGERSPELLING -> A visibly moves finger and hand bones', () => {
    verifySignAnimationMovement('FINGERSPELLING', 'A', [
      'mixamorigLeftHandIndex1',
      'mixamorigLeftHandMiddle1',
      'mixamorigLeftHand',
      'mixamorigLeftForeArm',
      'mixamorigRightHandPinky1'
    ]);
  });

  // Test 3: FINGERSPELLING -> HELLO
  test('Runtime Movement: FINGERSPELLING -> HELLO animates sequential hand gestures', () => {
    verifySignAnimationMovement('FINGERSPELLING', 'HELLO', [
      'mixamorigLeftArm',
      'mixamorigLeftForeArm',
      'mixamorigLeftHand',
      'mixamorigRightArm'
    ]);
  });

  // Test 4: ASL -> HELLO
  test('Runtime Movement: ASL -> HELLO visibly moves arm, forearm, hand, and fingers', () => {
    verifySignAnimationMovement('ASL', 'HELLO', [
      'mixamorigRightArm',
      'mixamorigRightForeArm',
      'mixamorigRightHand',
      'mixamorigRightHandThumb1'
    ]);
  });

  // Test 5: ASL -> YES
  test('Runtime Movement: ASL -> YES visibly moves fist and nods neck', () => {
    verifySignAnimationMovement('ASL', 'YES', [
      'mixamorigRightArm',
      'mixamorigRightForeArm',
      'mixamorigRightHand',
      'mixamorigNeck'
    ]);
  });

  // Test 6: ASL -> NO
  test('Runtime Movement: ASL -> NO visibly snaps fingers and shakes neck', () => {
    verifySignAnimationMovement('ASL', 'NO', [
      'mixamorigRightArm',
      'mixamorigRightForeArm',
      'mixamorigRightHand',
      'mixamorigNeck'
    ]);
  });

  // Test 7: ISL -> HELLO
  test('Runtime Movement: ISL -> HELLO visibly waves greeting hand and forearm', () => {
    verifySignAnimationMovement('ISL', 'HELLO', [
      'mixamorigRightArm',
      'mixamorigRightForeArm',
      'mixamorigRightHand',
      'mixamorigNeck'
    ]);
  });

  // Test 8: ISL -> YES
  test('Runtime Movement: ISL -> YES visibly nods hand and neck', () => {
    verifySignAnimationMovement('ISL', 'YES', [
      'mixamorigRightArm',
      'mixamorigRightForeArm',
      'mixamorigRightHand',
      'mixamorigNeck'
    ]);
  });

  // Test 9: ISL -> NO
  test('Runtime Movement: ISL -> NO visibly performs negation wave with headshake', () => {
    verifySignAnimationMovement('ISL', 'NO', [
      'mixamorigRightArm',
      'mixamorigRightForeArm',
      'mixamorigRightHand',
      'mixamorigNeck'
    ]);
  });

  console.log('\n====================================================');
  console.log(`RUNTIME MOVEMENT RESULTS: ${passed} PASSED | ${failed} FAILED`);
  console.log('====================================================');

  if (failed > 0) {
    process.exit(1);
  }
});
