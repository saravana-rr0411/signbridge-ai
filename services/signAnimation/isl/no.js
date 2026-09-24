const ISL_NO = (ref) => {
  let animations = [];
  animations.push(["mixamorigRightHandIndex1", "rotation", "z", 0, "-"]);
  animations.push(["mixamorigRightHandMiddle1", "rotation", "z", 0, "-"]);
  animations.push(["mixamorigRightHandRing1", "rotation", "z", 0, "-"]);
  animations.push(["mixamorigRightHandPinky1", "rotation", "z", 0, "-"]);
  animations.push(["mixamorigRightHandThumb1", "rotation", "x", Math.PI / 8, "+"]);
  animations.push(["mixamorigRightHandThumb2", "rotation", "y", 0, "+"]);
  animations.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 3.4, "-"]);
  animations.push(["mixamorigRightArm", "rotation", "z", Math.PI / 4.2, "-"]);
  animations.push(["mixamorigRightArm", "rotation", "y", Math.PI / 10, "+"]);
  animations.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 2.5, "+"]);
  animations.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 2, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "x", Math.PI / 6, "+"]);
  animations.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 8, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "z", Math.PI / 10, "+"]);
  animations.push(["mixamorigNeck", "rotation", "x", Math.PI / 16, "+"]);
  animations.push(["mixamorigNeck", "rotation", "y", 0, "+"]);
  ref.animations.push(animations);
  animations = [];
  animations.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 5.2, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 4, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "z", Math.PI / 4, "+"]);
  animations.push(["mixamorigNeck", "rotation", "y", -Math.PI / 7, "-"]);
  ref.animations.push(animations);
  animations = [];
  animations.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 3.2, "+"]);
  animations.push(["mixamorigRightHand", "rotation", "y", 0, "+"]);
  animations.push(["mixamorigRightHand", "rotation", "z", -Math.PI / 8, "-"]);
  animations.push(["mixamorigNeck", "rotation", "y", Math.PI / 7, "+"]);
  ref.animations.push(animations);
  animations = [];
  animations.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 5.2, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 4, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "z", Math.PI / 4, "+"]);
  animations.push(["mixamorigNeck", "rotation", "y", -Math.PI / 8, "-"]);
  ref.animations.push(animations);
  animations = [];
  animations.push(["mixamorigRightHandThumb1", "rotation", "x", 0, "-"]);
  animations.push(["mixamorigRightHandThumb2", "rotation", "y", 0, "+"]);
  animations.push(["mixamorigRightHand", "rotation", "x", 0, "-"]);
  animations.push(["mixamorigRightHand", "rotation", "y", 0, "+"]);
  animations.push(["mixamorigRightHand", "rotation", "z", 0, "-"]);
  animations.push(["mixamorigRightArm", "rotation", "x", 0, "+"]);
  animations.push(["mixamorigRightArm", "rotation", "y", 0, "-"]);
  animations.push(["mixamorigRightArm", "rotation", "z", Math.PI / 3, "+"]);
  animations.push(["mixamorigRightForeArm", "rotation", "z", 0, "-"]);
  animations.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.5, "+"]);
  animations.push(["mixamorigNeck", "rotation", "x", Math.PI / 12, "-"]);
  animations.push(["mixamorigNeck", "rotation", "y", 0, "+"]);
  ref.animations.push(animations);
  if (ref.pending === false && typeof ref.animate === "function") {
    ref.pending = true;
    ref.animate();
  }
};
export {
  ISL_NO
};
