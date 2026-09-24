const defaultPose = (ref) => {
  if (ref.characters) {
    ref.characters.push(" ");
  }
  const animations = [];
  animations.push(["mixamorigNeck", "rotation", "x", Math.PI / 12, "+"]);
  animations.push(["mixamorigLeftArm", "rotation", "z", -Math.PI / 3, "-"]);
  animations.push(["mixamorigLeftForeArm", "rotation", "y", -Math.PI / 1.5, "-"]);
  animations.push(["mixamorigRightArm", "rotation", "z", Math.PI / 3, "+"]);
  animations.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.5, "+"]);
  ref.animations.push(animations);
  if (ref.pending === false && typeof ref.animate === "function") {
    ref.pending = true;
    ref.animate();
  }
};
export {
  defaultPose
};
