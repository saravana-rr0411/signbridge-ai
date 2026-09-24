const ASL_GOOD_MORNING = (ref) => {
  const compoundSteps = [];
  const animations = [];
  let step = [];
  step.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 3, "-"]);
  step.push(["mixamorigRightArm", "rotation", "z", Math.PI / 5, "-"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 2.8, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 2.2, "-"]);
  step.push(["mixamorigRightHand", "rotation", "x", Math.PI / 5, "+"]);
  step.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 6, "-"]);
  step.push(["mixamorigRightHand", "rotation", "z", Math.PI / 8, "+"]);
  step.push(["mixamorigLeftArm", "rotation", "x", -Math.PI / 6, "-"]);
  step.push(["mixamorigLeftArm", "rotation", "z", -Math.PI / 4, "+"]);
  step.push(["mixamorigLeftForeArm", "rotation", "z", -Math.PI / 4, "-"]);
  step.push(["mixamorigLeftForeArm", "rotation", "y", -Math.PI / 1.8, "+"]);
  step.push(["mixamorigLeftHand", "rotation", "x", -Math.PI / 8, "-"]);
  step.push(["mixamorigLeftHand", "rotation", "z", -Math.PI / 6, "-"]);
  step.push(["mixamorigNeck", "rotation", "x", Math.PI / 10, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "GOOD", token: "GOOD", isWordEnd: false, bones: step });
  step = [];
  step.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 4.5, "+"]);
  step.push(["mixamorigRightArm", "rotation", "z", Math.PI / 4, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 4, "-"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.8, "-"]);
  step.push(["mixamorigRightHand", "rotation", "x", Math.PI / 8, "-"]);
  step.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 8, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "GOOD", token: "GOOD", isWordEnd: false, bones: step });
  step = [];
  step.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 5.5, "+"]);
  step.push(["mixamorigRightArm", "rotation", "z", Math.PI / 3.8, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 5, "-"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.6, "-"]);
  step.push(["mixamorigRightHand", "rotation", "x", 0, "-"]);
  step.push(["mixamorigRightHand", "rotation", "y", 0, "+"]);
  step.push(["mixamorigRightHand", "rotation", "z", 0, "-"]);
  step.push(["mixamorigNeck", "rotation", "x", Math.PI / 7, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "GOOD", token: "GOOD", isWordEnd: true, bones: step });
  step = [];
  step.push(["mixamorigNeck", "rotation", "x", Math.PI / 12, "-"]);
  step.push(["mixamorigLeftArm", "rotation", "x", -Math.PI / 4, "-"]);
  step.push(["mixamorigLeftArm", "rotation", "z", -Math.PI / 3.5, "+"]);
  step.push(["mixamorigLeftForeArm", "rotation", "z", -Math.PI / 3, "-"]);
  step.push(["mixamorigLeftForeArm", "rotation", "y", -Math.PI / 2, "+"]);
  step.push(["mixamorigLeftHand", "rotation", "x", 0, "+"]);
  step.push(["mixamorigLeftHand", "rotation", "y", -Math.PI / 6, "-"]);
  step.push(["mixamorigLeftHand", "rotation", "z", -Math.PI / 6, "-"]);
  step.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 5, "-"]);
  step.push(["mixamorigRightArm", "rotation", "z", Math.PI / 4, "-"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 4.5, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.5, "+"]);
  step.push(["mixamorigRightHand", "rotation", "x", Math.PI / 8, "+"]);
  step.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 6, "-"]);
  step.push(["mixamorigRightHand", "rotation", "z", 0, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "MORNING", token: "MORNING", isWordEnd: false, bones: step });
  step = [];
  step.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 4.5, "-"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 3.5, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.8, "-"]);
  step.push(["mixamorigRightHand", "rotation", "x", Math.PI / 6, "+"]);
  step.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 8, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "MORNING", token: "MORNING", isWordEnd: false, bones: step });
  step = [];
  step.push(["mixamorigRightArm", "rotation", "x", -Math.PI / 4, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", Math.PI / 2.6, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 2.2, "+"]);
  step.push(["mixamorigRightHand", "rotation", "x", Math.PI / 5, "+"]);
  step.push(["mixamorigRightHand", "rotation", "y", -Math.PI / 6, "-"]);
  step.push(["mixamorigNeck", "rotation", "x", Math.PI / 10, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "MORNING", token: "MORNING", isWordEnd: false, bones: step });
  step = [];
  step.push(["mixamorigRightHand", "rotation", "x", Math.PI / 5, "+"]);
  animations.push(step);
  compoundSteps.push({ word: "MORNING", token: "MORNING", isWordEnd: true, bones: step });
  step = [];
  step.push(["mixamorigLeftArm", "rotation", "x", 0, "+"]);
  step.push(["mixamorigLeftArm", "rotation", "z", -Math.PI / 3, "-"]);
  step.push(["mixamorigLeftForeArm", "rotation", "z", 0, "+"]);
  step.push(["mixamorigLeftForeArm", "rotation", "y", -Math.PI / 1.5, "-"]);
  step.push(["mixamorigLeftHand", "rotation", "x", 0, "+"]);
  step.push(["mixamorigLeftHand", "rotation", "y", 0, "+"]);
  step.push(["mixamorigLeftHand", "rotation", "z", 0, "+"]);
  step.push(["mixamorigRightArm", "rotation", "x", 0, "+"]);
  step.push(["mixamorigRightArm", "rotation", "z", Math.PI / 3, "+"]);
  step.push(["mixamorigRightForeArm", "rotation", "z", 0, "-"]);
  step.push(["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.5, "-"]);
  step.push(["mixamorigRightHand", "rotation", "x", 0, "-"]);
  step.push(["mixamorigRightHand", "rotation", "y", 0, "+"]);
  step.push(["mixamorigRightHand", "rotation", "z", 0, "-"]);
  step.push(["mixamorigNeck", "rotation", "x", Math.PI / 12, "-"]);
  animations.push(step);
  compoundSteps.push({
    word: "MORNING",
    token: "MORNING",
    isWordEnd: true,
    isReturnToRest: true,
    bones: step
  });
  ref.animations = animations;
  ref.compoundSteps = compoundSteps;
  if (ref.pending === false && typeof ref.animate === "function") {
    ref.pending = true;
    ref.animate();
  }
};
export {
  ASL_GOOD_MORNING
};
