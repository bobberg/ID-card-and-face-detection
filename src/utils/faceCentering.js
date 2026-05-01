/**
 * Face centering logic — determines if the face is well-positioned
 * inside the circle overlay for capture.
 *
 * Returns { status, guidance } where status is one of:
 *   "no_face" | "not_centered" | "too_far" | "too_close" | "centered"
 */

/**
 * @param {object} face - Detection from faceDetector: { box, score, keypoints }
 * @param {object} circle - { cx, cy, radius } in video pixel coordinates
 * @returns {{ status: string, guidance: string }}
 */
export function evaluateCentering(face, circle) {
  if (!face) {
    return { status: "no_face", guidance: "Position your face in the circle" };
  }

  const { box, keypoints } = face;

  // Use nose keypoint (index 2) as face center if available, else bbox center
  let faceCX, faceCY;
  if (keypoints && keypoints.length > 2) {
    faceCX = keypoints[2].x;
    faceCY = keypoints[2].y;
  } else {
    faceCX = box.x + box.width / 2;
    faceCY = box.y + box.height / 2;
  }

  // Check centering: face center within 15% of circle radius from circle center
  const dx = faceCX - circle.cx;
  const dy = faceCY - circle.cy;
  const distFromCenter = Math.sqrt(dx * dx + dy * dy);
  const centerThreshold = circle.radius * 0.15;
  const isCentered = distFromCenter < centerThreshold;

  // Check distance/size: face width should be 40-80% of circle diameter
  const faceSize = Math.max(box.width, box.height);
  const circleDiameter = circle.radius * 2;
  const sizeRatio = faceSize / circleDiameter;

  if (sizeRatio < 0.35) {
    return { status: "too_far", guidance: "Move closer" };
  }
  if (sizeRatio > 0.85) {
    return { status: "too_close", guidance: "Move back" };
  }
  if (!isCentered) {
    return { status: "not_centered", guidance: "Center your face" };
  }

  return { status: "centered", guidance: "Hold still…" };
}
