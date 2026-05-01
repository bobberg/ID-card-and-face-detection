import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const MIN_DETECTION_CONFIDENCE = 0.5;
const MIN_SUPPRESSION_THRESHOLD = 0.3;

let detector = null;
let detectorLoading = false;
let detectorError = null;

/**
 * Load the MediaPipe BlazeFace short-range model.
 */
export async function loadModel(onProgress) {
  if (detector) return detector;
  if (detectorLoading) {
    while (detectorLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    if (detector) return detector;
    throw detectorError || new Error("Face detector failed to load");
  }

  detectorLoading = true;
  try {
    if (onProgress) onProgress(0.3);

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );

    if (onProgress) onProgress(0.6);

    detector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      minDetectionConfidence: MIN_DETECTION_CONFIDENCE,
      minSuppressionThreshold: MIN_SUPPRESSION_THRESHOLD,
    });

    if (onProgress) onProgress(1);
    console.log("MediaPipe Face Detector loaded successfully");
    return detector;
  } catch (err) {
    detectorError = err;
    throw err;
  } finally {
    detectorLoading = false;
  }
}

/**
 * Run face detection on a video frame.
 * Returns array of { box: {x, y, width, height}, score, keypoints }
 *
 * Keypoints (6 points):
 *   0 - right eye, 1 - left eye, 2 - nose tip,
 *   3 - mouth center, 4 - right ear tragion, 5 - left ear tragion
 */
export function detect(videoElement) {
  if (!detector) throw new Error("Face detector not loaded");

  const result = detector.detectForVideo(videoElement, performance.now());

  return result.detections.map((d) => {
    const bb = d.boundingBox;
    return {
      box: {
        x: bb.originX,
        y: bb.originY,
        width: bb.width,
        height: bb.height,
      },
      score: d.categories[0]?.score ?? 0,
      keypoints: d.keypoints
        ? d.keypoints.map((kp) => ({
            x: kp.x * videoElement.videoWidth,
            y: kp.y * videoElement.videoHeight,
          }))
        : [],
    };
  });
}

/**
 * Check if a detection's bounding box is stable compared to previous.
 */
export function isStable(prev, curr, threshold = 20) {
  if (!prev || !curr) return false;
  return (
    Math.abs(prev.x - curr.x) < threshold &&
    Math.abs(prev.y - curr.y) < threshold &&
    Math.abs(prev.width - curr.width) < threshold &&
    Math.abs(prev.height - curr.height) < threshold
  );
}
