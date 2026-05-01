import * as tf from "@tensorflow/tfjs";

const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.5;
const MAX_DETECTIONS = 10;

let model = null;
let modelLoading = false;
let modelError = null;

/**
 * Load the YOLOv8 TF.js model from public/model/
 */
export async function loadModel(onProgress) {
  if (model) return model;
  if (modelLoading) {
    // Wait for the existing load to finish
    while (modelLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    if (model) return model;
    throw modelError || new Error("Model failed to load");
  }

  modelLoading = true;
  try {
    model = await tf.loadGraphModel("/model/model.json", {
      onProgress: (fraction) => {
        if (onProgress) onProgress(fraction);
      },
    });
    console.log("YOLOv8 model loaded successfully");
    console.log("Input shape:", model.inputs[0].shape);
    console.log("Output shape:", model.outputs.map((o) => o.shape));

    // Warm up with a dummy inference
    const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    const warmup = model.predict(dummy);
    if (Array.isArray(warmup)) warmup.forEach((t) => t.dispose());
    else warmup.dispose();
    dummy.dispose();

    return model;
  } catch (err) {
    modelError = err;
    throw err;
  } finally {
    modelLoading = false;
  }
}

/**
 * Run detection on a video frame.
 * Returns array of { box: {x, y, width, height}, score, class }
 */
export async function detect(videoElement) {
  if (!model) throw new Error("Model not loaded");

  const detections = tf.tidy(() => {
    // Read frame and preprocess
    const frame = tf.browser.fromPixels(videoElement);
    const [origH, origW] = [frame.shape[0], frame.shape[1]];

    // Resize to model input size
    const resized = tf.image.resizeBilinear(frame, [INPUT_SIZE, INPUT_SIZE]);

    // Normalize to [0, 1] and add batch dimension
    const normalized = resized.div(255.0);
    const batched = normalized.expandDims(0); // [1, 640, 640, 3]

    // Run inference
    const output = model.predict(batched);

    // YOLOv8 TF.js output: [1, num_values, num_detections]
    // where num_values = 4 (bbox) + num_classes
    // Transpose to [1, num_detections, num_values]
    let predictions;
    if (Array.isArray(output)) {
      predictions = output[0];
    } else {
      predictions = output;
    }

    // Shape could be [1, 5, 8400] or [1, 8400, 5] depending on export
    const shape = predictions.shape;
    let transposed;
    if (shape[1] < shape[2]) {
      // [1, 5, 8400] -> need to transpose to [1, 8400, 5]
      transposed = predictions.transpose([0, 2, 1]);
    } else {
      transposed = predictions;
    }

    return {
      data: transposed.squeeze(0), // [8400, 5]
      origW,
      origH,
    };
  });

  // Process detections outside tf.tidy to handle async
  const rawData = await detections.data.data();
  const numDetections = detections.data.shape[0];
  const numValues = detections.data.shape[1];
  const origW = detections.origW;
  const origH = detections.origH;
  detections.data.dispose();

  // Parse detections
  const boxes = [];
  const scores = [];

  for (let i = 0; i < numDetections; i++) {
    const offset = i * numValues;

    // YOLOv8 format: cx, cy, w, h, class_scores...
    const cx = rawData[offset];
    const cy = rawData[offset + 1];
    const w = rawData[offset + 2];
    const h = rawData[offset + 3];

    // For single class, just one score at index 4
    const score = numValues > 5
      ? Math.max(...Array.from(rawData.slice(offset + 4, offset + numValues)))
      : rawData[offset + 4];

    if (score < CONFIDENCE_THRESHOLD) continue;

    // Convert from model coordinates (0-640) to original image coordinates
    const x1 = ((cx - w / 2) / INPUT_SIZE) * origW;
    const y1 = ((cy - h / 2) / INPUT_SIZE) * origH;
    const bw = (w / INPUT_SIZE) * origW;
    const bh = (h / INPUT_SIZE) * origH;

    boxes.push({ x: x1, y: y1, width: bw, height: bh });
    scores.push(score);
  }

  // Apply NMS
  if (boxes.length === 0) return [];

  const boxesTensor = tf.tensor2d(
    boxes.map((b) => [b.y, b.x, b.y + b.height, b.x + b.width])
  );
  const scoresTensor = tf.tensor1d(scores);

  const nmsIndices = await tf.image.nonMaxSuppressionAsync(
    boxesTensor,
    scoresTensor,
    MAX_DETECTIONS,
    IOU_THRESHOLD,
    CONFIDENCE_THRESHOLD
  );

  const selectedIndices = await nmsIndices.data();
  nmsIndices.dispose();
  boxesTensor.dispose();
  scoresTensor.dispose();

  return Array.from(selectedIndices).map((idx) => ({
    box: boxes[idx],
    score: scores[idx],
    class: "id-card",
  }));
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
