# Face Detection Feature — Implementation Plan

## Goal

Add a face detection mode to the webcam app where:

- A **circle overlay** is shown on the webcam feed
- The user's face must be **centered inside the circle**
- Auto-capture when the face is detected, centered, and held still

## Recommended Approach: MediaPipe Face Detection

**Do NOT train a custom model.** Use Google's pre-trained MediaPipe Face Detection which runs in-browser via `@mediapipe/tasks-vision`. It's fast (~5ms on modern hardware), accurate, and provides face bounding boxes + landmarks out of the box.

### Why MediaPipe over YOLO for faces:

- Pre-trained, no dataset or training needed
- Optimized for real-time face detection in browsers
- Provides facial landmarks (useful for centering logic)
- Lightweight (~1MB model vs ~12MB for YOLO)
- Maintained by Google with excellent browser support

## Tasks

### 1. Install MediaPipe

```bash
npm install @mediapipe/tasks-vision
```

### 2. Create face detector utility (`src/utils/faceDetector.js`)

- Load the MediaPipe Face Detection model (BlazeFace short-range)
- Use `FaceDetector` from `@mediapipe/tasks-vision`
- Return face bounding box + keypoints (nose, eyes, ears, mouth)
- API should match yoloDetector pattern: `loadModel()`, `detect(videoElement)`

### 3. Create circle overlay component (`src/components/CircleOverlay.jsx`)

- Render an SVG or canvas circle overlay on top of the webcam feed
- Circle should be centered on the video, sized ~60-70% of the smaller dimension
- Outside the circle should be semi-transparent dark mask
- Circle border changes color based on state:
  - **White/gray** — no face detected
  - **Yellow** — face detected but not centered
  - **Green** — face centered and stable, about to capture

### 4. Implement centering logic (`src/utils/faceCentering.js`)

- Calculate if the face bounding box center is within the circle
- Check that the face fills an appropriate portion of the circle (not too far, not too close)
- Use nose keypoint (landmark index 2) as the primary center reference
- Thresholds:
  - Face center within ~15% of circle center = "centered"
  - Face width is 40-80% of circle diameter = "good distance"
- Provide guidance text: "Move closer", "Move back", "Center your face", "Hold still"

### 5. Create FaceCapture component (`src/components/FaceCapture.jsx`)

- Similar to WebcamCapture.jsx but with circle overlay
- Detection loop using faceDetector at ~100-150ms intervals
- Show guidance text overlay based on centering state
- Stability tracking: face must be centered + still for ~1.5s before auto-capture
- Capture only the circular region (crop to circle on the captured image)

### 6. Update App.jsx

- Add mode toggle: "ID Card" / "Face" capture modes
- Load appropriate model based on selected mode
- Render WebcamCapture or FaceCapture based on mode

### 7. Circular crop for captured image

- When capturing, crop the image to the circle region
- Use canvas clipping with `arc()` to create a circular image
- Save/display as PNG with transparent background outside circle

## File Structure

```
src/
├── utils/
│   ├── yoloDetector.js        # (existing) ID card detection
│   ├── faceDetector.js        # NEW — MediaPipe face detection
│   └── faceCentering.js       # NEW — centering + distance logic
├── components/
│   ├── WebcamCapture.jsx      # (existing) ID card capture
│   ├── FaceCapture.jsx        # NEW — face capture with circle
│   └── CircleOverlay.jsx      # NEW — circle mask overlay
└── App.jsx                    # Updated with mode toggle
```

## Key References

- MediaPipe Face Detection JS: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/web_js
- `@mediapipe/tasks-vision` npm package
- Model: BlazeFace short-range (built into the package, loaded from CDN or bundled)

## Notes

- The existing ID card detection (YOLO) should remain functional alongside face detection
- No Python training pipeline needed for this feature
- The MediaPipe model can be loaded from Google's CDN or bundled locally
- Consider adding `modelType` param to distinguish face vs ID card in the captured result
