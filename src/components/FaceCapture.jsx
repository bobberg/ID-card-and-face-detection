import { useRef, useEffect, useState, useCallback } from "react";
import { detect, isStable } from "../utils/faceDetector";
import { evaluateCentering } from "../utils/faceCentering";
import { drawCircleOverlay } from "./CircleOverlay";

const STABILITY_DURATION = 1500; // ms face must be centered + still
const DETECT_INTERVAL = 120; // ms between detection runs

export default function FaceCapture({ modelReady, onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const timerRef = useRef(null);
  const stableStartRef = useRef(null);
  const lastBoxRef = useRef(null);
  const capturedRef = useRef(false);

  const [status, setStatus] = useState("starting");
  const [guidance, setGuidance] = useState("");
  const [captured, setCaptured] = useState(null);
  const [progress, setProgress] = useState(0);

  // Start webcam (front-facing for face capture)
  useEffect(() => {
    let stream;
    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 1280, height: 720 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setStatus("searching");
        }
      } catch {
        setStatus("error");
      }
    }
    start();
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const takeSnapshot = useCallback(
    (circle) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      // Draw full frame, then crop to circle
      const size = circle.radius * 2;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");

      // Clip to circle
      ctx.beginPath();
      ctx.arc(circle.radius, circle.radius, circle.radius, 0, Math.PI * 2);
      ctx.closePath();
      ctx.clip();

      // Draw the video region that maps to the circle
      // The overlay circle is positioned in CSS-scaled coordinates,
      // but the video has its own intrinsic resolution.
      const sx = circle.cx - circle.radius;
      const sy = circle.cy - circle.radius;

      ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);

      const dataUrl = canvas.toDataURL("image/png");
      setCaptured(dataUrl);
      setStatus("captured");
      capturedRef.current = true;
      if (onCapture) onCapture(dataUrl);
    },
    [onCapture]
  );

  // Detection loop
  useEffect(() => {
    if (!modelReady || capturedRef.current) return;

    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) return;

    let running = true;
    let detecting = false;

    async function runDetection() {
      if (!running || detecting || video.readyState !== 4) return;
      detecting = true;

      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      const ctx = overlay.getContext("2d");

      try {
        const detections = detect(video);
        if (!running) return;

        // Pick best face
        const best =
          detections.length > 0
            ? detections.reduce((a, b) => (a.score > b.score ? a : b))
            : null;

        // Determine centering status — draw overlay first to get circle info
        // We need circle coords before evaluating centering
        const centeringStatus = best ? "checking" : "no_face";
        const circle = drawCircleOverlay(
          ctx,
          overlay.width,
          overlay.height,
          centeringStatus
        );

        const centering = evaluateCentering(best, circle);

        // Redraw overlay with correct status color
        drawCircleOverlay(ctx, overlay.width, overlay.height, centering.status);

        // Draw face bounding box if detected
        if (best) {
          ctx.strokeStyle =
            centering.status === "centered" ? "#00ff88" : "#ffaa00";
          ctx.lineWidth = 2;
          ctx.strokeRect(
            best.box.x,
            best.box.y,
            best.box.width,
            best.box.height
          );
        }

        setGuidance(centering.guidance);

        if (centering.status === "centered") {
          setStatus("hold_still");

          if (
            best &&
            isStable(lastBoxRef.current, best.box)
          ) {
            if (!stableStartRef.current) {
              stableStartRef.current = Date.now();
            }
            const elapsed = Date.now() - stableStartRef.current;
            const pct = Math.min(elapsed / STABILITY_DURATION, 1);
            setProgress(pct);

            if (elapsed >= STABILITY_DURATION) {
              takeSnapshot(circle);
              running = false;
              return;
            }
          } else {
            stableStartRef.current = null;
            setProgress(0);
          }
        } else {
          stableStartRef.current = null;
          setProgress(0);
          setStatus("searching");
        }

        if (best) lastBoxRef.current = best.box;
      } catch (err) {
        console.warn("Face detection error:", err);
      } finally {
        detecting = false;
      }
    }

    timerRef.current = setInterval(runDetection, DETECT_INTERVAL);

    return () => {
      running = false;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [modelReady, takeSnapshot]);

  function reset() {
    setCaptured(null);
    setStatus("searching");
    setGuidance("");
    setProgress(0);
    stableStartRef.current = null;
    lastBoxRef.current = null;
    capturedRef.current = false;
  }

  const statusMessages = {
    starting: "Starting camera…",
    searching: guidance || "Position your face in the circle",
    hold_still: "Hold still…",
    captured: "Captured!",
    error: "Could not access camera. Please allow camera permissions.",
  };

  return (
    <div className="webcam-capture">
      <div className="status-bar" data-status={status}>
        <span>{statusMessages[status]}</span>
        {status === "hold_still" && (
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        )}
      </div>

      {!captured ? (
        <div className="video-container face-video-container">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={overlayRef} className="overlay-canvas" />
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>
      ) : (
        <div className="capture-result face-capture-result">
          <img src={captured} alt="Captured face" />
          <button onClick={reset}>Retake</button>
        </div>
      )}
    </div>
  );
}
