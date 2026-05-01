import { useRef, useEffect, useState, useCallback } from "react";
import { detect, isStable } from "../utils/yoloDetector";

const STABILITY_DURATION = 1000; // ms the card must be held still
const DETECT_INTERVAL = 150; // ms between detection runs (model inference is async)

export default function WebcamCapture({ modelReady, onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const timerRef = useRef(null);
  const stableStartRef = useRef(null);
  const lastBoxRef = useRef(null);
  const capturedRef = useRef(false);

  const [status, setStatus] = useState("starting");
  const [captured, setCaptured] = useState(null);
  const [progress, setProgress] = useState(0);

  // Start webcam
  useEffect(() => {
    let stream;
    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: 1280, height: 720 },
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
    (box) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      const dataUrl = canvas.toDataURL("image/png");
      setCaptured(dataUrl);
      setStatus("captured");
      capturedRef.current = true;
      if (onCapture) onCapture(dataUrl, box);
    },
    [onCapture]
  );

  // Detection loop using setInterval (model inference is async)
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
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      try {
        const detections = await detect(video);
        if (!running) return;

        if (detections.length > 0) {
          // Use the highest-confidence detection
          const best = detections.reduce((a, b) =>
            a.score > b.score ? a : b
          );
          const { box, score } = best;

          // Draw bounding box
          ctx.strokeStyle = "#00ff88";
          ctx.lineWidth = 3;
          ctx.strokeRect(box.x, box.y, box.width, box.height);
          ctx.fillStyle = "rgba(0, 255, 136, 0.1)";
          ctx.fillRect(box.x, box.y, box.width, box.height);

          // Draw confidence label
          ctx.fillStyle = "#00ff88";
          ctx.font = "16px system-ui";
          ctx.fillText(
            `ID Card ${Math.round(score * 100)}%`,
            box.x,
            box.y > 24 ? box.y - 8 : box.y + box.height + 20
          );

          // Check stability
          if (isStable(lastBoxRef.current, box)) {
            if (!stableStartRef.current) {
              stableStartRef.current = Date.now();
            }
            const elapsed = Date.now() - stableStartRef.current;
            const pct = Math.min(elapsed / STABILITY_DURATION, 1);
            setProgress(pct);
            setStatus("hold_still");

            if (elapsed >= STABILITY_DURATION) {
              takeSnapshot(box);
              running = false;
              return;
            }
          } else {
            stableStartRef.current = null;
            setProgress(0);
            setStatus("searching");
          }
          lastBoxRef.current = box;
        } else {
          stableStartRef.current = null;
          lastBoxRef.current = null;
          setProgress(0);
          setStatus("searching");
        }
      } catch (err) {
        console.warn("Detection error:", err);
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
    setProgress(0);
    stableStartRef.current = null;
    lastBoxRef.current = null;
    capturedRef.current = false;
  }

  const statusMessages = {
    starting: "Starting camera…",
    searching: "Show your ID card to the camera",
    hold_still: "Card detected — hold still…",
    captured: "Captured!",
    error: "Could not access camera. Please allow camera permissions.",
  };

  return (
    <div className="webcam-capture">
      <div className="status-bar" data-status={status}>
        <span>{statusMessages[status]}</span>
        {status === "hold_still" && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
          </div>
        )}
      </div>

      {!captured ? (
        <div className="video-container">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={overlayRef} className="overlay-canvas" />
          {/* hidden canvas used for snapshot */}
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>
      ) : (
        <div className="capture-result">
          <img src={captured} alt="Captured ID card" />
          <button onClick={reset}>Retake</button>
        </div>
      )}
    </div>
  );
}
