import { useState, useEffect, useCallback } from "react";
import { loadModel as loadYoloModel } from "./utils/yoloDetector";
import { loadModel as loadFaceModel } from "./utils/faceDetector";
import WebcamCapture from "./components/WebcamCapture";
import FaceCapture from "./components/FaceCapture";
import "./App.css";

function App() {
  const [mode, setMode] = useState("face"); // "id-card" | "face"
  const [modelState, setModelState] = useState({
    ready: false,
    progress: 0,
    error: null,
  });

  const switchMode = useCallback((newMode) => {
    setMode(newMode);
    setModelState({ ready: false, progress: 0, error: null });
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loader =
      mode === "face"
        ? loadFaceModel((fraction) => {
            if (!cancelled) setModelState((s) => ({ ...s, progress: fraction }));
          })
        : loadYoloModel((fraction) => {
            if (!cancelled) setModelState((s) => ({ ...s, progress: fraction }));
          });

    loader
      .then(() => {
        if (!cancelled) setModelState((s) => ({ ...s, ready: true }));
      })
      .catch((err) => {
        if (!cancelled) {
          console.error(`Failed to load ${mode} model:`, err);
          setModelState((s) => ({ ...s, error: err.message }));
        }
      });

    return () => { cancelled = true; };
  }, [mode]);

  function handleCapture(dataUrl, box) {
    console.log(`${mode} captured!`, { box, preview: dataUrl.slice(0, 80) });
  }

  const title = mode === "face" ? "Face Capture" : "ID Card Scanner";

  return (
    <div className="app">
      <div className="mode-toggle">
        <button
          className={mode === "id-card" ? "active" : ""}
          onClick={() => switchMode("id-card")}
        >
          ID Card
        </button>
        <button
          className={mode === "face" ? "active" : ""}
          onClick={() => switchMode("face")}
        >
          Face
        </button>
      </div>
      <h1>{title}</h1>
      {modelState.error ? (
        <div className="loading">
          <p className="error-text">Failed to load model: {modelState.error}</p>
          {mode === "id-card" && (
            <>
              <p>Make sure you have trained and exported the model first.</p>
              <p>
                See <code>training/README.md</code> for instructions.
              </p>
            </>
          )}
        </div>
      ) : !modelState.ready ? (
        <div className="loading">
          <div className="spinner" />
          <p>Loading detection model… {Math.round(modelState.progress * 100)}%</p>
        </div>
      ) : mode === "face" ? (
        <FaceCapture modelReady={modelState.ready} onCapture={handleCapture} />
      ) : (
        <WebcamCapture modelReady={modelState.ready} onCapture={handleCapture} />
      )}
    </div>
  );
}

export default App;
