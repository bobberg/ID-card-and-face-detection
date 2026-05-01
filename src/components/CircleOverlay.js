/**
 * CircleOverlay — renders a circle mask on a canvas overlay.
 * Outside the circle is a semi-transparent dark mask.
 * Circle border color changes based on detection state.
 */
export default function CircleOverlay({ canvasRef, width, height, status }) {
  // Drawing is imperative via the parent passing canvasRef
  // This component just holds the canvas element
  return null;
}

/**
 * Draw the circle overlay onto a canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} width - canvas width
 * @param {number} height - canvas height
 * @param {string} status - "no_face" | "not_centered" | "too_far" | "too_close" | "centered"
 * @returns {{ cx, cy, radius }} circle info for centering calculations
 */
export function drawCircleOverlay(ctx, width, height, status) {
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.33;

  ctx.clearRect(0, 0, width, height);

  // Draw semi-transparent dark mask with circle cutout
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.beginPath();
  ctx.rect(0, 0, width, height);
  ctx.arc(cx, cy, radius, 0, Math.PI * 2, true); // counter-clockwise = cutout
  ctx.fill();
  ctx.restore();

  // Draw circle border
  const borderColors = {
    no_face: "rgba(255, 255, 255, 0.5)",
    not_centered: "#ffaa00",
    too_far: "#ffaa00",
    too_close: "#ffaa00",
    centered: "#00ff88",
  };

  ctx.save();
  ctx.strokeStyle = borderColors[status] || "rgba(255, 255, 255, 0.5)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  return { cx, cy, radius };
}
