import { useState, useEffect, useRef, useCallback } from "react";

// ─── Design tokens ────────────────────────────────────────────────────────────
const C = {
  bg: "#0A0F1E",
  bgCard: "#0F1628",
  bgInput: "#151C30",
  border: "#1E2A45",
  borderHov: "#2A3A5C",
  teal: "#00D4AA",
  tealDim: "#00D4AA22",
  tealMid: "#00D4AA66",
  amber: "#F5A623",
  amberDim: "#F5A62322",
  coral: "#FF6B6B",
  coralDim: "#FF6B6B22",
  purple: "#7C6FF7",
  purpleDim: "#7C6FF722",
  textPrim: "#E8EDF5",
  textSec: "#7A8BAA",
  textMut: "#3D4E6B",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
const glucoseColor = (g) => {
  if (g < 54) return C.coral;
  if (g < 70) return C.amber;
  if (g <= 180) return C.teal;
  if (g <= 250) return C.amber;
  return C.coral;
};
const glucoseLabel = (g) => {
  if (g < 54) return "CRITICAL LOW";
  if (g < 70) return "LOW";
  if (g <= 180) return "IN RANGE";
  if (g <= 250) return "HIGH";
  return "CRITICAL HIGH";
};
const trendArrow = (vals) => {
  if (vals.length < 2) return "→";
  const d =
    vals[vals.length - 1] - vals[vals.length - 3 < 0 ? 0 : vals.length - 3];
  if (d > 4) return "↑↑";
  if (d > 1) return "↑";
  if (d < -4) return "↓↓";
  if (d < -1) return "↓";
  return "→";
};
const actionColor = (id) => [C.teal, C.amber, C.coral, C.purple][id] ?? C.teal;
const actionIcon = (id) => ["✓", "💉", "💉💉", "🍬"][id] ?? "✓";

// ─── Default glucose history (realistic post-breakfast curve) ─────────────────
const DEFAULT_HISTORY = [
  118, 121, 126, 134, 142, 155, 162, 158, 152, 145, 139, 133,
];
const DEFAULT_HOUR = new Date().getHours() + new Date().getMinutes() / 60;

// ─── Glucose Sparkline Canvas ─────────────────────────────────────────────────
function GlucoseCanvas({ history, predicted, width, height }) {
  const ref = useRef();
  const animRef = useRef(0);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const all = predicted ? [...history, predicted] : history;
    const minV = Math.min(...all, 70) - 10;
    const maxV = Math.max(...all, 180) + 10;
    const pad = { l: 44, r: 28, t: 20, b: 32 };
    const cw = width - pad.l - pad.r;
    const ch = height - pad.t - pad.b;
    const toX = (i, total) => pad.l + (i / (total - 1)) * cw;
    const toY = (v) => pad.t + ch - ((v - minV) / (maxV - minV)) * ch;

    let frame = 0;
    const totalFrames = 48;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Grid lines + labels
      [54, 70, 140, 180, 250].forEach((v) => {
        if (v < minV || v > maxV) return;
        const y = toY(v);
        ctx.strokeStyle = v === 70 || v === 180 ? "#1E2A4566" : "#1E2A4533";
        ctx.lineWidth = 1;
        ctx.setLineDash(v === 70 || v === 180 ? [4, 4] : []);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + cw, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = v === 70 || v === 180 ? C.textSec : C.textMut;
        ctx.font = `10px 'DM Mono', monospace`;
        ctx.textAlign = "right";
        ctx.fillText(v, pad.l - 6, y + 4);
      });

      // Target range shading
      ctx.fillStyle = "#00D4AA09";
      ctx.fillRect(pad.l, toY(180), cw, toY(70) - toY(180));

      // Time labels
      const totalPts = predicted ? history.length + 1 : history.length;
      ctx.fillStyle = C.textMut;
      ctx.font = `9px 'DM Mono', monospace`;
      ctx.textAlign = "center";
      [-60, -30, 0, ...(predicted ? [5] : [])].forEach((mins, idx) => {
        const i = idx;
        const x = toX(
          i *
            (history.length / ([-60, -30, 0].length - 1 + (predicted ? 1 : 0))),
          totalPts,
        );
        ctx.fillText(
          mins === 5 ? "+5m" : `${mins}m`,
          pad.l + (idx / (totalPts - 1)) * cw,
          height - 4,
        );
      });

      // Animated progress
      const progress = Math.min(1, frame / totalFrames);
      const drawCount = Math.max(2, Math.floor(progress * history.length));

      // Gradient line for history
      const grad = ctx.createLinearGradient(pad.l, 0, pad.l + cw, 0);
      grad.addColorStop(0, "#00D4AA44");
      grad.addColorStop(0.7, "#00D4AA99");
      grad.addColorStop(1, "#00D4AADD");

      ctx.beginPath();
      history.slice(0, drawCount).forEach((v, i) => {
        const x = toX(i, predicted ? history.length + 1 : history.length);
        const y = toY(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2.5;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();

      // Area fill
      if (drawCount > 1) {
        ctx.beginPath();
        history.slice(0, drawCount).forEach((v, i) => {
          const x = toX(i, predicted ? history.length + 1 : history.length);
          const y = toY(v);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        const lastX = toX(
          drawCount - 1,
          predicted ? history.length + 1 : history.length,
        );
        ctx.lineTo(lastX, toY(minV));
        ctx.lineTo(
          toX(0, predicted ? history.length + 1 : history.length),
          toY(minV),
        );
        ctx.closePath();
        const areaGrad = ctx.createLinearGradient(0, toY(maxV), 0, toY(minV));
        areaGrad.addColorStop(0, "#00D4AA18");
        areaGrad.addColorStop(1, "#00D4AA00");
        ctx.fillStyle = areaGrad;
        ctx.fill();
      }

      // Predicted point
      if (predicted && progress === 1) {
        const px = toX(history.length, history.length + 1);
        const py = toY(predicted);
        const lastX = toX(history.length - 1, history.length + 1);
        const lastY = toY(history[history.length - 1]);

        // Dashed connector
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = glucoseColor(predicted) + "88";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(px, py);
        ctx.stroke();
        ctx.setLineDash([]);

        // Glow ring
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 500);
        ctx.beginPath();
        ctx.arc(px, py, 10 + pulse * 4, 0, Math.PI * 2);
        ctx.fillStyle = glucoseColor(predicted) + "18";
        ctx.fill();

        // Dot
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = glucoseColor(predicted);
        ctx.fill();
        ctx.strokeStyle = C.bgCard;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Current reading dot (last history point)
      if (drawCount === history.length) {
        const cx2 = toX(
          history.length - 1,
          predicted ? history.length + 1 : history.length,
        );
        const cy2 = toY(history[history.length - 1]);
        ctx.beginPath();
        ctx.arc(cx2, cy2, 5, 0, Math.PI * 2);
        ctx.fillStyle = glucoseColor(history[history.length - 1]);
        ctx.fill();
      }

      frame = Math.min(frame + 2, totalFrames);
      if (frame < totalFrames || predicted) {
        animRef.current = requestAnimationFrame(draw);
      }
    };

    cancelAnimationFrame(animRef.current);
    frame = 0;
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [history, predicted, width, height]);

  return <canvas ref={ref} style={{ width, height, display: "block" }} />;
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [history, setHistory] = useState(DEFAULT_HISTORY);
  const [historyInput, setHistoryInput] = useState(DEFAULT_HISTORY.join(", "));
  const [hour, setHour] = useState(DEFAULT_HOUR);
  const [prediction, setPrediction] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const canvasContainerRef = useRef();
  const [canvasW, setCanvasW] = useState(600);
  const CANVAS_H = 220;

  // Responsive canvas width
  useEffect(() => {
    const obs = new ResizeObserver((entries) => {
      setCanvasW(Math.floor(entries[0].contentRect.width));
    });
    if (canvasContainerRef.current) obs.observe(canvasContainerRef.current);
    return () => obs.disconnect();
  }, []);

  // Health check on mount
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  const parseHistory = useCallback((str) => {
    return str
      .split(/[,\s]+/)
      .map(Number)
      .filter((n) => !isNaN(n) && n > 0);
  }, []);

  const handleHistoryChange = (val) => {
    setHistoryInput(val);
    const parsed = parseHistory(val);
    if (parsed.length === 12) setHistory(parsed);
  };

  const runAnalysis = async () => {
    const parsed = parseHistory(historyInput);
    if (parsed.length !== 12) {
      setError("Please enter exactly 12 glucose readings separated by commas.");
      return;
    }
    setError(null);
    setLoading(true);
    setPrediction(null);
    setRecommendation(null);

    try {
      const current = parsed[parsed.length - 1];
      const trend =
        parsed[parsed.length - 1] -
        parsed[parsed.length - 3 < 0 ? 0 : parsed.length - 3];

      const [predRes, recRes] = await Promise.all([
        fetch("http://localhost:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ glucose_history: parsed, current_hour: hour }),
        }),
        fetch("http://localhost:8000/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            glucose_mgdl: current,
            glucose_trend: trend,
            current_hour: hour,
            insulin_on_board: 0.0,
          }),
        }),
      ]);

      const [pred, rec] = await Promise.all([predRes.json(), recRes.json()]);
      setPrediction(pred);
      setRecommendation(rec);
      setHistory(parsed);
    } catch (e) {
      setError(
        "Could not reach the API. Make sure uvicorn is running on port 8000.",
      );
    } finally {
      setLoading(false);
    }
  };

  const current = history[history.length - 1];
  const arrow = trendArrow(history);
  const gColor = glucoseColor(current);
  const gLabel = glucoseLabel(current);
  const recColor = recommendation
    ? actionColor(recommendation.action_id)
    : C.teal;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        color: C.textPrim,
        fontFamily: "'Inter', sans-serif",
      }}
    >
      {/* Top nav */}
      <nav
        style={{
          borderBottom: `1px solid ${C.border}`,
          padding: "0 32px",
          height: 56,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: C.teal,
              boxShadow: `0 0 8px ${C.teal}`,
            }}
          />
          <span
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 13,
              letterSpacing: "0.12em",
              color: C.textSec,
              textTransform: "uppercase",
            }}
          >
            Glucose AI
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background:
                health?.status === "healthy"
                  ? C.teal
                  : health?.status === "unreachable"
                    ? C.coral
                    : C.amber,
            }}
          />
          <span
            style={{
              fontSize: 11,
              color: C.textSec,
              fontFamily: "'DM Mono', monospace",
            }}
          >
            {health?.status === "healthy"
              ? "MODELS READY"
              : health?.status === "unreachable"
                ? "API OFFLINE"
                : "DEGRADED"}
          </span>
        </div>
      </nav>

      {/* Main layout */}
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "32px 24px",
          display: "grid",
          gridTemplateColumns: "1fr 380px",
          gap: 24,
        }}
      >
        {/* LEFT: Glucose monitor */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Big reading */}
          <div
            style={{
              background: C.bgCard,
              border: `1px solid ${C.border}`,
              borderRadius: 16,
              padding: "28px 32px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: 20,
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 11,
                    letterSpacing: "0.14em",
                    color: C.textSec,
                    textTransform: "uppercase",
                    fontFamily: "'DM Mono', monospace",
                    marginBottom: 6,
                  }}
                >
                  Current glucose
                </div>
                <div
                  style={{ display: "flex", alignItems: "flex-end", gap: 12 }}
                >
                  <span
                    style={{
                      fontSize: 72,
                      fontWeight: 700,
                      fontFamily: "'DM Mono', monospace",
                      color: gColor,
                      lineHeight: 1,
                      letterSpacing: "-2px",
                    }}
                  >
                    {current}
                  </span>
                  <div style={{ paddingBottom: 10 }}>
                    <div
                      style={{
                        fontSize: 14,
                        color: C.textSec,
                        fontFamily: "'DM Mono', monospace",
                      }}
                    >
                      mg/dL
                    </div>
                    <div style={{ fontSize: 22, color: gColor }}>{arrow}</div>
                  </div>
                </div>
                <div
                  style={{
                    marginTop: 8,
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 6,
                    background: gColor + "18",
                    border: `1px solid ${gColor}44`,
                    borderRadius: 6,
                    padding: "3px 10px",
                  }}
                >
                  <div
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: "50%",
                      background: gColor,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 10,
                      fontFamily: "'DM Mono', monospace",
                      color: gColor,
                      letterSpacing: "0.12em",
                    }}
                  >
                    {gLabel}
                  </span>
                </div>
              </div>

              {prediction && (
                <div style={{ textAlign: "right" }}>
                  <div
                    style={{
                      fontSize: 10,
                      letterSpacing: "0.12em",
                      color: C.textMut,
                      textTransform: "uppercase",
                      fontFamily: "'DM Mono', monospace",
                      marginBottom: 4,
                    }}
                  >
                    Forecast +5 min
                  </div>
                  <div
                    style={{
                      fontSize: 36,
                      fontWeight: 700,
                      fontFamily: "'DM Mono', monospace",
                      color: glucoseColor(prediction.predicted_glucose),
                    }}
                  >
                    {prediction.predicted_glucose}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: C.textMut,
                      fontFamily: "'DM Mono', monospace",
                    }}
                  >
                    ± {prediction.confidence_range} mg/dL
                  </div>
                  <div
                    style={{
                      marginTop: 6,
                      fontSize: 10,
                      color: glucoseColor(prediction.predicted_glucose),
                      background:
                        glucoseColor(prediction.predicted_glucose) + "18",
                      padding: "2px 8px",
                      borderRadius: 4,
                      fontFamily: "'DM Mono', monospace",
                      letterSpacing: "0.08em",
                    }}
                  >
                    {glucoseLabel(prediction.predicted_glucose)}
                  </div>
                </div>
              )}
            </div>

            {/* Canvas */}
            <div
              ref={canvasContainerRef}
              style={{
                width: "100%",
                borderRadius: 10,
                overflow: "hidden",
                background: "#0A0F1E",
                border: `1px solid ${C.border}`,
              }}
            >
              <GlucoseCanvas
                history={history}
                predicted={prediction?.predicted_glucose}
                width={canvasW || 600}
                height={CANVAS_H}
              />
            </div>

            {/* Legend */}
            <div
              style={{
                marginTop: 10,
                display: "flex",
                gap: 20,
                flexWrap: "wrap",
              }}
            >
              {[
                ["History (1hr)", C.teal],
                ["Target range", "#00D4AA33"],
                ["Forecast", "white"],
              ].map(([l, c]) => (
                <div
                  key={l}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 10,
                    color: C.textMut,
                    fontFamily: "'DM Mono', monospace",
                  }}
                >
                  <div
                    style={{
                      width: 20,
                      height: 2,
                      background: c,
                      borderRadius: 1,
                    }}
                  />
                  {l}
                </div>
              ))}
            </div>
          </div>

          {/* Recommendation card */}
          {recommendation && (
            <div
              style={{
                background: C.bgCard,
                border: `1px solid ${recColor}44`,
                borderRadius: 16,
                padding: "24px 28px",
                position: "relative",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 3,
                  background: `linear-gradient(90deg, ${recColor}, ${recColor}44)`,
                  borderRadius: "16px 16px 0 0",
                }}
              />
              <div
                style={{
                  marginBottom: 16,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <div
                    style={{
                      fontSize: 10,
                      letterSpacing: "0.14em",
                      color: C.textMut,
                      textTransform: "uppercase",
                      fontFamily: "'DM Mono', monospace",
                      marginBottom: 6,
                    }}
                  >
                    Recommendation
                  </div>
                  <div
                    style={{ fontSize: 22, fontWeight: 700, color: recColor }}
                  >
                    {actionIcon(recommendation.action_id)}{" "}
                    {recommendation.action_name}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div
                    style={{
                      fontSize: 9,
                      letterSpacing: "0.12em",
                      color: C.textMut,
                      textTransform: "uppercase",
                      fontFamily: "'DM Mono', monospace",
                      marginBottom: 4,
                    }}
                  >
                    URGENCY
                  </div>
                  <div
                    style={{
                      padding: "4px 12px",
                      borderRadius: 6,
                      background: recColor + "22",
                      border: `1px solid ${recColor}44`,
                      fontSize: 11,
                      color: recColor,
                      fontFamily: "'DM Mono', monospace",
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                    }}
                  >
                    {recommendation.urgency}
                  </div>
                </div>
              </div>
              <p
                style={{
                  fontSize: 14,
                  color: C.textSec,
                  lineHeight: 1.65,
                  maxWidth: 560,
                }}
              >
                {recommendation.reasoning}
              </p>
            </div>
          )}
        </div>

        {/* RIGHT: Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Glucose input */}
          <div
            style={{
              background: C.bgCard,
              border: `1px solid ${C.border}`,
              borderRadius: 16,
              padding: "24px 24px",
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.14em",
                color: C.textSec,
                textTransform: "uppercase",
                fontFamily: "'DM Mono', monospace",
                marginBottom: 12,
              }}
            >
              Last 12 readings (5-min intervals)
            </div>

            <textarea
              value={historyInput}
              onChange={(e) => handleHistoryChange(e.target.value)}
              rows={4}
              style={{
                width: "100%",
                background: C.bgInput,
                border: `1px solid ${C.border}`,
                borderRadius: 10,
                padding: "12px 14px",
                color: C.textPrim,
                fontFamily: "'DM Mono', monospace",
                fontSize: 13,
                lineHeight: 1.7,
                resize: "vertical",
                outline: "none",
                boxSizing: "border-box",
                transition: "border-color 0.15s",
              }}
              onFocus={(e) => (e.target.style.borderColor = C.tealMid)}
              onBlur={(e) => (e.target.style.borderColor = C.border)}
              placeholder="118, 121, 126, 134, 142, 155, 162, 158, 152, 145, 139, 133"
            />

            {/* Mini bar visualisation */}
            {history.length === 12 && (
              <div
                style={{
                  marginTop: 10,
                  display: "flex",
                  gap: 3,
                  height: 32,
                  alignItems: "flex-end",
                }}
              >
                {history.map((v, i) => {
                  const h = Math.max(4, ((v - 60) / (280 - 60)) * 32);
                  const c = glucoseColor(v);
                  return (
                    <div
                      key={i}
                      title={`${v} mg/dL`}
                      style={{
                        flex: 1,
                        height: h,
                        background: c + "66",
                        borderRadius: "2px 2px 0 0",
                        border: `1px solid ${c}44`,
                        position: "relative",
                        transition: "height 0.3s",
                      }}
                    >
                      {i === history.length - 1 && (
                        <div
                          style={{
                            position: "absolute",
                            top: -14,
                            left: "50%",
                            transform: "translateX(-50%)",
                            fontSize: 8,
                            color: c,
                            fontFamily: "'DM Mono', monospace",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {v}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {parseHistory(historyInput).length !== 12 &&
              historyInput.length > 0 && (
                <div
                  style={{
                    marginTop: 8,
                    fontSize: 11,
                    color: C.amber,
                    fontFamily: "'DM Mono', monospace",
                  }}
                >
                  {parseHistory(historyInput).length}/12 readings entered
                </div>
              )}
          </div>

          {/* Time input */}
          <div
            style={{
              background: C.bgCard,
              border: `1px solid ${C.border}`,
              borderRadius: 16,
              padding: "20px 24px",
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.14em",
                color: C.textSec,
                textTransform: "uppercase",
                fontFamily: "'DM Mono', monospace",
                marginBottom: 12,
              }}
            >
              Current time
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <input
                type="range"
                min={0}
                max={23.9}
                step={0.25}
                value={hour}
                onChange={(e) => setHour(parseFloat(e.target.value))}
                style={{ flex: 1, accentColor: C.teal }}
              />
              <span
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: 16,
                  color: C.teal,
                  minWidth: 48,
                  textAlign: "right",
                }}
              >
                {String(Math.floor(hour)).padStart(2, "0")}:
                {String(Math.round((hour % 1) * 60)).padStart(2, "0")}
              </span>
            </div>
            <div
              style={{
                marginTop: 8,
                display: "flex",
                justifyContent: "space-between",
                fontSize: 9,
                color: C.textMut,
                fontFamily: "'DM Mono', monospace",
              }}
            >
              {["00:00", "06:00", "12:00", "18:00", "23:45"].map((t) => (
                <span key={t}>{t}</span>
              ))}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div
              style={{
                background: C.coralDim,
                border: `1px solid ${C.coral}44`,
                borderRadius: 10,
                padding: "12px 16px",
                fontSize: 12,
                color: C.coral,
                lineHeight: 1.5,
                fontFamily: "'DM Mono', monospace",
              }}
            >
              {error}
            </div>
          )}

          {/* CTA button */}
          <button
            onClick={runAnalysis}
            disabled={loading}
            style={{
              background: loading
                ? C.tealDim
                : `linear-gradient(135deg, ${C.teal}, #00B896)`,
              border: `1px solid ${loading ? C.tealMid : C.teal}`,
              borderRadius: 12,
              padding: "16px 24px",
              color: loading ? C.tealMid : C.bg,
              fontSize: 14,
              fontWeight: 700,
              cursor: loading ? "not-allowed" : "pointer",
              letterSpacing: "0.04em",
              transition: "all 0.2s",
              width: "100%",
              fontFamily: "'Inter', sans-serif",
            }}
          >
            {loading ? "Analysing..." : "Run Analysis"}
          </button>

          {/* Stats bar */}
          {prediction && recommendation && (
            <div
              style={{
                background: C.bgCard,
                border: `1px solid ${C.border}`,
                borderRadius: 16,
                padding: "20px 24px",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  letterSpacing: "0.14em",
                  color: C.textMut,
                  textTransform: "uppercase",
                  fontFamily: "'DM Mono', monospace",
                  marginBottom: 14,
                }}
              >
                Analysis summary
              </div>
              {[
                [
                  "Forecast",
                  `${prediction.predicted_glucose} mg/dL`,
                  glucoseColor(prediction.predicted_glucose),
                ],
                [
                  "Confidence",
                  `± ${prediction.confidence_range} mg/dL`,
                  C.purple,
                ],
                [
                  "Status",
                  prediction.status.replace("_", " ").toUpperCase(),
                  glucoseColor(prediction.predicted_glucose),
                ],
                [
                  "Action",
                  `#${recommendation.action_id} — ${recommendation.action_name}`,
                  recColor,
                ],
                ["Urgency", recommendation.urgency.toUpperCase(), recColor],
              ].map(([label, value, color]) => (
                <div
                  key={label}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 10,
                    paddingBottom: 10,
                    borderBottom: `1px solid ${C.border}`,
                  }}
                >
                  <span
                    style={{
                      fontSize: 11,
                      color: C.textMut,
                      fontFamily: "'DM Mono', monospace",
                    }}
                  >
                    {label}
                  </span>
                  <span
                    style={{
                      fontSize: 12,
                      color,
                      fontFamily: "'DM Mono', monospace",
                      fontWeight: 600,
                      textAlign: "right",
                      maxWidth: "60%",
                    }}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Phase info card */}
          <div
            style={{
              background: C.bgCard,
              border: `1px solid ${C.border}`,
              borderRadius: 16,
              padding: "18px 20px",
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.12em",
                color: C.textMut,
                textTransform: "uppercase",
                fontFamily: "'DM Mono', monospace",
                marginBottom: 10,
              }}
            >
              Models
            </div>
            {[
              [
                "LSTM",
                "Glucose forecaster",
                "51,777 params · MAE 2.51 mg/dL",
                C.purple,
              ],
              [
                "PPO",
                "Lifestyle agent",
                "99.4% time-in-range · 0 hypo events",
                C.teal,
              ],
            ].map(([tag, name, detail, c]) => (
              <div
                key={tag}
                style={{
                  display: "flex",
                  gap: 10,
                  marginBottom: 10,
                  alignItems: "flex-start",
                }}
              >
                <div
                  style={{
                    background: c + "22",
                    border: `1px solid ${c}44`,
                    borderRadius: 5,
                    padding: "2px 7px",
                    fontSize: 9,
                    color: c,
                    fontFamily: "'DM Mono', monospace",
                    letterSpacing: "0.08em",
                    flexShrink: 0,
                    marginTop: 1,
                  }}
                >
                  {tag}
                </div>
                <div>
                  <div
                    style={{ fontSize: 12, color: C.textPrim, fontWeight: 600 }}
                  >
                    {name}
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      color: C.textMut,
                      fontFamily: "'DM Mono', monospace",
                      marginTop: 2,
                    }}
                  >
                    {detail}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div
        style={{
          borderTop: `1px solid ${C.border}`,
          padding: "16px 32px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: C.textMut,
            fontFamily: "'DM Mono', monospace",
          }}
        >
          AI Diabetic Lifestyle Optimizer · v1.0.0
        </span>
        <span
          style={{
            fontSize: 11,
            color: C.textMut,
            fontFamily: "'DM Mono', monospace",
          }}
        >
          For educational use only · Not a medical device
        </span>
      </div>
    </div>
  );
}
