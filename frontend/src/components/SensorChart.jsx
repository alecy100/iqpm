import React, { useMemo, useRef, useState } from "react";

const WIDTH = 640;
const HEIGHT = 260;
const PAD = { top: 16, right: 16, bottom: 28, left: 56 };
const TICKS = 4;

export default function SensorChart({ points, label, unit }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const chart = useMemo(() => buildChart(points), [points]);

  if (!chart) {
    return <div className="chart-empty">Waiting for {label.toLowerCase()} readings…</div>;
  }

  const { path, xScale, yScale, yTicks, values, times, niceMin, niceMax } = chart;

  const handleMove = (event) => {
    const svg = svgRef.current;
    if (!svg || values.length === 0) return;
    const rect = svg.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * WIDTH;
    const fraction = clamp((x - PAD.left) / (WIDTH - PAD.left - PAD.right), 0, 1);
    const index = Math.round(fraction * (values.length - 1));
    setHover(index);
  };

  const activeIndex = hover ?? values.length - 1;
  const activeValue = values[activeIndex];
  const activeTime = times[activeIndex];

  return (
    <div className="sensor-chart">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        onPointerMove={handleMove}
        onPointerLeave={() => setHover(null)}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="chart-grid" x1={PAD.left} x2={WIDTH - PAD.right} y1={yScale(tick)} y2={yScale(tick)} />
            <text className="chart-tick" x={PAD.left - 8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle">
              {formatTick(tick)}
            </text>
          </g>
        ))}
        <line className="chart-axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
        <line
          className="chart-axis"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
        />
        <path className="chart-line" d={path} />
        {hover !== null && (
          <line
            className="chart-crosshair"
            x1={xScale(activeIndex)}
            x2={xScale(activeIndex)}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
          />
        )}
        <circle className="chart-dot" cx={xScale(activeIndex)} cy={yScale(activeValue)} r={4} />
      </svg>
      <div className="chart-readout">
        <span className="chart-readout-value">
          {formatTick(activeValue)} <small>{unit}</small>
        </span>
        <span className="chart-readout-time">{activeTime}</span>
      </div>
      <div className="chart-range">
        <span>{formatTick(niceMin)}</span>
        <span className="chart-range-unit">{unit}</span>
        <span>{formatTick(niceMax)}</span>
      </div>
    </div>
  );
}

function buildChart(points) {
  if (!points || points.length < 2) return null;
  const values = points.map((point) => point.value);
  const times = points.map((point) => formatTime(point.timestamp));

  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const [niceMin, niceMax, ticks] = niceScale(rawMin, rawMax, TICKS);

  const innerWidth = WIDTH - PAD.left - PAD.right;
  const innerHeight = HEIGHT - PAD.top - PAD.bottom;
  const xScale = (index) => PAD.left + (index / (values.length - 1)) * innerWidth;
  const yScale = (value) => PAD.top + innerHeight - ((value - niceMin) / (niceMax - niceMin || 1)) * innerHeight;

  const path = values
    .map((value, index) => `${index === 0 ? "M" : "L"} ${xScale(index).toFixed(2)} ${yScale(value).toFixed(2)}`)
    .join(" ");

  return { path, xScale, yScale, yTicks: ticks, values, times, niceMin, niceMax };
}

function niceScale(min, max, targetTicks) {
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const span = max - min;
  const rawStep = span / targetTicks;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step = (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let value = niceMin; value <= niceMax + step / 2; value += step) ticks.push(Number(value.toFixed(6)));
  return [niceMin, niceMax, ticks];
}

function formatTick(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function formatTime(iso) {
  const date = new Date(iso);
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
