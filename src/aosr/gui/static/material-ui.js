"use strict";

let listing = null;

function freeId(base) {
  const taken = new Set(listing.materials.map((row) => row.entry.id));
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) if (!taken.has(`${base}_${n}`)) return `${base}_${n}`;
}

function freeLabel(base) {
  const taken = new Set(listing.materials.map((row) => row.entry.label));
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) if (!taken.has(`${base} ${n}`)) return `${base} ${n}`;
}

function designerUrl(id, copy = false) {
  const query = new URLSearchParams({[copy ? "copy" : "id"]: id});
  return "/materials/designer?" + query;
}

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === false || value === null || value === undefined) continue;
    if (name === "text") node.textContent = value;
    else if (name === "on") for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

const byId = (id) => document.getElementById(id);
async function api(path, options = {}) {
  const reply = await fetch(path, options);
  let body;
  try { body = await reply.json(); } catch { body = {error: `伺服器回的不是 JSON（狀態 ${reply.status}）`}; }
  return {ok: reply.ok, status: reply.status, body};
}

const SMALL_CURVE_HZ = {low: 31.5, high: 2000};
const SPAN_FILL = {extrapolation: "rgba(230,140,40,0.22)", model_fill: "rgba(110,125,135,0.2)"};

function overlapsSmallCurve(span) {
  return span.high_hz >= SMALL_CURVE_HZ.low && span.low_hz <= SMALL_CURVE_HZ.high;
}

// 取現有無規入射曲線的同一段對數頻軸，縱軸一律 0～1。實測表只在表內原有頻點加點。
function smallCurve(entry, result) {
  const w = 168, h = 56;
  const {low, high} = SMALL_CURVE_HZ;
  const x = hz => 2 + (w - 4) * Math.log(hz / low) / Math.log(high / low);
  const y = alpha => 2 + (h - 4) * (1 - alpha);
  const svg = materialSvg("svg", {class: "curve " + entry.provenance, viewBox: `0 0 ${w} ${h}`,
    role: "img", "aria-label": "無規入射吸音曲線，31.5 Hz 至 2 kHz，吸音率 0 至 1"});
  if (result.reference_only && result.reference_only.band_hz > low) {
    svg.append(materialSvg("rect", {class: "reference-only", width: x(Math.min(result.reference_only.band_hz, high)),
      height: h, fill: "rgba(60,120,210,.16)"}));
  }
  // 用同一條曲線，遮罩只淡化被標記的頻段；橘／灰底對應提醒原文中的顏色。
  const maskId = "preview-spans-" + entry.id;
  const mask = materialSvg("mask", {id: maskId, maskUnits: "userSpaceOnUse", x: 0, y: 0, width: w, height: h});
  mask.append(materialSvg("rect", {width: w, height: h, fill: "white"}));
  for (const span of result.spans) {
    if (!overlapsSmallCurve(span)) continue;
    const left = x(Math.max(low, span.low_hz)), right = x(Math.min(high, span.high_hz));
    const rect = {x: left, width: Math.max(2, right - left), height: h};
    mask.append(materialSvg("rect", {...rect, "data-kind": span.kind, fill: "#888"}));
    svg.append(materialSvg("rect", {...rect, fill: SPAN_FILL[span.kind]}));
  }
  svg.append(mask);
  for (const hz of [63, 125, 250, 500, 1000]) svg.append(materialSvg("line", {x1:x(hz), x2:x(hz), y1:0, y2:h}));
  svg.append(materialSvg("line", {x1:0, x2:w, y1:y(.5), y2:y(.5)}));
  const measured = new Set(entry.material?.points?.map(p => p.frequency_hz));
  const points = [];
  for (const [index, hz] of result.frequencies_hz.entries()) {
    // 多留左右各一點，由 SVG 視窗裁切，避免軸端取最近頻點而出現空隙。
    if (result.frequencies_hz[index + 1] < low || result.frequencies_hz[index - 1] > high) continue;
    const cx = x(hz).toFixed(2), cy = y(result.absorption[index]).toFixed(2);
    points.push(cx + "," + cy);
    if (entry.provenance === "measured" && measured.has(hz)) svg.append(materialSvg("circle", {cx, cy, r: 2.4}));
  }
  svg.append(materialSvg("polyline", {points: points.join(" "), fill: "none", stroke: "currentColor", "stroke-width": 2, mask: `url(#${maskId})`}));
  return svg;
}
