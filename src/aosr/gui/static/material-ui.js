"use strict";

let listing = null;

function freeId(base, unavailable = []) {
  const taken = new Set([...unavailable, ...listing.materials.map(row => row.entry.id)]);
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) if (!taken.has(`${base}_${n}`)) return `${base}_${n}`;
}

function freeLabel(base, exceptId) {
  const taken = new Set(listing.materials.filter(row => row.entry.id !== exceptId).map(row => row.entry.label));
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

let plot = null;

function destroyPlot() {
  if (plot) plot.destroy();
  plot = null;
  byId("curve-chart").replaceChildren();
}

function chartWidth() {
  return Math.max(byId("curve-chart").clientWidth, 240);
}

// 「鋼板直接貼在棉上」的構造（#1208）：界線頻率以下打淡藍底，頂端內側一行小字。字與界線都是伺服器給的（reply.reference_only）。
// 藍底是疊在圖上的兩個元素（不是畫在畫布上）：每次重畫照橫軸現在的位置重放，界線在橫軸左端以外就不放。
function placeBand(u, reference) {
  const over = u.over;
  let band = over.querySelector(".ref-band"), word = over.querySelector(".ref-band-word");
  if (!band) {
    band = el("div", {class: "ref-band", id: "curve-ref-band"}); word = el("div", {class: "ref-band-word", id: "curve-ref-band-word"});
    over.append(band, word);
  }
  const right = Math.min(u.valToPos(reference.band_hz, "x"), over.clientWidth);
  band.hidden = word.hidden = !(right > 0);
  band.style.width = `${Math.max(right, 0)}px`;
  band.dataset.hz = String(reference.band_hz);
  word.textContent = reference.band_words;
}

function drawPlot(reply, height = 320) {
  destroyPlot();
  const spans = reply.spans;
  const reference = reply.reference_only || null;
  // 底色畫在清畫布之後、畫線之前；頭尾同一點時至少畫 2 像素寬，看得到。
  const shade = (u) => {
    const ctx = u.ctx, box = u.bbox;
    ctx.save();
    for (const span of spans) {
      const left = u.valToPos(span.low_hz, "x", true), right = u.valToPos(span.high_hz, "x", true);
      ctx.fillStyle = SPAN_FILL[span.kind];
      ctx.fillRect(left, box.top, Math.max(right - left, 2 * devicePixelRatio), box.height);
    }
    ctx.restore();
  };
  plot = new uPlot({width: chartWidth(), height, legend: {show: false},
    scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}, y: {range: [0, 1]}},
    axes: [{label: "頻率（Hz）"}, {label: "無規入射吸音係數"}],
    series: [{}, {label: "吸音係數", stroke: "#15617b", width: 2, points: {show: false}}],
    hooks: {drawClear: [shade], draw: reference ? [(u) => placeBand(u, reference)] : []}},
    [reply.frequencies_hz, reply.absorption], byId("curve-chart"));
}

// 圖例那一行（#1208）：色塊、「藍底：」、那一句、「為什麼」（點開才看得到原因）。
function referenceNote(reference) {
  const more = el("details", {id: "curve-ref-why"}, el("summary", {text: "為什麼"}), el("p", {text: reference.why_only}));
  return el("li", {class: "ref-note", id: "curve-ref-note", "data-kind": "reference_only"},
    el("span", {class: "ref-sw"}), "藍底：", el("span", {class: "ref-say", text: reference.sentence}), more);
}

function showPlotKey(reply) {
  const drawn = reply.status === "drawn";
  const reference = reply.reference_only;
  const key = byId("curve-key");
  key.replaceChildren();
  key.hidden = !drawn || !(reply.spans.length || reference);
  if (!drawn) return;
  if (reference) key.append(referenceNote(reference));
  const kinds = new Set(reply.spans.map(span => span.kind));
  for (const [kind, text] of Object.entries({
    extrapolation: "橘底：材料模型外推（照算，不宣稱準確）",
    model_fill: "灰底：實測表外，用底構造算的模型補值",
  })) {
    key.append(el("li", {"data-kind": kind, hidden: !kinds.has(kind)}, el("span", {class: "swatch " + kind}), text));
  }
}
