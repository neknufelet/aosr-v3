// 自訂房間頁的高音那一段：按「算高音」→ 背景算 → 輪詢到結果，畫反射路徑（3D、可拖曳）、早期回聲、晚期衰減、逐頻能量。
// 全部用 canvas 與內嵌 SVG 畫，不載入外部程式庫。模型代號與「檢查過沒」從 #model 的 data-id／data-ok 讀（room_model.js 寫的）。
(() => {
  const $ = (id) => document.getElementById(id);
  const ORDER_COLORS = {1: "#1f7a6d", 2: "#3f7fb4", 3: "#8a7bb8"};
  const BLOCK = "#c23a3a", ROOM = "#15617b", SEAT = "#17212a", SPEAKER = "#e07b39";
  const NS = "http://www.w3.org/2000/svg";
  const show = {1: true, 2: true, 3: false, blocked: true};
  let result = null, speaker = 0, timer = null, active = null, yaw = -0.9, pitch = 0.55, drag = null;

  async function call(url, body) {
    const response = await fetch(url, body === undefined ? {} :
      {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }
  const short = (layer) => layer.split("_").slice(1).join("") || layer;

  // ---- 3D：正交投影、畫家演算法，拖曳改方位角與仰角 ----
  function frame3d() {
    const points = Object.values(result.triangles).flat(3);
    const xs = [], ys = [], zs = [];
    for (let i = 0; i < points.length; i += 3) { xs.push(points[i]); ys.push(points[i + 1]); zs.push(points[i + 2]); }
    const centre = [xs, ys, zs].map((v) => (Math.min(...v) + Math.max(...v)) / 2);
    const span = Math.max(...[xs, ys, zs].map((v) => Math.max(...v) - Math.min(...v)));
    return {centre, span};
  }
  function project(point, view, width, height) {
    const [x, y, z] = point.map((v, i) => v - view.centre[i]);
    const x1 = x * Math.cos(yaw) - y * Math.sin(yaw), y1 = x * Math.sin(yaw) + y * Math.cos(yaw);
    const depth = y1 * Math.cos(pitch) + z * Math.sin(pitch), up = -y1 * Math.sin(pitch) + z * Math.cos(pitch);
    const scale = Math.min(width, height) / view.span * 0.95;
    return [width / 2 + x1 * scale, height / 2 - up * scale, depth];
  }
  function rgba(hex, alpha) {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    return `rgba(${r},${g},${b},${alpha})`;
  }
  function draw3d() {
    if (!result) return;
    const canvas = $("high-3d"), ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    canvas.width = width * ratio; canvas.height = height * ratio;
    const g = canvas.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    g.clearRect(0, 0, width, height);
    const view = frame3d(), colors = Object.fromEntries(result.layers.map((l) => [l.name, l.color]));
    const faces = [];
    const solids = new Set(result.solid_layers);
    for (const [layer, triangles] of Object.entries(result.triangles)) {
      for (const triangle of triangles) {
        const q = triangle.map((p) => project(p, view, width, height));
        // 造型（天雲、板子）實心畫；外殼半透明，朝下的那幾面（天花板）幾乎透明，才看得到房內。
        const [a, b, c] = triangle, facing = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        const area = Math.hypot(...[0, 1, 2].map((i) => (b[(i + 1) % 3] - a[(i + 1) % 3]) * (c[(i + 2) % 3] - a[(i + 2) % 3]) -
                                                        (b[(i + 2) % 3] - a[(i + 2) % 3]) * (c[(i + 1) % 3] - a[(i + 1) % 3])));
        const ceiling = area > 0 && facing / area < -0.9;
        faces.push({q, depth: (q[0][2] + q[1][2] + q[2][2]) / 3, color: colors[layer] || "#999999",
                    alpha: solids.has(layer) ? 0.85 : (ceiling ? 0.06 : 0.18)});
      }
    }
    faces.sort((a, b) => b.depth - a.depth);
    for (const face of faces) {
      g.beginPath(); face.q.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath();
      g.fillStyle = rgba(face.color, face.alpha); g.fill();
      g.strokeStyle = rgba(face.color, Math.min(1, face.alpha + 0.3)); g.lineWidth = 0.5; g.stroke();
    }
    const line = (points, color, stroke, dashed) => {
      g.beginPath();
      points.forEach((p, i) => { const [x, y] = project(p, view, width, height); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.strokeStyle = color; g.lineWidth = stroke; g.setLineDash(dashed ? [7, 5] : []); g.stroke(); g.setLineDash([]);
    };
    const current = result.speakers[speaker];
    for (const order of [3, 2, 1]) {
      if (!show[order]) continue;
      current.paths.filter((p) => p.order === order).forEach((p) => line(p.points, ORDER_COLORS[order], order === 1 ? 2.6 : 1.2, false));
    }
    if (show.blocked) current.blocked.filter((p) => show[p.order]).forEach((p) => line(p.points, BLOCK, 2, true));
    for (const [point, label, color] of [[current.position, "喇叭", SPEAKER], [result.seat, "聽位", SEAT]]) {
      const [x, y] = project(point, view, width, height);
      g.beginPath(); g.arc(x, y, 6, 0, 2 * Math.PI); g.fillStyle = color; g.fill();
      g.fillStyle = SEAT; g.font = "13px system-ui, sans-serif"; g.fillText(label, x + 9, y - 8);
    }
  }
  $("high-3d").addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; $("high-3d").setPointerCapture(e.pointerId); });
  $("high-3d").addEventListener("pointermove", (e) => {
    if (!drag) return;
    yaw += (e.clientX - drag[0]) * 0.01;
    pitch = Math.max(-0.2, Math.min(1.5, pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
    draw3d();
  });
  $("high-3d").addEventListener("pointerup", () => { drag = null; });
  window.addEventListener("resize", draw3d);

  // ---- 平面圖：同一套座標框（格線、刻度、軸名）----
  function el(svg, tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    svg.appendChild(node);
    return node;
  }
  function axes(svg, box, xRange, yRange, xTicks, yTicks, xLabel, yLabel, log) {
    svg.replaceChildren();
    const m = {l: 56, r: 16, t: 12, b: 44}, [W, H] = box;
    const tx = (v) => (log ? Math.log(v) : v);
    const X = (v) => m.l + (tx(v) - tx(xRange[0])) / (tx(xRange[1]) - tx(xRange[0])) * (W - m.l - m.r);
    const Y = (v) => m.t + (yRange[1] - v) / (yRange[1] - yRange[0]) * (H - m.t - m.b);
    for (const v of yTicks) {
      el(svg, "line", {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "#e1e9ec"});
      el(svg, "text", {x: m.l - 8, y: Y(v) + 4, "text-anchor": "end"}, v);
    }
    for (const [v, label] of xTicks) {
      el(svg, "line", {x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: "#e1e9ec", "stroke-dasharray": "2 4"});
      el(svg, "text", {x: X(v), y: H - m.b + 18, "text-anchor": "middle"}, label);
    }
    el(svg, "text", {x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle"}, xLabel);
    el(svg, "text", {x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`}, yLabel);
    return {X, Y};
  }
  const steps = (from, to, step) => { const out = []; for (let v = from; v <= to + 1e-9; v += step) out.push(v); return out; };
  function polyline(svg, points, color, dots) {
    el(svg, "polyline", {points: points.map((p) => p.join(",")).join(" "), fill: "none", stroke: color, "stroke-width": 2});
    if (dots) points.forEach(([x, y]) => el(svg, "circle", {cx: x, cy: y, r: 2.5, fill: color}));
  }

  function drawEcho() {
    const svg = $("high-echo"), paths = result.speakers[speaker].paths, floor = -40;
    const end = Math.ceil(Math.max(...paths.map((p) => p.t_ms)) / 5) * 5;
    const f = axes(svg, [900, 300], [0, end], [floor, 2], steps(0, end, 5).map((v) => [v, v]), steps(floor, 0, 10).reverse(),
                   "抵達時間（毫秒）", "dB", false);
    for (const p of paths) {
      const x = f.X(p.t_ms), y = f.Y(Math.max(p.db, floor)), color = p.order === 0 ? SEAT : ORDER_COLORS[p.order];
      el(svg, "line", {x1: x, x2: x, y1: f.Y(floor), y2: y, stroke: color, "stroke-width": 2});
      const dot = el(svg, "circle", {cx: x, cy: y, r: 3, fill: color});
      el(dot, "title", {}, `${p.order === 0 ? "直達" : p.via.map(short).join("→")}  ${p.t_ms.toFixed(1)} ms  ${p.db.toFixed(1)} dB`);
    }
  }
  function drawDecay() {
    const svg = $("high-decay"), decay = result.decay;
    const low = Math.floor(Math.min(...decay.map((p) => p.db)) / 5) * 5, end = Math.ceil(Math.max(...decay.map((p) => p.t_ms)) / 20) * 20;
    const f = axes(svg, [900, 260], [0, end], [low, 1], steps(0, end, 20).map((v) => [v, v]), steps(low, 0, 5), "時間（毫秒）", "dB", false);
    polyline(svg, decay.map((p) => [f.X(p.t_ms), f.Y(p.db)]), ROOM, true);
  }
  function drawEnergy() {
    const svg = $("high-energy"), energy = result.speakers[speaker].energy_db, hz = result.frequencies_hz;
    const series = [["total", "總和", ROOM], ["early", "直達＋早期", "#e07b39"], ["late", "晚期", "#8a7bb8"]];
    const values = series.flatMap(([key]) => energy[key]);
    const low = Math.floor(Math.min(...values) / 5) * 5, high = Math.ceil(Math.max(...values) / 5) * 5;
    const ticks = [[300, "300"], [500, "500"], [1000, "1k"], [2000, "2k"], [4000, "4k"], [8000, "8k"]];
    const f = axes(svg, [900, 300], [hz[0], hz[hz.length - 1]], [low, high], ticks, steps(low, high, 5), "頻率（Hz）", "dB", true);
    series.forEach(([key, label, color], i) => {
      polyline(svg, hz.map((v, j) => [f.X(v), f.Y(energy[key][j])]), color, false);
      el(svg, "text", {x: 70 + i * 110, y: 26, fill: color, style: `fill:${color}`}, "— " + label);
    });
  }
  function legend() {
    const items = [[1, "一階", ORDER_COLORS[1], ""], [2, "二階", ORDER_COLORS[2], ""], [3, "三階", ORDER_COLORS[3], ""], ["blocked", "被擋掉", BLOCK, "dash"]];
    $("high-legend").replaceChildren(...items.map(([key, label, color, dash]) => {
      const box = document.createElement("label"), input = document.createElement("input"), swatch = document.createElement("span");
      input.type = "checkbox"; input.checked = show[key];
      input.addEventListener("change", () => { show[key] = input.checked; draw3d(); });
      swatch.className = `sw ${dash}`;
      box.style.color = color;
      box.append(input, swatch, label);
      return box;
    }));
  }
  function drawAll() {
    const current = result.speakers[speaker];
    $("high-facts").textContent = `有效反射 ${current.paths.length - 1} 條，被擋掉 ${current.blocked.length} 條；喇叭當全向；算了 ${result.seconds} 秒`;
    draw3d(); drawEcho(); drawDecay(); drawEnergy();
  }
  function showResult(data) {
    result = data; speaker = 0;
    $("high-speaker").replaceChildren(...data.speakers.map((s, i) => new Option(s.label, String(i))));
    $("high-body").hidden = false;
    legend();
    drawAll();
  }
  $("high-speaker").addEventListener("change", () => { speaker = Number($("high-speaker").value); drawAll(); });

  // ---- 起算、輪詢；換檔就放掉 ----
  function stop() { clearInterval(timer); timer = null; active = null; }
  async function poll(runId) {
    try {
      const data = await call(`/api/room-model-high-jobs/${runId}`);
      if (runId !== active) return;
      $("high-state").textContent = data.status === "done" ? "" : data.text + (data.error ? "\n" + data.error : "");
      if (data.status === "running") return;
      stop();
      $("run-high").disabled = false;
      if (data.result) showResult(data.result);
    } catch (error) {
      if (runId !== active) return;
      stop();
      $("run-high").disabled = false;
      $("high-state").textContent = error.message;
    }
  }
  new MutationObserver(() => { $("run-high").disabled = !($("model").dataset.ok === "true" && $("model").dataset.id) || timer !== null; })
    .observe($("model"), {attributes: true, attributeFilter: ["data-id", "data-ok"]});
  $("file").addEventListener("change", () => {
    stop(); result = null;
    delete $("model").dataset.id; delete $("model").dataset.ok;
    $("high").hidden = true; $("high-body").hidden = true; $("run-high").disabled = true;
  });
  $("run-high").addEventListener("click", async () => {
    const forModel = $("model").dataset.id, absorption = {};
    document.querySelectorAll("#layers select").forEach((select) => { absorption[select.dataset.layer] = select.value; });
    $("run-high").disabled = true;
    $("high").hidden = false; $("high-body").hidden = true;
    try {
      const job = await call(`/api/room-models/${forModel}/high-runs`, {absorption});
      if ($("model").dataset.id !== forModel) return;
      $("high-state").textContent = job.display_text || "計算高音中";
      stop();
      active = job.run_id;
      timer = setInterval(() => poll(job.run_id), 2000);
    } catch (error) {
      $("run-high").disabled = false;
      $("high-state").textContent = error.message;
    }
  });
})();
