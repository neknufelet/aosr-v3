/* 只收表單並畫伺服器給的座標；圖形運算只做 SVG 縮放。 */
let scheme;
let runId;
let timer;
const $ = (id) => document.getElementById(id);
const wallNames = {floor: "地板", ceiling: "天花", x0: "x 起點牆", xL: "x 終點牆", y0: "y 起點牆", yL: "y 終點牆"};
const coordNames = {x: "x", y: "y", z: "z"};

async function api(path, method = "GET", body) {
  const response = await fetch(path, {method, headers: {"Content-Type": "application/json"}, body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || JSON.stringify(data.problems));
  return data;
}
function numberField(id, label, value) {
  const wrap = document.createElement("label");
  wrap.textContent = label;
  const input = document.createElement("input");
  input.id = id; input.type = "number"; input.step = "any"; input.value = value;
  wrap.append(input);
  return wrap;
}
function rows(target, entries, prefix) {
  $(target).replaceChildren();
  for (const [name, point] of entries) {
    const row = document.createElement("div"); row.className = "row";
    const title = document.createElement("strong"); title.textContent = name; row.append(title);
    for (const axis of Object.keys(coordNames)) {
      const primary = scheme.receiver_set.points.find((item) => item.role === "primary");
      const label = prefix === "receiver" && name === primary.receiver_id && axis === "z" ? "主位 z 座標（公尺）" : coordNames[axis];
      row.append(numberField(`${prefix}-${name}-${axis}`, label, point[axis]));
    }
    $(target).append(row);
  }
}
function renderForm() {
  $("save-id").value = scheme.scheme_id;
  $("source-model").value = scheme.source_model;
  $("room-fields").replaceChildren();
  for (const [key, label] of [["Lx", "長 Lx（公尺）"], ["Ly", "寬 Ly（公尺）"], ["Lz", "高 Lz（公尺）"]])
    $("room-fields").append(numberField(`room-${key}`, label, scheme.scene.room_m[key]));
  $("walls").replaceChildren();
  for (const [name, value] of Object.entries(scheme.scene.impedance_pa_s_per_m_by_wall)) {
    const row = document.createElement("div"); row.className = "row";
    row.append(numberField(`wall-${name}`, `${wallNames[name]}阻抗（帕·秒／公尺）`, value));
    const multiple = document.createElement("span"); multiple.id = `multiple-${name}`; row.append(multiple);
    $("walls").append(row);
  }
  $("scattering").replaceChildren();
  for (const name of Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall))
    $("scattering").append(numberField(`scatter-${name}`, `${wallNames[name]}散射`, scheme.scene.scattering_by_wall?.[name] ?? ""));
  $("use-scattering").checked = scheme.scene.scattering_by_wall !== null && scheme.scene.scattering_by_wall !== undefined;
  rows("speakers", Object.entries(scheme.speakers), "speaker");
  rows("receivers", scheme.receiver_set.points.map((point) => [point.receiver_id,
       {x: point.position_m[0], y: point.position_m[1], z: point.position_m[2]}]), "receiver");
  action(updateMultiples);
}
async function updateMultiples() {
  const checked = await api("/api/validate", "POST", collect());
  for (const [name, label] of Object.entries(checked.impedance_labels))
    $(`multiple-${name}`).textContent = label;
}
function collect() {
  scheme.scheme_id = $("save-id").value;
  scheme.source_model = $("source-model").value;
  const numeric = (id) => {
    const input = $(id);
    return input.value.trim() === "" ? null : Number(input.value);
  };
  for (const key of Object.keys(scheme.scene.room_m)) scheme.scene.room_m[key] = numeric(`room-${key}`);
  for (const name of Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall))
    scheme.scene.impedance_pa_s_per_m_by_wall[name] = numeric(`wall-${name}`);
  scheme.scene.scattering_by_wall = $("use-scattering").checked ? Object.fromEntries(
    Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall).map((name) => [name, numeric(`scatter-${name}`)])) : null;
  for (const [name, point] of Object.entries(scheme.speakers))
    for (const axis of Object.keys(coordNames)) point[axis] = numeric(`speaker-${name}-${axis}`);
  for (const receiver of scheme.receiver_set.points)
    receiver.position_m = Object.keys(coordNames).map((axis) => numeric(`receiver-${receiver.receiver_id}-${axis}`));
  return scheme;
}
function svgNode(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
function separateLabels(labels, text) {
  for (let step = 0; step < 24; step++) {
    const box = text.getBBox();
    const crosses = labels.some((other) => {
      const old = other.getBBox();
      return box.x < old.x + old.width && old.x < box.x + box.width &&
        box.y < old.y + old.height && old.y < box.y + box.height;
    });
    if (!crosses) break;
    text.setAttribute("y", Number(text.getAttribute("y")) + 16);
  }
  labels.push(text);
}
function draw(svgId, range, markers, zoomRange, primaryId, zoomed, speakers, vertical) {
  const svg = $(svgId); svg.replaceChildren();
  const width = range.u[1] - range.u[0];
  const height = range.v[1] - range.v[0];
  const scale = Math.min(520 / width, 320 / height);
  const x = (u) => 40 + (u - range.u[0]) * scale;
  const y = (v) => 360 - (v - range.v[0]) * scale;
  svg.append(svgNode("rect", {x: 40, y: 360 - height * scale,
    width: width * scale, height: height * scale, fill: "none", stroke: "#334b58"}));
  if (!zoomed) svg.append(svgNode("rect", {x: x(zoomRange.u[0]), y: y(zoomRange.v[1]),
    width: (zoomRange.u[1] - zoomRange.u[0]) * scale,
    height: (zoomRange.v[1] - zoomRange.v[0]) * scale,
    fill: "none", stroke: "#167997", "stroke-dasharray": "5 4"}));
  if (!zoomed) for (const speaker of speakers) {
    if (speaker.aim) svg.append(svgNode("line", {
      x1: x(speaker.point.x), y1: y(speaker.point[vertical ? "z" : "y"]),
      x2: x(speaker.aim.x), y2: y(speaker.aim[vertical ? "z" : "y"]),
      stroke: "#db6b3a"}));
  }
  const labels = [];
  for (const item of markers) {
    if (zoomed && (item.u < range.u[0] || item.u > range.u[1] ||
                   item.v < range.v[0] || item.v > range.v[1])) continue;
    if (!zoomed && item.kind === "receiver" && !item.ids.includes(primaryId)) continue;
    const detail = item.detail_lines.join("\n");
    const group = svgNode("g", {"data-ids": item.ids.join(" "), tabindex: 0});
    const title = svgNode("title", {}); title.textContent = detail; group.append(title);
    group.append(svgNode("circle", {cx: x(item.u), cy: y(item.v), r: 5,
      fill: item.kind === "receiver" ? "#167997" : "#db6b3a"}));
    const caption = svgNode("text", {x: x(item.u) + 8, y: y(item.v) - 8});
    caption.textContent = !zoomed && item.kind === "receiver" ? "主" :
      !zoomed && item.kind === "mixed" && !item.ids.includes(primaryId) ?
        speakers.filter((speaker) => item.ids.includes(speaker.id)).map((speaker) => speaker.marker).join("／") :
        item.marker;
    group.append(caption); svg.append(group);
    separateLabels(labels, caption);
    group.addEventListener("click", () => { $("plan-detail").textContent = detail; });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") $("plan-detail").textContent = detail;
    });
  }
}
function drawPlan(plan) {
  const primary = plan.receivers.find((item) => item.role === "primary");
  const limits = {plan: {u: [0, plan.room.Lx], v: [0, plan.room.Ly]},
    side: {u: [0, plan.room.Lx], v: [0, plan.room.Lz]}};
  draw("plan-xy", limits.plan, plan.views.plan, plan.listening_zoom.plan, primary.id, false,
    plan.speakers, false);
  draw("plan-xz", limits.side, plan.views.side, plan.listening_zoom.side, primary.id, false,
    plan.speakers, true);
  draw("zoom-xy", plan.listening_zoom.plan, plan.views.plan, plan.listening_zoom.plan, primary.id, true,
    plan.speakers, false);
  draw("zoom-xz", plan.listening_zoom.side, plan.views.side, plan.listening_zoom.side, primary.id, true,
    plan.speakers, true);
  $("plan-legend").replaceChildren();
  for (const item of [...plan.speakers, ...plan.receivers]) {
    const line = document.createElement("li");
    line.dataset.id = item.id;
    line.textContent = `${item.marker}－${item.detail_text}`;
    $("plan-legend").append(line);
  }
}
async function save() {
  const document = collect();
  if (!await refreshPlan()) return false;
  const saved = await api(`/api/schemes/${encodeURIComponent(document.scheme_id)}`, "PUT", document);
  $("messages").textContent = saved.message; return true;
}
async function refreshPlan() {
  const response = await fetch("/api/plan", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(collect())});
  const plan = await response.json();
  if (!response.ok) {
    for (const name of ["plan-xy", "plan-xz", "zoom-xy", "zoom-xz"])
      $(name).replaceChildren();
    $("messages").textContent = plan.problems ?
      plan.problems.map((problem) => `${problem.path}：${problem.message}`).join("\n") :
      (plan.error || "圖面檢查失敗");
    return false;
  }
  drawPlan(plan);
  $("messages").textContent = plan.message;
  return true;
}
async function poll() {
  const state = await api(`/api/runs/${runId}`);
  $("run-state").textContent = `${state.display_text}；${state.stderr_tail.join("\n")}`;
  if (state.status !== "running") {
    clearInterval(timer); $("stop").disabled = true;
    if (state.status === "done") {
      $("messages").textContent = `${state.result_path}；${state.next_step_note}`;
      $("result-link").href = state.result_url;
      $("result-link").hidden = false;
    }
  }
}
async function action(work) {
  try { await work(); } catch (error) { $("messages").textContent = String(error); }
}
window.addEventListener("DOMContentLoaded", () => action(async () => {
  const example = await api("/api/example"); scheme = example.scheme;
  $("feature-note").textContent = example.feature_match_note;
  $("rho-c").textContent = example.rho_c_label;
  renderForm();
  await refreshPlan();
  $("walls").addEventListener("input", () => action(updateMultiples));
  $("check").onclick = () => action(refreshPlan);
  $("save").onclick = () => action(save);
  $("calculate").onclick = () => action(async () => {
    if (!await save()) return;
    const state = await api("/api/runs", "POST", {scheme_id: scheme.scheme_id});
    runId = state.run_id; $("stop").disabled = false; await poll();
    if (!$("stop").disabled) timer = setInterval(() => action(poll), 1000);
  });
  $("stop").onclick = () => action(async () => { await api(`/api/runs/${runId}/stop`, "POST"); await poll(); });
}));
