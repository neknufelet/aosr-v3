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
function draw(svgId, room, speakers, receivers, vertical) {
  const svg = $(svgId); svg.replaceChildren();
  const height = vertical ? room.Lz : room.Ly;
  const scale = Math.min(520 / room.Lx, 320 / height);
  const x = (point) => 40 + point.x * scale;
  const y = (point) => 360 - (vertical ? point.z : point.y) * scale;
  svg.append(svgNode("rect", {x: 40, y: 360 - height * scale, width: room.Lx * scale, height: height * scale, fill: "none", stroke: "#334b58"}));
  for (const speaker of speakers) {
    if (speaker.aim) svg.append(svgNode("line", {x1: x(speaker.point), y1: y(speaker.point), x2: x(speaker.aim), y2: y(speaker.aim), stroke: "#db6b3a"}));
    svg.append(svgNode("circle", {cx: x(speaker.point), cy: y(speaker.point), r: 6, fill: "#db6b3a"}));
    const text = svgNode("text", {x: x(speaker.point) + 8, y: y(speaker.point) - 8}); text.textContent = speaker.role_label; svg.append(text);
  }
  for (const receiver of receivers) {
    svg.append(svgNode("circle", {cx: x(receiver.point), cy: y(receiver.point), r: 5, fill: "#167997"}));
    const text = svgNode("text", {x: x(receiver.point) + 8, y: y(receiver.point) + 14}); text.textContent = `${receiver.id}（${receiver.role_label}）`; svg.append(text);
  }
}
async function save() {
  const document = collect();
  const check = await api("/api/validate", "POST", document);
  if (check.problems.length) { $("messages").textContent = JSON.stringify(check.problems, null, 2); return false; }
  const saved = await api(`/api/schemes/${encodeURIComponent(document.scheme_id)}`, "PUT", document);
  const plan = await api(`/api/plan/${encodeURIComponent(document.scheme_id)}`);
  draw("plan-xy", plan.room, plan.speakers, plan.receivers, false);
  draw("plan-xz", plan.room, plan.speakers, plan.receivers, true);
  $("messages").textContent = saved.message; return true;
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
  $("walls").addEventListener("input", () => action(updateMultiples));
  $("check").onclick = () => action(async () => { $("messages").textContent = JSON.stringify((await api("/api/validate", "POST", collect())).problems, null, 2); });
  $("save").onclick = () => action(save);
  $("calculate").onclick = () => action(async () => {
    if (!await save()) return;
    const state = await api("/api/runs", "POST", {scheme_id: scheme.scheme_id});
    runId = state.run_id; $("stop").disabled = false; await poll();
    if (!$("stop").disabled) timer = setInterval(() => action(poll), 1000);
  });
  $("stop").onclick = () => action(async () => { await api(`/api/runs/${runId}/stop`, "POST"); await poll(); });
}));
