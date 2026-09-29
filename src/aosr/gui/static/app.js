/* 只收表單並畫伺服器給的座標；圖形運算只做 SVG 縮放。 */
let scheme;
let openedId = null;
let runId;
let timer;
// 這一次開算之後表單有沒有被改過；改過的話算完不把連結掛在表單旁邊。
let runEdited = false;
const compareChoice = {a: null, b: null};
const $ = (id) => document.getElementById(id);
const wallNames = {floor: "地板", ceiling: "天花", x0: "x 起點牆", xL: "x 終點牆", y0: "y 起點牆", yL: "y 終點牆"};
const coordNames = {x: "x", y: "y", z: "z"};

async function api(path, method = "GET", body, headers = {}) {
  const response = await fetch(path, {method, headers: {"Content-Type": "application/json", ...headers}, body: body === undefined ? undefined : JSON.stringify(body)});
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
async function save() {
  const document = collect();
  if (!await refreshPlan()) return false;
  const headers = openedId === null || openedId !== document.scheme_id ? {"If-None-Match": "*"} : {};
  const saved = await api(`/api/schemes/${encodeURIComponent(document.scheme_id)}`, "PUT", document, headers);
  openedId = document.scheme_id;
  $("messages").textContent = saved.message;
  await loadSchemeList();
  return true;
}
async function loadSchemeList() {
  const data = await api("/api/schemes");
  const list = $("scheme-list");
  const selected = list.value;
  list.replaceChildren();
  for (const name of data.schemes) {
    const option = document.createElement("option");
    option.value = name; option.textContent = name; list.append(option);
  }
  if (data.schemes.includes(selected)) list.value = selected;
}
async function openScheme() {
  const name = $("scheme-list").value;
  if (!name) return;
  scheme = (await api(`/api/schemes/${encodeURIComponent(name)}`)).scheme;
  openedId = name;
  renderForm();
  $("result-link").hidden = true;
  $("result-stale").hidden = true;
  await refreshPlan();
}
async function saveAs() {
  const name = $("save-as-id").value.trim();
  if (!name) { $("messages").textContent = "請填另存的新代號"; return; }
  if (!await refreshPlan()) return;
  // 方案代號欄只顯示現在開著哪一份：伺服器存好了才換成新名字，存失敗就維持原樣。
  const saved = await api(`/api/schemes/${encodeURIComponent(name)}`, "PUT",
    {...collect(), scheme_id: name}, {"If-None-Match": "*"});
  scheme.scheme_id = name; $("save-id").value = name; openedId = name;
  $("messages").textContent = saved.message;
  await loadSchemeList();
  $("scheme-list").value = name;
}
function markStale() {
  runEdited = true;
  if (!$("result-link").hidden) $("result-stale").hidden = false;
}
function chooseCompare(side, item) {
  compareChoice[side] = item;
  $(side === "a" ? "compare-a" : "compare-b").textContent =
    `${side.toUpperCase()}：${item.scheme_id}（${item.finished_text}）`;
  const link = $("compare-link");
  link.hidden = !compareChoice.a || !compareChoice.b;
  if (!link.hidden) link.href = `/compare/${compareChoice.a.run_id}/${compareChoice.b.run_id}`;
}
async function loadResultList() {
  const data = await api("/api/results");
  const list = $("results-list"); list.replaceChildren();
  for (const item of data.results) {
    const row = document.createElement("tr");
    for (const value of [item.scheme_id, item.finished_text, item.duration_text,
      item.engine_text, item.registry_text]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    const cell = document.createElement("td");
    const link = document.createElement("a");
    link.href = item.result_url; link.textContent = "查看"; cell.append(link); row.append(cell);
    for (const side of ["a", "b"]) {
      const choiceCell = document.createElement("td");
      const choice = document.createElement("button");
      choice.type = "button";
      choice.textContent = `選為 ${side.toUpperCase()}`;
      choice.onclick = () => chooseCompare(side, item);
      choiceCell.append(choice); row.append(choiceCell);
    }
    list.append(row);
  }
}
async function resumeRuns() {
  // 只接回還在算的；已結束的那一筆不貼，表單開的是範本，貼上去會讓人以為是這份的結果。
  const state = (await api("/api/runs")).running[0];
  if (!state) return;
  runId = state.run_id;
  $("run-state").textContent = state.display_text;
  $("stop").disabled = false;
  clearInterval(timer);
  timer = setInterval(() => action(poll), 1000);
}
async function refreshPlan() {
  const response = await fetch("/api/plan", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(collect())});
  const plan = await response.json();
  if (!response.ok) {
    for (const name of ["plan-xy", "plan-xz", "zoom-xy", "zoom-xz", "plan-legend", "plan-detail"])
      $(name).replaceChildren();
    $("messages").textContent = plan.problems ?
      plan.problems.map((problem) => `${problem.path}：${problem.message}`).join("\n") :
      (plan.error || "圖面檢查失敗");
    return false;
  }
  drawPlan(plan, {planXY: "plan-xy", planXZ: "plan-xz", zoomXY: "zoom-xy",
    zoomXZ: "zoom-xz", detail: $("plan-detail"), legend: $("plan-legend")});
  $("messages").textContent = plan.message;
  return true;
}
async function poll() {
  const state = await api(`/api/runs/${runId}`);
  // 計算沒有輸出時不補一個空的分號。
  $("run-state").textContent = state.stderr_tail.length ?
    `${state.display_text}；${state.stderr_tail.join("\n")}` : state.display_text;
  if (state.status !== "running") {
    clearInterval(timer); $("stop").disabled = true;
    if (state.status === "done") {
      // 表單還是算的那一份、開算後也沒改過，連結才掛在表單旁邊；不然會讓人以為是表單上這份的結果。
      if (state.scheme_id === openedId && !runEdited) {
        $("messages").textContent = `${state.result_path}；${state.next_step_note}`;
        $("result-link").href = state.result_url;
        $("result-link").hidden = false;
        $("result-stale").hidden = true;
      } else {
        $("messages").textContent = `「${state.scheme_id}」算完了；表單上現在不是算的那一份（開算後改過或換了方案），結果在下方結果清單`;
      }
      await loadResultList();
    }
  }
}
async function action(work) {
  // 只印伺服器給的中文訊息，不帶 JS 的英文字首「Error: 」。
  try { await work(); } catch (error) { $("messages").textContent = error instanceof Error ? error.message : String(error); }
}
window.addEventListener("DOMContentLoaded", () => action(async () => {
  const example = await api("/api/example"); scheme = example.scheme;
  $("feature-note").textContent = example.feature_match_note;
  $("rho-c").textContent = example.rho_c_label;
  renderForm();
  // 按鈕先綁：下面任一份清單載入失敗，也不能讓整頁按鈕都沒反應。
  for (const id of ["room-fields", "walls", "scattering", "speakers", "receivers", "source-model", "use-scattering"])
    $(id).addEventListener("input", markStale);
  $("walls").addEventListener("input", () => action(updateMultiples));
  $("check").onclick = () => action(refreshPlan);
  $("save").onclick = () => action(save);
  $("open-scheme").onclick = () => action(openScheme);
  $("save-as").onclick = () => action(saveAs);
  $("calculate").onclick = () => action(async () => {
    if (!await save()) return;
    const state = await api("/api/runs", "POST", {scheme_id: scheme.scheme_id});
    // 真的開算了才清記號：存檔被擋（例如正在算的那一份改不得）時，記號要留著。
    runEdited = false;
    runId = state.run_id; $("stop").disabled = false;
    // 上一筆的計時器先停：留著的話它會每秒去跑算完那一段，蓋掉訊息、藏掉舊結果標示。
    clearInterval(timer);
    await poll();
    if (!$("stop").disabled) timer = setInterval(() => action(poll), 1000);
  });
  $("stop").onclick = () => action(async () => { await api(`/api/runs/${runId}/stop`, "POST"); await poll(); });
  await refreshPlan();
  for (const load of [loadSchemeList, loadResultList, resumeRuns]) await action(load);
}));
