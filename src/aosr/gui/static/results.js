/* 結果頁只排版伺服器資料；聲學數值、暫定線與文字都由伺服器給。 */
const $ = (id) => document.getElementById(id);
let view;
let selectedRole;
let selectedPair;
let plot;
const resultId = location.pathname.split("/").pop();

function label(code) { return view.labels[code] || "尚無中文標籤"; }
function node(name, value) {
  const element = document.createElement(name);
  element.textContent = value;
  return element;
}
function table(target, headings, rows) {
  target.replaceChildren();
  const element = document.createElement("table");
  const head = document.createElement("tr");
  for (const value of headings) head.append(node("th", value));
  element.append(head);
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const value of row) line.append(node("td", value));
    element.append(line);
  }
  target.append(element);
}
function button(target, caption, handler) {
  const element = node("button", caption);
  element.onclick = handler;
  target.append(element);
}
function visibleResponses() {
  const candidates = view.frequency_responses.filter((item) => item.role === selectedRole &&
    (item.receiver_role === "primary" || item.receiver_role === "surrounding"));
  return selectedPair ? candidates.filter((item) =>
    item.receiver_id === selectedPair.receiver_id || item.receiver_id === selectedPair.reference_id) : candidates;
}
// 曲線顏色（只是畫面上分得出哪一條，不帶任何意義）。
const LINE_COLORS = ["#1b6f8a", "#c0392b", "#27ae60", "#8e44ad", "#d68910", "#2c3e50", "#16a085", "#7f8c8d"];
function drawChart() {
  const responses = visibleResponses();
  if (plot) plot.destroy();
  $("chart").replaceChildren();
  if (!responses.length) return;
  const all = view.frequency_responses.filter((item) => item.role === selectedRole);
  const aligned = view.frequency_plot_data[selectedRole];
  const series = responses.map((item) => aligned[all.indexOf(item) + 1]);
  // 每條線要指定顏色：uPlot 的曲線沒給 stroke 就不畫線。x 軸範圍照資料頭尾，不讓對數刻度自動拉到整十倍。
  plot = new uPlot({width: Math.min($("chart").clientWidth || 900, 900), height: 420,
    scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}},
    axes: [{label: view.labels.frequency_axis}, {label: view.labels.level_axis}],
    series: [{}, ...responses.map((item, index) => ({
      label: `${label(item.role)}・${item.receiver_id}（${item.receiver_label}）`,
      stroke: LINE_COLORS[index % LINE_COLORS.length], width: 1.5}))]},
    [aligned[0], ...series], $("chart"));
}
function drawPair() {
  $("pair-detail").replaceChildren();
  if (!selectedPair) return;
  const rows = view.listening_area.pairs.filter((item) => item.role === selectedRole &&
    item.group === selectedPair.group && item.receiver_id === selectedPair.receiver_id &&
    item.reference_id === selectedPair.reference_id).map((item) => [
      label(item.metric), `${item.value_text} ${item.unit}`,
      `${item.limit_text} ${item.unit}`, item.over_limit ? `${item.excess_text} ${item.unit}` : item.excess_text,
      item.status_text,
    ]);
  table($("pair-detail"), ["量", "位置差", "暫定線", "超出多少", "狀態"], rows);
}
function choosePair(pair) { selectedPair = pair; drawChart(); drawPair(); }
function drawListening() {
  $("listening-scope").textContent = view.listening_area.scope_note;
  $("listening-status").textContent = view.listening_area.state === "unavailable" ?
    `不可估：${view.listening_area.reason_codes.map(label).join("、")}` : "";
  const buttons = $("pair-buttons"); buttons.replaceChildren();
  const seen = new Set();
  for (const pair of view.listening_area.pairs.filter((item) => item.role === selectedRole)) {
    const key = `${pair.group}/${pair.receiver_id}/${pair.reference_id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    button(buttons, `${label(pair.group)}：${pair.reference_id} ↔ ${pair.receiver_id}`,
      () => choosePair(pair));
  }
  const summaries = view.listening_area.summaries.filter((item) => item.role === selectedRole);
  table($("summary"), ["喇叭", "量", "組", "重要性加權平均", "最差位置對", "最差差值", "最差差距暫定線", "超出多少"],
    summaries.map((item) => [label(item.role), label(item.metric), label(item.group),
      `${item.weighted_mean_text} ${item.unit}`,
      `${item.worst_reference_id} ↔ ${item.worst_receiver_id}`,
      `${item.worst_value_text} ${item.unit}`, `${item.limit_text} ${item.unit}；${item.baseline_note}`,
      item.over_limit ? `超過 ${item.excess_text} ${item.unit}` : item.excess_text]));
  drawPair();
}
function drawSpeakers() {
  const target = $("speaker-buttons"); target.replaceChildren();
  for (const role of [...new Set(view.frequency_responses.map((item) => item.role))])
    button(target, label(role), () => { selectedRole = role; selectedPair = null; drawChart(); drawListening(); });
}
function drawCategories() {
  $("ranking-state").textContent = `排名位置：${label(view.ranking_status)}；淘汰原因：${view.ranking_reasons.map(label).join("、") || "無"}；缺的類：${view.missing_categories.map(label).join("、") || "無"}`;
  table($("categories"), ["類別", "狀態", "代價", "旗標", "原因碼", "評估器版本", "說明"],
    view.categories.map((item) => [label(item.category), item.state_label, item.cost_text,
      item.flags.map(label).join("、"), item.reason_codes.map(label).join("、"),
      item.evaluator_version || "—", item.note]));
}
function drawAlerts() {
  const target = $("alerts"); target.replaceChildren();
  if (!view.alerts.length) { target.append(node("p", "沒有警戒")); return; }
  for (const item of view.alerts) {
    const block = document.createElement("article");
    block.append(node("h3", `${label(item.kind)}・${item.speaker_id || ""}・${item.role ? label(item.role) : ""}`));
    block.append(node("p", [item.reference_id, item.receiver_id].filter(Boolean).join(" ↔ ")));
    block.append(node("p", item.fields.map(([name, value]) => `${name}：${value}`).join("；")));
    if (item.excess_text !== null)
      block.append(node("p", `超出多少：${item.excess_text}${item.baseline_note ? `；${item.baseline_note}` : ""}`));
    target.append(block);
  }
}
function drawReverb() {
  $("reverb-note").textContent = `${view.reverberation.note}；${label(view.reverberation.role)}・${view.reverberation.receiver_id}`;
  table($("reverb"), ["頻帶", "T20（秒）", "T30（秒）", "目標下限", "目標上限", "狀態與原因"],
    view.reverberation.bands.map((item) => [item.center_text, item.t20_text, item.t30_text,
      item.target_low_text || "—", item.target_high_text || "—",
      [item.t20_state, item.t30_state, ...item.t20_reason_codes, ...item.t30_reason_codes].map(label).join("、")]));
}
function drawReflections() {
  const target = $("reflections"); target.replaceChildren();
  for (const channel of view.reflections) {
    target.append(node("h3", `${label(channel.role)}・${channel.speaker_id}・${channel.receiver_id}`));
    target.append(node("p", `結論：${label(channel.state)}；涵蓋：${label(channel.coverage)}；驗證：${label(channel.validation)}；旗標：${channel.flags.map(label).join("、")}；原因：${channel.reason_codes.map(label).join("、")}`));
    const box = document.createElement("div"); target.append(box);
    table(box, ["延遲", "相對直達音量", "水平角", "仰角", "方向", "牆序列", "時間窗"],
      channel.paths.map((path) => [path.delay_text, path.level_text, path.azimuth_text,
        path.elevation_text, label(path.zone), path.wall_sequence.join(" → "),
        path.within_window ? "窗內" : "窗外"]));
  }
}
async function rerun() {
  const response = await fetch(`/api/results/${resultId}/rerun`, {method: "POST",
    headers: {"Content-Type": "application/json"}, body: "{}"});
  const data = await response.json();
  $("rerun-state").textContent = response.ok ? `已開始重算，計算代號：${data.run_id}` :
    (data.error || `這份結果的方案過不了現行檢查，請在輸入頁重新存一份再算：\n${
      (data.problems || []).map((item) => `${item.path}：${item.message}`).join("\n")}`);
}
async function load() {
  const response = await fetch(`/api/results/${resultId}`);
  const data = await response.json();
  if (!response.ok) {
    $("rejection").hidden = false;
    $("rejection-title").textContent = response.status === 409 ? "結果被拒收" : "結果讀取失敗";
    $("reject-reason").textContent = response.status === 409 ?
      `被拒收：${data.reason}` : `伺服器回應 ${response.status}：${data.error || data.reason}`;
    $("rerun").hidden = response.status !== 409;
    if (response.status === 409) $("rerun").onclick = rerun;
    return;
  }
  view = data; $("content").hidden = false;
  $("identity").textContent = `方案：${view.scheme_id}；引擎提交：${view.engine_commit}；日期：${view.run_date}；求解 ${view.timing_texts.solve_s}；輸出 ${view.timing_texts.output_s}；評估 ${view.timing_texts.evaluate_s}；全程 ${view.timing_texts.total_s}`;
  selectedRole = view.frequency_responses[0]?.role;
  drawSpeakers(); drawChart(); drawListening(); drawCategories(); drawAlerts(); drawReverb(); drawReflections();
}
window.addEventListener("DOMContentLoaded", () => load().catch((error) => {
  $("rejection").hidden = false; $("reject-reason").textContent = String(error);
}));
