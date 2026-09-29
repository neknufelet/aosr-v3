/* 比較頁只排版伺服器給的文字與曲線資料。 */
const $ = (id) => document.getElementById(id);
let view;
let plot;
const [aId, bId] = location.pathname.split("/").slice(-2);
const LINE_COLORS = ["#1b6f8a", "#c0392b"];

function node(name, value) {
  const element = document.createElement(name);
  element.textContent = value;
  return element;
}
function table(target, headings, rows) {
  target.replaceChildren();
  const element = document.createElement("table");
  const head = document.createElement("tr");
  for (const heading of headings) head.append(node("th", heading));
  element.append(head);
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const value of row) line.append(node("td", value));
    element.append(line);
  }
  target.append(element);
}
function selected(pair) {
  return [pair.a_key, pair.b_key].map((key) =>
    view.overlay.series.find((item) => item.key === key));
}
function drawChart(pair) {
  const lines = selected(pair);
  if (plot) plot.destroy();
  $("chart").replaceChildren();
  // 圖例只列兩條線的名字，不跟著游標印數值（uPlot 預設會多一格英文「Value」）。
  plot = new uPlot({width: Math.min($("chart").clientWidth || 900, 900), height: 420,
    // 圖例記號跟線一樣分實線、虛線，不只靠顏色分 A、B。
    legend: {live: false, markers: {dash: (u, i) => (u.series[i].dash || []).length ? "dashed" : "solid"}},
    scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}},
    axes: [{label: view.labels.frequency_axis}, {label: view.labels.level_axis}],
    series: [{}, ...lines.map((item, index) => ({
      label: item.legend_text, stroke: LINE_COLORS[index], width: 1.5,
      dash: index === 1 ? [8, 5] : [], spanGaps: true}))]},
    [view.overlay.frequency_hz, ...lines.map((item) => item.levels_db)], $("chart"));
}
function drawPairs() {
  const target = $("pair-buttons"); target.replaceChildren();
  for (const pair of view.overlay.pairs) {
    const button = node("button", pair.label);
    button.type = "button";
    button.onclick = () => choosePair(pair);
    target.append(button);
  }
  const first = view.overlay.pairs[0];
  if (first) choosePair(first);
}
function choosePair(pair) {
  // 目前選的那一顆按鈕要看得出來（aria-pressed，樣式在 style.css）。
  view.overlay.pairs.forEach((item, index) =>
    $("pair-buttons").children[index].setAttribute("aria-pressed", String(item === pair)));
  drawChart(pair);
}
function identity(side, letter) {
  return `${letter}：${side.scheme_id}；引擎提交：${side.engine_text}；日期：${side.run_date}；全程：${side.total_text}`;
}
function draw() {
  $("identity-a").textContent = identity(view.a, "A");
  $("identity-b").textContent = identity(view.b, "B");
  $("summary-text").textContent = view.summary_text;
  $("table-a").textContent = `A：${view.table.a_text}`;
  $("table-b").textContent = `B：${view.table.b_text}`;
  $("table-reason").textContent = view.table.reason_text;
  $("table-calibration").textContent = view.table.calibration_text;
  $("content").hidden = false;
  drawPairs();
  if (view.changes.length) {
    table($("changes"), ["項目", "A", "B"],
      view.changes.map((item) => [item.label, item.a_text, item.b_text]));
  } else $("changes").append(node("p", "兩份方案設定相同"));
  for (const side of ["a", "b"]) {
    drawPlan(view.plans[side], {planXY: `plan-${side}-xy`, planXZ: `plan-${side}-xz`,
      detail: $(`plan-${side}-detail`), legend: $(`plan-${side}-legend`)},
    view.plan_scale_room, view.changed_keys);
  }
  table($("fingerprints"), ["指紋", "核對"],
    view.fingerprints.map((item) => [item.label, item.text]));
  table($("categories"), ["類別", "A 狀態", "A 代價", "B 狀態", "B 代價", "說明"],
    view.categories.map((item) => [item.label, item.a.state_label, item.a.cost_text,
      // 兩邊說明一樣（例如都寫尚未評估）只印一次。
      item.b.state_label, item.b.cost_text, [...new Set([item.a.note, item.b.note, item.comparison_text].filter(Boolean))].join("；")]));
  const notes = $("notes"); notes.replaceChildren();
  for (const note of view.notes) notes.append(node("p", note));
}
async function rerun(url) {
  const response = await fetch(url, {method: "POST",
    headers: {"Content-Type": "application/json"}, body: "{}"});
  const data = await response.json();
  $("rerun-state").textContent = response.ok ? `已開始重算，計算代號：${data.run_id}` :
    (data.error || `這份結果的方案過不了現行檢查，請在輸入頁重新存一份再算：\n${
      (data.problems || []).map((item) => `${item.path}：${item.message}`).join("\n")}`);
}
function reject(response, data) {
  const reason = $("reject-reason"); reason.replaceChildren();
  $("rerun-holder").replaceChildren();
  if (Array.isArray(data.problems)) {
    const list = document.createElement("ul");
    for (const problem of data.problems) list.append(node("li", problem));
    reason.append(list);
    reason.append(node("p", "（第 1 份是 A，第 2 份是 B）"));
    reason.append(node("p", "這兩份不能直接比較；要比較請確認兩份是不同方案、在同一版引擎下算的"));
  } else if (data.rejected) {
    reason.append(node("p", `${data.side.toUpperCase()} 讀回被拒收：${data.reason}`));
    const button = node("button", "用現在的引擎重算這一份");
    button.onclick = () => rerun(data.rerun_url);
    $("rerun-holder").append(button);
  } else reason.append(node("p", data.error || data.reason));
  $("rejection").hidden = false;
}
async function load() {
  let response, data;
  try {
    response = await fetch(`/api/compare/${aId}/${bId}`);
    data = await response.json();
  } finally { $("loading").hidden = true; }
  if (!response.ok) { reject(response, data); return; }
  view = data; draw();
}
window.addEventListener("DOMContentLoaded", () => load().catch((error) => {
  $("loading").hidden = true;
  $("rejection").hidden = false;
  $("reject-reason").textContent = String(error);
}));
