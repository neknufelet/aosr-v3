/* 結果頁只排版伺服器資料；聲學數值、暫定線與文字都由伺服器給。 */
const $ = (id) => document.getElementById(id);
let view;
let selectedRole;
let selectedPair;
let plot;
const resultId = location.pathname.split("/").pop();

function label(code) { return view.labels[code] || "尚無中文標籤"; }
// 喇叭、座位的顯示名由伺服器照顯示名稱表給；表上沒有的伺服器已填原代號。
function speakerName(role) { return view.speaker_names[role] || role; }
function pointName(id) { return view.point_names[id] || id; }
function node(name, value) {
  const element = document.createElement(name);
  element.textContent = value;
  return element;
}
function tableElement(headings, rows) {
  const element = document.createElement("table");
  const head = document.createElement("tr");
  for (const value of headings) head.append(node("th", value));
  element.append(head);
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const value of row) line.append(node("td", value));
    element.append(line);
  }
  return element;
}
function table(target, headings, rows) {
  const element = tableElement(headings, rows);
  target.replaceChildren(element);
  return element;
}
// 摺疊區：預設收起，摘要行由伺服器的一句話說裡面有什麼。
function folded(summary, content) {
  const element = document.createElement("details");
  element.append(node("summary", summary), content);
  return element;
}
function button(target, caption, handler) {
  const element = node("button", caption);
  element.type = "button";
  element.onclick = handler;
  target.append(element);
  return element;
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
    // 圖例只列每條線的名字與顏色，不跟著游標印數值（uPlot 預設會多一格英文「Value」和一排「--」）。
    legend: {live: false},
    scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}},
    axes: [{label: view.labels.frequency_axis}, {label: view.labels.level_axis}],
    series: [{}, ...responses.map((item, index) => ({
      label: `${speakerName(item.role)} → ${pointName(item.receiver_id)}`,
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
  const heading = node("h3", selectedPair.text);
  $("pair-detail").append(heading, tableElement(["量", "位置差", "暫定線", "超出多少", "狀態"], rows));
}
// 目前選的那一顆按鈕看得出來（aria-pressed）；選的不是周圍點彼此時，下拉選單回到提示那一格。
function markPairs() {
  for (const element of $("pair-buttons").querySelectorAll("button"))
    element.setAttribute("aria-pressed", String(element.choice ? element.choice === selectedPair : !selectedPair));
  const select = $("peer-pairs");
  if (!select) return;
  if (!(selectedPair && selectedPair.group === "surrounding_to_surrounding")) select.value = "";
  // 選單選到一對時，選單本身也亮起來（跟按鈕一樣看得出目前選的是哪一對）。
  select.classList.toggle("chosen", select.value !== "");
}
function choosePair(pair) { selectedPair = pair; markPairs(); drawChart(); drawPair(); }
function drawPairChoices() {
  const holder = $("pair-buttons"); holder.replaceChildren();
  const choices = view.listening_area.pair_choices.filter((item) => item.role === selectedRole);
  if (!choices.length) return;
  button(holder, "全部位置", () => choosePair(null));
  // 主位對周圍點：前面寫一次組名，按鈕只寫周圍點那一端（伺服器給的 button_text），六顆加選單才排得進一列。
  const primary = choices.filter((item) => item.group === "primary_to_surrounding");
  if (primary.length) {
    const group = node("span", `${label("primary_to_surrounding")}：`);
    group.className = "pair-group";
    holder.append(group);
  }
  for (const choice of primary) {
    const element = button(holder, choice.button_text, () => choosePair(choice));
    element.choice = choice;
  }
  // 周圍點彼此的配對多（六個周圍點就十五對），收進一個下拉選單。
  const peers = choices.filter((item) => item.group === "surrounding_to_surrounding");
  if (peers.length) {
    const select = document.createElement("select");
    select.id = "peer-pairs";
    select.setAttribute("aria-label", label("surrounding_to_surrounding"));
    select.append(new Option(`${label("surrounding_to_surrounding")}：選一對`, ""));
    peers.forEach((choice, index) => select.append(new Option(choice.text, String(index))));
    // 選回提示那一格＝不選任何一對：回到全部曲線（不然圖上還是剛才那一對，按鈕與選單卻看不出來）。
    select.onchange = () => choosePair(select.value === "" ? null : peers[Number(select.value)]);
    holder.append(select);
  }
  markPairs();
}
function drawListening() {
  $("listening-scope").textContent = view.listening_area.scope_note;
  $("listening-status").textContent = view.listening_area.state === "unavailable" ?
    `不可估：${view.listening_area.reason_codes.map(label).join("、")}` : "";
  drawPairChoices();
  const summaries = view.listening_area.summaries.filter((item) => item.role === selectedRole);
  const element = table($("summary"), ["喇叭", "量", "組", "重要性加權平均", "最差位置對", "最差差值", "最差差距暫定線", "超出多少"],
    summaries.map((item) => [speakerName(item.role), label(item.metric), label(item.group),
      `${item.weighted_mean_text} ${item.unit}`, item.worst_pair_text,
      `${item.worst_value_text} ${item.unit}`, "",
      item.over_limit ? `超過 ${item.excess_text} ${item.unit}` : item.excess_text]));
  // 「最差差距暫定線」那一格：線與「尚未正式校準」各自不拆開，放不下時在分號後面換行（不會剩一個「準」字在下一行）。
  summaries.forEach((item, index) => {
    const parts = [`${item.limit_text} ${item.unit}`, item.baseline_note].filter(Boolean);
    element.rows[index + 1].cells[6].replaceChildren(
      ...parts.map((part, at) => node("span", at < parts.length - 1 ? `${part}；` : part)));
  });
  drawPair();
}
function markSpeakers() {
  for (const element of $("speaker-buttons").children)
    element.setAttribute("aria-pressed", String(element.dataset.role === selectedRole));
}
function drawSpeakers() {
  const target = $("speaker-buttons"); target.replaceChildren();
  for (const role of [...new Set(view.frequency_responses.map((item) => item.role))]) {
    const element = button(target, speakerName(role), () => {
      selectedRole = role; selectedPair = null; markSpeakers(); drawChart(); drawListening();
    });
    element.dataset.role = role;
  }
  markSpeakers();
}
function drawCategories() {
  $("ranking-state").textContent = view.ranking_text;
  $("ranking-approximation").textContent = view.ranking_approximation_text;
  $("ranking-approximation").hidden = !view.ranking_approximation_text;
  $("cost-note").textContent = view.cost_note;
  table($("categories"), ["類別", "狀態", "代價", "注意事項", "說明"],
    view.categories.map((item) => [label(item.category), item.state_label, item.cost_text,
      item.flags_text, item.note]));
  // 評估器版本與原因碼是給查問題用的，收在「技術細節」摺疊區。
  table($("category-technical"), ["類別", "評估器版本", "原因碼"],
    view.categories.map((item) => [label(item.category), item.evaluator_version || "—",
      item.reason_codes.map(label).join("、") || "無"]));
}
function fieldsText(fields) { return fields.map(([name, value]) => `${name}：${value}`).join("；"); }
function drawAlerts() {
  const target = $("alerts"); target.replaceChildren();
  if (view.furniture_flutter_text) target.append(node("p", view.furniture_flutter_text));
  if (!view.alerts.length && !view.flutter_groups.length) { target.append(node("p", "沒有警戒")); return; }
  for (const item of view.alerts) {
    const block = document.createElement("article");
    block.append(node("h3", item.heading_text));
    block.append(node("p", [item.place_text, fieldsText(item.fields)].join("；")));
    if (item.excess_text !== null)
      block.append(node("p", `超出多少：${item.excess_text}${item.baseline_note ? `；${item.baseline_note}` : ""}`));
    target.append(block);
  }
  // 牆間顫動：一對牆一筆，逐帶明細收進摺疊區。
  for (const group of view.flutter_groups) {
    const block = document.createElement("article");
    block.className = "flutter";
    block.append(node("h3", group.heading_text), node("p", fieldsText(group.fields)),
      node("p", group.summary_text),
      folded(group.detail_summary_text, tableElement(["名義中心頻率", "中心頻率", "持續度", "本房同帶 T20", "超出多少"],
        group.bands.map((band) => [band.nominal_text, band.center_text, band.duration_text,
          band.room_t20_text, band.excess_text]))));
    target.append(block);
  }
}
function drawReverb() {
  $("reverb-note").textContent = view.reverberation.caption_text;
  $("reverb-compare-note").textContent = view.reverberation.compare_note;
  const element = table($("reverb"), ["頻帶", "T20（秒）", "T30（秒）", "目標下限", "目標上限", "跟目標比", "量得到嗎"],
    view.reverberation.bands.map((item) => [item.center_text, item.t20_text, item.t30_text,
      item.target_low_text || "—", item.target_high_text || "—", item.verdict_text, item.measured_text]));
  // 「跟目標比」照伺服器的判定上色，只是好認；判定本身是伺服器給的。
  view.reverberation.bands.forEach((item, index) => {
    element.rows[index + 1].cells[5].dataset.verdict = item.verdict;
  });
}
const PATH_HEADINGS = ["延遲", "相對直達音量", "水平角", "仰角", "方向", "反射經過的面"];
function pathRows(paths) {
  return paths.map((path) => [path.delay_text, path.level_text, path.azimuth_text,
    path.elevation_text, label(path.zone), path.surface_text || path.wall_sequence.join(" → ")]);
}
function drawReflections() {
  const target = $("reflections"); target.replaceChildren();
  // 一支喇叭一塊（article 有分隔線），兩支的表才不會黏在一起。
  for (const channel of view.reflections) {
    const block = document.createElement("article");
    block.append(node("h3", channel.heading_text));
    // 「路徑數值驗證」只講反射路徑算到的階數在數值驗證範圍內；注意事項是伺服器給的白話句子，自己一行。
    block.append(node("p", `結論：${label(channel.state)}；涵蓋：${channel.coverage_text}；路徑數值驗證：${channel.validation_text}；原因：${channel.reason_codes.map(label).join("、") || "無"}`),
      node("p", `注意事項：${channel.flags_text || "無"}`));
    if (channel.window_text) block.append(node("p", channel.window_text));
    // 時間窗內的直接列；窗外的收進摺疊區，摘要行說有幾條。
    const inside = channel.paths.filter((path) => path.within_window);
    const outside = channel.paths.filter((path) => !path.within_window);
    if (inside.length) block.append(tableElement(PATH_HEADINGS, pathRows(inside)));
    if (outside.length)
      block.append(folded(channel.outside_summary_text, tableElement(PATH_HEADINGS, pathRows(outside))));
    target.append(block);
  }
}
function drawFurniture() {
  $("furniture").hidden = !view.furniture_reason;
  if (view.furniture.length) table($("furniture-list"), ["家具", "材質", "未知頻帶"], view.furniture);
  $("furniture-reason").textContent = view.furniture_reason;
  const notes = $("furniture-notes"); notes.replaceChildren();
  for (const text of view.furniture_notes) notes.append(node("p", text));
  $("frequency-furniture-note").hidden = !view.frequency_note;
  $("frequency-furniture-note").querySelector("a").textContent = view.frequency_note;
}
// 重算：按下就停用，開始了就一直停用（再按一次會多起一份好幾分鐘的計算）；沒開始才放回來。
async function rerun(url, stateId, trigger) {
  const target = $(stateId);
  trigger.disabled = true;
  target.textContent = "正在送出重算…";
  let response;
  try {
    response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  } catch {
    trigger.disabled = false;
    target.textContent = "重算沒有送出：連不上網頁伺服器；確認伺服器開著再按一次";
    return;
  }
  const data = await response.json().catch(() => ({}));
  if (response.ok) { rerunStarted(target, data); return; }
  trigger.disabled = false;
  target.textContent = data.error || (data.problems ?
    `這份結果的方案過不了現在的檢查；請在方案輸入頁打開這個方案、改好下面幾項，另存新名字再算：\n${
      (data.problems || []).map((item) => item.text).join("\n")}` : `重算沒有開始（網頁伺服器回應 ${response.status}）`);
}
// 開始了：說要等多久（伺服器給的參考秒數）、新結果去哪裡看；計算代號不印。
function rerunStarted(target, data) {
  const wait = Number.isFinite(data.reference_s) ? `，參考時間約 ${Math.round(data.reference_s / 60)} 分鐘` : "";
  const link = node("a", "前往方案輸入頁");
  link.href = "/";
  target.replaceChildren(`已開始重算${wait}。\n計算進度和算好的新結果都在方案輸入頁看（新結果會列在「結果清單」）；這一頁不會自己換成新結果。`, link);
}
function showRejection(response, data) {
  if (data.server_notice) {
    $("server-notice").hidden = false;
    $("server-notice").textContent = data.server_notice;
  }
  // 舊格式：伺服器給種類與一句白話；技術原因收進摺疊區，要查時再點開。
  const oldFormat = !data.server_notice && data.reason_kind === "old_format";
  $("rejection").hidden = false;
  $("rejection-title").textContent = data.server_notice ? "網頁伺服器要重開" : oldFormat ? "舊格式的結果" :
    (response.status === 409 ? "結果被拒收" : "結果讀取失敗");
  $("reject-reason").textContent = data.server_notice ? "" : oldFormat ? data.reason_text :
    (response.status === 409 ? `被拒收：${data.reason}` : `伺服器回應 ${response.status}：${data.error || data.reason}`);
  $("reject-detail").hidden = !oldFormat;
  $("reject-technical").textContent = oldFormat ? data.reason : "";
  $("rerun").hidden = response.status !== 409 || !!data.server_notice || !data.rerun_url;
  if (!$("rerun").hidden) $("rerun").onclick = () => rerun(data.rerun_url, "rerun-state", $("rerun"));
}
async function load() {
  let response, data;
  try {
    response = await fetch(`/api/results/${resultId}`);
    data = await response.json();
  } finally { $("loading").hidden = true; }
  $("run-notice").hidden = !data.run_notice;
  $("run-notice").textContent = data.run_notice || "";
  if (!response.ok) { showRejection(response, data); return; }
  view = data; $("content").hidden = false;
  drawPlan(view.plan, {planXY: "result-plan-xy", planXZ: "result-plan-xz",
    detail: $("result-plan-detail"), legend: $("result-plan-legend")});
  drawCapabilities(view);
  $("identity").textContent = `方案：${view.scheme_id}；日期：${view.run_date}`;
  $("speaker-setup").textContent = view.speaker_setup_line;
  $("speaker-setup").hidden = !view.speaker_setup_line;
  $("timings").textContent = `計算時間：求解 ${view.timing_texts.solve_s}；輸出 ${view.timing_texts.output_s}；評估 ${view.timing_texts.evaluate_s}；全程 ${view.timing_texts.total_s}`;
  // 讀回等級與白話由伺服器判；物理改過才給重算網址，版本碼與物理身分收進技術細節。
  $("fingerprint-status").textContent = view.standing_text;
  $("fingerprint-status").classList.toggle("notice", view.standing !== "current");
  // 技術細節只在讀到結果時才有東西：拒收頁不顯示這個空的摺疊區。
  $("header-technical").textContent = `程式版本碼：${view.engine_commit_text}；物理身分前 12 碼：${view.fingerprint_text}`;
  $("header-details").hidden = false;
  $("fingerprint-rerun").hidden = !view.rerun_url;
  if (view.rerun_url) $("fingerprint-rerun").onclick = () =>
    rerun(view.rerun_url, "fingerprint-rerun-state", $("fingerprint-rerun"));
  selectedRole = view.frequency_responses[0]?.role;
  drawSpeakers(); drawChart(); drawListening(); drawCategories(); drawAlerts(); drawReverb(); drawReflections(); drawFurniture();
  loadModalDiagnosis(resultId);
}
window.addEventListener("DOMContentLoaded", () => load().catch((error) => {
  $("loading").hidden = true;
  $("rejection").hidden = false; $("reject-reason").textContent = String(error);
}));
