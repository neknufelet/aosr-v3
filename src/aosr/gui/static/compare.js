/* 比較頁只排版伺服器給的文字與曲線資料。 */
const $ = (id) => document.getElementById(id);
let view;
let plot;
let currentPair;
const [aId, bId] = location.pathname.split("/").slice(-2);
// A、B 兩條線的顏色與線型（只為分得出哪一條）；圖上與匯出圖片的圖例共用這一份。
// 不讀 uPlot 畫完的 series.stroke：畫完後那一格被換成函式，拿去當顏色會退回黑色。
const LINE_COLORS = ["#1b6f8a", "#c0392b"];
const LINE_DASHES = [[], [8, 5]];

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
  // 圖佔滿整個區塊的寬（區塊本身在寬螢幕是整頁寬）。
  plot = new uPlot({width: $("chart").clientWidth || 900, height: 420,
    // 圖例記號跟線一樣分實線、虛線，不只靠顏色分 A、B。
    legend: {live: false, markers: {dash: (u, i) => (u.series[i].dash || []).length ? "dashed" : "solid"}},
    scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}},
    axes: [{label: view.labels.frequency_axis}, {label: view.labels.level_axis}],
    series: [{}, ...lines.map((item, index) => ({
      label: item.legend_text, stroke: LINE_COLORS[index], width: 1.5,
      dash: LINE_DASHES[index], spanGaps: true}))]},
    [view.overlay.frequency_hz, ...lines.map((item) => item.levels_db)], $("chart"));
}
function choiceButtons(target, choices, onPick) {
  target.replaceChildren();
  for (const choice of choices) {
    const button = node("button", choice.label);
    button.type = "button";
    button.dataset.key = choice.key;
    button.onclick = () => onPick(choice.key);
    target.append(button);
  }
}
function pairFor(channel, position) {
  return view.overlay.pairs.find((item) => item.channel === channel && item.position === position);
}
// 聲道切換加位置按鈕：每一對（聲道、位置）都挑得到，A、B 兩條線一起換。
function drawSwitch() {
  choiceButtons($("channel-buttons"), view.overlay.channels, (channel) =>
    // 換聲道時位置不動；新聲道沒有這個位置就退回那個聲道排第一的那一對。
    choosePair(pairFor(channel, currentPair.position) ||
      view.overlay.pairs.find((item) => item.channel === channel)));
  choiceButtons($("position-buttons"), view.overlay.positions, (position) =>
    choosePair(pairFor(currentPair.channel, position)));
  const first = view.overlay.pairs[0];
  if (first) choosePair(first);
}
function choosePair(pair) {
  if (!pair) return;
  currentPair = pair;
  // 目前選的聲道與位置要看得出來（aria-pressed，樣式在 compare.css）。
  for (const button of $("channel-buttons").children) {
    button.setAttribute("aria-pressed", String(button.dataset.key === pair.channel));
  }
  for (const button of $("position-buttons").children) {
    button.setAttribute("aria-pressed", String(button.dataset.key === pair.position));
    // 這個聲道在這個位置沒有兩份都有的曲線，就不能按。
    button.disabled = !pairFor(pair.channel, button.dataset.key);
  }
  drawChart(pair);
}
// 一行放不下就換行（逐字量，中英混排都行），不縮字、不切掉：窄畫面時但書的後半句最要緊。
function wrapLines(context, text, width) {
  const lines = [];
  let line = "";
  for (const char of text) {
    if (line && context.measureText(line + char).width > width) {
      lines.push(line);
      line = char;
    } else line += char;
  }
  if (line) lines.push(line);
  return lines;
}
function downloadPng() {
  const source = $("chart").querySelector("canvas");
  if (!source || !currentPair || !plot) return;
  // 螢幕倍率：圖上的 canvas 是 CSS 寬度乘倍率；字、位置、線寬跟著乘（虛線照 uPlot 不乘）。
  const scale = source.width / plot.width;
  const font = (size) => `${size * scale}px sans-serif`;
  const room = source.width - 32 * scale;
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  const context = canvas.getContext("2d");
  // 先量好每一段要幾行，再定畫布高度（改高度會清掉畫布與字型設定，所以量完才設）。
  const lines = selected(currentPair);
  context.font = font(16);
  const titleLines = wrapLines(context, `A：${view.a.scheme_id}　B：${view.b.scheme_id}　${currentPair.label}`, room);
  const legendLines = lines.map((item) => wrapLines(context, item.legend_text, room - 54 * scale));
  context.font = font(12);
  const noteLines = wrapLines(context, view.level_note, room);
  const titleHeight = (20 + 22 * titleLines.length) * scale;
  const legendRows = legendLines.reduce((count, rows) => count + rows.length, 0);
  const legendHeight = (25 + 22 * legendRows + 10 + 18 * noteLines.length + 12) * scale;
  canvas.height = titleHeight + source.height + legendHeight;
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(source, 0, titleHeight);
  context.fillStyle = getComputedStyle($("chart")).color;
  context.font = font(16);
  titleLines.forEach((text, row) => context.fillText(text, 16 * scale, (32 + 22 * row) * scale));
  let y = titleHeight + source.height + 25 * scale;
  lines.forEach((item, index) => {
    const line = plot.series[index + 1];
    context.strokeStyle = LINE_COLORS[index];
    context.lineWidth = line.width * scale;
    // uPlot 畫虛線時不乘螢幕倍率，圖例照圖上一樣，不乘。
    context.setLineDash(LINE_DASHES[index]);
    context.beginPath();
    context.moveTo(16 * scale, y);
    context.lineTo(58 * scale, y);
    context.stroke();
    context.setLineDash([]);
    context.font = font(16);
    legendLines[index].forEach((text, row) => context.fillText(text, 70 * scale, y + (5 + 22 * row) * scale));
    y += 22 * legendLines[index].length * scale;
  });
  // 圖片傳出去也要帶著音量基準的但書（字由伺服器給）。
  context.font = font(12);
  noteLines.forEach((text, row) => context.fillText(text, 16 * scale, y + (10 + 18 * row) * scale));
  // 檔名帶著是哪一對，同一組 A、B 換一對再存不會只多一個 (1)。
  const which = currentPair.a_key.split(":").slice(1).join("-");
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `compare-${aId.slice(0, 8)}-${bId.slice(0, 8)}-${which}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
}
function identity(side, letter) {
  return `${letter}：${side.scheme_id}；計算指紋：${side.fingerprint_text}；引擎提交：${side.engine_text}；日期：${side.run_date}；全程：${side.total_text}`;
}
function drawSummary() {
  $("summary-text").textContent = view.summary_text;
  $("table-reason").textContent = view.table.reason_text;
  $("table-a").textContent = `A：${view.table.a_text}`;
  $("table-b").textContent = `B：${view.table.b_text}`;
  // 比較好的那一份（伺服器判）加重標出；相同或不列總代價就都不標。
  $("table-a").classList.toggle("better", view.table.better === "a");
  $("table-b").classList.toggle("better", view.table.better === "b");
  for (const [id, text] of [["table-verdict", view.table.verdict_text], ["pending-text", view.pending_text]]) {
    $(id).textContent = text;
    $(id).hidden = !text;
  }
  $("table-calibration").textContent = view.table.calibration_text;
}
function drawCategories() {
  const target = $("categories"); target.replaceChildren();
  const element = document.createElement("table");
  const head = document.createElement("tr");
  for (const heading of ["類別", "A 狀態", "A 代價", "B 狀態", "B 代價", "哪一份較好", "說明"]) {
    head.append(node("th", heading));
  }
  element.append(head);
  for (const item of view.categories) {
    const line = document.createElement("tr");
    const cells = [item.label, item.a.state_label, item.a.cost_text, item.b.state_label,
      item.b.cost_text, item.better_text, item.note_text].map((value) => node("td", value));
    // 代價比較低的那一格加重（伺服器判）：A 代價是第 3 格、B 代價是第 5 格。
    if (item.better === "a") cells[2].classList.add("better");
    if (item.better === "b") cells[4].classList.add("better");
    line.append(...cells);
    element.append(line);
  }
  target.append(element);
}
function draw() {
  $("download-png").onclick = downloadPng;
  $("download-curves").href = `/api/compare/${aId}/${bId}/export/curves`;
  $("download-summary").href = `/api/compare/${aId}/${bId}/export/summary`;
  $("identity-a").textContent = identity(view.a, "A");
  $("identity-b").textContent = identity(view.b, "B");
  drawSummary();
  $("level-note").textContent = view.level_note;
  $("content").hidden = false;
  drawSwitch();
  if (view.changes.length) {
    table($("changes"), ["項目", "A", "B"],
      view.changes.map((item) => [item.label, item.a_text, item.b_text]));
  } else $("changes").append(node("p", "兩份方案設定相同"));
  for (const side of ["a", "b"]) {
    drawPlan(view.plans[side], {planXY: `plan-${side}-xy`, planXZ: `plan-${side}-xz`,
      detail: $(`plan-${side}-detail`), legend: $(`plan-${side}-legend`)},
    view.plan_scale_room, view.changed_keys);
  }
  table($("fingerprints"), ["項目", "兩份比對"],
    view.fingerprints.map((item) => [item.label, item.text]));
  const outdated = Object.entries(view.outdated_schemes || {});
  for (const [side, scheme] of outdated) {
    $("fingerprints").append(node("p", `${side.toUpperCase()}（${scheme}）是用舊程式算的（計算指紋跟現在不同）`));
  }
  // 比得成就代表兩份指紋相同；兩份都舊時要說清楚：彼此能比，跟現在算的不能比。
  if (outdated.length > 1) {
    $("fingerprints").append(node("p", "兩份是同一版舊程式算的，彼此可以比較；要跟現在算的結果比，兩份都要重算"));
  }
  drawCategories();
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
  $("server-notice").hidden = !data.server_notice;
  $("server-notice").textContent = data.server_notice || "";
  $("rejection-title").textContent = data.server_notice ? "網頁伺服器要重開" :
    data.reason_kind === "old_format" ? "要先重算才能比較" : "比較讀取失敗";
  if (Array.isArray(data.problems)) {
    const list = document.createElement("ul");
    for (const problem of data.problems) list.append(node("li", problem));
    reason.append(list);
    reason.append(node("p", "（第 1 份是 A，第 2 份是 B）"));
    reason.append(node("p", "這兩份不能直接比較；要比較請確認兩份是不同方案、用相同計算指紋算的"));
    for (const side of data.outdated_sides || []) {
      reason.append(node("p", `${side.toUpperCase()}（${data.outdated_schemes[side]}）是用舊程式算的（計算指紋跟現在不同）`));
    }
    for (const side of data.outdated_sides || []) {
      const url = data.rerun_urls?.[side];
      if (!url) continue;
      const button = node("button", `用現在的引擎重算 ${side.toUpperCase()} 這一份`);
      button.onclick = () => rerun(url);
      $("rerun-holder").append(button);
    }
  } else if (data.rejected) {
    // 舊格式那一份伺服器給一句白話（句首已經寫了是哪一份），不再貼技術原因；其他原因照舊寫哪一份被拒收。
    reason.append(node("p", data.reason_text || `${data.side.toUpperCase()} 讀回被拒收：${data.reason}`));
    if (data.rerun_url) {
      const button = node("button", `用現在的引擎重算 ${data.side.toUpperCase()} 這一份`);
      button.onclick = () => rerun(data.rerun_url);
      $("rerun-holder").append(button);
    }
  } else if (!data.server_notice) reason.append(node("p", data.error || data.reason));
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
// 視窗寬度變了，圖跟著區塊寬度重畫（高度不變就不重畫）。
window.addEventListener("resize", () => {
  if (plot && currentPair && $("chart").clientWidth !== plot.width) drawChart(currentPair);
});
window.addEventListener("DOMContentLoaded", () => load().catch((error) => {
  $("loading").hidden = true;
  $("rejection").hidden = false;
  $("reject-reason").textContent = String(error);
}));
