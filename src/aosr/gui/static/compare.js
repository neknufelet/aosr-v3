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
// 頁首：主畫面只寫哪個方案、哪天算的、花多久，計算版本相同或不同由伺服器判；
// 計算指紋與程式提交代號收進摺起來的技術細節。
function identity(side, letter) {
  return `${letter}：${side.scheme_id}；計算日期 ${side.run_date}；計算時間（全程）${side.total_text}`;
}
function drawIdentity() {
  $("identity-a").textContent = identity(view.a, "A");
  $("identity-b").textContent = identity(view.b, "B");
  $("version-text").textContent = view.version_text;
  $("identity-codes").textContent = ["a", "b"].map((side) =>
    `${side.toUpperCase()}：計算指紋前 12 碼 ${view[side].fingerprint_text}、程式提交代號 ${view[side].engine_text}`).join("；");
  $("identity").hidden = false;
}
function drawSummary() {
  $("summary-text").textContent = view.summary_text;
  $("table-reason").textContent = view.table.reason_text;
  $("table-a").textContent = `A：${view.table.a_text}`;
  $("table-b").textContent = `B：${view.table.b_text}`;
  // 比較好的那一份（伺服器判）加重標出；相同或不列總代價就都不標，判勝負那句也不用綠色粗體。
  $("table-a").classList.toggle("better", view.table.better === "a");
  $("table-b").classList.toggle("better", view.table.better === "b");
  $("table-verdict").classList.toggle("decided", ["a", "b"].includes(view.table.better));
  for (const [id, text] of [["table-verdict", view.table.verdict_text],
    ["table-review", view.table.review_text], ["pending-text", view.pending_text]]) {
    $(id).textContent = text;
    $(id).hidden = !text;
  }
  $("table-calibration").textContent = view.table.calibration_text;
}
// 寬螢幕兩欄：「座位與聲道設定核對」接在右欄「改了哪裡」下面、左欄摘要下面、或兩欄底下橫跨整排，
// 三種擺法挑兩欄高度最接近的那一種（改了很多處時接在摘要下面，摘要卡下面不留一大片空白）；
// 窄螢幕上下排時照閱讀順序接在「改了哪裡」後面。每張卡照內容自己的高度（compare.css），量到的就是自然高度。
function placeCheck() {
  const top = $("compare-top"), check = $("compare-check");
  const tryPlace = (where) => {
    (where === "full" ? top : $(`compare-${where}`)).append(check);
    const [left, right] = ["compare-left", "compare-right"].map((id) => $(id).getBoundingClientRect());
    return {where, stacked: left.left === right.left, gap: Math.abs(left.height - right.height)};
  };
  const tried = ["right", "left", "full"].map(tryPlace);
  const best = tried[0].stacked ? tried[0] : tried.reduce((most, next) => (next.gap < most.gap ? next : most));
  tryPlace(best.where);
}
// 圖上記號說明：只留圖上真的畫出來的記號（沒有改動就沒有圓圈、全向聲源沒有指向線）。
function drawPlanKey() {
  const drawn = (selector) => ["a", "b"].some((side) =>
    ["xy", "xz"].some((plane) => $(`plan-${side}-${plane}`).querySelector(selector)));
  const key = $("plan-key");
  key.querySelector('[data-mark="ring"]').hidden = !drawn(".changed-ring");
  key.querySelector('[data-mark="aim"]').hidden = !drawn("line");
}
function drawCategories() {
  const target = $("categories"); target.replaceChildren();
  const element = document.createElement("table");
  const head = document.createElement("tr");
  // 說明欄每一類都空著時不畫（空欄還佔一欄寬，窄畫面會把別欄擠成一字一行）；摘要 CSV 照樣有這一欄。
  const withNotes = view.categories.some((item) => item.note_text);
  for (const heading of ["類別", "A 狀態", "A 代價", "B 狀態", "B 代價", "哪一份較好",
    ...(withNotes ? ["說明"] : [])]) {
    head.append(node("th", heading));
  }
  element.append(head);
  for (const item of view.categories) {
    const line = document.createElement("tr");
    const cells = [item.label, item.a.state_label, item.a.cost_text, item.b.state_label,
      item.b.cost_text, item.better_text].map((value) => node("td", value));
    if (withNotes) {
      cells.push(node("td", item.note_text));
      cells[cells.length - 1].classList.add("note");
    }
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
  drawIdentity();
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
  drawPlanKey();
  // 三項都相同時伺服器給一句話，不畫三列「相同」的表。
  if (view.fingerprints_text) {
    $("fingerprints").replaceChildren(node("p", view.fingerprints_text));
  } else {
    table($("fingerprints"), ["項目", "兩份比對"], view.fingerprints.map((item) => [item.label, item.text]));
  }
  const outdated = Object.entries(view.outdated_schemes || {});
  for (const [side, scheme] of outdated) {
    $("fingerprints").append(node("p", `${side.toUpperCase()}（${scheme}）是用舊程式算的（計算版本跟現在不同）`));
  }
  // 比得成就代表兩份指紋相同；兩份都舊時要說清楚：彼此能比，跟現在算的不能比。
  if (outdated.length > 1) {
    $("fingerprints").append(node("p", "兩份是同一版舊程式算的，彼此可以比較；要跟現在算的結果比，兩份都要重算"));
  }
  drawCategories();
  const notes = $("notes"); notes.replaceChildren();
  for (const note of view.notes) notes.append(node("p", note));
  placeCheck();
}
// 技術細節：摺起來，要查時再點開；主畫面不放英文欄名、雜湊、計算代號與瀏覽器的英文錯誤。
function technical(lines, title = "技術細節") {
  const details = document.createElement("details");
  details.className = "technical";
  details.append(node("summary", title));
  for (const line of lines) details.append(node("p", line));
  return details;
}
// 按下重算：送出期間與開始之後按鈕停用（不會重複開好幾份）；不印計算代號，告訴老闆算完去哪裡找。
// 方案過不了現在的檢查時，逐條印伺服器寫好的白話（表單上的中文欄名加說明，同一句只一條），一條一行
// （#rerun-state 在共用樣式照換行排）。
async function rerun(url, button) {
  button.disabled = true;
  $("rerun-detail").replaceChildren();
  let response;
  try {
    response = await fetch(url, {method: "POST",
      headers: {"Content-Type": "application/json"}, body: "{}"});
  } catch (error) {
    button.disabled = false;
    $("rerun-state").textContent = "重算沒有送出：連不上網頁伺服器；確認伺服器開著再按一次";
    return;
  }
  const data = await response.json().catch(() => ({}));
  button.disabled = response.ok;
  $("rerun-state").textContent = response.ok ?
    "已開始重算。算完後回方案輸入頁的結果清單，選新算好的那一份再比較" : rerunProblem(response, data);
}
// 重算沒開始的原因：伺服器寫好的中文照印；結果檔不見了（404，伺服器只回計算代號）寫一句白話；
// 讀檔、解析出錯的英文原文收進技術細節。
function rerunProblem(response, data) {
  const problems = (data.problems || []).map((item) => item.text);
  if (problems.length) {
    return `這份結果的方案過不了現在的檢查；請在方案輸入頁打開這個方案、改好下面幾項，另存新名字再算：\n${
      problems.join("\n")}`;
  }
  if (response.status === 404) return "重算沒有開始：這份結果檔找不到（可能已被移走），請回方案輸入頁的結果清單重新選";
  if (!data.error) return `重算沒有開始：網頁伺服器回應 ${response.status}`;
  if (/^[一-鿿]/.test(data.error)) return `重算沒有開始：${data.error}`;
  $("rerun-detail").append(technical([data.error], "重算的技術細節"));
  return "重算沒有開始：這份結果檔讀不了（原文在下面「重算的技術細節」）";
}
function rerunButton(side, url) {
  const button = node("button", `用現在的程式重算 ${side.toUpperCase()} 這一份`);
  button.onclick = () => rerun(url, button);
  $("rerun-holder").append(button);
}
// 兩份不能直接比的原因：伺服器給的原句（比較層的拒收理由，含雜湊、英文欄名與「第 1 份、第 2 份」）
// 照字樣認出是哪一種，主畫面換成 A、B 的白話；原句收進技術細節。原句的字樣由考卷拿真的拒收理由餵頁面釘住。
const PROBLEM_SENTENCES = [
  ["計算指紋", "A 和 B 的計算版本不同（算的時候程式或設定不一樣）；要用同一版程式算的兩份才能比較"],
  ["候選代號重複", "A 和 B 的方案代號相同；比較頁只比兩個不同代號的方案。要比同一個方案改前改後，" +
    "請在方案輸入頁把改過的方案另存新名字再算，拿新算好的那一份來比"],
  ["用途（purpose）", "A 和 B 的方案用途不同；用途不同的兩份不能直接比較"],
  ["聲道組指紋", "A 和 B 的聲道設定不同（哪支喇叭接哪個聲道、哪兩個聲道互相比對、峰谷配對容差，" +
    "至少一項不一樣）；聲道設定不同的兩份不能直接比較"],
];
const OTHER_PROBLEM = "A 和 B 有一項固定設定對不上，不能直接比較（原文在下面的技術細節）";
function plainProblems(problems) {
  return [...new Set(problems.map((text) =>
    (PROBLEM_SENTENCES.find(([marker]) => text.includes(marker)) || ["", OTHER_PROBLEM])[1]))];
}
function reject(response, data) {
  const reason = $("reject-reason"); reason.replaceChildren();
  $("rerun-holder").replaceChildren();
  $("server-notice").hidden = !data.server_notice;
  $("server-notice").textContent = data.server_notice || "";
  const problems = Array.isArray(data.problems);
  const rejectedSide = (data.side || "").toUpperCase();
  $("rejection-title").textContent = data.server_notice ? "網頁伺服器要重開" :
    data.reason_kind === "old_format" ? "要先重算才能比較" :
    data.rejected ? `${rejectedSide} 那一份現在的程式讀不了` :
    problems || response.status === 409 ? "這兩份不能直接比較" : "比較讀取失敗";
  if (problems) {
    const list = document.createElement("ul");
    for (const text of plainProblems(data.problems)) list.append(node("li", text));
    reason.append(list);
    for (const side of data.outdated_sides || []) {
      reason.append(node("p", `${side.toUpperCase()}（${data.outdated_schemes[side]}）是用舊程式算的（計算版本跟現在不同）`));
    }
    reason.append(technical(["（原文的第 1 份是 A，第 2 份是 B）", ...data.problems]));
    for (const side of data.outdated_sides || []) {
      if (data.rerun_urls?.[side]) rerunButton(side, data.rerun_urls[side]);
    }
  } else if (data.rejected) {
    // 舊格式那一份伺服器給一句白話（句首已經寫了是哪一份）；其他原因（欄位對不上、登記簿改過、檔案被改過等）
    // 網頁分不出是哪一種，寫一句不多說的白話，伺服器的原因收進技術細節。
    if (data.reason_text) reason.append(node("p", data.reason_text));
    else {
      reason.append(node("p", `${rejectedSide} 那份結果檔的內容跟現在的程式對不上。可以按下面的按鈕用現在的程式重算這一份；` +
        "存的方案過不了現在的檢查時，按下去會列出要改哪幾項"));
      reason.append(technical([data.reason]));
    }
    if (data.rerun_url) rerunButton(data.side, data.rerun_url);
  } else if (!data.server_notice) reason.append(node("p", data.error || data.reason));
  $("rejection").hidden = false;
}
// 讀不到或畫不出來：主畫面寫一句白話，瀏覽器的英文錯誤收進技術細節。
function failed(title, text, error) {
  $("loading").hidden = true;
  $("rejection-title").textContent = title;
  $("reject-reason").replaceChildren(node("p", text), technical([String(error)]));
  $("rejection").hidden = false;
}
async function load() {
  let response, data;
  try {
    response = await fetch(`/api/compare/${aId}/${bId}`);
  } catch (error) {
    failed("比較讀取失敗", "連不上網頁伺服器；確認伺服器開著，再重新整理這一頁", error);
    return;
  }
  try {
    data = await response.json();
  } catch (error) {
    failed("比較讀取失敗", `網頁伺服器的回應讀不懂（回應 ${response.status}）；請重新整理這一頁`, error);
    return;
  }
  $("loading").hidden = true;
  if (!response.ok) { reject(response, data); return; }
  view = data; draw();
}
// 視窗寬度變了，圖跟著區塊寬度重畫（高度不變就不重畫）；核對那一卡重新挑擺法。
window.addEventListener("resize", () => {
  if (!view) return;
  placeCheck();
  if (plot && currentPair && $("chart").clientWidth !== plot.width) drawChart(currentPair);
});
window.addEventListener("DOMContentLoaded", () => load().catch((error) =>
  failed("比較頁畫不出來", "這一頁畫到一半出錯了；請重新整理這一頁", error)));
