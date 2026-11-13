/* 結果頁的分區附表（#1092）：逐區表、原牆基準、改善多少。字與數字都由伺服器排好，這裡只照印。
   node／tableElement 借 results.js 的（同一頁的一般腳本共用全域）。
   查附表（GET）不會開算；只有按「算原牆基準」才開。算中只輪詢那一筆計算的狀態（輕），算完才重讀整份附表。 */
let zonePollTimer;
// 位置與尺寸、原牆基準與它怎麼來，各併成一格兩行（伺服器的字照原樣接，不另外算）。
// 材料兩格先寫名字，收位後的工程參數收在「參數」裡（#1119）；沒有參數的格（同左、—）不出細節。
function materialDetail(cell, detail) {
  if (!detail) return;
  const box = document.createElement("details"); box.className = "zone-material-detail";
  box.append(node("summary", "參數"), node("div", detail));
  cell.append(box);
}
// 名字旁邊的標語、名字底下的出處（#1082 段 2-9）：標語同材料頁（實測／近似／模擬，滑過去看整句）；
// 出處與網址、實測表自己帶的量測出處與條件緊接在名字下面；實測表逐列數字收在自己那一層、預設收起來。
function materialCell(cell, name, facts, detail, tail) {
  cell.replaceChildren(name);
  for (const tag of facts.tags) {
    const badge = node("span", tag.text); badge.className = `badge basis-${tag.code}`; badge.title = tag.title;
    cell.append(" ", badge);
  }
  if (tail) cell.append(`\n${tail}`);
  if (facts.source) {
    const line = node("p", `出處與條件：${facts.source}`); line.className = "zone-material-source";
    for (const link of facts.links) {
      const anchor = node("a", link); anchor.href = link; anchor.target = "_blank"; anchor.rel = "noopener";
      line.append(" ", anchor);
    }
    cell.append(line);
  }
  if (facts.conditions && facts.conditions !== facts.source) {
    const line = node("p", `量測出處與條件：${facts.conditions}`); line.className = "zone-material-conditions";
    cell.append(line);
  }
  materialDetail(cell, detail);
  if (facts.table_rows.length) {
    const box = document.createElement("details"); box.className = "zone-measured-table";
    const list = document.createElement("ol");
    list.append(...facts.table_rows.map((text) => node("li", text)));
    box.append(node("summary", facts.table), list);
    cell.append(box);
  }
}
function zoneRowsTable(rows) {
  const element = tableElement(["牆", "區", "位置與尺寸（公尺）", "做法", "設計的材料", "原牆基準的材料（怎麼來）"],
    rows.map((row) => [row.wall, row.zone, `${row.place}\n${row.size}，${row.area}`, row.mode, row.design,
      `${row.baseline}\n（${row.baseline_source}）`]));
  rows.forEach((row, index) => {
    const line = element.rows[index + 1];
    line.dataset.changed = String(row.changed);
    // 有白話一句就先寫白話，原本的名字（含原牆）併進「參數」（#1134）。
    materialCell(line.cells[4], row.design_plain || row.design, row.design_facts,
      row.design_plain ? `${row.design}\n${row.design_detail}` : row.design_detail, "");
    materialCell(line.cells[5], row.baseline, row.baseline_facts, row.baseline_detail, `（${row.baseline_source}）`);
  });
  return element;
}
function zoneImprovementTable(improvement) {
  const lines = [...improvement.rows, ...(improvement.total ? [improvement.total] : [])];
  const element = tableElement(["類別", "原牆基準（扣分）", "這一套（扣分）", "差多少（扣分越低越好，綠色＝改善）"],
    lines.map((row) => [row.category, row.baseline, row.design, row.change]));
  lines.forEach((row, index) => { element.rows[index + 1].cells[3].dataset.verdict = row.verdict; });
  if (improvement.total) element.rows[lines.length].classList.add("zone-total");
  return element;
}
// 「這一間挑材料」只給存著的方案（第三階段頁只收存檔裡的方案）；讀不到清單就不給，不猜（#1134）。
async function showMaterialsLink() {
  try {
    const listing = await (await fetch("/api/schemes")).json();
    if (!listing.schemes.includes(view.scheme_id)) return;
    $("next-materials").href = `/gradient?scheme=${encodeURIComponent(view.scheme_id)}`;
    $("next-materials").hidden = false;
  } catch { /* 清單讀不到：不給這個連結 */ }
}
function drawZoneResult(runId, data) {
  $("zone-result").hidden = false;
  showMaterialsLink();
  $("zone-notes").replaceChildren(...data.notes.map((text) => node("p", text)));
  $("zone-rows").replaceChildren(zoneRowsTable(data.rows));
  $("zone-state").textContent = data.state_text;
  if (!data.running) $("zone-start-state").textContent = "";
  $("zone-baseline-start").hidden = !data.can_start;
  $("zone-baseline-start").disabled = false;
  $("zone-baseline-start").onclick = () => startZoneBaseline(runId);
  $("zone-baseline-link").hidden = !data.baseline_result_url;
  if (data.baseline_result_url) $("zone-baseline-link").querySelector("a").href = data.baseline_result_url;
  const improvement = data.improvement;
  $("zone-improvement").replaceChildren(...(improvement ? [zoneImprovementTable(improvement)] : []));
  $("zone-total-note").textContent = improvement ? improvement.total_note : "";
  $("zone-improvement-note").textContent = improvement ? improvement.note : "";
  clearTimeout(zonePollTimer);
  if (data.running && data.baseline_run_id) zonePollTimer = setTimeout(() => pollZoneBaseline(runId, data.baseline_run_id), 5000);
}
async function loadZoneResult(runId) {
  let response, data;
  try {
    response = await fetch(`/api/results/${runId}/zones`);
    data = await response.json();
  } catch {
    $("zone-result").hidden = false;
    $("zone-state").textContent = "分區附表沒有讀到：連不上網頁伺服器，重新整理這一頁再試";
    return;
  }
  if (!response.ok) {
    $("zone-result").hidden = false;
    $("zone-state").textContent = data.server_notice || data.error || `分區附表沒有讀到（網頁伺服器回應 ${response.status}）`;
    return;
  }
  if (!data.zoned) { $("zone-result").hidden = true; return; }
  drawZoneResult(runId, data);
}
async function pollZoneBaseline(runId, baselineRunId) {
  let state;
  try { state = await (await fetch(`/api/runs/${baselineRunId}`)).json(); } catch { state = {status: "running"}; }
  if (state.status === "running") {
    if (state.display_text) $("zone-state").textContent = state.display_text;
    zonePollTimer = setTimeout(() => pollZoneBaseline(runId, baselineRunId), 5000);
    return;
  }
  loadZoneResult(runId);
}
// 按下就停用：再按一次會多開一份半小時的計算；沒開始才放回來。
async function startZoneBaseline(runId) {
  const trigger = $("zone-baseline-start");
  trigger.disabled = true;
  $("zone-start-state").textContent = "正在送出…";
  let response, data;
  try {
    response = await fetch(`/api/results/${runId}/zones/baseline`, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
    data = await response.json();
  } catch {
    trigger.disabled = false;
    $("zone-start-state").textContent = "沒有送出：連不上網頁伺服器；確認伺服器開著再按一次";
    return;
  }
  $("zone-start-state").textContent = response.ok ? "原牆基準開始算了；算完這一節會自己換成改善多少。" :
    (data.server_notice || data.error || `原牆基準沒有開始（網頁伺服器回應 ${response.status}）`);
  if (!response.ok) { trigger.disabled = false; return; }
  loadZoneResult(runId);
}
