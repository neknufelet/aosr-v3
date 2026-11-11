/* 結果頁的分區附表（#1092）：逐區表、原牆基準、改善多少。字與數字都由伺服器排好，這裡只照印。
   node／tableElement 借 results.js 的（同一頁的一般腳本共用全域）。
   查附表（GET）不會開算；只有按「算原牆基準」才開。算中只輪詢那一筆計算的狀態（輕），算完才重讀整份附表。 */
let zonePollTimer;
// 位置與尺寸、原牆基準與它怎麼來，各併成一格兩行（伺服器的字照原樣接，不另外算）。
function zoneRowsTable(rows) {
  const element = tableElement(["牆", "區", "位置與尺寸（公尺）", "做法", "設計的材料", "原牆基準的材料（怎麼來）"],
    rows.map((row) => [row.wall, row.zone, `${row.place}\n${row.size}，${row.area}`, row.mode, row.design,
      `${row.baseline}\n（${row.baseline_source}）`]));
  rows.forEach((row, index) => { element.rows[index + 1].dataset.changed = String(row.changed); });
  return element;
}
function zoneImprovementTable(improvement) {
  const lines = [...improvement.rows, ...(improvement.total ? [improvement.total] : [])];
  const element = tableElement(["類別", "原牆基準", "設計", "差（越低越好）"],
    lines.map((row) => [row.category, row.baseline, row.design, row.change]));
  lines.forEach((row, index) => { element.rows[index + 1].cells[3].dataset.verdict = row.verdict; });
  if (improvement.total) element.rows[lines.length].classList.add("zone-total");
  return element;
}
function drawZoneResult(runId, data) {
  $("zone-result").hidden = false;
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
