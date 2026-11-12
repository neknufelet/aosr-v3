"use strict";
// 開始擺位搜尋頁（#700）：選方案 → 設定（可從之前一場帶入）→ 預覽 → 開始 → 清單、停止、接續、放進結果清單。
// 判斷全交給伺服器（gui/search_jobs.py）；這裡只把表單排成搜尋設定、把回話照實印出來。
// 所有 POST 都帶 content-type: application/json（網頁中介層不收別的）。

const POLL_MS = 5000;
const WALLS = {x0: "x 起點牆", xL: "x 終點牆", y0: "y 起點牆", yL: "y 終點牆"};
const RUNNING = new Set(["starting", "searching", "finishing", "stopping"]);

const byId = (id) => document.getElementById(id);
const params = new URLSearchParams(window.location.search);
let schemeId = params.get("scheme");
let selectedJob = params.get("job");
let facts = null;
let previous = [];
// 最後一次預覽成功送出的設定（JSON 字串）：開始只送這一份；表單改過就要重新預覽，畫面上的預覽才是真的會跑的。
let previewedBody = null;
let knownJobs = [];
let machineBusy = null;
let busyProblem = null;
let polling = false;

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === false || value === null || value === undefined) continue;
    if (name === "text") node.textContent = value;
    else if (name === "on") for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

function say(id, text, bad = false) {
  const node = byId(id);
  node.textContent = text || "";
  node.className = bad ? "notice" : "";
}

async function api(path, body = undefined) {
  const options = body === undefined ? {} :
    {method: "POST", headers: {"content-type": "application/json"}, body: typeof body === "string" ? body : JSON.stringify(body)};
  try {
    const reply = await fetch(path, {cache: "no-store", ...options});
    let data;
    try { data = await reply.json(); } catch {
      data = {error: `伺服器出錯（狀態 ${reply.status}），回的不是 JSON；請把這一行交給維護的人`};
    }
    return {ok: reply.ok, status: reply.status, body: data};
  } catch (error) {
    return {ok: false, status: 0, body: {error: `連不到網頁伺服器：${error}`}};
  }
}

function when(epoch) {
  return new Date(epoch * 1000).toLocaleString("zh-TW", {hour12: false});
}

// ---- 表單 → 搜尋設定 ----

class FormProblem extends Error {}

function label(id) {
  const node = byId(id).closest("label");
  return node ? node.firstChild.textContent.trim() : id;
}

function num(id, {integer = false, required = false} = {}) {
  const raw = byId(id).value.trim();
  if (raw === "") {
    if (required) throw new FormProblem(`「${label(id)}」要填`);
    return null;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new FormProblem(`「${label(id)}」要是${integer ? "整數" : "數字"}`);
  }
  return value;
}

function span(lowId, highId, {required = false} = {}) {
  const low = num(lowId);
  const high = num(highId);
  if (low === null && high === null) {
    if (required) throw new FormProblem(`「${label(lowId)}」兩格都要填`);
    return null;
  }
  if (low === null || high === null) throw new FormProblem(`「${label(lowId)}」要兩格都填，或兩格都空白`);
  return {low, high};
}

const AXES = ["x", "y", "z"];

function boxRow(container, box = null) {
  const row = el("div", {class: "box-row"});
  for (const axis of AXES) {
    row.append(el("span", {text: `${axis}`}),
      el("input", {type: "number", step: "any", "aria-label": `${axis} 最小`, value: box ? box[axis].low : null}),
      el("span", {text: "～"}),
      el("input", {type: "number", step: "any", "aria-label": `${axis} 最大`, value: box ? box[axis].high : null}));
  }
  row.append(el("button", {type: "button", class: "quiet", text: "刪掉這一塊", on: {click: () => { row.remove(); changed(); }}}));
  row.addEventListener("input", changed);
  byId(container).append(row);
}

function boxes(container, word) {
  return [...byId(container).querySelectorAll(".box-row")].map((row, index) => {
    const values = [...row.querySelectorAll("input")].map((input) => input.value.trim());
    if (values.some((value) => value === "" || !Number.isFinite(Number(value)))) {
      throw new FormProblem(`第 ${index + 1} 塊${word}的六格都要填數字`);
    }
    const numbers = values.map(Number);
    return {x: {low: numbers[0], high: numbers[1]}, y: {low: numbers[2], high: numbers[3]}, z: {low: numbers[4], high: numbers[5]}};
  });
}

function stageOne() {
  return document.querySelector("input[name=stage]:checked").value === "one";
}

function readForm() {
  const one = stageOne();
  const layout = {front_wall: byId("front-wall").value, axis_offset_m: num("axis-offset") ?? 0,
    front_distance_m: span("front-low", "front-high", {required: true}),
    spacing_m: span("spacing-low", "spacing-high", {required: true}), wall_gap_m: num("wall-gap") ?? 0};
  if (byId("seat-locked").checked) layout.seat_locked = true;
  else {
    layout.listening_distance_m = span("listen-low", "listen-high", {required: true});
    layout.ear_height_m = num("ear-height", {required: true});
  }
  if (facts && !facts.has_speaker_setup) {
    layout.speaker_height_m = num("speaker-height", {required: true});
    layout.cabinet = {width_m: num("cabinet-width", {required: true}), depth_m: num("cabinet-depth", {required: true}),
      height_m: num("cabinet-height", {required: true}), acoustic_center_behind_front_m: num("cabinet-behind") ?? 0,
      acoustic_center_above_bottom_m: num("cabinet-above")};
  }
  const range = span("range-low", "range-high");
  if (range) layout.listening_range_m = range;
  const angle = span("angle-low", "angle-high");
  if (angle) layout.base_angle_deg = angle;
  layout.keep_out = boxes("keep-out", "禁區");
  const areas = boxes("speaker-areas", "可用區");
  if (areas.length) layout.speaker_areas = areas;
  const body = {layout, budget: num("budget", {integer: true, required: true}),
    convergence_run: num("convergence", {integer: true, required: true}),
    seed: num("seed", {integer: true, required: true}), n_startup_trials: num("startup", {integer: true, required: true}),
    batch_size: num("batch", {integer: true, required: true}), max_workers: num("workers", {integer: true, required: true}),
    constant_liar: byId("liar").checked};
  const hours = num("time-budget");
  if (hours !== null) body.search_time_budget_s = hours * 3600;
  if (one) {
    body.stage = "stage_one_fem_only";
    const room = {Lx: span("room-x-low", "room-x-high"), Ly: span("room-y-low", "room-y-high"), Lz: span("room-z-low", "room-z-high")};
    const area = num("room-area");
    if (room.Lx || room.Ly || room.Lz || area !== null) {
      layout.room_range = Object.fromEntries(Object.entries(room).filter(([, value]) => value));
      if (area !== null) layout.room_range.min_floor_area_m2 = area;
    }
    const cap = num("candidate-cap");
    if (cap !== null) body.stage_one_candidate_cap_s = cap;
    const top = num("top-n", {integer: true});
    if (top !== null) {
      body.stage_one_refine = {top_n: top, far_control_min_front_distance_m: num("far-control", {required: true})};
      const spread = num("area-spread");
      const difference = num("area-difference");
      if (spread !== null || difference !== null) {
        body.stage_one_refine.area_control_max_spread = spread;
        body.stage_one_refine.area_control_min_difference = difference;
      }
    }
  } else {
    const refineBudget = num("refine-budget", {integer: true});
    if (refineBudget !== null) body.refine = {budget: refineBudget, convergence_run: num("refine-convergence", {integer: true, required: true})};
    const offset = num("feedback-offset");
    if (offset !== null) body.feedback = {offset};
  }
  return body;
}

function put(id, value) {
  byId(id).value = value === null || value === undefined ? "" : String(value);
}

function putSpan(lowId, highId, value) {
  put(lowId, value ? value.low : null);
  put(highId, value ? value.high : null);
}

// 之前一場的設定填回表單；表單沒有的格不偷偷帶（帶不進來的照講）。
function fillForm(settings) {
  const layout = settings.layout;
  document.querySelector(`input[name=stage][value=${settings.stage ? "one" : "two"}]`).checked = true;
  byId("front-wall").value = layout.front_wall;
  put("axis-offset", layout.axis_offset_m);
  putSpan("front-low", "front-high", layout.front_distance_m);
  putSpan("spacing-low", "spacing-high", layout.spacing_m);
  byId("seat-locked").checked = Boolean(layout.seat_locked);
  putSpan("listen-low", "listen-high", layout.listening_distance_m);
  if (!layout.seat_locked) put("ear-height", layout.ear_height_m);
  if (facts && !facts.has_speaker_setup) {
    put("speaker-height", layout.speaker_height_m);
    put("cabinet-width", layout.cabinet.width_m);
    put("cabinet-depth", layout.cabinet.depth_m);
    put("cabinet-height", layout.cabinet.height_m);
    put("cabinet-behind", layout.cabinet.acoustic_center_behind_front_m);
    put("cabinet-above", layout.cabinet.acoustic_center_above_bottom_m);
  }
  put("wall-gap", layout.wall_gap_m);
  putSpan("range-low", "range-high", layout.listening_range_m);
  putSpan("angle-low", "angle-high", layout.base_angle_deg);
  byId("keep-out").replaceChildren();
  for (const box of layout.keep_out || []) boxRow("keep-out", box);
  byId("speaker-areas").replaceChildren();
  for (const box of layout.speaker_areas || []) boxRow("speaker-areas", box);
  const room = layout.room_range || {};
  putSpan("room-x-low", "room-x-high", room.Lx);
  putSpan("room-y-low", "room-y-high", room.Ly);
  putSpan("room-z-low", "room-z-high", room.Lz);
  put("room-area", room.min_floor_area_m2);
  put("budget", settings.budget);
  put("convergence", settings.convergence_run);
  put("time-budget", settings.search_time_budget_s ? settings.search_time_budget_s / 3600 : null);
  put("refine-budget", settings.refine ? settings.refine.budget : null);
  put("refine-convergence", settings.refine ? settings.refine.convergence_run : null);
  put("feedback-offset", settings.feedback ? settings.feedback.offset : null);
  put("candidate-cap", settings.stage_one_candidate_cap_s);
  const refined = settings.stage_one_refine;
  put("top-n", refined ? refined.top_n : null);
  put("far-control", refined ? refined.far_control_min_front_distance_m : null);
  put("area-spread", refined ? refined.area_control_max_spread : null);
  put("area-difference", refined ? refined.area_control_min_difference : null);
  put("startup", settings.n_startup_trials);
  put("batch", settings.batch_size);
  put("workers", settings.max_workers);
  put("seed", settings.seed);
  byId("liar").checked = settings.constant_liar !== false;
  showStage();
  changed();
}

function showStage() {
  const one = stageOne();
  byId("room-fields").hidden = !one;
  byId("stage-one-budget").hidden = !one;
  byId("stage-two-budget").hidden = one;
  byId("free-seat").hidden = byId("seat-locked").checked;
}

function changed() {
  previewedBody = null;
  byId("preview-block").hidden = true;
  showStage();
}

// ---- 方案與之前的場次 ----

async function loadSchemes() {
  const reply = await api("/api/schemes");
  const pick = byId("scheme-pick");
  pick.replaceChildren();
  if (!reply.ok) { say("scheme-message", reply.body.error, true); return; }
  for (const name of reply.body.schemes) pick.append(el("option", {value: name, text: name}));
  if (!reply.body.schemes.length) say("scheme-message", "還沒有存過的方案：先到方案輸入頁存一份");
  if (schemeId && reply.body.schemes.includes(schemeId)) { pick.value = schemeId; await loadScheme(); }
}

// 方案事實只給人看：公尺取到公釐（表單與送出的設定照原值，不經過這裡）。
const mm = (value) => String(Number(value.toFixed(3)));

function factLines(data) {
  const room = data.room_m;
  const [x, y, z] = data.primary_m.map(mm);
  return [`房間：x ${mm(room.Lx)} × y ${mm(room.Ly)} × 高 ${mm(room.Lz)} 公尺`, `主位（耳朵）：x ${x}、y ${y}、高 ${z} 公尺`,
    data.speaker_setup || "喇叭：這份方案沒有設喇叭類型與擺法",
    data.furniture ? `家具：${data.furniture} 件（跟著主位一起移動，不是搜尋變數）` : "家具：沒有",
    `前牆預設：${WALLS[data.front_wall_guess]}（兩支喇叭離這面牆最近；可以改）`];
}

async function loadScheme() {
  schemeId = byId("scheme-pick").value;
  if (!schemeId) return;
  const reply = await api(`/api/search-jobs/schemes/${encodeURIComponent(schemeId)}`);
  if (!reply.ok) { say("scheme-message", reply.body.error, true); return; }
  facts = reply.body;
  say("scheme-message", `已讀方案「${schemeId}」`);
  byId("facts-list").replaceChildren(...factLines(facts).map((line) => el("li", {text: line})));
  if (facts.zoned) byId("facts-list").append(el("li", {class: "notice", text: facts.zoned}));
  byId("scheme-facts").hidden = false;
  byId("speaker-fields").hidden = facts.has_speaker_setup;
  byId("form-block").hidden = false;
  byId("front-wall").value = facts.front_wall_guess;
  put("ear-height", facts.primary_m[2]);
  if (!facts.has_speaker_setup) put("speaker-height", facts.speaker_height_m);
  await loadPrevious();
  history.replaceState(null, "", `?scheme=${encodeURIComponent(schemeId)}${selectedJob ? `&job=${selectedJob}` : ""}`);
  changed();
}

async function loadPrevious() {
  const reply = await api("/api/search-jobs/previous");
  const pick = byId("previous-pick");
  pick.replaceChildren();
  if (!reply.ok) { say("previous-message", reply.body.error, true); return; }
  previous = reply.body.searches;
  byId("previous-row").hidden = !previous.length;
  for (const item of previous) {
    pick.append(el("option", {value: item.search_id,
      text: `${when(item.created_at)}　方案「${item.scheme_id}」${item.stage_one ? "（第一階段）" : "（第二階段）"}`}));
  }
  const skipped = reply.body.skipped.length ? `；${reply.body.skipped.length} 場讀不懂、沒列出來` : "";
  const same = previous.find((item) => item.scheme_id === facts.scheme_id);
  if (same) {
    pick.value = same.search_id;
    fillForm(same.settings);
    say("previous-message", `已帶入 ${when(same.created_at)} 那一場（同一份方案）的設定，可以再改${skipped}`);
  } else {
    say("previous-message", previous.length ? `這份方案還沒搜過；要沿用別份方案的設定就選一場按「帶入」${skipped}`
      : "還沒有任何一場搜尋可以帶入：預算與停止那幾格要自己填");
    defaultAdvanced();
  }
}

// 取樣與工作行程的預設（主對話定的工程值，非產品規格）：一間房 6 支工作行程、每批 6 題、起點隨機 12 題、種子取今天日期。
function defaultAdvanced() {
  const today = new Date();
  put("startup", 12);
  put("batch", 6);
  put("workers", 6);
  put("seed", today.getFullYear() * 10000 + (today.getMonth() + 1) * 100 + today.getDate());
}

function loadPreviousChoice() {
  const item = previous.find((entry) => entry.search_id === byId("previous-pick").value);
  if (!item) { say("previous-message", "沒有選到一場", true); return; }
  fillForm(item.settings);
  say("previous-message", `已帶入 ${when(item.created_at)} 那一場（方案「${item.scheme_id}」）的設定，可以再改`);
}

// ---- 預覽與開始 ----

function spanText(value, unit) {
  return value ? `${value.low}～${value.high} ${unit}` : "不限";
}

function previewLines(settings) {
  const layout = settings.layout;
  const lines = [settings.stage ? "第一階段：只看低頻" : "第二階段：擺位（低頻、反射、殘響）",
    `前牆：${WALLS[layout.front_wall]}，中軸偏移 ${layout.axis_offset_m} 公尺`,
    `離前牆 ${spanText(layout.front_distance_m, "公尺")}；間距 ${spanText(layout.spacing_m, "公尺")}`,
    layout.seat_locked ? facts.seat_locked_note : `聆聽距離 ${spanText(layout.listening_distance_m, "公尺")}；耳高 ${layout.ear_height_m} 公尺`,
    `喇叭聲學中心高 ${layout.speaker_height_m} 公尺；箱體 寬 ${layout.cabinet.width_m} × 深 ${layout.cabinet.depth_m} × 高 ${layout.cabinet.height_m} 公尺`,
    `離牆至少 ${layout.wall_gap_m} 公尺；禁區 ${layout.keep_out.length} 塊；喇叭可用區 ${layout.speaker_areas ? `${layout.speaker_areas.length} 塊` : "不限"}`,
    `型號適用聆聽距離 ${spanText(layout.listening_range_m, "公尺")}；水平夾角 ${spanText(layout.base_angle_deg, "度")}`,
    `搜尋 ${settings.budget} 題為上限、連續 ${settings.convergence_run} 個沒改善就停` +
      (settings.search_time_budget_s ? `、搜尋最多 ${settings.search_time_budget_s / 3600} 小時` : "")];
  if (layout.room_range) {
    const room = layout.room_range;
    lines.push(`房間尺寸：x ${spanText(room.Lx, "公尺")}、y ${spanText(room.Ly, "公尺")}、高 ${spanText(room.Lz, "公尺")}` +
      (room.min_floor_area_m2 ? `、地板至少 ${room.min_floor_area_m2} 平方公尺` : ""));
  }
  if (settings.refine) lines.push(`細算最多 ${settings.refine.budget} 個、第一名連續 ${settings.refine.convergence_run} 個沒換就停`);
  else if (!settings.stage) lines.push("不細算");
  if (settings.stage_one_refine) lines.push(`細算保留前 ${settings.stage_one_refine.top_n} 名，加較遠對照（離前牆至少 ${settings.stage_one_refine.far_control_min_front_distance_m} 公尺）`);
  if (settings.feedback) lines.push(`回饋偏移 ${settings.feedback.offset}`);
  lines.push(`同時算 ${settings.max_workers} 個、每批 ${settings.batch_size} 題、起點隨機 ${settings.n_startup_trials} 題、種子 ${settings.seed}`);
  return lines;
}

function withoutNulls(key, value) {
  return value === null ? undefined : value;
}

async function preview() {
  let body;
  try { body = readForm(); } catch (error) {
    if (!(error instanceof FormProblem)) throw error;
    say("form-message", error.message, true);
    return;
  }
  say("form-message", "預覽中…");
  const text = JSON.stringify(body);
  const reply = await api(`/api/search-jobs/schemes/${encodeURIComponent(schemeId)}/preview`, text);
  if (!reply.ok) { say("form-message", `設定過不了：${reply.body.error}`, true); return; }
  say("form-message", "");
  previewedBody = text;
  byId("preview-list").replaceChildren(...previewLines(reply.body.settings).map((line) => el("li", {text: line})));
  byId("preview-json").textContent = JSON.stringify(reply.body.settings, withoutNulls, 1);
  byId("preview-block").hidden = false;
  machineBusy = reply.body.busy;
  drawBusy();
}

async function start() {
  if (previewedBody === null) { say("start-message", "設定改過了，要先重新預覽", true); return; }
  byId("start").disabled = true;
  say("start-message", "開跑中…");
  const reply = await api(`/api/search-jobs/schemes/${encodeURIComponent(schemeId)}/start`, previewedBody);
  if (!reply.ok) {
    say("start-message", reply.body.error, true);
    byId("start").disabled = false;
    return;
  }
  selectedJob = reply.body.job_id;
  changed();
  say("form-message", "已開跑；下面這一跑會跟著更新。要再開一跑，改完設定重新預覽");
  await refresh();
}

function busyWho(busy) {
  const job = knownJobs.find((item) => item.job_id === busy);
  if (job) return `方案「${job.scheme_id}」的擺位搜尋（${job.state_text}）`;
  if (busy.startsWith("aosr-gradient-")) return `第三階段搜尋或建替身（服務 ${busy}）`;
  if (busy.startsWith("aosr-search-")) return `別的網頁開的擺位搜尋（服務 ${busy}）`;
  if (busy.startsWith("搜尋 ")) return `命令列開的${busy}（資料夾正被計算行程拿著）`;
  return busy;
}

function drawBusy() {
  const line = byId("busy-line");
  const problem = busyProblem || (machineBusy ? `整台機器正在算：${busyWho(machineBusy)}；等它做完或先停掉才能開新的一跑` : "");
  line.textContent = problem;
  line.hidden = !problem;
  byId("start").disabled = Boolean(problem) || previewedBody === null;
}

// ---- 清單與這一跑 ----

function drawList() {
  const box = byId("job-list");
  if (!knownJobs.length) { box.replaceChildren(el("p", {text: "還沒有從網頁開的擺位搜尋。"})); return; }
  const table = el("table", {}, el("tr", {}, el("th", {text: "開始"}), el("th", {text: "方案"}), el("th", {text: "階段"}),
    el("th", {text: "狀態"}), el("th", {text: ""})));
  for (const job of knownJobs) {
    const row = el("tr", {class: job.job_id === selectedJob ? "selected" : null}, el("td", {text: when(job.created_at)}),
      el("td", {text: job.scheme_id}), el("td", {text: job.stage_one ? "第一階段" : "第二階段"}), el("td", {text: job.state_text}),
      el("td", {}, el("button", {type: "button", class: "quiet", text: "看這一跑", on: {click: () => { selectedJob = job.job_id; drawJob(); drawList(); }}})));
    table.append(row);
  }
  box.replaceChildren(table);
}

function drawJob() {
  const job = knownJobs.find((item) => item.job_id === selectedJob);
  byId("job-block").hidden = !job;
  if (!job) return;
  byId("job-title").textContent = `這一跑：方案「${job.scheme_id}」（${when(job.created_at)} 開始）`;
  byId("job-state").textContent = `狀態：${job.state_text}`;
  byId("job-state").className = ["failed", "interrupted", "paused_time_limit"].includes(job.state) ? "notice" : "";
  byId("job-summary").textContent = job.summary ? `進度：${job.summary}` : "進度：搜尋資料夾還沒建好";
  byId("job-outer").textContent = job.outer ? `外圈結論：${job.outer}` : "";
  byId("job-progress-link").hidden = !job.search_url;
  if (job.search_url) byId("job-progress").href = job.search_url;
  byId("job-stop").hidden = !job.actions.stop;
  byId("job-resume").hidden = !job.actions.resume;
  byId("job-select").hidden = !job.actions.select;
  // 沒完成的那幾種結論（細算用完上限、使用者停止…）細算第一名照樣拿得出來，但要講清楚是未完成的結果。
  byId("job-select").textContent = job.state === "done" ? "把細算第一名放進結果清單、打開結果頁"
    : "把細算第一名放進結果清單（這場沒有完成，是未完成的結果）、打開結果頁";
  const steps = byId("job-steps");
  steps.replaceChildren(...job.steps.map((step, index) => el("div", {},
    el("p", {text: `第 ${index + 1} 步（${step.kind === "start" ? "開跑：搜尋再細算與收尾" : "接續：細算與收尾"}）` +
      `${step.exit_code === null ? "" : `，離開碼 ${step.exit_code}`}${step.error ? `：${step.error}` : ""}`}),
    el("pre", {text: step.log_tail.join("\n") || "（還沒有輸出）"}))));
}

async function act(action) {
  say("job-message", "送出中…");
  const reply = await api(`/api/search-jobs/${selectedJob}/${action}`, {});
  if (!reply.ok) { say("job-message", reply.body.error, true); return; }
  if (action === "select") {
    const result = byId("job-result");
    result.replaceChildren(`${reply.body.is_new ? "已放進" : "早就在"}結果清單（細算第一名：${reply.body.candidate}）：`,
      el("a", {href: reply.body.url, text: "打開結果頁"}));
    result.hidden = false;
    say("job-message", "");
    return;
  }
  say("job-message", action === "stop" ? "已送出停止" : "已接續");
  await refresh();
}

async function refresh() {
  if (polling) return;
  polling = true;
  try {
    const reply = await api("/api/search-jobs");
    if (!reply.ok) { say("connection", reply.body.error, true); return; }
    knownJobs = reply.body.jobs;
    machineBusy = reply.body.busy;
    busyProblem = reply.body.busy_problem;
    if (!selectedJob && knownJobs.length && RUNNING.has(knownJobs[0].state)) selectedJob = knownJobs[0].job_id;
    drawList();
    drawJob();
    drawBusy();
    say("connection", `資料讀取於 ${new Date().toLocaleTimeString("zh-TW", {hour12: false})}；每 5 秒更新`);
  } finally {
    polling = false;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  byId("load-scheme").addEventListener("click", loadScheme);
  byId("load-previous").addEventListener("click", loadPreviousChoice);
  byId("add-keep-out").addEventListener("click", () => { boxRow("keep-out"); changed(); });
  byId("add-speaker-area").addEventListener("click", () => { boxRow("speaker-areas"); changed(); });
  byId("form-block").addEventListener("input", changed);
  byId("form-block").addEventListener("change", changed);
  byId("preview").addEventListener("click", preview);
  byId("start").addEventListener("click", start);
  byId("job-stop").addEventListener("click", () => act("stop"));
  byId("job-resume").addEventListener("click", () => act("resume"));
  byId("job-select").addEventListener("click", () => act("select"));
  loadSchemes();
  refresh();
  setInterval(refresh, POLL_MS);
});
