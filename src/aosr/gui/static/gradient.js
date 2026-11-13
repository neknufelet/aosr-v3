"use strict";
// 第三階段搜尋頁（#1082 段 1-6b）：選方案 → 開跑選項 → 預覽 → 替身（沒有就建）→ 開跑 → 進度、停止、接續、終點 → 存成方案算結果頁。
// 判斷全交給伺服器（#1093 開跑後端、#1098 建替身、本頁補的進度細節與接結果頁）；這裡只把選項排成請求、把回話照實印出來。
// 所有 POST 都帶 content-type: application/json，沒有內容就送 {}（網頁中介層不收別的）。

const POLL_MS = 5000;
const SLOW_MS = 15000;
// 牆名先用座標名，讀到方案後換成伺服器依喇叭與主位排的前後左右（#1134，預覽回話的 wall_names）。
const WALLS = {floor: "地板", ceiling: "天花", x0: "x 起點牆", xL: "x 終點牆", y0: "y 起點牆", yL: "y 終點牆"};
const MODE_WORDS = {fixed: "固定這一組材料", keep_original: "保留原牆", search: "交給搜尋"};
const WHOLE_WALL = "whole_wall";
const RUNNING_BUILD = new Set(["preparing", "building", "fixed_checking", "validating", "checking", "installing", "stopping"]);
const RUN_WORDS = {running: "一般計算中", done: "一般計算做完了", failed: "一般計算失敗", stopped: "一般計算已停止",
  interrupted: "一般計算中斷", unknown: "一般計算狀態讀不到"};

const byId = (id) => document.getElementById(id);
const params = new URLSearchParams(window.location.search);
let schemeId = params.get("scheme");
let selectedJob = params.get("job");
let defaults = null;
let lastPreview = null;
// 最後一次預覽送出的選項（JSON 字串）：開跑只送這一份；表單改過就要先重新預覽，畫面上的預覽才是真的會跑的。
let previewedRequest = null;
// 伺服器最後一次說的整機忙碌（一跑代號或服務名）；建替身那一顆按鈕的停用要同時看它，不被建模狀態蓋掉。
let machineBusy = null;
let watchedBuild = null;
let surrogateDefaults = null;
let busyProblem = null;
let watchedBuildState = null;
let budgetHeldByBuild = false;
// 使用者自己在兩格預算裡打過字（#1082 段 2-9）：重讀建替身選項時不拿預設值蓋掉；建模結束放回預設時一起清掉。
let budgetTyped = false;
// 這一頁看過的每一份建模（含剛按下開建、輪詢清單還沒帶到的那一份），各記上一次畫到的樣子（#1082 段 2-9）：
// 忙碌行認得是哪份方案在建；只在同一份從建模中變成「已放進格子」那一刻重新預覽一次（原本每次畫到已放進格子、
// 預覽又說沒涵蓋就再預覽，預覽又畫回同一份，幾分鐘送了六千多次）。
const seenBuilds = new Map();
let knownJobs = [];
let knownBuilds = [];
let polling = false;
let pollAgain = false;

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

// 伺服器回的錯誤一律照原文；不是 JSON（例如 500）時說狀態碼。等太久先說在等什麼（開跑鎖沒有逾時）。
async function api(path, body = undefined, messageId = null) {
  const options = body === undefined ? {} :
    {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body)};
  const slow = messageId ? setTimeout(() => say(messageId, "還在等伺服器回應（可能在等開跑鎖：別的網頁正在開一跑）…"), SLOW_MS) : null;
  try {
    const reply = await fetch(path, options);
    let data;
    try { data = await reply.json(); } catch {
      data = {error: `伺服器出錯（狀態 ${reply.status}），回的不是 JSON；可能是開跑鎖或服務那一層出問題，請把這一行交給維護的人`};
    }
    return {ok: reply.ok, status: reply.status, body: data};
  } catch (error) {
    return {ok: false, status: 0, body: {error: `連不到網頁伺服器：${error}`}};
  } finally {
    if (slow) clearTimeout(slow);
  }
}

// 忙碌時伺服器只給代號（一跑的代號，或別處開的服務名）：照清單翻成「哪份方案、在搜尋還是建替身」。
function busyWho(busy) {
  const job = knownJobs.find((item) => item.job_id === busy);
  if (job) return {text: `方案「${job.scheme_id}」的第三階段搜尋（${job.state_text}）`, job: job.job_id};
  const build = [...knownBuilds, ...seenBuilds.values()].find((item) => item.unit === busy || item.build_id === busy);
  if (build) return {text: `方案「${build.scheme_id}」正在建替身（${build.state_text}）`, build: build.build_id};
  if (busy.startsWith("aosr-search-")) return {text: `網頁開的擺位搜尋（服務 ${busy}）`};
  if (busy.startsWith("搜尋 ")) return {text: `命令列開的${busy}（資料夾正被計算行程拿著）`};
  return {text: `別的資料夾或命令列開的第三階段計算（服務 ${busy}）`};
}

function busyText(body) {
  return body.busy ? `整台機器同時只跑一件：${busyWho(body.busy).text}正在算，等它做完或先停掉` : body.error;
}

function showBusy(busy) {
  const line = byId("busy-line");
  line.hidden = !busy && !busyProblem;
  if (!busy) { line.replaceChildren(...(busyProblem ? [busyProblem] : [])); return; }
  const who = busyWho(busy);
  line.replaceChildren(`整台機器同時只跑一件：${who.text}正在算，等它做完或先停掉。`);
  if (who.job) line.append(" ", el("button", {type: "button", class: "quiet", id: "busy-open", text: "看那一跑",
    on: {click: () => selectJob(who.job)}}));
  if (who.build) line.append(" ", el("button", {type: "button", class: "quiet", id: "busy-open", text: "看建替身進度",
    on: {click: () => { watchedBuild = who.build; poll(); byId("surrogate-block").scrollIntoView(); }}}));
}

function errorText(reply) {
  if (reply.status === 409 && reply.body.needs === "surrogate") return `${reply.body.error}（替身該放在 ${reply.body.slot}）`;
  if (reply.status === 409 && reply.body.busy) return busyText(reply.body);
  return reply.body.error || `失敗（狀態 ${reply.status}）`;
}

function duration(seconds) {
  if (seconds === null || seconds === undefined) return "還推不出來";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `約 ${minutes} 分`;
  return `約 ${Math.floor(minutes / 60)} 小時 ${minutes % 60} 分`;
}

// 整頁的時間都經過這一支、照看網頁那台電腦的時區印：伺服器給的是秒數（epoch）或帶時區的 ISO 8601 字串
// （搜尋本體寫的是世界時 +00:00）。沒寫時區的字串分不出是哪一區，照原文印、標明沒寫時區，不猜。
const ZONED = /(Z|[+-]\d\d:?\d\d)$/;
function when(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string" && !ZONED.test(value)) return `${value}（沒寫時區）`;
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-TW", {hour12: false});
}

// 每區准用清單（#1082 段 2-3b；轉好的參數 plan.allowed：面片 → 家族代號、fixed:<代號>、none）；沒有清單的區照家族全部試。
function allowedText(plan, surface, mode) {
  if (mode !== "search") return "—";
  const targets = (plan.allowed || {})[surface];
  if (!targets) return "沒有限制（家族都試、也試不貼）";
  return targets.map((target) => target === "none" ? "不貼" : target.startsWith("fixed:") ?
    `${materialLabel(target.slice("fixed:".length))}（固定，整塊選或不選）` : materialLabel(target)).join("、");
}

// 代號照預覽回的材料庫顯示名印（#1119）；對不到就照印代號。代號本身看「交給搜尋命令列的參數」。
function materialLabel(id) {
  return (lastPreview && lastPreview.labels && lastPreview.labels[id]) || id;
}

function surfaceParts(surface) {
  const [wall, zone] = surface.split("@");
  return {wall, zone: zone === undefined ? WHOLE_WALL : zone};
}

function zoneText(zone) {
  return zone === WHOLE_WALL ? "整面一區" : zone;
}

// ---- 1. 方案 ----

async function loadSchemes() {
  const reply = await api("/api/schemes");
  if (!reply.ok) { say("connection", `讀不到方案清單：${errorText(reply)}`, true); return; }
  const pick = byId("scheme-pick");
  pick.replaceChildren(...reply.body.schemes.map((name) => el("option", {value: name, text: name})));
  if (schemeId && reply.body.schemes.includes(schemeId)) pick.value = schemeId;
  say("connection", reply.body.schemes.length ? "" : "還沒有存好的方案；先在方案輸入頁存一份、在立面圖把要搜尋的區設成「交給搜尋」。");
  if (schemeId && reply.body.schemes.includes(schemeId)) await loadPlan();
}

function request() {
  const value = {family: byId("family").value || null, mount: byId("mount").value, device: byId("device").value};
  const steps = byId("max-steps").value.trim();
  if (steps !== "") value.max_steps = Number(steps);
  const batch = byId("frequency-batch").value.trim();
  if (batch !== "") value.frequency_batch = Number(batch);
  const depths = {};
  for (const input of byId("depth-table").querySelectorAll("input[data-wall]")) {
    if (input.value.trim() === "") continue;
    (depths[input.dataset.wall] ||= {})[input.dataset.zone] = Number(input.value);
  }
  value.max_depths = depths;
  return value;
}

async function loadPlan() {
  // 換方案：上一份打的預算不帶過去（#1082 段 2-9），這一份照它自己的預設。
  if (byId("scheme-pick").value !== schemeId) budgetTyped = false;
  schemeId = byId("scheme-pick").value;
  if (!schemeId) return;
  history.replaceState(null, "", `/gradient?scheme=${encodeURIComponent(schemeId)}${selectedJob ? `&job=${selectedJob}` : ""}`);
  say("scheme-message", "正在讀這份方案的開跑選項…");
  const reply = await api(`/api/gradient/schemes/${encodeURIComponent(schemeId)}/plan`, {}, "scheme-message");
  for (const id of ["form-block", "plan-block", "surrogate-block", "start-block"]) byId(id).hidden = !reply.ok;
  if (!reply.ok) { say("scheme-message", `這份方案開不了第三階段搜尋：${errorText(reply)}`, true); return; }
  say("scheme-message", "");
  Object.assign(WALLS, reply.body.wall_names);
  defaults = reply.body.defaults;
  byId("max-steps").value = defaults.max_steps;
  byId("max-steps-default").textContent = `步數上限預設 ${defaults.max_steps} 步（設定檔的值）；這一跑改了只換這一跑，數字記進搜尋資料夾、報告會印出來。`;
  byId("family").replaceChildren(...defaults.families.map((id) => el("option", {value: id, text: (reply.body.labels || {})[id] || id})));
  byId("family").value = defaults.family;
  byId("mount").value = defaults.mount;
  byId("device").value = defaults.device;
  byId("frequency-batch").value = "";
  mountWarning();
  drawDepthTable(reply.body.plan);
  showPreview(reply.body, request());
}

// 結果頁沒指原方案時，是拆「貼在原牆上」的加貼層找回原牆；貼在硬底上的區拆不回去，會被當成沒動、改善算不進來。
function mountWarning() {
  const rigid = byId("mount").value === "rigid";
  byId("mount-warning").hidden = !rigid;
  byId("mount-warning").textContent = rigid ? "貼在硬底上：結果頁要算「改善多少」得知道原牆。設計設定沒指原方案時，結果頁是拆" +
    "「貼在原牆上」的加貼層找回原牆；貼在硬底上的區拆不回去，會被當成沒動、改善算不進來。要用硬底，先在設計設定指原方案。" : "";
}

// 只給人看：六位有效數字、千分位，同立面圖 walls.js 的 shownNumber（這一頁不載入那一支）；送伺服器的值不經過這裡。
function shownNumber(value) {
  // 格子清空時表單回 null：照舊原樣印，不丟錯（丟錯會讓整段畫面畫不出來）。
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
  return Number(value.toPrecision(6)).toLocaleString("en-US", {maximumFractionDigits: 20});
}

// 深度上限是哪一層擋的（#1111）：設計設定的三層取最嚴，這一跑另填的只能更嚴。
const DEPTH_SOURCES = {zone: "這一塊", wall: "這面牆", room: "房間預設", request: "這一跑另填"};
function depthText(plan, surface, mode) {
  const depth = plan.max_depths[surface];
  if (depth === undefined) return mode === "search" ? "照材料自己的限制" : "不加層";
  const source = (plan.depth_sources || {})[surface];
  return source ? `${shownNumber(depth)} 公尺（${DEPTH_SOURCES[source] || source}）` : `${shownNumber(depth)} 公尺`;
}

// 深度上限那一格：個案上限一行，下面逐組跟自己的上限取較嚴（#1082 段 2-4，伺服器逐組寫好一句；
// 放不下的那一組這一跑不換過去、起點也跳過）。
function depthCell(plan, surface, mode) {
  const lines = (plan.depth_targets || {})[surface] || [];
  if (mode === "fixed") return fixedDepthCell(surface);
  return el("td", {class: "depth-cell"}, el("p", {text: depthText(plan, surface, mode)}),
    ...lines.map((line) => el("p", {class: line.usable ? "note" : "limit-warn", text: line.text})));
}

// 固定區那一格（#1082 段 2-9）：跟立面圖同一次限制檢查（伺服器 fixed_depths）——生效上限是哪一層，
// 加上深度那幾條原句（選了材料庫的實測表就是「總深 … 放得進 …」，方案原本的牆是「比不了」）；沒給上限照立面圖寫「沒給上限」。
function fixedDepthCell(surface) {
  const found = ((lastPreview && lastPreview.fixed_depths) || {})[surface];
  if (!found) return el("td", {class: "depth-cell"}, el("p", {text: "不加層"}));
  const cap = found.cap ? `${shownNumber(found.cap.value_m)} 公尺（${DEPTH_SOURCES[found.cap.level] || found.cap.level}）` : "沒給上限";
  return el("td", {class: "depth-cell"}, el("p", {text: cap}),
    ...found.lines.map((line) => el("p", {class: line.level === "red" ? "limit-warn" : "note", text: line.text})));
}

function drawDepthTable(plan) {
  const rows = plan.surfaces.map((surface, index) => ({surface, mode: plan.modes[index]})).filter((row) => row.mode === "search");
  const body = el("tbody");
  for (const row of rows) {
    const {wall, zone} = surfaceParts(row.surface);
    const designed = plan.max_depths[row.surface];
    body.append(el("tr", {}, el("td", {text: WALLS[wall] || wall}), el("td", {text: zoneText(zone)}),
      el("td", {}, el("input", {type: "number", min: "0", step: "any", "data-wall": wall, "data-zone": zone,
        placeholder: designed === undefined ? "" : `設計設定 ${designed}`,
        "aria-label": `${WALLS[wall] || wall} ${zoneText(zone)} 深度上限（公尺）`})),
      el("td", {text: designed === undefined ? "照材料自己的限制" : depthText(plan, row.surface, row.mode)})));
  }
  const head = el("tr", {}, ...["牆", "區", "這一跑另填（只能更嚴）", "設計設定的上限"].map((text) => el("th", {text})));
  byId("depth-table").replaceChildren(rows.length ? el("table", {}, el("thead", {}, head), body) :
    el("p", {text: "沒有交給搜尋的區。"}));
}

// ---- 3. 預覽 ----

async function preview() {
  say("form-message", "正在預覽…");
  const body = request();
  const reply = await api(`/api/gradient/schemes/${encodeURIComponent(schemeId)}/plan`, body, "form-message");
  if (!reply.ok) { say("form-message", errorText(reply), true); return; }
  say("form-message", "預覽好了，看下面。");
  showPreview(reply.body, body);
}

// 「開始搜尋」與「建替身」亮不亮只由這一支決定（#1119）：開始要有預覽、選項沒改過、這份切法有替身、整台機器現在沒在算；
// 建替身要整台沒在算、看著的那一份也沒在建。忙不忙每次輪詢從清單回應重拿（搜尋、接續、終點、建替身、別處開的
// 第三階段服務都算，同伺服器開跑鎖 active_job），不靠預覽那一刻的舊值；查不到忙不忙也不准開。
function refreshStart() {
  showBusy(machineBusy);
  const blocked = Boolean(machineBusy) || Boolean(busyProblem);
  byId("build-surrogate").disabled = RUNNING_BUILD.has(watchedBuildState) || blocked;
  if (previewedRequest === null || !lastPreview) { byId("start").disabled = true; return; }
  const stale = JSON.stringify(request()) !== previewedRequest;
  byId("start").disabled = stale || !lastPreview.surrogate.found || blocked;
}

// 表單改過：舊預覽不再是會跑的那一份，先停用開跑、請使用者重新預覽。
function markStale() {
  if (previewedRequest === null) return;
  const stale = JSON.stringify(request()) !== previewedRequest;
  say("start-message", stale ? "選項改過了：先按「預覽」看過這一份，才能開跑。" : "", stale);
  refreshStart();
}

function showPreview(data, sent) {
  lastPreview = data;
  previewedRequest = JSON.stringify(sent);
  machineBusy = data.busy || null;
  say("start-message", "");
  const plan = data.plan;
  const searched = plan.modes.filter((mode) => mode === "search").length;
  byId("plan-summary").textContent = `共 ${plan.surfaces.length} 區，交給搜尋 ${searched} 區；家族 ${materialLabel(plan.family)}；` +
    `往下走步數上限 ${plan.max_steps} 步；裝法「${plan.request.mount === "on_wall" ? "貼在原牆上" : "貼在硬底上"}」；` +
    `用${plan.request.device === "gpu" ? "顯卡" : "處理器"}算。`;
  // 結果頁會把哪些區當成沒動（#1128）：開跑前先講，要指原方案就在跑之前指。
  byId("plan-baseline-warning").hidden = !data.baseline_warning;
  byId("plan-baseline-warning").textContent = data.baseline_warning || "";
  const body = el("tbody");
  plan.surfaces.forEach((surface, index) => {
    const {wall, zone} = surfaceParts(surface);
    const mode = plan.modes[index];
    body.append(el("tr", {"data-surface": surface}, el("td", {text: WALLS[wall] || wall}), el("td", {text: zoneText(zone)}),
      el("td", {text: MODE_WORDS[mode] || mode, class: mode === "search" ? "mode-search" : null}),
      el("td", {text: plan.presets[index] ? materialLabel(plan.presets[index]) : (mode === "search" ? "" : "方案裡這一區的材料，不調")}),
      el("td", {text: allowedText(plan, surface, mode)}),
      depthCell(plan, surface, mode)));
  });
  const head = el("tr", {}, ...["牆", "區", "做法", "起跑樣板", "准用材料", "深度上限"].map((text) => el("th", {text})));
  byId("plan-table").replaceChildren(el("table", {}, el("thead", {}, head), body));
  byId("plan-arguments").textContent = data.arguments.join(" ");
  const surrogate = data.surrogate;
  // 三件都在、但這一場的材料有替身沒訓練過的（#1082 段 2-3c）：伺服器把原因放在 message，照講、要重建。
  byId("surrogate-state").textContent = surrogate.found ? "這份切法已經有替身，可以直接開搜尋。" :
    surrogate.missing.length === 0 ? `${surrogate.message}。` :
    `這份切法還沒有替身，要先建（缺：${surrogate.missing.join("、")}）。`;
  byId("surrogate-slot").textContent = `替身放在：${surrogate.slot}`;
  byId("surrogate-state").className = surrogate.found ? "" : "notice";
  refreshStart();
  if (!surrogate.found) loadSurrogatePlan(false); else byId("surrogate-build-form").hidden = true;
}

// ---- 4. 替身 ----

async function loadSurrogatePlan(replace) {
  const reply = await api(`/api/gradient/schemes/${encodeURIComponent(schemeId)}/surrogate/plan`, {replace});
  if (!reply.ok) { say("surrogate-message", `建不了替身：${errorText(reply)}`, true); return; }
  const data = reply.body;
  byId("surrogate-build-form").hidden = false;
  byId("build-surrogate").dataset.replace = String(Boolean(replace || data.exists));
  surrogateDefaults = data.defaults;
  if (!budgetTyped) {
    byId("max-full-solves").value = data.defaults.max_full_solves;
    byId("stop-residual").value = data.defaults.stop_residual;
  }
  const last = data.history[0];
  byId("surrogate-history").textContent = (last ?
    `這台機器上一次建替身花了 ${duration(last.total_seconds)}（${last.patches} 區、完整解 ${last.full_solves} 次）。` :
    "這台機器還沒建過替身，開建前推不出要多久。") + data.estimate_note;
  machineBusy = data.busy || null;
  refreshStart();
  // 忙的正是下面畫的這一份建模：進度就在下面，不另寫一句（舊的那句也清掉，免得留著別份的話）。
  const own = data.latest && data.busy && (data.latest.unit === data.busy || data.latest.build_id === data.busy);
  if (data.busy && !own) say("surrogate-message", busyText({busy: data.busy}), true);
  else if (own) say("surrogate-message", "");
  if (data.latest) { watchedBuild = data.latest.build_id; showBuild(data.latest); }
}

async function buildSurrogate() {
  const body = {replace: byId("build-surrogate").dataset.replace === "true"};
  // 准用清單有固定材料時，建替身照開跑選項的起跑配置先核過才裝（#1082 段 2-8）。送表單上現在的那一份（#1128）：
  // 沒先按預覽、或預覽之後改過選項，也是開跑會用的這一組，不送預設、不送舊預覽。
  body.search = request();
  const solves = byId("max-full-solves").value.trim();
  if (solves !== "" && Number(solves) !== surrogateDefaults.max_full_solves) body.max_full_solves = Number(solves);
  const residual = byId("stop-residual").value.trim();
  if (residual !== "" && Number(residual) !== surrogateDefaults.stop_residual) body.stop_residual = Number(residual);
  say("surrogate-message", "正在開建…");
  const reply = await api(`/api/gradient/schemes/${encodeURIComponent(schemeId)}/surrogate`, body, "surrogate-message");
  if (!reply.ok) { say("surrogate-message", errorText(reply), true); return; }
  say("surrogate-message", "開建了。");
  watchedBuild = reply.body.build_id;
  showBuild(reply.body);
}

// 這一份實際用的完整解上限與停止殘差（#1119）：開建時另填的照它，沒填的是設定檔預設。建模中重開頁面也照這一份印，
// 兩格輸入在建模中換成這一份的值並鎖住；建完放回預設值，下一次開建不會悄悄沿用這一份另填的數字。
function buildBudget(build) {
  const given = build.request || {};
  const solves = given.max_full_solves ?? (surrogateDefaults ? surrogateDefaults.max_full_solves : null);
  const residual = given.stop_residual ?? (surrogateDefaults ? surrogateDefaults.stop_residual : null);
  const word = (value, unit, own) => `${value === null ? "照設定檔預設" : `${value}${unit}`}（${own ? "這一份另填" : "預設"}）`;
  return {solves, residual,
    text: `完整解上限 ${word(solves, " 次", given.max_full_solves != null)}、停止殘差 ${word(residual, "", given.stop_residual != null)}`};
}
function holdBudget(build, budget) {
  const running = RUNNING_BUILD.has(build.state);
  if (running) {
    if (budget.solves !== null) byId("max-full-solves").value = budget.solves;
    if (budget.residual !== null) byId("stop-residual").value = budget.residual;
  } else if (budgetHeldByBuild && surrogateDefaults) {
    byId("max-full-solves").value = surrogateDefaults.max_full_solves;
    byId("stop-residual").value = surrogateDefaults.stop_residual;
    budgetTyped = false;
  }
  if (running || budgetHeldByBuild) {
    byId("max-full-solves").disabled = running;
    byId("stop-residual").disabled = running;
  }
  budgetHeldByBuild = running;
}

function showBuild(build) {
  const before = seenBuilds.has(build.build_id) ? seenBuilds.get(build.build_id).state : null;
  seenBuilds.set(build.build_id, build);
  byId("surrogate-build").hidden = false;
  watchedBuildState = build.state;
  const budget = buildBudget(build);
  holdBudget(build, budget);
  byId("build-state").textContent = `${build.state_text}（開建 ${when(build.created_at)}；${budget.text}）`;
  byId("build-state").className = ["failed", "not_validated", "gate_refused", "interrupted"].includes(build.state) ? "notice" : "";
  const estimate = build.estimate;
  if (estimate) {
    const left = estimate.remaining_s === null ? "還推不出還要多久" : `還要${duration(estimate.remaining_s)}`;
    byId("build-estimate").textContent = `${estimate.label}：${left}（已經建了${duration(estimate.elapsed_s)}）。${estimate.basis}`;
  } else {
    byId("build-estimate").textContent = build.result ?
      `完整解 ${build.result.full_solves} 次、秩 ${build.result.rank}。${build.result.message}` : "";
  }
  const checks = build.fixed_checks || [];
  byId("build-checks").hidden = !checks.length;
  byId("build-checks").querySelector("ul").replaceChildren(...checks.map((text) => el("li", {text})));
  byId("build-error").hidden = !build.error;
  byId("build-error").textContent = build.error || "";
  byId("build-log").textContent = (build.log_tail || []).join("\n");
  byId("stop-build").hidden = !build.actions.stop;
  refreshStart();
  // 只在這一份剛建好放進格子（上一次畫它還在建）時重新預覽一次，看這一場的材料它涵不涵蓋；之後不再自己送。
  if (build.state === "installed" && RUNNING_BUILD.has(before) && lastPreview && !lastPreview.surrogate.found) preview();
}

async function stopBuild() {
  if (!watchedBuild) return;
  const reply = await api(`/api/gradient/surrogates/${watchedBuild}/stop`, {}, "surrogate-message");
  if (!reply.ok) { say("surrogate-message", errorText(reply), true); return; }
  showBuild(reply.body);
}

// ---- 5. 開跑 ----

async function start() {
  if (previewedRequest === null || JSON.stringify(request()) !== previewedRequest) { markStale(); return; }
  say("start-message", "正在開跑…");
  byId("start").disabled = true;
  const reply = await api(`/api/gradient/schemes/${encodeURIComponent(schemeId)}/start`, JSON.parse(previewedRequest), "start-message");
  if (!reply.ok) {
    refreshStart();
    say("start-message", errorText(reply), true);
    if (reply.body.needs === "surrogate") loadSurrogatePlan(false);
    return;
  }
  say("start-message", "開跑了；進度看下面。");
  selectJob(reply.body.job_id);
}

// ---- 清單與這一跑 ----

async function refreshJobs() {
  const reply = await api("/api/gradient/jobs");
  if (!reply.ok) { say("connection", `讀不到搜尋清單：${errorText(reply)}`, true); return; }
  say("connection", "");
  const jobs = reply.body.jobs;
  knownJobs = jobs;
  machineBusy = reply.body.busy || null;
  busyProblem = reply.body.busy_problem || null;
  refreshStart();
  if (!jobs.length) { byId("job-list").replaceChildren(el("p", {text: "還沒有第三階段搜尋。"})); return; }
  const body = el("tbody");
  for (const job of jobs) {
    body.append(el("tr", {"data-job": job.job_id, class: job.job_id === selectedJob ? "selected" : null},
      el("td", {text: when(job.created_at)}), el("td", {text: job.scheme_id}),
      el("td", {text: job.state_text, class: ["failed", "surrogate_refused", "interrupted", "endpoint_failed"].includes(job.state) ? "notice" : null}),
      el("td", {}, el("button", {type: "button", text: "看這一跑", on: {click: () => selectJob(job.job_id)}}))));
  }
  const head = el("tr", {}, ...["開始", "方案", "狀態", ""].map((text) => el("th", {text})));
  byId("job-list").replaceChildren(el("table", {}, el("thead", {}, head), body));
}

function selectJob(jobId) {
  selectedJob = jobId;
  const query = new URLSearchParams();
  if (schemeId) query.set("scheme", schemeId);
  query.set("job", jobId);
  history.replaceState(null, "", `/gradient?${query}`);
  say("job-message", "");
  refreshJob();
  refreshJobs();
}

async function refreshJob() {
  if (!selectedJob) return;
  const [view, detail] = await Promise.all([api(`/api/gradient/jobs/${selectedJob}`), api(`/api/gradient/runs/${selectedJob}/detail`)]);
  byId("job-block").hidden = false;
  if (!view.ok) { say("job-state", `讀不到這一跑：${errorText(view)}`, true); return; }
  showJob(view.body, detail.ok ? detail.body : null, detail.ok ? null : errorText(detail));
}

function progressLines(job, detail) {
  const lines = [];
  if (detail && detail.descent_steps === 0 && job.state === "searching") {
    // 搜尋本體把往下走那一段整段算完才一次寫進帳本，算的途中帳本裡一列都沒有。
    lines.push(`往下走（上限 ${job.max_steps} 步）：這一段整段算完才一次記帳，正在算；算完會直接跳到實際走了幾步`);
  } else if (detail && detail.descent_steps !== null && detail.descent_steps !== undefined) {
    lines.push(`往下走：${detail.descent_steps}／${job.max_steps} 步`);
  } else {
    lines.push(`往下走步數上限：${job.max_steps} 步（帳本還沒開始寫）`);
  }
  const progress = job.progress;
  if (progress) {
    lines.push(`候選輪做完 ${progress.rounds_done} 輪、累計求值 ${progress.evaluations} 次` +
      (progress.template_rounds_done ? `；換樣板做完 ${progress.template_rounds_done} 輪、求值 ${progress.template_evaluations} 次` : ""));
    if (progress.last_value !== null) lines.push(`目前合分（越小越好）：${progress.last_value.toFixed(4)}`);
    if (progress.stop_reason) lines.push(`停止原因：${progress.stop_reason}`);
    if (progress.message) lines.push(`搜尋本體的話：${progress.message}`);
    lines.push(`狀態檔最後更新：${when(progress.updated_at)}`);
  }
  if (job.progress_note) lines.push(job.progress_note);
  return lines;
}

// 這一跑開跑時每區的深度上限判決（#1082 段 2-4；工作記錄存的那一份計畫）：個案上限是哪一層、每一組生效多少、哪一條擋住。
// 牆名用這一跑自己那份方案的（#1134，伺服器排；清單跨方案，不借頁面上選的那一份）。
function showJobDepths(plan, names) {
  const found = Object.entries((plan && plan.depth_targets) || {});
  byId("job-depths").hidden = !found.length;
  byId("job-depths").querySelector("div").replaceChildren(...found.map(([surface, lines]) => {
    const {wall, zone} = surfaceParts(surface);
    return el("div", {class: "depth-cell", "data-surface": surface},
      el("p", {text: `${(names || WALLS)[wall] || wall}「${zoneText(zone)}」：個案上限 ${depthText(plan, surface, "search")}`}),
      ...lines.map((line) => el("p", {class: line.usable ? "note" : "limit-warn", text: line.text})));
  }));
}

// 換樣板試過、沒選的（#1128）：固定材料候選一句寫在外面（整塊試過、差多少），每一輪全部候選收在表裡。
// 快篩合分是搜尋自己的梯度目標（越小越好），跟結果頁的總分不是同一把尺；快篩沒重調，換不換照重調後跟現狀比。
const TRY_SCALE_NOTE = "合分是搜尋自己的尺（越小越好），跟結果頁的總分不是同一把尺。快篩是換上去先算一次的粗篩；" +
  "快篩排前面的幾個再重調空腔，換不換是拿重調後跟這一輪現狀（起點配置重調後）比、要好過改善門檻才換。" +
  "表裡「比現狀」有重調的拿重調後比；沒進重調的只能拿快篩比，標「（快篩）」。";
function score(value) { return value === null || value === undefined ? "—" : value.toFixed(4); }
function signed(value) { return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`; }
function showTemplateTries(rounds, names) {
  const box = byId("job-template-tries");
  box.hidden = !rounds.length;
  if (!rounds.length) { box.replaceChildren(); return; }
  const place = (surface) => { const {wall, zone} = surfaceParts(surface); return `${(names || WALLS)[wall] || wall}「${zoneText(zone)}」`; };
  const lines = [];
  for (const round of rounds) {
    const fixed = round.tries.filter((item) => item.fixed);
    if (!fixed.length) continue;
    const tried = fixed.map((item) => `${place(item.surface)}整塊換成「${item.label}」快篩 ${score(item.value)}` +
      (item.retuned !== null ? `、重調後 ${score(item.retuned)}` : "") + (item.chosen ? "（換上了）" : "（沒選）")).join("；");
    lines.push(el("p", {class: "tried-fixed", text: `換樣板第 ${round.round} 輪試過固定材料：${tried}；這一輪現狀 ${score(round.baseline)}。`}));
  }
  const tables = rounds.map((round) => {
    const outcome = round.accepted ? "換上了一個候選" : (round.stop_reason ? `都沒換：${round.stop_reason}` : "都沒換");
    if (!round.tries.length) {
      return el("p", {text: `第 ${round.round} 輪沒開` + (round.stop_reason ? `：${round.stop_reason}` : "")});
    }
    const head = el("tr", {}, ...["牆／區", "換成", "快篩合分", "重調後", "比現狀", "結果"].map((text) => el("th", {text})));
    const body = el("tbody", {}, ...round.tries.map((item) => {
      const value = item.retuned !== null ? item.retuned : item.value;
      return el("tr", {class: item.fixed ? "tried-fixed-row" : null},
        el("td", {text: place(item.surface)}), el("td", {text: item.label}), el("td", {text: score(item.value)}),
        el("td", {text: item.retuned !== null ? score(item.retuned) : "沒進重調"}),
        el("td", {text: round.baseline === null ? "—" : signed(value - round.baseline) + (item.retuned !== null ? "" : "（快篩）")}),
        el("td", {text: item.chosen ? "換上了" : "沒選"}));
    }));
    return el("div", {}, el("p", {text: `第 ${round.round} 輪：現狀 ${score(round.baseline)}；${outcome}`}),
      el("table", {}, el("thead", {}, head), body));
  });
  box.replaceChildren(el("h3", {text: "換樣板試過什麼"}), ...lines, el("p", {class: "hint", text: TRY_SCALE_NOTE}),
    el("details", {}, el("summary", {text: "每一輪全部候選的合分"}), ...tables));
}

function showJob(job, detail, detailError) {
  // 狀態換了，上一個動作留下的話（例如「接續了。」）就過期了，不留著誤導。
  if (byId("job-block").dataset.state && byId("job-block").dataset.state !== `${job.job_id}:${job.state}`) say("job-message", "");
  byId("job-block").dataset.state = `${job.job_id}:${job.state}`;
  byId("job-title").textContent = `這一跑：${job.scheme_id}（${job.job_id.slice(0, 8)}，開始 ${when(job.created_at)}）`;
  byId("job-state").textContent = job.state_text;
  byId("job-state").className = ["failed", "surrogate_refused", "interrupted", "endpoint_failed", "endpoint_interrupted"].includes(job.state) ? "notice" : "";
  const lines = progressLines(job, detail);
  if (detailError) lines.push(`進度細節讀不到：${detailError}`);
  byId("job-progress").replaceChildren(...lines.map((text) => el("li", {text})));
  const notes = detail ? detail.notes : [];
  byId("job-notes").hidden = !notes.length;
  byId("job-notes").querySelector("ul").replaceChildren(...notes.map((text) => el("li", {text})));
  showJobDepths(job.plan, job.wall_names);
  const lastStep = job.steps[job.steps.length - 1];
  const failures = [...(detail ? detail.failures : [])];
  if (lastStep && lastStep.error) failures.unshift(`開服務那一層：${lastStep.error}`);
  const failed = ["failed", "surrogate_refused", "endpoint_failed"].includes(job.state);
  if (failed && !failures.length && lastStep) failures.push(...lastStep.log_tail);
  byId("job-failures").hidden = !failures.length;
  byId("job-failures").querySelector("ul").replaceChildren(...failures.map((text) => el("li", {text})));
  byId("job-stop").hidden = !job.actions.stop;
  byId("job-resume").hidden = !job.actions.resume;
  byId("job-endpoint").hidden = !job.actions.endpoint;
  byId("job-rebuild").hidden = job.state !== "surrogate_refused";
  const adopted = detail ? detail.adopted : null;
  byId("job-adopt").hidden = job.state !== "done" || Boolean(adopted);
  byId("job-endpoint-note").hidden = !job.endpoint;
  byId("job-endpoint-note").textContent = job.endpoint ? job.endpoint.baseline_note : "";
  showTemplateTries(detail ? (detail.template_tries || []) : [], job.wall_names);
  const warning = detail ? detail.baseline_warning : null;
  byId("job-baseline-warning").hidden = !warning;
  byId("job-baseline-warning").textContent = warning || "";
  const adoptedLine = byId("job-adopted");
  adoptedLine.hidden = !adopted;
  if (adopted) {
    const words = RUN_WORDS[adopted.run_status] || adopted.run_status;
    adoptedLine.replaceChildren(`已存成方案「${adopted.scheme_id}」，${words}。`,
      adopted.run_error ? el("span", {class: "notice", text: ` ${adopted.run_error}`}) : "",
      " ", el("a", {href: adopted.result_url, id: "result-link", text: "看結果頁（每一塊貼了什麼、改善多少）"}));
  }
  byId("job-steps").replaceChildren(...job.steps.map((step, index) => el("div", {},
    el("h4", {text: `第 ${index + 1} 步：${{start: "開跑", resume: "接續", endpoint: "終點"}[step.kind]}（${when(step.launched_at)}）` +
      (step.exit_code === null || step.exit_code === undefined ? "" : `，離開碼 ${step.exit_code}`)}),
    el("pre", {text: step.log_tail.join("\n")}))));
}

async function jobAction(action) {
  say("job-message", "送出中…");
  const reply = await api(`/api/gradient/jobs/${selectedJob}/${action}`, {}, "job-message");
  if (!reply.ok) { say("job-message", errorText(reply), true); return; }
  say("job-message", {stop: "已送出停止（下一次求值之前停）。", resume: "接續了。", endpoint: "終點開跑了。"}[action]);
  await refreshJob();
  refreshJobs();
}

async function adopt() {
  say("job-message", "正在存成方案、開一般計算…");
  const reply = await api(`/api/gradient/runs/${selectedJob}/adopt`, {}, "job-message");
  if (!reply.ok) { say("job-message", errorText(reply), true); return; }
  say("job-message", "存好了，一般計算開跑；做完點「看結果頁」。");
  await refreshJob();
}

async function rebuild() {
  const reply = await api(`/api/gradient/jobs/${selectedJob}`);
  if (!reply.ok) { say("job-message", errorText(reply), true); return; }
  schemeId = reply.body.scheme_id;
  byId("scheme-pick").value = schemeId;
  await loadPlan();
  await loadSurrogatePlan(true);
  byId("surrogate-block").scrollIntoView();
  say("surrogate-message", "這一跑的替身被准用閘門拒收：按「建替身」重建一份（新的建好、核過才換掉舊的）。", true);
}

async function refreshBuilds() {
  const reply = await api("/api/gradient/surrogates");
  if (reply.ok) knownBuilds = reply.body.builds;
}

// 上一次還沒回來又輪到：記一筆、做完立刻補一次，不吞掉（#1128：被吞掉的那一次若剛好是「查得到忙不忙了」，
// 開始鈕要等下一輪才亮；假時鐘的考卷一次只推一格，吞掉就永遠不亮）。
async function poll() {
  if (polling) { pollAgain = true; return; }
  polling = true;
  try {
    do {
      pollAgain = false;
      await refreshBuilds();
      await refreshJobs();
      if (selectedJob) await refreshJob();
      if (watchedBuild) {
        const reply = await api(`/api/gradient/surrogates/${watchedBuild}`);
        if (reply.ok) showBuild(reply.body);
      }
    } while (pollAgain);
  } finally {
    polling = false;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  byId("load-plan").addEventListener("click", loadPlan);
  byId("preview").addEventListener("click", preview);
  byId("build-surrogate").addEventListener("click", buildSurrogate);
  byId("stop-build").addEventListener("click", stopBuild);
  byId("start").addEventListener("click", start);
  byId("job-stop").addEventListener("click", () => jobAction("stop"));
  byId("job-resume").addEventListener("click", () => jobAction("resume"));
  byId("job-endpoint").addEventListener("click", () => jobAction("endpoint"));
  byId("job-adopt").addEventListener("click", adopt);
  byId("job-rebuild").addEventListener("click", rebuild);
  byId("mount").addEventListener("change", mountWarning);
  byId("form-block").addEventListener("input", markStale);
  byId("form-block").addEventListener("change", markStale);
  for (const id of ["max-full-solves", "stop-residual"]) byId(id).addEventListener("input", () => { budgetTyped = true; });
  await refreshBuilds();
  await refreshJobs();
  await loadSchemes();
  await poll();
  setInterval(poll, POLL_MS);
});
