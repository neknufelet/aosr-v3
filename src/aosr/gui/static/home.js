/* 首頁第一屏（#1134）：最近算的那間房（俯視圖）、三個入口、存著的方案每份一句白話。
   字、座標、連到哪都由伺服器排好（/api/home）；這裡只把它畫成 SVG、排成連結。前牆朝上、左牆在左。 */
const HOME_SVG = "http://www.w3.org/2000/svg";
function homeSvg(tag, attrs = {}, text = "") {
  const node = document.createElementNS(HOME_SVG, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== "") node.setAttribute(key, String(value));
  if (text) node.textContent = text;
  return node;
}
function homeElement(tag, text = "", className = "") {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}
// 俯視圖：房間外框、有貼東西的那幾段牆（實測藍、模擬橘）、喇叭、主位、四面牆名寫在牆外側。
function drawHomeRoom(room) {
  const svg = document.getElementById("home-room");
  const pad = 30, longest = 240;
  const scale = longest / Math.max(room.width_m, room.depth_m);
  const width = room.width_m * scale, depth = room.depth_m * scale;
  svg.setAttribute("viewBox", `0 0 ${width + 2 * pad} ${depth + 2 * pad}`);
  svg.replaceChildren();
  const at = (point) => [pad + point.x * scale, pad + point.y * scale];
  svg.append(homeSvg("rect", {x: pad, y: pad, width, height: depth, class: "home-wall"}));
  for (const segment of room.segments) {
    const [x1, y1] = at(segment.from), [x2, y2] = at(segment.to);
    svg.append(homeSvg("line", {x1, y1, x2, y2, class: `home-treated home-${segment.basis}`, "data-wall": segment.wall}));
  }
  const [px, py] = at(room.primary);
  for (const speaker of room.speakers) {
    const [sx, sy] = at(speaker.at);
    svg.append(homeSvg("line", {x1: sx, y1: sy, x2: px, y2: py, class: "home-beam"}));
  }
  for (const speaker of room.speakers) {
    const [sx, sy] = at(speaker.at);
    svg.append(homeSvg("rect", {x: sx - 6, y: sy - 6, width: 12, height: 12, rx: 2, class: "home-speaker"}));
    svg.append(homeSvg("text", {x: sx, y: sy + 20, "text-anchor": "middle", class: "home-small"}, speaker.name));
  }
  svg.append(homeSvg("circle", {cx: px, cy: py, r: 7, class: "home-seat"}));
  svg.append(homeSvg("text", {x: px, y: py + 22, "text-anchor": "middle", class: "home-small"}, "主位"));
  for (const wall of room.walls) {
    const [wx, wy] = at(wall.at);
    const side = wy <= pad + 0.5 ? "top" : wy >= pad + depth - 0.5 ? "bottom" : wx <= pad + 0.5 ? "left" : "right";
    const [x, y, turn] = {top: [wx, pad - 10, 0], bottom: [wx, pad + depth + 20, 0],
                          left: [pad - 10, wy, -90], right: [pad + width + 10, wy, 90]}[side];
    svg.append(homeSvg("text", {x, y, "text-anchor": "middle", class: "home-wall-name", "data-wall": wall.wall,
                                transform: turn ? `rotate(${turn} ${x} ${y})` : ""}, wall.name));
  }
}
function homeLink(id, link) {
  const anchor = document.getElementById(id);
  anchor.href = link.url;
  anchor.hidden = !link.url;
  anchor.querySelector(".home-note").textContent = link.note;
}
// 每讀一次加一：晚到的舊回覆不蓋掉新的（開頁時這一支與結果清單各叫一次）。
let homeSeq = 0;
async function refreshHome() {
  const mine = ++homeSeq;
  try { await drawHome(mine); } catch (error) {
    if (mine === homeSeq) document.getElementById("home-facts").textContent = `首頁摘要讀不到：${error.message}`;
  }
}
async function drawHome(mine) {
  const reply = await fetch("/api/home", {cache: "no-store"});
  let data;
  try { data = await reply.json(); } catch { throw new Error(`網頁伺服器回應 ${reply.status}，內容讀不懂`); }
  if (!reply.ok) throw new Error(data.error || `網頁伺服器回應 ${reply.status}`);
  if (mine !== homeSeq) return;
  document.getElementById("home-title").textContent = data.title;
  document.getElementById("home-facts").textContent = data.facts;
  document.getElementById("home-picture").hidden = !data.room;
  document.getElementById("home-unnamed").hidden = !data.room || data.room.named;
  if (data.room) drawHomeRoom(data.room);
  for (const [id, key] of [["home-result", "result"], ["home-materials", "materials"], ["home-layout", "layout"]])
    homeLink(id, data.links[key]);
  schemeSummaries = new Map(data.schemes.map(item => [item.name, item.text]));
  if ($("scheme-menu").hidden) drawSchemeMenu();
}
window.addEventListener("DOMContentLoaded", refreshHome);

// 輸入頁兩條操作列。存檔與計算仍走 app.js 共用的鎖；狀態用同一份表單的改動記號。
let schemeNames = [], savedTimes = {}, schemeSummaries = new Map(), finishedNames = new Set();
let activeRuns = [], followedState = null, dirtyField = "", allSchemes = false;
const copyFirst = "改了以後先按右上「另存新名字」存成新的一份，才能再算";

function menuItems() {
  return [...$("scheme-menu").querySelectorAll('[role="menuitem"]')].filter(node => node.offsetHeight);
}
function closeBarBoxes() {
  for (const id of ["scheme-menu", "save-as-box", "template-box"]) $(id).hidden = true;
  for (const id of ["scheme-menu-button", "save-as-toggle"]) $(id).setAttribute("aria-expanded", "false");
}
function toggleMenu() {
  const open = $("scheme-menu").hidden;
  closeBarBoxes();
  if (!open) return;
  allSchemes = false;
  drawSchemeMenu();
  $("scheme-menu").hidden = false;
  $("scheme-menu-button").setAttribute("aria-expanded", "true");
  menuItems()[0]?.focus();
}
function drawSchemeMenu() {
  const rows = $("home-schemes");
  const names = [...schemeNames].sort((a, b) => (savedTimes[b] || 0) - (savedTimes[a] || 0));
  rows.replaceChildren(...names.map((name, index) => {
    const row = document.createElement("div");
    row.dataset.scheme = name;
    row.hidden = !allSchemes && index >= 5;
    const open = homeElement("button"), archive = homeElement("button", "封存");
    for (const button of [open, archive]) button.setAttribute("role", "menuitem");
    open.className = "home-open"; open.tabIndex = archive.tabIndex = -1;
    const saved = savedTimes[name];
    const when = saved ? new Date(saved * 1000).toLocaleString("zh-TW", {hour12: false}) + " 存的，" : "";
    const status = runningFor(name) ? "正在算" : finishedNames.has(name) ? "算過，鎖住" : "還沒算過";
    open.append(homeElement("b", name), homeElement("small", (name === openedId ? "現在開著；" : "") + when + status));
    if (schemeSummaries.get(name)) open.append(homeElement("small", schemeSummaries.get(name)));
    open.setAttribute("aria-label", name);
    open.setAttribute("aria-current", String(name === openedId));
    open.onclick = () => action(async () => {
      closeBarBoxes();
      $("scheme-menu-button").focus();
      if (!await canLeaveForm()) return;
      if (!inputBusy) $("scheme-section").scrollIntoView();
      await openScheme(name);
    });
    archive.setAttribute("aria-label", "封存 " + name);
    archive.title = "收起來、拿得回來";
    archive.onclick = () => action(async () => {
      closeBarBoxes(); $("scheme-menu-button").focus(); await archiveScheme(name);
    });
    row.append(open, archive);
    return row;
  }));
  $("all-schemes").hidden = allSchemes || names.length <= 5;
  $("all-schemes").textContent = `看全部方案（${names.length} 份）`;
}
function formHasEdits() { return runEdited || window.zonePageHasEdits(); }
async function canLeaveForm() {
  // 鎖頁中的程式點擊照走 openScheme 自己的拒收提示。
  if (!formHasEdits() || inputBusy) return true;
  const dialog = $("leave-edits"), locked = formRunning() || finishedNames.has(openedId);
  dialog.querySelector('[value="save"]').disabled = Boolean(locked);
  $("leave-note").textContent = locked ? "要留下改動，請取消後按「另存新名字」。" : "要先儲存目前的改動嗎？";
  dialog.returnValue = "cancel";
  dialog.showModal();
  await new Promise(resolve => dialog.addEventListener("close", resolve, {once: true}));
  if (dialog.returnValue === "save") return await save() && !formHasEdits();
  return dialog.returnValue === "discard";
}
function bindBars() {
  $("nav-zones").onclick = event => { event.preventDefault(); window.openZoneOverview(); };
  document.querySelector(".input-nav").addEventListener("click", event => {
    const link = event.target.closest("a");
    if (!link || link.hash || !formHasEdits() || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    action(async () => { if (await canLeaveForm()) location.assign(link.href); });
  });
  $("scheme-menu-button").onclick = toggleMenu;
  $("scheme-menu-button").disabled = false;
  $("scheme-menu-button").onkeydown = event => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault(); toggleMenu();
    if (event.key === "ArrowUp") menuItems().at(-1)?.focus();
  };
  $("scheme-menu").onkeydown = event => {
    const items = menuItems(), index = items.indexOf(document.activeElement);
    const next = {ArrowDown: (index + 1) % items.length, ArrowUp: (index + items.length - 1) % items.length,
                  Home: 0, End: items.length - 1}[event.key];
    if (next !== undefined) { event.preventDefault(); items[next].focus(); }
    if (event.key === "Tab") { closeBarBoxes(); $("scheme-menu-button").focus(); }
  };
  $("show-results").onclick = () => openResults();
  $("all-schemes").onclick = () => { allSchemes = true; drawSchemeMenu(); menuItems()[0]?.focus(); };
  $("choose-template").onclick = () => {
    closeBarBoxes(); $("template-box").hidden = false; $("template-list").focus();
  };
  // 頂條選單「看已封存的」：有封存就展開那一塊、游標到它的標題；沒有封存（第二趟 F5）照樣打開抽屜，
  // 那一塊只寫一句「還沒有封存的結果」，不展開空表。
  $("show-archived").onclick = () => {
    closeBarBoxes(); openResults();
    archiveAsked = true;
    if ($("archived").hidden) {
      $("archive-none").hidden = false; $("archive-none").focus({preventScroll: true});
      return;
    }
    $("archived").open = true;
    $("archived").scrollIntoView({block: "nearest"}); $("archived").querySelector("summary").focus({preventScroll: true});
  };
  bindResultsDrawer();
  $("save-as-toggle").onclick = () => {
    const open = $("save-as-box").hidden;
    closeBarBoxes();
    if (open) {
      $("save-as-box").hidden = false; $("save-as-toggle").setAttribute("aria-expanded", "true"); $("save-as-id").focus();
    }
  };
  $("cancel-save-as").onclick = () => { closeBarBoxes(); $("save-as-toggle").focus(); };
  $("save-as-box").onsubmit = event => { event.preventDefault(); action(saveAs); };
  document.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    const button = $("save-as-box").hidden ? $("scheme-menu-button") : $("save-as-toggle");
    if (["scheme-menu", "save-as-box", "template-box"].every(id => $(id).hidden)) return;
    closeBarBoxes(); button.focus();
  });
  document.addEventListener("pointerdown", event => {
    if (!event.target.closest?.(".bar-anchor")) closeBarBoxes();
  });
  $("zone-to-search").onclick = () => action(sendToSearch);
}
// 結果抽屜（#1265 單元 4）：頂條「算過的結果與比較 ▸」打開；關閉鈕、Esc、點遮罩關，關了游標回那顆鈕。
// 開著時 Tab 只在抽屜裡轉。抽屜住在 main 裡：存檔、計算、開檔、切牆等把 main 鎖住（inert）的時候，
// 抽屜裡的封存、搬回、刪除也一起按不了（這道鎖不另外接）。
let archiveAsked = false;
function openResults() {
  closeBarBoxes();
  $("results-note").textContent = ""; $("results-note").className = "";
  archiveAsked = false; $("archive-none").hidden = true;
  drawerFocus = null;
  $("results-drawer").hidden = false;
  document.documentElement.classList.add("drawer-open");
  drawResultRows();
  $("results-close").focus();
}
function closeResults() {
  if ($("results-drawer").hidden) return;
  $("results-drawer").hidden = true;
  document.documentElement.classList.remove("drawer-open");
  $("show-results").focus();
}
// 抽屜開著時清單重畫（重讀、封存、搬回、刪除、換方案）或鎖頁解開（第二趟 F1、F3）：節點被換掉、或 main 被鎖住
// 那一下，游標會無聲掉回 body，不會有 focusin。所以游標進抽屜時先記住它在哪（哪張表、哪一筆、哪一種按鈕，
// 照按鈕種類認：勾選格、看、封存、搬回、永久刪除），重畫完與解鎖後放回同一筆的同一顆；那一筆不在了，
// 放到原位置的下一列同一顆，沒有就放到「關閉 ✕」。main 鎖著時 .focus() 沒作用，留到解鎖再放。
const drawerLists = {"results-list": "runId", "archived-packages-list": "packageId", "archived-list": "runId"};
let drawerFocus = null;
const drawerKind = node => `${node.tagName}:${node.textContent}`;
function rowStops(row) {
  return [...row.querySelectorAll("input, a[href], button")];
}
function noteDrawerFocus(node = document.activeElement) {
  if (!(node instanceof HTMLElement) || !$("results-panel").contains(node) || !node.getClientRects().length) return;
  const row = node.closest("tr"), body = row?.parentElement;
  drawerFocus = body && body.id in drawerLists
    ? {list: body.id, id: row.dataset[drawerLists[body.id]], kind: drawerKind(node), index: [...body.children].indexOf(row)}
    : {node};
}
function keepDrawerFocus() {
  const active = document.activeElement;
  if ($("results-drawer").hidden || $("leave-edits").open || document.querySelector("main").inert) return;
  if ($("results-panel").contains(active) && active.getClientRects().length) return;
  const key = drawerFocus;
  let target = key?.node?.isConnected ? key.node : null;
  if (key?.list) {
    const rows = [...$(key.list).children];
    const row = rows.find(item => item.dataset[drawerLists[key.list]] === key.id) || rows[key.index];
    target = row && rowStops(row).find(node => drawerKind(node) === key.kind);
  }
  if (!target || target.disabled || !target.getClientRects().length) target = $("results-close");
  target.focus({preventScroll: true});
  noteDrawerFocus(target);
}
function drawerStops() {
  return [...$("results-panel").querySelectorAll("button, a[href], input, summary")]
    .filter(node => !node.disabled && node.getClientRects().length);
}
function bindResultsDrawer() {
  $("results-close").onclick = closeResults;
  $("results-mask").onclick = closeResults;
  document.addEventListener("keydown", event => {
    if ($("results-drawer").hidden || $("leave-edits").open) return;
    // 整頁鎖著（存檔、計算、換方案…，main.inert）：Esc 不關、Tab 不動，解鎖後游標還在抽屜裡（第二趟 F2）。
    if (document.querySelector("main").inert) {
      if (["Escape", "Tab"].includes(event.key)) event.preventDefault();
      return;
    }
    if (event.key === "Escape") { event.preventDefault(); closeResults(); return; }
    if (event.key !== "Tab") return;
    const stops = drawerStops(), index = stops.indexOf(document.activeElement);
    if (!stops.length) return;
    if (event.shiftKey && index <= 0) { event.preventDefault(); stops.at(-1).focus(); }
    else if (!event.shiftKey && (index === -1 || index === stops.length - 1)) { event.preventDefault(); stops[0].focus(); }
  });
  // 游標被別處搶走（鎖頁解開後放回、點到遮罩外）時拉回抽屜。
  document.addEventListener("focusin", event => {
    if (!$("results-drawer").hidden && !$("results-drawer").contains(event.target) &&
        !$("leave-edits").open) $("results-close").focus();
    else noteDrawerFocus(event.target);
  });
  // 鎖頁解開（main 拿掉 inert）：游標若在鎖住那一下掉回 body，放回記住的那一顆。
  new MutationObserver(keepDrawerFocus).observe(document.querySelector("main"), {attributes: true, attributeFilter: ["inert"]});
}
function runningFor(name) {
  if (!name) return null;
  if (followedState?.status === "running" && followedState.scheme_id === name) return followedState;
  return activeRuns.find(item => item.scheme_id === name);
}
function formRunning() { return runningFor(openedId); }
function renderInputBars() {
  const running = formRunning(), locked = running || finishedNames.has(openedId);
  const job = followedState, failed = job && job.scheme_id === openedId && ["failed", "stopped"].includes(job.status);
  const dirty = formHasEdits(), state = $("save-state");
  state.dataset.kind = dirty ? "dirty" : locked ? "locked" : openedId ? "saved" : "template";
  state.textContent = dirty ? "有改動還沒存" + (dirtyField ? "：" + dirtyField : "") +
    (locked ? "。請另存新名字" : "。按右邊「儲存」") : running ? "正在算，這一份已鎖住" :
    locked ? "算過了，內容鎖住留著比較" : openedId ? "都存好了（還沒算過）" : $("scheme-archive-notice").hidden ? "範本還沒存" : "這份方案已封存";
  $("save").hidden = Boolean(locked);
  $("save-as-toggle").classList.toggle("primary", Boolean(locked));
  $("lock-bar").hidden = !locked;
  $("lock-bar").textContent = (running ? "這一份正在算，內容鎖住。" : "這一份算過了，內容鎖住，之後拿來跟別份比。") +
    "要改的話照樣在下面改，改完按「另存新名字」存成新的一份；原來這份不動。";
  const done = !$("result-link").hidden && !dirty;
  $("calculate-action").hidden = Boolean(running || done);
  $("calculate").disabled = Boolean(locked && dirty);
  $("calculate").textContent = failed ? "再算一次" : "照現在牆上貼的算一次";
  $("calculate-note").hidden = Boolean(failed);
  const other = job && job.scheme_id !== openedId;
  const stopHome = $(other ? "followed-stop" : "current-stop");
  if ($("stop").parentNode !== stopHome) stopHome.append($("stop"));
  $("stop").hidden = job?.status !== "running";
  $("run-log").hidden = !job || (other && openedId !== null) || (!failed && !job.stderr_tail?.length);
  $("run-state").hidden = Boolean(other && openedId && job.status !== "running");
  $("search-action").hidden = Boolean(failed);
  const reason = runFailureReason(job);
  $("next-title").textContent = running ? `正在算這一份：已跑 ${running.elapsed_s} 秒` :
    failed ? "沒算完：" + reason :
    done ? "算好了" : locked ? "這一份已經算過了" : "改好了就選一個往下走";
  $("input-footer").dataset.state = running ? "running" : failed ? "failed" : done ? "done" : locked ? "locked" : "idle";
  $("next-note").textContent = running ? "算的是按下那一刻存好的版本。可以離開這頁，回來再看。" :
    failed ? "方案已經存好。先看出錯紀錄，再決定要不要再算。" : locked && dirty ? copyFirst :
    done ? "這一份現在鎖住，留著比較。要再改就另存新名字。" : locked ? "沒改過可以直接再算。要修改這份方案，請另存新名字。" :
    "不用先按儲存：按這兩顆會自動先存。算過的方案會留著比較。";
  if (scheme) renderSearchAction();
  // 第 1 段右卡（room_facts.js）：結果那一行跟這一條同一份狀態與原因（有沒有改過也是頂條那一份 dirty）；
  // 總厚度上限、改造前跟著設計設定換字。
  renderRoomResult(running, failed, reason, dirty);
  renderRoomEditors();
}
// 沒算完的原因：計算程式自己寫的那一句，沒有就拿出錯紀錄最後一行（英文的加一句是誰說的），都沒有照狀態講。
function runFailureReason(job) {
  const lastLine = job?.stderr_tail?.filter(line => line.trim()).at(-1);
  return job?.process_note || (lastLine && (/[\u3400-\u9fff]/u.test(lastLine) ? lastLine : "計算程式回報：" + lastLine)) ||
    (job?.status === "stopped" ? "已停止" : "原因不明，看出錯紀錄");
}
