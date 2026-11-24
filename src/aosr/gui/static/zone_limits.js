/* 超過限制的區塊標紅（#1111，#1082 第 1-4 段）：每次確定的改動之後（切完一面、換了做法或材料、改了深度上限、
   改了房間尺寸、打開方案、按「檢查」）把表單現在的樣子與設計設定交伺服器判（/api/zone-limits），回來的每一條照原文
   寫在那一區旁邊；紅的擋「拿去搜尋」（開跑那邊伺服器也照同一把再擋一次）。判法只在伺服器一份，這裡只畫。
   不從每個輸入事件發（房間、喇叭、座位每打一個字都會觸發），不鎖頁；晚到的回覆照世代號丟掉。
   回覆到的時候使用者正在立面圖那一區的某一格打字、或正在拖框：先不重畫（重畫會換掉那一格或那張圖，打到一半的字、
   拖到一半的框就沒了），離開那一格、放開滑鼠再畫。 */
let zoneLimits = null;
let zoneLimitsGeneration = 0;
let zoneLimitsReceived = 0;  // 收下的是第幾次的判決（最新那次回來才收）
let zoneLimitsShown = 0;     // 畫面上畫的是第幾次的判決；等於 zoneLimitsGeneration 表示最新的一次已經畫上去
let zoneLimitsRequest = Promise.resolve();
let zoneLimitsAfterDrag = 0; // 拖框中收下、放開滑鼠才畫的那一次
const DEPTH_LEVEL_WORDS = {zone: "這一塊", wall: "這面牆", room: "房間預設"};

// 這兩區的格子判決回來會重畫：立面圖那一區（第 2 段整段，#1265），和第 1 段右卡的總厚度上限（房間預設）那一格。
const LIMIT_TYPING_AREAS = "#elevation-panel, #depth-editor";
// 牆面表單（第 2 段「換原本的牆」打開的那一塊，#1265 搬進立面圖那一區）不在內：判決重畫不換它的格子，
// 以前它在立面圖那一區外面，打字不延後判決、也不被放回焦點反白。
function inLimitArea(node) { return Boolean(node?.closest?.(LIMIT_TYPING_AREAS)) && !node.closest("#walls"); }
// 正在打字＝焦點在上面兩區的某一格、而且那一格有還沒確定的字（跟畫出來時或上次確定的值不同；
// 數字格打了「.」這種還不成數字的字時，值讀起來是空的，瀏覽器另外標 badInput，也算）。
// 按 Tab 換到別格、按 Enter 確定之後那一格是乾淨的：照畫，畫完把焦點放回同一格，並把那一格整格反白（showZoneLimits）。
function typingInElevation() {
  const active = document.activeElement;
  return active instanceof HTMLInputElement && active.type !== "checkbox" && inLimitArea(active)
    && (active.value !== active.defaultValue || active.validity.badInput) ? active : null;
}
function showZoneLimits(generation) {
  if (generation !== zoneLimitsGeneration) return;
  zoneLimitsShown = generation;
  const active = document.activeElement;
  const focused = active?.id && inLimitArea(active) ? active.id : null;
  if (typeof renderElevation === "function") renderElevation();
  if (!focused) return;
  const again = document.getElementById(focused);
  again?.focus();
  // 放回去的是新畫的一格，原本那一格的反白與游標位置帶不過來（數字格讀不到選取範圍）：打字的格子一律整格反白，
  // 跟 Tab 進去時一樣，接著打的字是換掉舊值。不反白的話游標落在最前面，打的字會黏在舊值前面：
  // 「0.45」打 0.50 變成 0500.45，送出去就是一個超出牆面的框（#1231）。
  if (again instanceof HTMLInputElement && again.type !== "checkbox") again.select();
}

function refreshZoneLimits() {
  zoneLimitsRequest = checkZoneLimits();
  return zoneLimitsRequest;
}
// 「拿去搜尋」要的是最新那次的判決：還在路上就等它回來（不會丟錯：檢查失敗也是一份判決）。
async function settledZoneLimits() {
  while (zoneLimitsReceived !== zoneLimitsGeneration) await zoneLimitsRequest;
  return zoneLimits;
}
function zoneLimitsDragEnded() {
  const generation = zoneLimitsAfterDrag;
  zoneLimitsAfterDrag = 0;
  if (generation) showZoneLimits(generation);
}
async function checkZoneLimits() {
  zoneLimitsGeneration += 1;
  const generation = zoneLimitsGeneration;
  let result;
  try {
    const document = wallDocument();
    const {response, data} = await reply(fetch("/api/zone-limits", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scheme: document, design: zoneDesign})}));
    result = response.ok ? data : {status: "error", message: data.error || "限制檢查失敗", red: 0, zones: {}, walls: {}, caps: {}, refusals: []};
  } catch (error) {
    result = {status: "error", message: `限制檢查沒有做成：${errorText(error)}`, red: 0, zones: {}, walls: {}, caps: {}, refusals: []};
  }
  if (generation !== zoneLimitsGeneration) return;
  zoneLimits = result;
  zoneLimitsReceived = generation;
  const typing = typingInElevation();
  if (typing) { typing.addEventListener("focusout", () => showZoneLimits(generation), {once: true}); return; }
  if (window.elevationDrawing?.()) { zoneLimitsAfterDrag = generation; return; }
  showZoneLimits(generation);
}
function zoneLimitLines(wall, key) { return zoneLimits?.zones?.[wall]?.[key] ?? []; }
function wallLimitLines(wall) { return zoneLimits?.walls?.[wall] ?? []; }
function zoneIsRed(wall, key) { return zoneLimitLines(wall, key).some(line => line.level === "red"); }
function wallIsRed(wall) { return wallLimitLines(wall).some(line => line.level === "red"); }
function zoneCap(wall, key) { return zoneLimits?.caps?.[wall]?.[key] ?? null; }
function limitsRedCount() { return zoneLimits?.red ?? 0; }

function limitLine(line, id) {
  const item = document.createElement("p");
  item.className = `limit-${line.level}`;
  if (id) item.id = id;
  item.textContent = line.text;
  return item;
}
function limitLinesBlock(lines, idPrefix) {
  const block = document.createElement("div"); block.className = "limit-lines";
  lines.forEach((line, index) => block.append(limitLine(line, `${idPrefix}-${index}`)));
  return block;
}

// 填公分的深度格共用的換算（#1265：輸入頁兩格與「這間房貼什麼」那一頁的「最深」同一支）：記到公釐、大於 0 才收，
// 回公尺；不收（不是數字、不到 0.1 公分、0 或負的）回 NaN。空白怎麼算由各格自己定，這裡不收空白。
function depthFromCentimetres(text) {
  const millimetres = Math.round(Number(text) * 10);
  return Number.isFinite(millimetres) && millimetres > 0 ? millimetres / 1000 : NaN;
}
// 深度上限（填公分，存公尺）：空白＝不給（照下一層或材料自己的限制）；給了要大於 0。改了就重判。
// 總厚度上限那一格底下的提示句（第 1 段右卡房間預設、第 2 段「這一面另設」）：重畫時搬回同一個節點、不新建，
// 還沒處理的提示句才不會被洗掉（#1265 單元 2 審查發現 2）；記著是哪一份方案（哪一面）寫的，換了就清——那句講的是別一格。
// 要在清掉那一塊之前拿（清掉之後節點不在頁面上，找不到就會新建一個空的）。
function depthNote(id, owner) {
  const line = document.getElementById(id) ?? document.createElement("p");
  line.id = id; line.setAttribute("aria-live", "polite");
  if (line.dataset.owner !== owner) { line.textContent = ""; line.className = ""; line.dataset.owner = owner; }
  return line;
}
// 總厚度上限那幾格共用：收不收只在這裡判一次（空白，或大於 0 的公分數），提示句寫進 note（depthNote 那一行），
// 收了就清掉。拒收時格子退回最後一次確認的值（defaultValue）：那一格有焦點時不重建，建格當時的值早就過期
// （#1265 單元 2 審查發現 1：確認過 15 公分再填錯，退回的是空白，還被當成打到一半）。
function depthInput(id, label, value, blankMeans, note, apply) {
  const wrap = document.createElement("label"); wrap.textContent = label;
  const input = document.createElement("input"); input.type = "number"; input.min = "0"; input.step = "0.1"; input.id = id;
  input.defaultValue = value == null ? "" : String(centimetres(value));  // 確定過的值：跟它不同才算「正在打字」
  input.addEventListener("change", () => {
    const text = input.value.trim();
    const metres = text === "" ? null : depthFromCentimetres(text);
    if (Number.isNaN(metres)) {
      note.textContent = `總厚度上限要填大於 0 的數字（公分）；空白表示${blankMeans}`; note.className = "notice";
      input.value = input.defaultValue;
      return;
    }
    note.textContent = ""; note.className = "";
    apply(metres);
    input.defaultValue = input.value;
    markStale();
    refreshZoneLimits();
  });
  wrap.append(input);
  return wrap;
}
// 房間預設那一格在第 1 段右卡（room_facts.js，#1265）；這裡只留這一面。
function depthRow(wall, note) {
  const row = document.createElement("div"); row.className = "fields"; row.id = "depth-limits";
  row.append(depthInput(`depth-wall-${wall}`, `${wallNames[wall]}總厚度最多（公分；空白表示照房間預設）`, zoneDesign?.wall_max_depth_m?.[wall], "照房間預設", note, value => {
    const design = designForEdit();
    design.wall_max_depth_m ??= {};
    if (value === null) delete design.wall_max_depth_m[wall]; else design.wall_max_depth_m[wall] = value;
    if (!Object.keys(design.wall_max_depth_m).length) delete design.wall_max_depth_m;
  }));
  const hint = document.createElement("span"); hint.className = "note";
  hint.textContent = "三層（房間預設、這一面、每一區）取最小的；每一區的最深在「這間房貼什麼」選那一區設。指定貼一種的區：貼上去的總厚不能超過它；由程式最佳化的區：程式選的不會超過它";
  row.append(hint, note);
  return row;
}
function zoneDepthCell(wall, key, setting, cell) {
  if (setting.mode === "keep_original") { cell.textContent = "不加層"; return; }
  const depth = document.createElement("p"); depth.className = "note"; depth.id = `depth-zone-${wall}-${key}`;
  depth.textContent = setting.max_depth_m == null ? "這一區沒給" : `這一區給 ${shownNumber(setting.max_depth_m)}`;
  cell.append(depth);
  const cap = zoneCap(wall, key);
  const now = document.createElement("p"); now.className = "note"; now.id = `depth-cap-${wall}-${key}`;
  const targets = cap?.targets ?? [];
  now.textContent = !cap ? (setting.mode === "search" ? "沒給：照材料自己的限制" : "沒給上限")
    : targets.length ? `個案上限 ${shownNumber(cap.value_m)} 公尺（${DEPTH_LEVEL_WORDS[cap.level]}）；每一組跟自己的上限取較嚴：`
    : `生效：${shownNumber(cap.value_m)} 公尺（${DEPTH_LEVEL_WORDS[cap.level]}擋的）`;
  cell.append(now);
  // 交給搜尋的區（#1082 段 2-4）：准用的每一組各自的生效上限、哪一條擋住；放不下的那一組標出來（搜尋不換過去）。
  targets.forEach((target, index) => {
    const line = document.createElement("p");
    line.className = target.usable ? "note" : "limit-warn";
    line.id = `depth-target-${wall}-${key}-${index}`;
    line.textContent = target.text;
    cell.append(line);
  });
}
function limitsSummary() {
  const line = document.createElement("p"); line.id = "zone-limits-summary";
  if (!zoneLimits) { line.className = "note"; line.textContent = "還沒檢查限制"; return line; }
  if (zoneLimits.status !== "checked") { line.className = "notice"; line.textContent = `限制沒檢查成：${zoneLimits.message}`; return line; }
  const red = limitsRedCount();
  line.className = red ? "limit-red" : "ok";
  line.textContent = red ? `有 ${red} 條超過限制（標紅的區），改好之前不能拿去搜尋：${zoneLimits.refusals.join("；")}` : "限制都沒超過";
  return line;
}
window.refreshZoneLimits = refreshZoneLimits;
