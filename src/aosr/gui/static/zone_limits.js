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

// 正在打字＝焦點在立面圖那一區的某一格、而且那一格有還沒確定的字（跟畫出來時或上次確定的值不同；
// 數字格打了「.」這種還不成數字的字時，值讀起來是空的，瀏覽器另外標 badInput，也算）。
// 按 Tab 換到別格、按 Enter 確定之後那一格是乾淨的：照畫，畫完把焦點放回同一格。
function typingInElevation() {
  const active = document.activeElement;
  return active instanceof HTMLInputElement && active.type !== "checkbox" && active.closest("#elevation-panel")
    && (active.value !== active.defaultValue || active.validity.badInput) ? active : null;
}
function showZoneLimits(generation) {
  if (generation !== zoneLimitsGeneration) return;
  zoneLimitsShown = generation;
  const active = document.activeElement;
  const focused = active?.id && active.closest?.("#elevation-panel") ? active.id : null;
  if (typeof renderElevation === "function") renderElevation();
  if (focused) document.getElementById(focused)?.focus();
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

// 深度上限：空白＝不給（照下一層或材料自己的限制）；給了要大於 0。改了就重判。
function depthInput(id, label, value, apply) {
  const wrap = document.createElement("label"); wrap.textContent = label;
  const input = document.createElement("input"); input.type = "number"; input.min = "0"; input.step = "0.01"; input.id = id;
  input.defaultValue = value ?? "";  // 確定過的值：跟它不同才算「正在打字」
  input.addEventListener("change", () => {
    const text = input.value.trim();
    const number = Number(text);
    if (text !== "" && !(Number.isFinite(number) && number > 0)) {
      zoneNote("深度上限要填大於 0 的數字（公尺）；空白表示不給", "notice");
      input.value = value ?? "";
      return;
    }
    apply(text === "" ? null : number);
    input.defaultValue = input.value;
    markStale();
    refreshZoneLimits();
  });
  wrap.append(input);
  return wrap;
}
function depthRow(wall) {
  const row = document.createElement("div"); row.className = "fields"; row.id = "depth-limits";
  row.append(depthInput("depth-room", "深度上限：房間預設（公尺）", zoneDesign?.room_max_depth_m, value => {
    const design = designForEdit();
    if (value === null) delete design.room_max_depth_m; else design.room_max_depth_m = value;
  }));
  row.append(depthInput(`depth-wall-${wall}`, `${wallNames[wall]}（公尺）`, zoneDesign?.wall_max_depth_m?.[wall], value => {
    const design = designForEdit();
    design.wall_max_depth_m ??= {};
    if (value === null) delete design.wall_max_depth_m[wall]; else design.wall_max_depth_m[wall] = value;
    if (!Object.keys(design.wall_max_depth_m).length) delete design.wall_max_depth_m;
  }));
  const hint = document.createElement("span"); hint.className = "note";
  hint.textContent = "三層（房間預設、這面牆、每一塊）取最嚴的；固定的區比貼上去的總深，交給搜尋的區當搜尋的上限";
  row.append(hint);
  return row;
}
function zoneDepthCell(wall, key, setting, cell) {
  if (setting.mode === "keep_original") { cell.textContent = "不加層"; return; }
  cell.append(depthInput(`depth-zone-${wall}-${key}`, "", setting.max_depth_m, value => {
    const next = {...zoneSetting(wall, key)};
    if (value === null) delete next.max_depth_m; else next.max_depth_m = value;
    setZoneSetting(wall, key, next);
  }));
  const cap = zoneCap(wall, key);
  const now = document.createElement("p"); now.className = "note"; now.id = `depth-cap-${wall}-${key}`;
  const targets = cap?.targets ?? [];
  now.textContent = !cap ? (setting.mode === "search" ? "沒給：照材料自己的限制" : "沒給上限")
    : targets.length ? `個案上限 ${cap.value_m} 公尺（${DEPTH_LEVEL_WORDS[cap.level]}）；每一組跟自己的上限取較嚴：`
    : `生效：${cap.value_m} 公尺（${DEPTH_LEVEL_WORDS[cap.level]}擋的）`;
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
