/* 超過限制的區塊標紅（#1111，#1082 第 1-4 段）：每次確定的改動之後（切完一面、換了做法或材料、改了深度上限、
   改了房間尺寸、打開方案、改了第 1 段任一格，#1265 單元 6 起不再有「檢查」鈕）把表單現在的樣子與設計設定交伺服器判（/api/zone-limits），回來的每一條照原文
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
// 判決在路上（#1265 單元 3 之二第二趟追加第 1 條）：切牆、清掉、復原、改做法這幾條路先重畫、後重問（refreshPlan 才叫
// refreshZoneLimits），中間那一下收著的判決對的是舊切法、舊做法。重畫之前先 staleZoneLimits() 標好：這之前收的判決不算數，
// 要等之後新問的那一次回來。配對（zone_overview.js 的 readPairing）、立面圖的鏡像（elevation.js 的 symmetryResult）、
// 右欄那一區的上限（wall_section.js 的 capWords／noCapWords）、逐組上限（depthTargetLines）都看 zoneLimitsFresh() 這一個訊號。
let zoneLimitsFloor = 0;
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

// 頂條那一句（check_state.js）跟著這三個點換：標「之前收的不算數」、送出新的一次、收下最新那一次。
function staleZoneLimits() { zoneLimitsFloor = zoneLimitsGeneration + 1; renderCheckState(); }
function zoneLimitsFresh() { return Boolean(zoneLimits) && zoneLimitsReceived === zoneLimitsGeneration && zoneLimitsReceived >= zoneLimitsFloor; }
function refreshZoneLimits() {
  zoneLimitsRequest = checkZoneLimits();
  // 畫著的是一份沒做成的判決（鏡子那一句寫「沒確認成」）：新的一次已經出去了，那一句先改回「正在確認」（#1265 單元 3 之二收尾
  // 第二趟 B4 之一）。只重寫那一句、不整頁重畫；做成的判決照舊等新的回來才換（每次重問都閃一下更糟）。
  if (zoneLimits?.status !== "checked") window.refreshAxisWords();
  return zoneLimitsRequest;
}
// 「拿去搜尋」要的是最新那次的判決：還在路上就等它回來（不會丟錯：檢查失敗也是一份判決）。
// 手上那一個請求回來了、判決卻還不是最新（中間又出去了一次，或 zoneLimitsRequest 指的是先出去的那一個）：
// 不能再 await 同一個已經好了的請求——那只讓出微任務、不讓出事件迴圈，整頁會卡死（雲端考卷關分頁時卡住 30 分鐘）。
// 換等目前的請求；還是同一個就讓出事件迴圈一下再看。
async function settledZoneLimits() {
  while (zoneLimitsReceived !== zoneLimitsGeneration) {
    const pending = zoneLimitsRequest;
    await pending;
    if (pending === zoneLimitsRequest && zoneLimitsReceived !== zoneLimitsGeneration)
      await new Promise(resolve => setTimeout(resolve, 50));
  }
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
  renderCheckState();
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
  renderCheckState();
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
// 數字格裡的公分（#1265 單元 3 審查真錯 1）：數字格收不下千分位逗號（「1,000」會變空白、又被當成正在打字），
// 格子的值一律用這一支；給人看的字照用 centimetres（帶千分位）。
function centimetresBox(metres) { return String(Math.round(metres * 1000) / 10); }
// 公分深度格填的是什麼（輸入頁兩格與「這間房貼什麼」小框的「最深」同一支判法，#1265 單元 3 審查真錯 2）：
// 瀏覽器讀不出來（unreadable＝validity.badInput，例如打了 --1，這時 value 是空字串）NaN，不當成空白；
// 空白 null；其餘照 depthFromCentimetres（不收回 NaN）。
function depthBoxValue(text, unreadable) {
  if (unreadable) return NaN;
  const trimmed = text.trim();
  return trimmed === "" ? null : depthFromCentimetres(trimmed);
}
// 不收時那一句（同一支寫）：subject 是哪一格，blank 是那一格空白的意思（各格自己的）。
function depthRefusedWords(subject, blank) { return `${subject}要填大於 0 的數字（公分，最少 0.1）；空白${blank}`; }
// 深度上限（填公分，存公尺）：空白＝不給（照下一層或材料自己的限制）；給了要大於 0。改了就重判。
// 總厚度上限那一格底下的提示句（第 1 段右卡房間預設、第 2 段「這一面另設」）：重畫時搬回同一個節點、不新建，
// 還沒處理的提示句才不會被洗掉（#1265 單元 2 審查發現 2）；記著是哪一次打開（哪一面）寫的，換了就清——那句講的是別一格。
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
  input.defaultValue = value == null ? "" : centimetresBox(value);  // 確定過的值：跟它不同才算「正在打字」
  input.addEventListener("change", () => {
    const metres = depthBoxValue(input.value, input.validity.badInput);
    if (Number.isNaN(metres)) {
      note.textContent = depthRefusedWords("總厚度上限", `表示${blankMeans}`); note.className = "notice";
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
// 交給搜尋的區（#1082 段 2-4）：「個案上限…每一組跟自己的上限取較嚴」那半句，接准用的每一組各自的生效上限、哪一條擋住；
// 放不下的那一組標出來（橘字，搜尋不換過去）。伺服器逐組寫好一句（zoneLimits.caps[牆][區].targets）。
// #1265 單元 3 之二第 C 條起畫在第 2 段右欄那一區底下（wall_section.js）；整頁一份、id 照舊。
// 沒有逐組的（固定的區、沒給上限的）回空的：那一區的上限照右欄那一句（capWords、noCapWords）寫。
// 判決在路上就不畫上一版的（做法剛改、最深剛改、切法剛換，那幾行對的是舊的）。不另判做法（#1265 單元 3 之二收尾）：
// 伺服器只替由程式最佳化的區寫 targets（zone_limits.py 的 zone_limits，mode == "search" 那一段），判決新鮮時它看的就是
// 現在這一份做法——改做法的每一條路（小框套用、切牆、清掉再復原）重畫之前都先 staleZoneLimits()，換方案把判決清成 null。
// 一組都放不下、整區紅的時候（那一區有 depth_no_group 那一條紅）不畫（#1265 單元 3 之二收尾第二趟 A）：那一條紅字就是伺服器把
// 每一組這一句接起來（zone_limits.py 的 no_group_words），已經寫在這一區底下（limit-<牆>-<區>-<i>），再列一次是同一件事寫兩遍。
// 留紅字那一份：它是擋住的原因（開跑那邊擋的也是這一句），而且開頭那半句「這一區什麼都貼不上」只有它講；上限從哪裡來，
// 每一組那一句裡有（「（房間預設的上限）」），右欄那一區那一句也寫（capWords）。認法照那一條的代號，不比對字。
// 不紅、只有幾組放不下的區照畫：那幾行告訴人搜尋會跳過哪幾組。
const NO_GROUP_FITS = "depth_no_group";
function depthTargetLines(wall, key) {
  if (!zoneLimitsFresh()) return [];
  if (zoneLimitLines(wall, key).some(line => line.code === NO_GROUP_FITS)) return [];
  const cap = zoneCap(wall, key), targets = cap?.targets ?? [];
  if (!targets.length) return [];
  const now = document.createElement("p"); now.className = "note"; now.id = `depth-cap-${wall}-${key}`;
  now.textContent = `個案上限 ${shownNumber(cap.value_m)} 公尺（${DEPTH_LEVEL_WORDS[cap.level]}）；每一組跟自己的上限取較嚴：`;
  return [now, ...targets.map((target, index) => {
    const line = document.createElement("p");
    line.className = target.usable ? "note" : "limit-warn";
    line.id = `depth-target-${wall}-${key}-${index}`;
    line.textContent = target.text;
    return line;
  })];
}
// 擋下的每一條（zoneLimits.refusals）屬於誰：照伺服器排的順序從回覆本身認（zone_limits.py 的 zone_limits 最後那幾行：最上層
// ＝design 每一條、整面紅的＝walls 逐面、各區紅的＝zones 逐面逐區），不比對字串（#1265 單元 3 之二收尾第二趟 A；伺服器考卷
// test_gui_zone_limits.py 的 test_refusals_come_in_the_order_the_reply_itself_lays_out 釘住這個排法）。回每一面整面那一段、
// 那一面各區那一段各是 refusals 的第幾條到第幾條（[起, 迄)）。面的代號不是數字，物件照回覆的順序走；同一面裡區的先後不用（區代號
// 可以是純數字，瀏覽器會把它排到前面），只算那一面各區一共幾條。
function refusalSpans() {
  const reds = lines => lines.filter(line => line.level === "red").length;
  const spans = {walls: {}, zones: {}};
  let at = zoneLimits.design.length;
  for (const [wall, lines] of Object.entries(zoneLimits.walls)) spans.walls[wall] = [at, at += reds(lines)];
  for (const [wall, items] of Object.entries(zoneLimits.zones)) {
    spans.zones[wall] = [at, at += Object.values(items).reduce((sum, lines) => sum + reds(lines), 0)];
  }
  return spans;
}
// 右欄上方的總述（#1265 單元 3 之二收尾 K、第二趟 A：同一件事整欄只寫一次）。wall：第 2 段右欄現在列的那一面（沒列是 null）；
// keys：那一面右欄列出來的區（代號，照號碼排）。那一面整面的判決寫在總述底下（limit-wall-<牆>-<i>），那一面紅的區每一條寫在
// 那一區底下（limit-<牆>-<區>-<i>，wall_section.js）：總述不再抄，改說寫在哪裡。別面的照列（畫面上看不到）。
// 那一面紅的區只要有一區沒列在右欄（判決還是上一版切法的），那一面各區的就照抄：寧可多寫一次，不藏掉。
function limitsSummary(wall, keys) {
  const line = document.createElement("p"); line.id = "zone-limits-summary";
  if (!zoneLimits) { line.className = "note"; line.textContent = "還沒檢查限制"; return line; }
  if (zoneLimits.status !== "checked") { line.className = "notice"; line.textContent = `限制沒檢查成：${zoneLimits.message}`; return line; }
  const red = limitsRedCount();
  // 都沒超過：右欄不寫（#1265 單元 6）。「檢查通過」只寫在頂條狀態旁（check_state.js），回 null 由呼叫的人略過。
  if (!red) return null;
  line.className = "limit-red";
  const spans = refusalSpans(), none = [0, 0];
  const redKeys = Object.keys(zoneLimits.zones[wall] ?? {}).filter(key => zoneIsRed(wall, key));
  const zoned = keys.filter(key => redKeys.includes(key));
  const [wallFrom, wallTo] = spans.walls[wall] ?? none;
  const [zoneFrom, zoneTo] = zoned.length === redKeys.length ? spans.zones[wall] ?? none : none;
  const below = index => (index >= wallFrom && index < wallTo) || (index >= zoneFrom && index < zoneTo);
  const whole = zoneLimits.refusals.filter((text, index) => index >= wallFrom && index < wallTo).length;
  const inZones = zoneLimits.refusals.filter((text, index) => index >= zoneFrom && index < zoneTo).length;
  const parts = [...zoneLimits.refusals.filter((text, index) => !below(index)),
    ...(whole ? [`${wallNames[wall]}整面的 ${whole} 條寫在下面`] : []),
    ...(inZones ? [`${zoned.map(key => zoneCalled(wall, key)).join("、")}的 ${inZones} 條寫在${zoned.length > 1 ? "各區" : "那一區"}底下`] : [])];
  line.textContent = `有 ${red} 條超過限制（標紅的區），改好之前不能拿去搜尋：${parts.join("；")}`;
  return line;
}
window.refreshZoneLimits = refreshZoneLimits;
