/* 每一區的做法（#1105，#1082 第 1-3 段）：每一區（畫的、自動補的、整面一區）是「指定貼一種」「不貼，保留原樣」「由程式最佳化」
   （設計檔裡叫 fixed／keep_original／search）；由程式最佳化的區一律存一份明確清單（allowed）。
   #1265 單元 3 之二第二趟起輸入頁那張舊做法表拆了：做法、材料、候選、最深都在「這間房貼什麼」那一頁的小框改（zone_overview.js，
   換材料走 /api/zone-material/batch），輸入頁第 2 段右欄只列出來看（wall_section.js）；這一支留著做法的狀態、材料庫、共用的字與「拿去搜尋」。
   做法存在設計設定（/api/designs，跟方案同名、跟著方案存）。材料選單讀材料庫（/api/materials），不另做一份。
   設計設定整份原樣帶著走（原方案、之後的段加的欄位都不丟），只改做法管的那幾格。 */
let zoneDesign = null;
let zoneDesignStored = false;
let zoneDesignNote = "";
// 伺服器上那一份設計設定讀不到（壞檔或連不上）：這一頁不准存回去，免得一份空的蓋掉原本每一區的做法。
let zoneDesignBroken = false;
// 最近一次存方案時每一區的做法有沒有存成（沒設過、不必寫也算存成）；「拿去搜尋」靠它決定跳不跳。
let zoneDesignSaved = false;
// 頁面上那一份做法換過幾次（打開方案、載入範本各算一次）：存檔途中使用者換了方案，晚到的回覆不准把舊做法寫回來。
let zoneDesignGeneration = 0;
let zoneLibrary = null;
let zoneLibraryProblem = "";
// 每一組材料的標題（伺服器 material_surfaces.GROUP_WORDS）；小框「顯示全部材料」打開才列不是這一面用的那幾組。
let zoneSurfaceGroups = {};
let zoneNoteText = "";
let zoneNoteKind = "";
// 「拿去搜尋」（sendToSearch）沒送出去的那一句：{text, kind}。輸入頁寫在頁底那顆鈕底下（zone-to-search-note，renderSearchAction），
// 「這間房貼什麼」那一頁照舊寫在 zone-page-note；下一次有別的回話（zoneNote）、換牆、換方案就拿掉。
let zoneSearchSaid = null;
// 同一份方案上次套用的候選清單，給新開的區當起點。
let zoneLastAllowed = null;
// 選的跟原牆一模一樣、做法自己跳成「不貼，保留原樣」的區（#1175／#1176）：那一區自己寫為什麼（八區的牆上，回話那一行常在畫面外）。
// #1265 單元 3 之二第二趟起寫在第 2 段右欄那一區底下（wall_section.js），由「這間房貼什麼」小框套用時記（zone_overview.js 的 applyMethod）。
// 「牆/區」→ 選的那一筆的名字；只記在這一頁，那一區再改做法、選別的、重切或換方案就拿掉。
const zoneSameAsOriginal = new Map();
const ZONE_NONE = "none";
const WOOL_48K = "glass_wool_48k";
// 改成由程式最佳化時一開始放的就是這三種內建玻璃棉＋不貼（#1176，跟老闆講好的起點，不隨材料庫多了幾筆而變）。
const THREE_WOOLS = [WOOL_48K, "glass_wool_24k", "glass_wool_96k"];
// 做法名稱六面同一句（#1176 用語表）：天花、地板沒有「原牆」，「不貼」那一種叫「保留原樣」。
const ZONE_MODE_WORDS = {fixed: "指定貼一種", keep_original: "不貼，保留原樣", search: "由程式最佳化"};
const KEEP_WORDS = ZONE_MODE_WORDS.keep_original;
// 這一區原本的面怎麼叫（#1176 用語表）：牆面「原牆」、地板「原本的地板」、天花「原本的天花」；這一面本身叫牆面／地板／天花。
function originalFace(wall) { return {floor: "原本的地板", ceiling: "原本的天花"}[wall] ?? "原牆"; }
function faceWord(wall) { return {floor: "地板", ceiling: "天花"}[wall] ?? "牆面"; }

// 設計設定裡有沒有東西：每一區的做法、原方案、深度上限（房間預設與每面牆，#1111）任一樣有寫就算。
function designHasContent(design) {
  return Object.keys(design.zones || {}).length > 0 || design.original_scheme_id != null
    || design.room_max_depth_m != null || Object.keys(design.wall_max_depth_m || {}).length > 0
    || design.left_right_symmetry != null;  // 左右對稱關著（#1206；開著不寫這一格）
}
function freshZoneDesign() {
  return {schema_version: "aosr.zone_design.v1", scheme_id: scheme.scheme_id, zones: {}};
}
function designForEdit() {
  if (!zoneDesign) zoneDesign = freshZoneDesign();
  zoneDesign.zones ??= {};
  return zoneDesign;
}
// 區代號：矩形用自己的代號、整面一區用保留字（同伺服器 zone_design.zone_keys）。
function zoneKey(zone) { return zone.area === "rectangle" ? zone.id : "whole_wall"; }
function zoneList(wall) {
  return scheme.scene.surface_zones_by_wall?.[wall] ?? [{area: "whole_wall", material: wallMaterial(wall)}];
}
function zoneSetting(wall, key) { return zoneDesign?.zones?.[wall]?.[key] ?? {mode: "fixed"}; }
// 「鋼板直接貼在棉上」的構造（#1208）：哪幾筆是、畫面上那幾句字，都是伺服器給的（/api/materials 的 reference_only：
// {ids, sentence, why, search_why, listed_why, map_words, band_hz…}；材料庫裡一筆都沒有就沒有這一格，這裡是 null）。
// 這一類不進程式最佳化：一區的清單不准「新加」這種代號，本來就存著的照樣留著、可以拿掉（伺服器開跑時會點名拒收）。
let zoneReference = null;
function referenceOnly(id) { return Boolean(zoneReference?.ids.includes(id)) && libraryEntry(id)?.kind !== "construction"; }
// 由程式最佳化的清單裡程式真的會挑的那幾種（#1208：「鋼板直接貼在棉上」那一類不算）；「這間房貼什麼」小框「從 N 種裡選」
// 與第 2 段右欄「從 N 種裡挑」同一支（#1265 單元 3 審查設計洞 2）。
function pickableChoices(list) { return list.filter(id => !referenceOnly(id)); }
// 一份新清單裡這一類的代號，只留那一區本來就存著的（kept）。每一條會寫清單的路都過這一支。
function keepReferenceOut(next, kept) { return next.filter(id => !referenceOnly(id) || (kept ?? []).includes(id)); }
// 一區現在的材料是不是這一類：貼上去的那一筆，或原本的面本身（原牆就是材料庫裡那一筆）。材料庫認不到的層管不到。
function zoneReferenceOnly(material, wall, key) {
  return [libraryLabel(material, wall, key), zoneLibraryMatch(originalMaterial(material), wall, key)].some(entry => entry && zoneReference?.ids.includes(entry.id));
}
// 伺服器說開跑會拒收的那幾句（限制檢查順便回的 reference_only：一組一句 {id, zones: [{wall, zone}…], text}，字與順序都是
// 伺服器排的，這裡不接字）。一句是不是還「活著」只有這一支判斷（清單卡、頁底、兩顆「拿去搜尋」都只吃這裡回的）：
// 它點名的區，恰好等於現在由程式最佳化、清單去重排序後是同一份的所有區（方案裡有的），而且那份清單還有那一筆。
// （不是由程式最佳化的區沒有清單，自己一堆、對不上任何一句。）
// 剛改過清單（取消那一筆、拆開、併進來）、新的檢查結果還沒回來時對不上，整句先不顯示，等新的結果回來再畫。
function referenceRefusals() {
  const groups = new Map();
  for (const [wall, zones] of Object.entries(zoneDesign?.zones || {})) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const [key, setting] of Object.entries(zones)) {
      if (!keys.has(key)) continue;
      const list = [...new Set(setting.allowed)].sort().join("\n");
      groups.set(list, [...(groups.get(list) ?? []), `${wall}/${key}`]);
    }
  }
  const alive = new Set([...groups].flatMap(([list, names]) => list.split("\n").map(id => `${id}\n${names.sort().join("\n")}`)));
  return (zoneLimits?.reference_only ?? []).filter(line => alive.has(`${line.id}\n${line.zones.map(({wall, zone}) => `${wall}/${zone}`).sort().join("\n")}`));
}
// 一區要寫進去的那一份，照存檔規矩收好（每一條寫做法的路都過這一支；kept 是這一區原本的清單）：
// - 最後一道（#1208）：新清單不准多出本來沒有的「鋼板直接貼在棉上」的構造；拿掉之後空了就只放不貼。
// - 這一區自己的可調範圍（#1199，ranges：材料代號 → {porous_m, cavity_m}）只給由程式最佳化的區、只給清單裡有的那幾種：
//   做法換掉、清單拿掉那一種，那一份跟著拿掉；一種都不剩就整格不寫（伺服器不收空的，存檔也回到沒寫範圍的樣子）。
function settledSetting(setting, kept) {
  if (setting.mode !== "fixed" && "picked_entry" in setting) {
    setting = {...setting};
    delete setting.picked_entry;
  }
  if (setting.allowed) {
    const allowed = keepReferenceOut(setting.allowed, kept);
    if (allowed.length !== setting.allowed.length) setting = {...setting, allowed: allowed.length ? allowed : [ZONE_NONE]};
  }
  if (setting.ranges) {
    const ranges = Object.entries(setting.ranges).filter(([id]) => setting.mode === "search" && (!setting.allowed || setting.allowed.includes(id)));
    if (ranges.length !== Object.keys(setting.ranges).length) {
      setting = {...setting};
      if (ranges.length) setting.ranges = Object.fromEntries(ranges); else delete setting.ranges;
    }
  }
  return setting;
}
function setZoneSetting(wall, key, setting) {
  const design = designForEdit();
  design.zones[wall] ??= {};
  design.zones[wall][key] = settledSetting(setting, design.zones[wall][key]?.allowed);
}
function zoneNote(text, kind) {
  zoneNoteText = text; zoneNoteKind = kind; zoneSearchSaid = null;
  // 六面一起看那一頁（zone_overview.js）結果列上面那一行。輸入頁舊做法表底下那一行跟著表拆了（#1265 單元 3 之二第二趟）：
  // 那幾個動作只有舊表按得到；輸入頁還按得到、會回話的是頁底「交給程式最佳化 ▸」，回話走 searchNote。
  const line = $("zone-page-note");
  if (line) { line.textContent = text; line.className = kind; }
}
// 「拿去搜尋」的回話：那一頁照舊（zoneNote），輸入頁寫在頁底那顆鈕底下（renderSearchAction 讀 zoneSearchSaid）。
function searchNote(text, kind) {
  zoneNote(text, kind);
  zoneSearchSaid = {text, kind};
  renderSearchAction();
}

// 開舊方案：讀它的設計設定（沒有就是全部固定）；讀不到照講、不當成沒有。
// 先讀、後套：讀的時候不碰頁面狀態，打開方案時跟方案一起在鎖頁期間讀完，成功才一口氣套上（中間沒有等待點）。
async function fetchZoneDesign(name) {
  try {
    const data = await api(`/api/designs/${encodeURIComponent(name)}`);
    const stored = designHasContent(data.design);
    const note = data.problems.length ? `設計設定有幾條跟方案對不上（下次儲存時拿掉）：${data.problems.join("；")}` : "";
    return {design: data.design, stored, broken: false, note};
  } catch (error) {
    return {design: null, stored: false, broken: true,
            note: `每一區的做法讀不到（${errorText(error)}）：這一頁改的做法不會存回去，免得蓋掉原本存的；先處理好再重新打開方案`};
  }
}
function applyZoneDesign(state) {
  zoneDesignGeneration += 1;
  zoneDesign = state.design; zoneDesignStored = state.stored; zoneDesignBroken = state.broken;
  zoneDesignNote = state.note; zoneNote("", ""); zoneLimits = null; zoneSameAsOriginal.clear();
  zoneLastAllowed = null;  // 「上一次的組合」不跨方案（審查 11-15）
}
// 載入範本：新的方案，沒有設計設定。
function resetZoneDesign() {
  zoneDesignGeneration += 1; zoneDesign = null; zoneDesignStored = false; zoneDesignBroken = false; zoneDesignNote = ""; zoneNote("", "");
  zoneLimits = null; zoneSameAsOriginal.clear(); zoneLastAllowed = null;
}
// 存方案之後接著存設計設定（整份換上）；對不上方案的區先拿掉（伺服器會整份拒收），原方案指到自己就清掉。
// 從來沒設過任何一區、伺服器上也沒有這一份，就不寫（舊方案存檔不多出一個檔）。
// 分兩步（#1187）：按下儲存那一刻先把要送的那一份做好（這裡，沒有等待點），方案存好之後才送（回傳的那一支）；
// 中間隔著好幾個等待點，送出去的仍是按下那一刻的做法，不是那時畫面上的。
function prepareZoneDesignSave(name) {
  zoneDesignSaved = false;
  if (zoneDesignBroken) return async () => { throw new Error("原本的設計設定讀不到，這次沒存每一區的做法（免得蓋掉原本存的）"); };
  const touched = zoneDesign && designHasContent(zoneDesign);
  // 沒內容、也沒存過：不寫檔。那一頁要是標過「改的還沒存」（例如左右對稱關了又開回來，#1206），這裡一樣要清掉那一句。
  if (!touched && !zoneDesignStored) return async () => { zoneDesignSaved = true; window.zonePageSaved?.(); };
  const design = structuredClone(designForEdit());
  design.scheme_id = name;
  for (const wall of Object.keys(design.zones)) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const key of Object.keys(design.zones[wall])) if (!keys.has(key)) delete design.zones[wall][key];
    if (!Object.keys(design.zones[wall]).length) delete design.zones[wall];
  }
  if (design.original_scheme_id === name) design.original_scheme_id = null;
  if (design.wall_max_depth_m && !Object.keys(design.wall_max_depth_m).length) delete design.wall_max_depth_m;
  // 由程式最佳化的區一律寫明確清單（#1176）：舊的沒寫清單（null）照伺服器的意思寫成全部可調材料＋不貼；材料庫沒讀全就照舊不寫。
  const everything = serverDefaultAllowed();
  for (const zones of Object.values(design.zones)) {
    for (const [key, setting] of Object.entries(zones)) {
      if (setting.mode === "search" && !setting.allowed && everything) zones[key] = {...setting, allowed: everything};
    }
  }
  const generation = zoneDesignGeneration;
  return async () => {
    await api(`/api/designs/${encodeURIComponent(name)}`, "PUT", design);
    zoneDesignSaved = true;
    window.zonePageSaved?.();  // 「這間房貼什麼」那一頁的「改的還沒存進檔案」那一句（zone_overview.js）
    if (generation === zoneDesignGeneration) { zoneDesign = design; zoneDesignStored = true; zoneDesignNote = ""; }
  };
}
// 各面交給搜尋的區有幾塊（只算現在切法裡還在的區）。
function searchZoneCount() {
  let count = 0;
  for (const [wall, zones] of Object.entries(zoneDesign?.zones || {})) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const [key, setting] of Object.entries(zones)) if (keys.has(key) && setting.mode === "search") count += 1;
  }
  return count;
}
// 由程式最佳化、清單只放了固定成品的區（表上紅框那幾區，只算現在切法裡還在的）：名字列出來，處理好之前不准拿去搜尋（#1176）。
function stuckZoneNames() {
  const names = [];
  for (const [wall, zones] of Object.entries(zoneDesign?.zones || {})) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const [key, setting] of Object.entries(zones)) {
      if (keys.has(key) && setting.mode === "search" && allowedProblem(setting.allowed)) names.push(zoneCalled(wall, key));
    }
  }
  return names;
}
function stuckWords(names) { return `先處理${names.join("、")}的紅框（只放了固定成品，程式沒得選），處理好才能拿去搜尋`; }
// 「拿去搜尋」：搜尋頁讀的是存好的方案與做法，先照畫面存一份；兩樣都存成才跳過去，沒存成留在這頁照講。
// 鎖（app.js 的 holdInput）從存檔一路拿到離開這一頁（#1188 第 5 條，同「計算」把重讀清單包在鎖裡）：存好之後還要重讀清單、
// 搜尋頁也要一點時間才回來，那一段要是點得到、改得動，改的東西沒存就被帶去搜尋頁，搜尋頁讀到的比畫面上少。
async function sendToSearch() {
  const cut = window.pendingMirrorCut?.();
  if (cut) await cut;
  const stuck = stuckZoneNames();
  if (stuck.length) { searchNote(stuckWords(stuck), "notice"); return; }
  // 房間尺寸調整還在路上（#1266）：先等它落地，底下等到的限制判決才是照新尺寸重問的那一次。
  const resize = window.pendingRoomResize?.();
  if (resize) await resize;
  const limits = await settledZoneLimits();
  if (limits?.status !== "checked") {
    searchNote(`限制沒檢查成（${limits?.message ?? "還沒檢查"}），先處理好再拿去搜尋`, "notice"); return;
  }
  if (limitsRedCount()) { searchNote(`有 ${limitsRedCount()} 條超過限制（標紅的區），改好之前不能拿去搜尋`, "notice"); return; }
  // 清單裡有「鋼板直接貼在棉上」的構造（#1208）：開跑一定被拒收，不存不跳，照伺服器那一句講。
  const refused = referenceRefusals();
  if (refused.length) { searchNote(refused.map(line => line.text).join("；"), "notice"); return; }
  if (inputBusy) {
    // 等限制檢查的時候又送了一批改動、還在路上：這一次不存不跳。那一批落地後平面圖檢查會換掉訊息列，
    // 「這間房貼什麼」那一頁另外留著這一句（zone_overview.js 的 zonePageKeep，#1188 第 6 條）。
    const refused = `${busyWords}，這一次沒有儲存；等它做完再按一次`;
    say(refused, "notice");
    window.zonePageKeep?.(refused);
    return;
  }
  await holdInput(async () => {
    const generation = zoneDesignGeneration, name = $("save-id").value;
    const stored = await storeScheme();
    if (!stored) return;
    await refreshAfterSave(stored.message);
    // 鎖著的時候換不了方案；這一條是鎖漏掉時的第二道（同 pressedCopy 的 samePage）。
    if (generation !== zoneDesignGeneration) { searchNote(`存檔途中換了方案：「${name}」照樣存好，這次不跳搜尋頁`, "notice"); return; }
    if (!zoneDesignSaved) { searchNote("每一區的做法沒存成，搜尋頁會讀到舊的做法：先處理好訊息列寫的問題再按一次", "notice"); return; }
    await leaveForSearch(`/gradient?scheme=${encodeURIComponent(scheme.scheme_id)}`);
  });
}
// 送出跳頁、等到離開這一頁（呼叫的人拿著鎖）：搜尋頁回來之前舊的這一頁還在畫面上。離開了這個等待就不會回來，鎖跟著頁面一起消失；
// 兩種情況會回來、把鎖放開：按「上一頁」時瀏覽器把這一頁原樣搬回來（pageshow），或等了一陣子還在這一頁（搜尋頁沒打開成；
// 伺服器關著時瀏覽器會換成錯誤頁、走不到這裡）。等多久是隨手定的，沒有量測依據：本機搜尋頁一般幾十毫秒就回來。
const LEAVE_WAIT_MS = 10000;
async function leaveForSearch(url) {
  const stayed = await new Promise(resolve => {
    window.addEventListener("pageshow", () => resolve(false), {once: true});
    setTimeout(() => resolve(true), LEAVE_WAIT_MS);
    window.location.href = url;
  });
  if (stayed) searchNote("方案與每一區的做法都存好了，但等了一陣子搜尋頁還沒打開：再按一次「拿去搜尋」", "notice");
}
function renderSearchAction() {
  const count = searchZoneCount(), button = $("zone-to-search"), note = $("zone-to-search-note");
  const red = limitsRedCount(), stuck = stuckZoneNames(), refused = referenceRefusals().map(line => line.text);
  const lockedEdit = finishedNames.has(openedId) && formHasEdits();
  button.disabled = !count || !!formRunning() || lockedEdit || red > 0 || stuck.length > 0 || refused.length > 0;
  const blocked = [...(stuck.length ? [stuckWords(stuck)] : []), ...refused, ...(red ? [`有 ${red} 條超過限制（標紅的區），改好之前不能拿去搜尋`] : [])];
  // 按了沒送出去的那一句（zoneSearchSaid）排在擋住的理由後面：擋住時那一句就是其中一條，照擋住的寫；沒擋才寫它，不被下一次重畫換掉。
  const said = formRunning() || lockedEdit || blocked.length ? null : zoneSearchSaid;
  note.className = blocked.length ? "limit-red" : said ? said.kind : "note";
  note.textContent = formRunning() ? "等這一次算完" : lockedEdit ? "先另存新名字" : blocked.length ? blocked.join("；") : said ? said.text :
    count ? `交出黃色的 ${count} 區；下一頁按「開始搜尋」才開始算` : "到「這間房貼什麼」設定要交給程式的區";

}
// 切完一面：自動補的代號每次重編，做法跟著 /api/zone-cut 回的 inherited（每一塊沿用了切之前哪一區的材料）走。
function remapZoneDesign(wall, inherited) {
  // 自動補的代號重編了：「改之前指定貼的那一筆」記的是舊代號，這一面的全部丟掉（不然會掛到別一塊上）。
  for (const name of [...zoneSameAsOriginal.keys()]) if (name.startsWith(`${wall}/`)) zoneSameAsOriginal.delete(name);
  const old = zoneDesign?.zones?.[wall];
  if (!old) return;
  const next = {};
  for (const [key, source] of Object.entries(inherited)) if (old[source]) next[key] = structuredClone(old[source]);
  if (Object.keys(next).length) zoneDesign.zones[wall] = next; else delete zoneDesign.zones[wall];
}

async function fetchZoneLibrary() {
  try {
    const data = await api("/api/materials");
    // 可調多層構造（現在只有可調 VPR，#1200 第 1 支）先不進分區頁：電腦挑它還沒接上，選單、勾選表、不列的那幾組都不出現。
    return {library: data.materials.filter(row => row.entry.kind !== "tunable_layers")
              .map(row => ({...row.entry, basis: row.basis, placement: row.placement})),
            groups: data.surface_groups, reference: data.reference_only ?? null,
            problem: data.user_file_problem ? `使用者材料庫讀不進來，選單只列內建材料：${data.user_file_problem}` : ""};
  } catch (error) {
    return {library: [], groups: {}, reference: null, problem: `材料庫讀不到：${errorText(error)}`};
  }
}
function applyZoneLibrary(state) {
  zoneLibrary = state.library; zoneSurfaceGroups = state.groups; zoneLibraryProblem = state.problem;
  zoneReference = state.reference ?? null;
}
async function loadZoneLibrary() { applyZoneLibrary(await fetchZoneLibrary()); }
function libraryEntries(kind) { return (zoneLibrary || []).filter(entry => entry.kind === kind); }
function assignableEntries() { return (zoneLibrary || []).filter(entry => entry.kind === "fixed" || entry.kind === "construction"); }
function fixedCandidate(entry) { return entry?.kind === "fixed" || (entry?.kind === "construction" && entry.backing !== "zone_wall"); }
function appendFixedCandidateNote(container, allowed) {
  if (!allowed.some(id => fixedCandidate(libraryEntry(id)))) return;
  const note = document.createElement("p");
  note.className = "zone-candidate-cost";
  note.textContent = "清單裡有整塊貼上去的材料。建替身時，每筆都要放進訓練，再多核對一次。"
    + "之後只要清單、深度上限、裝法或方案有改，就要重建替身。"
    + "存檔後，開跑前到第三階段搜尋頁按「預覽轉好的參數」，就能看到是否需要重建及原因。";
  container.append(note);
}
function fixedSurface(entry) {
  if (entry.kind === "construction") {
    return entry.backing === "zone_wall" ? null : {kind: "assembly", layers: entry.layers, backing: entry.backing};
  }
  return entry.kind === "fixed" ? entry.material : null;
}
function fixedLayers(entry) { return entry.kind === "construction" ? entry.layers : fixedSurface(entry)?.layers ?? []; }
function libraryEntry(id) { return (zoneLibrary || []).find(entry => entry.id === id) ?? null; }
// 這一筆列不列在這一面（伺服器每一筆都帶分組，跟網頁一起出貨）。
function usableHere(entry, wall) { return entry.placement.surfaces.includes(wall); }
// 不列在這一面的，照組排（組的順序同伺服器標題表）：[[組的代號, [材料…]]…]。walls 是一面或幾面（「這間房貼什麼」小框一次選
// 好幾面，#1265）：小框只列每一面都能貼的，所以有任何一面不能貼就算不列的（照 usableHere 反過來列）。
function hiddenGroups(walls, entries) {
  const faces = [walls].flat();
  const groups = new Map(Object.keys(zoneSurfaceGroups).map(group => [group, []]));
  for (const entry of entries) if (!faces.every(wall => usableHere(entry, wall))) groups.get(entry.placement.group).push(entry);
  return [...groups].filter(([, items]) => items.length);
}
// 可調材料的顯示名拿掉尾巴的「（可調厚度與空腔）」：能調什麼另外照條目寫（24K、96K 空腔其實不調）。
function shortLabel(entry) { return entry.kind === "optimizable" ? entry.label.replace(/（可調[^）]*）$/, "") : entry.label; }
function centimetres(metres) { return shownNumber(Math.round(metres * 1000) / 10); }
// 可調的到底調什麼（#1176）：照條目的多孔層厚度與空腔寫；空腔起點沒有（null 或 0）搜尋就沒有空腔這一格，寫「空腔：沒有（直接貼牆）」（#1199 用語表：「空腔不調」被讀成棉厚也不調）。
// 材料自己的總深上限只在真的會擋時寫：比厚度上限加空腔上限（或固定空腔）還淺才寫「加起來最深」（審查 11-15：24K 空腔不調、
// 最深就是 20 cm，寫 30 cm 是錯的）。
function adjustWords(entry) {
  const thick = entry.porous_thickness, cavity = entry.cavity;
  const thickTuned = thick.adjustable && thick.range_m[1] > thick.range_m[0];
  const thickText = thickTuned
    ? `厚 ${centimetres(thick.range_m[0])}～${centimetres(thick.range_m[1])} cm` : `厚度不調（${centimetres(entry.porous.thickness_m)} cm）`;
  const noCavity = !cavity || !cavity.initial_m;
  const cavityTuned = !noCavity && cavity.adjustable && cavity.range_m[1] > cavity.range_m[0];
  const cavityText = noCavity ? "空腔：沒有（直接貼牆）"
    : cavityTuned ? `空腔 ${centimetres(cavity.range_m[0])}～${centimetres(cavity.range_m[1])} cm` : `空腔不調（${centimetres(cavity.initial_m)} cm）`;
  const deepest = (thickTuned ? thick.range_m[1] : entry.porous.thickness_m) + (noCavity ? 0 : cavityTuned ? cavity.range_m[1] : cavity.initial_m);
  const cap = entry.max_total_depth_m;
  const depth = cap && cap < deepest - 1e-9 ? `，加起來最深 ${centimetres(cap)} cm` : "";
  return `可調：${thickText}、${cavityText}${depth}`;
}
// 這一區現在的材料是材料庫哪一筆：多層構造比貼上去的那幾層，固定阻抗與實測表比整份。
// 裸構造只讀這一面、這一區的伺服器判決。上一份草稿的判決不能套到新位置。
function constructionAt(wall, key) {
  return zoneLimitsReceived === zoneLimitsGeneration ? zoneLimits?.picked_constructions?.[wall]?.[key] : null;
}
function uncertainWords(verdict) { return `看不出：跟材料庫「${verdict.label}」一樣`; }
function libraryLabel(material, wall, key) {
  const verdict = constructionAt(wall, key);
  if (material.kind === "assembly") return verdict?.state === "picked" ? libraryEntry(verdict.id) : null;
  for (const entry of assignableEntries()) {
    const chosen = fixedSurface(entry);
    if (material.kind === "mounted_on_wall") {
      if ((entry.kind === "fixed" || entry.backing === "zone_wall") && sameMaterial(fixedLayers(entry), material.layers)) return entry;
    } else if (chosen?.kind === "measured_impedance" && material.kind === "measured_impedance") {
      // 實測表進物理時，原本空白的虛部補成 0。
      const points = chosen.points.map(point => ({...point, imag_pa_s_m: point.imag_pa_s_m ?? 0}));
      if (sameMaterial({...chosen, points}, material)) return entry;
    } else if (entry.kind === "fixed" && chosen?.kind !== "assembly" && sameMaterial(chosen, material)) {
      return entry;
    }
  }
  return null;
}
function materialWords(material, wall, key) {
  const entry = libraryLabel(material, wall, key);
  if (entry) return `${entry.label}（${material.kind === "mounted_on_wall" ? `貼在${originalFace(wall)}上` : `直接取代${faceWord(wall)}`}）`;
  return material.kind === "mounted_on_wall" ? `${wallText(material)}（${originalFace(wall)}：${wallText(material.wall)}）` : wallText(material);
}
// 搜尋存成的方案裡那幾層（#1128）：每層「種類 厚度」，多孔層加流阻（兩種棉才分得出來）；名字來自伺服器的模型描述。
function layersText(material) {
  const spec = inputDefaults.material_form;
  return material.layers.map(layer => {
    const kind = spec[layer.kind]?.label ?? layer.kind;
    const thick = typeof layer.thickness_m === "number" ? ` ${shownNumber(layer.thickness_m * 1000)} mm` : "";
    const flow = typeof layer.flow_resistivity_pa_s_m2 === "number" ? `（流阻 ${shownNumber(layer.flow_resistivity_pa_s_m2)} Pa·s/m²）` : "";
    return `${kind}${thick}${flow}`;
  }).join("＋");
}
function nowWords(material, wall, key) {
  if (material.kind !== "mounted_on_wall" || libraryLabel(material, wall, key)) return materialWords(material, wall, key);
  return `${layersText(material)}，貼在${originalFace(wall)}上`;
}
function originalMaterial(material) { return material.kind === "mounted_on_wall" ? material.wall : material; }
// 這一區現在的材料對到材料庫哪一筆：貼上去的（libraryLabel），或這一區本身就是材料庫某一筆多層構造（例如天花那一套吊頂、
// 門、窗這種整塊就是那一筆、不是貼上去的）。區名認門窗地毯（zone_names.js）與原牆的名字都用它。
// 「指定貼一種」選單上這一區選的那一筆照舊只認 libraryLabel（整塊就是門的那一區沒貼東西，選單寫「現在：沒貼東西」）。
function zoneLibraryMatch(material, wall, key) {
  if (!material) return null;
  if (constructionAt(wall, key)?.state === "unknown") return null;
  return libraryLabel(material, wall, key) ?? libraryEntries("fixed").find(item => material.kind === "assembly" && sameMaterial(fixedSurface(item), material)) ?? null;
}
// 白話（#1176）：原牆不寫阻抗數字（數字收在點開的「數字」細節）；材料庫認得的寫它的名字。
function originalWords(material, wall, key) {
  const verdict = constructionAt(wall, key);
  if (verdict?.state === "unknown") return zoneSetting(wall, key).mode === "keep_original" ? KEEP_WORDS : uncertainWords(verdict);
  const label = zoneLibraryMatch(material, wall, key)?.label ?? (verdict?.state === "original" ? verdict.label : null);
  return label ? `${originalFace(wall)}「${label}」原樣` : `${originalFace(wall)}原樣`;
}
function plainNow(material, wall, key) {
  const verdict = constructionAt(wall, key);
  if (verdict?.state === "unknown") return zoneSetting(wall, key).mode === "keep_original" ? KEEP_WORDS : uncertainWords(verdict);
  if (verdict?.state === "picked") return materialWords(material, wall, key);
  return material.kind === "mounted_on_wall" ? `已貼 ${nowWords(material, wall, key)}` : `沒貼東西（${originalWords(material, wall, key)}）`;
}
// 固定成品的總厚（每層有厚度的加起來，所以寫「約」）：薄板只給面密度、沒有厚度，有薄板就註明不含（不然跟顯示名裡
// 「總厚約 14 cm」這種把鋼板算進去的數字打架）；不是多層構造的（固定阻抗、實測表）沒有厚度可寫，回空字串。
function fixedDepthWords(entry) {
  const layers = fixedLayers(entry);
  const depth = layers.reduce((sum, layer) => sum + (typeof layer.thickness_m === "number" ? layer.thickness_m : 0), 0);
  if (!(depth > 0)) return "";
  return `總厚約 ${centimetres(depth)} cm${layers.some(layer => layer.kind === "board") ? "（不含薄板）" : ""}`;
}

async function afterZoneChange() {
  staleZoneLimits();  // 先畫後問：上一版判決對的是改之前的做法（#1265 單元 3 之二第二趟追加第 1、3、4 條）
  renderWalls();
  markStale();
  action(updateMultiples);
  action(updateInputPreview);
  await refreshPlan();
}
// 訊息與選單旁白裡叫一區（#1176）：牆名、號碼、位置名，例如「前牆 ④「左聲道喇叭後面那塊」」（跟立面圖與表上同一個號碼）。
function zoneCalled(wall, key) {
  if (key === "whole_wall") return `${wallNames[wall]}「整面」`;  // 一面只有一區，不必號碼
  const label = zoneLabels(wall).get(key);
  return `${wallNames[wall]} ${label.numberText}「${label.name}」`;
}

// 選的跟這一區原牆一模一樣（#1175）那一句：「這間房貼什麼」小框套用的回話（#1265 單元 3 之二第 E 條）。
function sameAsOriginalWords(wall, key) { return `${zoneCalled(wall, key)}：選的跟原本一樣，當成不貼（做法改成「${KEEP_WORDS}」）`; }
// 伺服器沒寫清單（null）的意思：材料庫全部可調材料＋不貼。材料庫沒讀全（讀不到或使用者檔壞了）就回 null，不替伺服器縮小範圍。
function serverDefaultAllowed() {
  if (!zoneLibrary?.length || zoneLibraryProblem) return null;
  return [...libraryEntries("optimizable").map(entry => entry.id), ZONE_NONE];
}
function fitsHere(id, wall) {
  const entry = libraryEntry(id);
  return id === ZONE_NONE || (entry !== null && entry.backing !== "zone_wall" && usableHere(entry, wall));
}
// 改成由程式最佳化時一開始放的清單：這一頁上一次改過的組合優先（這一面用不了的拿掉；拿完沒有起點就不用它），
// 不然一律是三種內建玻璃棉＋不貼（材料庫沒有的、不是這一面用的拿掉，地板只剩不貼）；使用者自己加的可調材料要用就在分區頁小框加。
// 材料庫沒讀全就回 null（照舊不寫清單、照講；使用者材料庫壞了的時候伺服器本來就不准換做法，輪不到這裡）。
function startingAllowed(wall) {
  if (!serverDefaultAllowed()) return null;
  const last = carriedList(zoneLastAllowed ?? [], id => fitsHere(id, wall));
  if (last?.length && !allowedProblem(last)) return last;
  return defaultAllowed(wall);
}
function defaultAllowed(wall) { return [...THREE_WOOLS.filter(id => fitsHere(id, wall)), ZONE_NONE]; }
// 別區的一份清單帶到新的一區（#1208）：「鋼板直接貼在棉上」的構造不帶（別區本來就存著的那一筆，不准跟著帶進新的一區）。
// 拿掉它之後只剩「不貼」或什麼都不剩，回 null（呼叫的人改用預設的三種玻璃棉＋不貼）：那一份清單本來是「它或不貼」，
// 帶過來變成「只有不貼」等於程式什麼都不做，不是使用者選的。來源本來就沒有它的照舊（本來就只有「不貼」的也照舊）。
function carriedList(list, keeps) {
  const kept = list.filter(id => !referenceOnly(id) && keeps(id));
  return list.some(referenceOnly) && !kept.some(id => id !== ZONE_NONE) ? null : kept;
}
function choiceName(id) {
  const entry = libraryEntry(id);
  return id === ZONE_NONE ? "不貼" : entry ? shortLabel(entry) : `${id}（材料庫找不到）`;
}
// 清單裡每一種都是材料庫認得的固定成品（沒有可調材料、沒有不貼）：程式沒得選（伺服器開跑時 gradient_plan 同一句白話再擋一次）。
// 材料庫找不到的代號不算固定成品（材料庫沒讀全時可能是可調材料；那一種照「材料庫找不到」那一行講，開跑時伺服器點名）。
function allowedProblem(allowed) {
  if (!allowed?.length) return "";
  return allowed.every(id => fixedCandidate(libraryEntry(id))) ? "只放了固定成品，程式沒得選。" : "";
}

// 「顯示全部材料」那幾組的組標題、開關的字、「不是這一面用的」那一句：「這間房貼什麼」小框（#1265）用。
function hiddenGroupLabel(group) { return `不列的：${zoneSurfaceGroups[group]}`; }
const SHOW_ALL_WORDS = "顯示全部材料（門、窗、整片天花構造、另一間示範房的這些平常不列；要選它們才打開）";
function awayWords(entry, wall) { return `「${entry.label}」不是${wallNames[wall]}用的：${entry.placement.reason}`; }
function sameList(left, right) { return left.length === right.length && left.every(id => right.includes(id)); }
function originalSchemeRow() {
  const row = document.createElement("div"); row.className = "fields";
  const wrap = document.createElement("label"); wrap.textContent = "改造前用哪一份方案（算「改善多少」時每一區原本的樣子；也給拆不出原樣的區用）";
  const select = document.createElement("select"); select.id = "original-scheme";
  const none = document.createElement("option"); none.value = ""; none.textContent = "沒指定：每一面都只有原本的牆（拆掉貼上去的層）"; select.append(none);
  const self = $("save-id").value;
  for (const name of schemeNames) {
    if (name === self) continue;
    const item = document.createElement("option"); item.value = name; item.textContent = name; select.append(item);
  }
  const chosen = zoneDesign?.original_scheme_id ?? "";
  if (chosen && ![...select.options].some(option => option.value === chosen)) {
    const missing = document.createElement("option"); missing.value = chosen; missing.textContent = `${chosen}（不在清單裡）`; select.append(missing);
  }
  select.value = chosen;
  select.onchange = () => { designForEdit().original_scheme_id = select.value || null; markStale(); refreshZoneLimits(); };
  wrap.append(select); row.append(wrap);
  return row;
}
// 「改這一區 ▸」（#1258 2b）：回「這間房貼什麼」打開那一區的小框。#1265 單元 3 起住在第 2 段右欄唯讀清單
// （wall_section.js）；一區一顆、整頁只有一處。
function zoneGotoButton(wall, key) {
  const go = document.createElement("button"); go.type = "button"; go.id = `zone-goto-${wall}-${key}`; go.className = "lnk";
  go.textContent = "改這一區 ▸"; go.setAttribute("aria-label", `到「這間房貼什麼」改${zoneCalled(wall, key)}`);
  go.onclick = () => window.openZoneOverview(`${wall}/${key}`);
  return go;
}
// 保留原樣的區寫什麼（第 2 段右欄清單那一句）：原本的面叫什麼、不貼東西。
function keepWords(zone, wall, key) {
  const words = originalWords(originalMaterial(zone.material), wall, key);
  return words === KEEP_WORDS ? words : `${words}，不貼東西`;
}
// 設計設定讀不到、材料庫有問題那兩句：第 2 段右欄清單上方（wall_section.js，#1265 單元 3 之二第 D 條）。
function zoneProblemLines() {
  return [zoneDesignNote, zoneLibraryProblem].filter(Boolean).map(text => {
    const line = document.createElement("p"); line.className = "notice"; line.textContent = text;
    return line;
  });
}
// 輸入頁跟方案一起重畫的那一條路（名字是舊做法表留下的；#1265 單元 3 之二第二趟起舊做法表拆了，表上每一樣都有新家）：
// 開檔、切牆、換材料、改做法都會走到這裡。
function renderZoneTable(wall) {
  // 六面一起看的那一頁（zone_overview.js）跟著重畫。
  window.renderZoneOverview?.();
  // 原方案與總厚度上限（房間預設）兩格在第 1 段右卡（#1265）。
  renderRoomEditors();
  // 第 2 段牆那一行、右欄唯讀清單（上方是限制總結、整面判決、兩句錯誤提示）、「每一區的位置與大小」（#1265 單元 3）；
  // 這一面的總厚度上限那一格在牆那一行「這一面另設」打開的那一塊。
  renderWallSection(wall);
  refreshClearUndo();
  // 清掉那一塊之前拿；擁有者是第幾次打開＋哪一面（同 room_facts.js 房間預設那一句，#1265 單元 3 審查真錯 4）。
  const depthSaid = wall && scheme ? depthNote("depth-wall-note", `${zoneDesignGeneration}/${wall}`) : null;
  $("wall-depth-editor").replaceChildren();
  if (!wall || !scheme) return;
  $("wall-depth-editor").append(depthRow(wall, depthSaid));
  renderSearchAction();
}
window.renderZoneTable = renderZoneTable;
