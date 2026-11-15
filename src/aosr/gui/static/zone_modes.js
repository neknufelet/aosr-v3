/* 每一區的做法（#1105，#1082 第 1-3 段）：立面圖下面一張表，這一面每一區（畫的、自動補的、整面一區）選
   「指定貼一種」「不貼，保留原牆」「交給電腦挑」（設計檔裡叫 fixed／keep_original／search）；交給電腦挑的區一律列出
   一份明確清單（電腦可以從哪幾種裡挑，存檔也寫明確的 allowed），改成交給電腦挑時直接帶好一組，不用先全勾再取消。材料選單只列這一面能貼的（#1176，分組由伺服器 /api/materials 的 placement 現算），其餘收在
   最下面「顯示全部材料」裡分組列出、各一句理由；已經存在某一區的代號不管分到哪一組都照樣顯示。
   做法存在設計設定（/api/designs，跟方案同名、跟著方案存）；方案那一區的材料由伺服器換（/api/zone-material）：
   固定材料貼在原牆上、不貼與交給搜尋放回原牆。材料選單讀材料庫（/api/materials），不另做一份。
   設計設定整份原樣帶著走（原方案、之後的段加的欄位都不丟），只改這張表管的那幾格。 */
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
// 每一組材料的標題（伺服器 material_surfaces.GROUP_WORDS）；「顯示全部材料」打開才列不是這一面用的那幾組，只記在這一頁。
let zoneSurfaceGroups = {};
let zoneShowAll = false;
let zoneNoteText = "";
let zoneNoteKind = "";
const zoneAllowedOpen = new Set();
// 交給電腦挑的清單（#1176）：這一頁上一次改過的組合（改成交給電腦挑時優先放它）、每一區改成交給電腦挑之前指定貼的那一筆
// （「只讓電腦決定：這一筆貼或不貼」用）。兩樣都只記在這一頁，不跨重開。
let zoneLastAllowed = null;
const zoneFormerFixed = new Map();
const ZONE_NONE = "none";
const WOOL_48K = "glass_wool_48k";
// 改成交給電腦挑時一開始放的就是這三種內建玻璃棉＋不貼（#1176，跟老闆講好的起點，不隨材料庫多了幾筆而變）。
const THREE_WOOLS = [WOOL_48K, "glass_wool_24k", "glass_wool_96k"];
// 一鍵常用組合：只用內建代號；材料庫裡沒有、或不是這一面用的，那一顆就不出現。
const ZONE_PRESETS = [
  {id: "three_wools", words: "三種玻璃棉都比＋可以不貼", members: [...THREE_WOOLS, ZONE_NONE]},
  {id: "wool48", words: "48K 玻璃棉＋可以不貼", members: [WOOL_48K, ZONE_NONE]},
];
const ZONE_MODE_WORDS = {fixed: "指定貼一種", keep_original: "不貼，保留原牆", search: "交給電腦挑"};

// 設計設定裡有沒有東西：每一區的做法、原方案、深度上限（房間預設與每面牆，#1111）任一樣有寫就算。
function designHasContent(design) {
  return Object.keys(design.zones || {}).length > 0 || design.original_scheme_id != null
    || design.room_max_depth_m != null || Object.keys(design.wall_max_depth_m || {}).length > 0;
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
function setZoneSetting(wall, key, setting) {
  const design = designForEdit();
  design.zones[wall] ??= {};
  design.zones[wall][key] = setting;
}
function zoneNote(text, kind) { zoneNoteText = text; zoneNoteKind = kind; const line = $("zone-note"); if (line) { line.textContent = text; line.className = kind; } }

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
  zoneDesignNote = state.note; zoneNote("", ""); zoneLimits = null; zoneFormerFixed.clear();
}
// 載入範本：新的方案，沒有設計設定。
function resetZoneDesign() { zoneDesignGeneration += 1; zoneDesign = null; zoneDesignStored = false; zoneDesignBroken = false; zoneDesignNote = ""; zoneNote("", ""); zoneLimits = null; zoneFormerFixed.clear(); }
// 存方案之後接著存設計設定（整份換上）；對不上方案的區先拿掉（伺服器會整份拒收），原方案指到自己就清掉。
// 從來沒設過任何一區、伺服器上也沒有這一份，就不寫（舊方案存檔不多出一個檔）。
async function saveZoneDesign(name) {
  zoneDesignSaved = false;
  if (zoneDesignBroken) throw new Error("原本的設計設定讀不到，這次沒存每一區的做法（免得蓋掉原本存的）");
  const touched = zoneDesign && designHasContent(zoneDesign);
  if (!touched && !zoneDesignStored) { zoneDesignSaved = true; return; }
  const design = structuredClone(designForEdit());
  design.scheme_id = name;
  for (const wall of Object.keys(design.zones)) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const key of Object.keys(design.zones[wall])) if (!keys.has(key)) delete design.zones[wall][key];
    if (!Object.keys(design.zones[wall]).length) delete design.zones[wall];
  }
  if (design.original_scheme_id === name) design.original_scheme_id = null;
  if (design.wall_max_depth_m && !Object.keys(design.wall_max_depth_m).length) delete design.wall_max_depth_m;
  // 交給電腦挑的區一律寫明確清單（#1176）：舊的沒寫清單（null）照伺服器的意思寫成全部可調材料＋不貼；材料庫沒讀全就照舊不寫。
  const everything = serverDefaultAllowed();
  for (const zones of Object.values(design.zones)) {
    for (const [key, setting] of Object.entries(zones)) {
      if (setting.mode === "search" && !setting.allowed && everything) zones[key] = {...setting, allowed: everything};
    }
  }
  const generation = zoneDesignGeneration;
  await api(`/api/designs/${encodeURIComponent(name)}`, "PUT", design);
  zoneDesignSaved = true;
  if (generation === zoneDesignGeneration) { zoneDesign = design; zoneDesignStored = true; zoneDesignNote = ""; }
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
// 交給電腦挑、清單只放了固定成品的區（表上紅框那幾區，只算現在切法裡還在的）：名字列出來，處理好之前不准拿去搜尋（#1176）。
function stuckZoneNames() {
  const names = [];
  for (const [wall, zones] of Object.entries(zoneDesign?.zones || {})) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const [key, setting] of Object.entries(zones)) {
      if (keys.has(key) && setting.mode === "search" && allowedProblem(setting.allowed)) names.push(`${wallNames[wall]}「${zoneName(key)}」`);
    }
  }
  return names;
}
function stuckWords(names) { return `先處理${names.join("、")}的紅框（只放了固定成品，電腦沒得挑），處理好才能拿去搜尋`; }
// 「拿去搜尋」：搜尋頁讀的是存好的方案與做法，先照畫面存一份；兩樣都存成才跳過去，沒存成留在這頁照講。
async function sendToSearch() {
  const stuck = stuckZoneNames();
  if (stuck.length) { zoneNote(stuckWords(stuck), "notice"); return; }
  const limits = await settledZoneLimits();
  if (limits?.status !== "checked") {
    zoneNote(`限制沒檢查成（${limits?.message ?? "還沒檢查"}），先處理好再拿去搜尋`, "notice"); return;
  }
  if (limitsRedCount()) { zoneNote(`有 ${limitsRedCount()} 條超過限制（標紅的區），改好之前不能拿去搜尋`, "notice"); return; }
  const generation = zoneDesignGeneration, name = $("save-id").value;
  if (!await save()) return;
  if (generation !== zoneDesignGeneration) { zoneNote(`存檔途中換了方案：「${name}」照樣存好，這次不跳搜尋頁`, "notice"); return; }
  if (!zoneDesignSaved) { zoneNote("每一區的做法沒存成，搜尋頁會讀到舊的做法：先處理好訊息列寫的問題再按一次", "notice"); return; }
  window.location.href = `/gradient?scheme=${encodeURIComponent(scheme.scheme_id)}`;
}
function searchButtonRow() {
  const count = searchZoneCount();
  if (!count) return null;
  const row = document.createElement("p"); row.className = "fields";
  const button = document.createElement("button"); button.type = "button"; button.id = "zone-to-search";
  button.textContent = `拿去搜尋（${count} 區交給電腦挑）`;
  button.onclick = () => action(sendToSearch);
  const red = limitsRedCount(), stuck = stuckZoneNames();
  button.disabled = red > 0 || stuck.length > 0;
  const note = document.createElement("span"); note.id = "zone-to-search-note";
  const blocked = [...(stuck.length ? [stuckWords(stuck)] : []), ...(red ? [`有 ${red} 條超過限制（標紅的區），改好之前不能拿去搜尋`] : [])];
  note.className = blocked.length ? "limit-red" : "note";
  note.textContent = blocked.length ? blocked.join("；") : "會先照畫面儲存，再打開第三階段搜尋頁";
  row.append(button, " ", note);
  return row;
}
// 切完一面：自動補的代號每次重編，做法跟著 /api/zone-cut 回的 inherited（每一塊沿用了切之前哪一區的材料）走。
function remapZoneDesign(wall, inherited) {
  // 自動補的代號重編了：「改之前指定貼的那一筆」記的是舊代號，這一面的全部丟掉（不然會掛到別一塊上）。
  for (const name of [...zoneFormerFixed.keys()]) if (name.startsWith(`${wall}/`)) zoneFormerFixed.delete(name);
  const old = zoneDesign?.zones?.[wall];
  if (!old) return;
  const next = {};
  for (const [key, source] of Object.entries(inherited)) if (old[source]) next[key] = structuredClone(old[source]);
  if (Object.keys(next).length) zoneDesign.zones[wall] = next; else delete zoneDesign.zones[wall];
}

async function fetchZoneLibrary() {
  try {
    const data = await api("/api/materials");
    return {library: data.materials.map(row => ({...row.entry, basis: row.basis, placement: row.placement})),
            groups: data.surface_groups ?? {},
            problem: data.user_file_problem ? `使用者材料庫讀不進來，選單只列內建材料：${data.user_file_problem}` : ""};
  } catch (error) {
    return {library: [], groups: {}, problem: `材料庫讀不到：${errorText(error)}`};
  }
}
function applyZoneLibrary(state) { zoneLibrary = state.library; zoneSurfaceGroups = state.groups ?? {}; zoneLibraryProblem = state.problem; }
async function loadZoneLibrary() { applyZoneLibrary(await fetchZoneLibrary()); }
function libraryEntries(kind) { return (zoneLibrary || []).filter(entry => entry.kind === kind); }
function libraryEntry(id) { return (zoneLibrary || []).find(entry => entry.id === id) ?? null; }
// 這一筆列不列在這一面（伺服器沒給分組的舊回覆照舊全列）。
function usableHere(entry, wall) { return !entry.placement || entry.placement.surfaces.includes(wall); }
function groupWords(entry) { return zoneSurfaceGroups[entry.placement?.group] ?? "其他"; }
// 不列在這一面的，照組排（組的順序同伺服器標題表）：[[組的標題, [材料…]]…]。
function hiddenGroups(wall, entries) {
  const groups = new Map(Object.keys(zoneSurfaceGroups).map(group => [group, []]));
  for (const entry of entries) {
    if (usableHere(entry, wall)) continue;
    const group = entry.placement?.group ?? "other";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(entry);
  }
  return [...groups].filter(([, items]) => items.length).map(([group, items]) => [zoneSurfaceGroups[group] ?? group, items]);
}
// 可調材料的顯示名拿掉尾巴的「（可調厚度與空腔）」：能調什麼另外照條目寫（24K、96K 空腔其實不調）。
function shortLabel(entry) { return entry.kind === "optimizable" ? entry.label.replace(/（可調[^）]*）$/, "") : entry.label; }
function centimetres(metres) { return shownNumber(Math.round(metres * 1000) / 10); }
// 可調的到底調什麼（#1176）：照條目的多孔層厚度與空腔寫；空腔起點沒有（null 或 0）搜尋就沒有空腔這一格，寫「空腔不調」。
function adjustWords(entry) {
  const thick = entry.porous_thickness, cavity = entry.cavity;
  const thickText = thick.adjustable && thick.range_m[1] > thick.range_m[0]
    ? `厚 ${centimetres(thick.range_m[0])}～${centimetres(thick.range_m[1])} cm` : `厚度不調（${centimetres(entry.porous.thickness_m)} cm）`;
  const cavityText = !cavity || !cavity.initial_m ? "空腔不調"
    : cavity.adjustable && cavity.range_m[1] > cavity.range_m[0]
      ? `空腔 ${centimetres(cavity.range_m[0])}～${centimetres(cavity.range_m[1])} cm` : `空腔不調（${centimetres(cavity.initial_m)} cm）`;
  const depth = entry.max_total_depth_m ? `，加起來最深 ${centimetres(entry.max_total_depth_m)} cm` : "";
  return `可調：${thickText}、${cavityText}${depth}`;
}
const FIXED_WORDS = "固定成品：整塊貼或不貼，不能調";
function choiceWords(entry) { return entry.kind === "optimizable" ? adjustWords(entry) : FIXED_WORDS; }
// 這一區現在的材料是材料庫哪一筆：多層構造比貼上去的那幾層，固定阻抗與實測表比整份。
function libraryLabel(material) {
  for (const entry of libraryEntries("fixed")) {
    const chosen = entry.material.kind === "measured_impedance" ? null : entry.material;
    if (material.kind === "mounted_on_wall" && chosen?.kind === "assembly" && sameMaterial(chosen.layers, material.layers)) return entry;
    if (chosen && chosen.kind !== "assembly" && sameMaterial(chosen, material)) return entry;
    // 實測表：材料庫存的虛部可以整張沒有（物理那邊補 0），其餘欄與每一列都要一樣才算同一筆。
    if (entry.material.kind === "measured_impedance" && material.kind === "measured_impedance"
        && sameMaterial({...entry.material, points: entry.material.points.map(point => ({...point, imag_pa_s_m: point.imag_pa_s_m ?? 0}))}, material))
      return entry;
  }
  return null;
}
function materialWords(material) {
  const entry = libraryLabel(material);
  if (entry) return `${entry.label}（${material.kind === "mounted_on_wall" ? "貼在原牆上" : "直接取代牆面"}）`;
  return material.kind === "mounted_on_wall" ? `${wallText(material)}（原牆：${wallText(material.wall)}）` : wallText(material);
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
function nowWords(material) {
  if (material.kind !== "mounted_on_wall" || libraryLabel(material)) return materialWords(material);
  return `${layersText(material)}，貼在原牆上`;
}
function originalMaterial(material) { return material.kind === "mounted_on_wall" ? material.wall : material; }
// 白話（#1176）：原牆不寫阻抗數字（數字收在點開的「數字」細節）；材料庫認得的寫它的名字。
function originalWords(material) {
  // 原牆本身就是材料庫某一筆多層構造（例如天花那一套吊頂）也寫它的名字。
  const entry = libraryLabel(material) ?? libraryEntries("fixed").find(item => item.material.kind === "assembly"
    && material.kind === "assembly" && sameMaterial(item.material, material));
  return entry ? `原牆「${entry.label}」原樣` : "原牆原樣";
}
function plainNow(material) {
  return material.kind === "mounted_on_wall" ? `已貼 ${nowWords(material)}` : `沒貼東西（${originalWords(material)}）`;
}
function numbersDetail(material, id) {
  const details = document.createElement("details"); details.className = "zone-numbers"; details.id = id;
  const summary = document.createElement("summary"); summary.textContent = "數字（給要核對的人）";
  const line = document.createElement("p"); line.className = "note";
  line.textContent = `原牆：${wallText(originalMaterial(material))}`;
  details.append(summary, line);
  return details;
}
// 固定成品的總厚（每層有厚度的加起來；薄板只給面密度、不算厚度，所以寫「約」）。
function fixedDepthWords(entry) {
  const layers = entry.material.kind === "assembly" ? entry.material.layers : [];
  const depth = layers.reduce((sum, layer) => sum + (typeof layer.thickness_m === "number" ? layer.thickness_m : 0), 0);
  return depth > 0 ? `${FIXED_WORDS}（總厚約 ${centimetres(depth)} cm）` : FIXED_WORDS;
}

// 交伺服器換這一區的材料；成功才原地換進方案（同切牆面：等回覆時整頁鎖住）。回伺服器的回覆，沒換成回 null。
async function changeZoneMaterial(wall, key, choice) {
  if (inputBusy) { zoneNote("上一個改動還在處理，請稍等再選", "notice"); return null; }
  if (measuredDrafts[wall]) { zoneNote(`${wallNames[wall]}有還沒套用的實測阻抗草稿：先套用或取消`, "notice"); return null; }
  const draft = wallDocument();
  inputBusy = true;
  document.querySelector("main").inert = true;
  let data;
  try {
    let response;
    ({response, data} = await reply(fetch("/api/zone-material", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scheme: draft, wall, zone: key, ...choice, original_scheme_id: zoneDesign?.original_scheme_id ?? null})})));
    if (!response.ok) { zoneNote(`沒有換：${data.error}`, "notice"); return null; }
    if (!scheme.scene.surface_zones_by_wall) {
      scheme.scene.surface_zones_by_wall = draft.scene.surface_zones_by_wall;
      delete scheme.scene.impedance_pa_s_per_m_by_wall;
    }
    const zone = scheme.scene.surface_zones_by_wall[wall].find(item => zoneKey(item) === key);
    zone.material = data.material;
  } finally {
    document.querySelector("main").inert = false;
    inputBusy = false;
  }
  zoneNote(`${wallNames[wall]}「${zoneName(key)}」：${data.note}`, data.origin === "unknown" ? "notice" : "ok");
  return data;
}
async function afterZoneChange() {
  renderWalls();
  markStale();
  action(updateMultiples);
  action(updateInputPreview);
  await refreshPlan();
}
function zoneName(key) { return key === "whole_wall" ? "整面" : key.startsWith(inputDefaults.zone_cut.fill_prefix) ? `自動 ${key}` : key; }

// 換材料連不上或回覆讀不出來：照講、把選單畫回原樣（方案與做法都沒動），不留在新選項上。
async function zoneMaterialChanged(wall, key, choice) {
  try { return await changeZoneMaterial(wall, key, choice); } catch (error) {
    zoneNote(`沒有換：${errorText(error)}`, "notice");
    return null;
  }
}
async function changeZoneMode(wall, key, mode) {
  const before = zoneSetting(wall, key);
  const held = libraryLabel(zoneList(wall).find(zone => zoneKey(zone) === key).material);
  if (mode !== "fixed" && !await zoneMaterialChanged(wall, key, {choice: "original"})) { renderZoneTable(wall); return; }
  if (mode === "search" && held && usableHere(held, wall)) zoneFormerFixed.set(`${wall}/${key}`, held.id);
  else zoneFormerFixed.delete(`${wall}/${key}`);
  const next = {...before, mode};
  if (mode !== "search") delete next.allowed;
  else if (!next.allowed) {
    const start = startingAllowed(wall);
    if (start) next.allowed = start;
  }
  setZoneSetting(wall, key, next);
  if (mode === "fixed") zoneNote(`${wallNames[wall]}「${zoneName(key)}」：指定貼一種，在右邊選要貼哪一種；不選就照方案現在的材料`, "ok");
  if (mode === "search" && next.allowed) {
    zoneNote(`${wallNames[wall]}「${zoneName(key)}」：交給電腦挑，先放了 ${next.allowed.map(choiceName).join("、")}；要改就按「改清單」或下面的常用組合`, "ok");
  }
  await afterZoneChange();
}
async function chooseFixed(wall, key, materialId) {
  const data = await zoneMaterialChanged(wall, key, {choice: "fixed", material_id: materialId});
  if (!data) { renderZoneTable(wall); return; }
  // 從交給電腦挑的紅框按「就貼這一筆」也走這裡：清單只給交給電腦挑的區，改成指定貼一種就拿掉（伺服器不收固定區帶清單）。
  // 選的跟這一區原牆一模一樣（#1175）：伺服器放回原牆，這裡做法跟著改成「不貼，保留原牆」並說明。
  const next = {...zoneSetting(wall, key), mode: data.same_as_original ? "keep_original" : "fixed"};
  delete next.allowed;
  setZoneSetting(wall, key, next);
  if (data.same_as_original) {
    zoneNote(`${wallNames[wall]}「${zoneName(key)}」：選的跟原本一樣，當成不貼（做法改成「不貼，保留原牆」）`, "ok");
  }
  await afterZoneChange();
}
// 改清單（清單上的 ×、勾選表、常用組合都走這裡）：存成明確清單、記成這一頁「上一次的組合」
// （起點要是可調材料或不貼，伺服器開跑時再核一次）。
function changeAllowed(wall, key, allowed) {
  setZoneSetting(wall, key, {...zoneSetting(wall, key), mode: "search", allowed});
  zoneLastAllowed = [...allowed];
  markStale();
  renderZoneTable(wall);
}
// 伺服器沒寫清單（null）的意思：材料庫全部可調材料＋不貼。材料庫沒讀全（讀不到或使用者檔壞了）就回 null，不替伺服器縮小範圍。
function serverDefaultAllowed() {
  if (!zoneLibrary?.length || zoneLibraryProblem) return null;
  return [...libraryEntries("optimizable").map(entry => entry.id), ZONE_NONE];
}
function fitsHere(id, wall) { return id === ZONE_NONE || (libraryEntry(id) !== null && usableHere(libraryEntry(id), wall)); }
// 改成交給電腦挑時一開始放的清單：這一頁上一次改過的組合優先（這一面用不了的拿掉；拿完沒有起點就不用它），
// 不然一律是三種內建玻璃棉＋不貼（材料庫沒有的、不是這一面用的拿掉，地板只剩不貼）；使用者自己加的可調材料要用就在「改清單」勾。
// 材料庫整份讀不到就回 null（照舊不寫清單、照講）。
function startingAllowed(wall) {
  if (!zoneLibrary?.length) return null;
  const last = (zoneLastAllowed ?? []).filter(id => fitsHere(id, wall));
  if (last.length && !allowedProblem(last)) return last;
  return [...THREE_WOOLS.filter(id => libraryEntry(id)?.kind === "optimizable" && fitsHere(id, wall)), ZONE_NONE];
}
function choiceName(id) {
  const entry = libraryEntry(id);
  return id === ZONE_NONE ? "不貼" : entry ? shortLabel(entry) : `${id}（材料庫找不到）`;
}
// 清單裡每一種都是材料庫認得的固定成品（沒有可調材料、沒有不貼）：電腦沒得挑（伺服器開跑時 gradient_plan 同一句白話再擋一次）。
// 材料庫找不到的代號不算固定成品（材料庫沒讀全時可能是可調材料；那一種照「材料庫找不到」那一行講，開跑時伺服器點名）。
function allowedProblem(allowed) {
  if (!allowed?.length) return "";
  return allowed.every(id => libraryEntry(id)?.kind === "fixed") ? "只放了固定成品，電腦沒得挑。" : "";
}
// 紅框直接給按鈕（#1176，老闆看不懂「准用清單只有固定材料：搜尋要一個起點…」）：只有一筆就能直接改成指定貼它；
// 或加上「不貼」、加上 48K 玻璃棉（這一面用不了就不出現）。
function problemBox(wall, key, allowed, edit) {
  const box = document.createElement("div"); box.className = "zone-problem"; box.id = `zone-allowed-problem-${wall}-${key}`;
  box.setAttribute("role", "alert");
  const head = document.createElement("p"); head.className = "limit-red"; head.textContent = allowedProblem(allowed);
  const why = document.createElement("p");
  why.textContent = "電腦要從一個能調的東西開始：可調的材料（例如玻璃棉），或先不貼。你要的是哪一個？";
  const buttons = document.createElement("div"); buttons.className = "zone-problem-actions";
  const button = (id, text, onclick) => {
    const item = document.createElement("button"); item.type = "button"; item.id = `zone-fix-${id}-${wall}-${key}`;
    item.textContent = text; item.onclick = onclick; buttons.append(item);
  };
  const only = allowed.length === 1 ? libraryEntry(allowed[0]) : null;
  if (only?.kind === "fixed") button("only", "就貼這一筆（改成指定貼一種）", () => action(() => chooseFixed(wall, key, only.id)));
  button("add-none", "加上「不貼」", () => edit([...allowed, ZONE_NONE]));
  if (fitsHere(WOOL_48K, wall)) button("add-wool", "加上 48K 玻璃棉", () => edit([...allowed, WOOL_48K]));
  box.append(head, why, buttons);
  return box;
}

function modeSelect(wall, key, setting) {
  const select = document.createElement("select"); select.id = `zone-mode-${wall}-${key}`;
  select.setAttribute("aria-label", `${wallNames[wall]} ${zoneName(key)} 的做法`);
  for (const [value, text] of Object.entries(ZONE_MODE_WORDS)) {
    const option = document.createElement("option"); option.value = value; option.textContent = text; select.append(option);
  }
  select.value = setting.mode;
  select.onchange = () => action(() => changeZoneMode(wall, key, select.value));
  return select;
}
function fixedOption(entry, text = `${entry.label}（${entry.basis}）`) {
  const option = document.createElement("option"); option.value = entry.id; option.textContent = text;
  return option;
}
function optionGroup(label, entries) {
  const group = document.createElement("optgroup"); group.label = label;
  for (const entry of entries) group.append(fixedOption(entry));
  return group;
}
// 指定貼一種（#1176）：選單只列這一面能貼的固定成品；這一區現在放的那一筆就算不是這一面用的也照樣列（不然選單跟方案對不上）。
function fixedCell(wall, key, zone, cell) {
  const select = document.createElement("select"); select.id = `zone-fixed-${wall}-${key}`;
  select.setAttribute("aria-label", `${wallNames[wall]} ${zoneName(key)} 要貼哪一種`);
  const current = libraryLabel(zone.material);
  if (!current) {
    const now = document.createElement("option"); now.value = ""; now.textContent = `現在：${plainNow(zone.material)}`;
    select.append(now);
  }
  const fixed = libraryEntries("fixed");
  const here = fixed.filter(entry => usableHere(entry, wall));
  for (const entry of here.filter(item => item.placement?.group !== "other")) select.append(fixedOption(entry));
  const other = here.filter(item => item.placement?.group === "other");
  if (other.length) select.append(optionGroup(zoneSurfaceGroups.other ?? "其他", other));
  const away = current && !usableHere(current, wall);
  // 貼上去的才會「不是這一面用的」；直接取代整面的（例如地板本身就是木地板那一筆固定阻抗）就是這一面本身，照實寫、不標紅。
  const pasted = zone.material.kind === "mounted_on_wall";
  if (away && !zoneShowAll) {
    select.append(fixedOption(current, pasted ? `${current.label}（現在用的；${groupWords(current)}）`
      : `${current.label}（現在用的：整面就是它，不是貼上去的）`));
  }
  if (zoneShowAll) for (const [title, entries] of hiddenGroups(wall, fixed)) select.append(optionGroup(`不列的：${title}`, entries));
  select.value = current?.id ?? "";
  select.onchange = () => { if (select.value) action(() => chooseFixed(wall, key, select.value)); };
  cell.append(select);
  if (away && pasted) {
    const line = document.createElement("p"); line.className = "notice"; line.id = `zone-fixed-away-${wall}-${key}`;
    line.textContent = `「${current.label}」不是${wallNames[wall]}用的：${current.placement.reason}`;
    cell.append(line);
  }
  if (!current && zone.material.kind !== "mounted_on_wall") cell.append(numbersDetail(zone.material, `zone-numbers-${wall}-${key}`));
}
// 勾選表（#1176）：只列這一面能貼的，分「可調」「不貼」「固定成品」「其他」幾組；「顯示全部材料」打開才分組列出不是這一面
// 用的那幾組（各一個標題說為什麼）。onToggle(代號, 勾選格) 交呼叫的人決定清單怎麼改。
function allowedChecklist(wall, key, allowed, onToggle) {
  const list = document.createElement("div"); list.className = "zone-allowed"; list.id = `zone-allowed-${wall}-${key}`;
  const heading = text => { const line = document.createElement("p"); line.className = "zone-group"; line.textContent = text; return line; };
  const box = (id, name, words, reason) => {
    const wrap = document.createElement("label"); wrap.className = "zone-choice";
    if (reason) wrap.title = reason;
    const input = document.createElement("input"); input.type = "checkbox"; input.id = `zone-allow-${wall}-${key}-${id}`;
    input.checked = allowed.includes(id);
    input.onchange = () => onToggle(id, input);
    const text = document.createElement("span"); text.textContent = name;
    const small = document.createElement("small"); small.textContent = words;
    wrap.append(input, text, small);
    return wrap;
  };
  const here = (zoneLibrary || []).filter(entry => usableHere(entry, wall));
  const optimizable = here.filter(entry => entry.kind === "optimizable" && entry.placement?.group !== "other");
  const fixed = here.filter(entry => entry.kind === "fixed" && entry.placement?.group !== "other");
  const other = here.filter(entry => entry.placement?.group === "other");
  list.append(heading("可調：電腦會調厚度（有的連空腔一起調）"));
  if (!optimizable.length) {
    const line = document.createElement("p"); line.className = "note";
    line.textContent = `${wallNames[wall]}沒有可調的材料：電腦只能從「不貼」開始，可以再勾固定成品讓它比`;
    list.append(line);
  }
  for (const entry of optimizable) list.append(box(entry.id, shortLabel(entry), adjustWords(entry), entry.placement?.reason));
  list.append(box(ZONE_NONE, "不貼，保留原牆", "放進來，電腦就可以決定這裡不貼"));
  if (fixed.length) list.append(heading("固定成品：整塊貼上去，不能調（電腦只決定貼不貼）"));
  for (const entry of fixed) list.append(box(entry.id, entry.label, fixedDepthWords(entry), entry.placement?.reason));
  if (other.length) list.append(heading(zoneSurfaceGroups.other ?? "其他"));
  for (const entry of other) list.append(box(entry.id, shortLabel(entry), entry.kind === "fixed" ? fixedDepthWords(entry) : adjustWords(entry)));
  const hidden = hiddenGroups(wall, zoneLibrary || []);
  const count = hidden.reduce((sum, [, entries]) => sum + entries.length, 0);
  if (zoneShowAll) {
    for (const [title, entries] of hidden) {
      list.append(heading(`不列的：${title}`));
      for (const entry of entries) {
        const words = entry.kind === "fixed" ? fixedDepthWords(entry) : adjustWords(entry);
        list.append(box(entry.id, shortLabel(entry), `${words}；${entry.placement?.reason ?? ""}`));
      }
    }
  } else if (count) {
    const line = document.createElement("p"); line.className = "note"; line.id = `zone-allowed-hidden-${wall}-${key}`;
    line.textContent = `還有 ${count} 筆不是${wallNames[wall]}用的、不列出來（${hidden.map(([title]) => title).join("；")}）。要選就打開表格下面「顯示全部材料」`;
    list.append(line);
  }
  return list;
}
function sameList(left, right) { return left.length === right.length && left.every(id => right.includes(id)); }
// 清單上每一種一個方塊，按 × 拿掉；不是這一面用的、材料庫找不到的照樣列（存著的清單原樣帶回）。
function allowedChips(wall, key, allowed, edit) {
  const row = document.createElement("div"); row.className = "zone-chips"; row.id = `zone-allowed-chips-${wall}-${key}`;
  const lead = document.createElement("span"); lead.className = "zone-chips-lead"; lead.textContent = "電腦從這幾種裡挑：";
  row.append(lead);
  for (const id of allowed) {
    const entry = libraryEntry(id);
    const chip = document.createElement("span"); chip.className = "zone-chip"; chip.id = `zone-chip-${wall}-${key}-${id}`;
    const name = document.createElement("strong"); name.textContent = choiceName(id);
    const words = document.createElement("small");
    words.textContent = id === ZONE_NONE ? "保留原牆，電腦可以決定這裡不貼" : !entry ? "照原樣留著，開跑時伺服器會點名"
      : usableHere(entry, wall) ? choiceWords(entry) : `${choiceWords(entry)}；不是${wallNames[wall]}用的（${groupWords(entry)}）`;
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "zone-chip-remove";
    remove.id = `zone-chip-remove-${wall}-${key}-${id}`; remove.textContent = "×";
    remove.setAttribute("aria-label", `拿掉「${choiceName(id)}」`);
    remove.onclick = () => edit(allowed.filter(choice => choice !== id));
    chip.append(name, words, remove);
    row.append(chip);
  }
  return row;
}
// 一鍵常用組合（這一面用得到的才出現）；這一區改成交給電腦挑之前指定貼了某一筆，多一顆「只讓電腦決定：這一筆貼或不貼」。
function presetRow(wall, key, zone, allowed, edit) {
  const presets = ZONE_PRESETS.filter(preset => preset.members.every(id => fitsHere(id, wall)));
  const held = zoneFormerFixed.get(`${wall}/${key}`) ?? libraryLabel(zone.material)?.id;
  if (held && fitsHere(held, wall)) {
    presets.push({id: "this_or_none", words: `只讓電腦決定：「${libraryEntry(held).label}」貼或不貼`, members: [held, ZONE_NONE]});
  }
  const row = document.createElement("div"); row.className = "zone-presets"; row.id = `zone-presets-${wall}-${key}`;
  if (!presets.length) return row;
  const lead = document.createElement("span"); lead.textContent = "一鍵換成常用組合：";
  row.append(lead);
  for (const preset of presets) {
    const button = document.createElement("button"); button.type = "button"; button.id = `zone-preset-${wall}-${key}-${preset.id}`;
    const current = sameList(preset.members, allowed);
    button.textContent = current ? `${preset.words}（現在用的）` : preset.words;
    button.disabled = current;
    button.onclick = () => edit([...preset.members]);
    row.append(button);
  }
  return row;
}
function searchCell(wall, key, zone, setting, cell) {
  // 寫方案裡這一區現在真的放什麼（#1128）：網頁改成交給電腦挑時放原牆，但搜尋存成的方案（或它的另存）放的是搜尋選的材料。
  const start = document.createElement("p"); start.className = "note"; start.id = `zone-search-now-${wall}-${key}`;
  start.textContent = zone.material.kind === "mounted_on_wall"
    ? `現在：${plainNow(zone.material)}`
    : `現在：${plainNow(zone.material)}，電腦挑了要貼的就貼在它上面`;
  cell.append(start, numbersDetail(zone.material, `zone-numbers-${wall}-${key}`));
  // 舊的沒寫清單（null）照伺服器的意思列成明確清單（全部可調材料＋不貼）；材料庫沒讀全就只能照講，不猜。
  const allowed = setting.allowed ?? serverDefaultAllowed();
  if (!allowed) {
    const line = document.createElement("p"); line.className = "notice"; line.id = `zone-allowed-unknown-${wall}-${key}`;
    line.textContent = "電腦從全部可調材料與「不貼」裡挑（材料庫沒讀全，列不出是哪幾種；等材料庫讀得到再改清單）";
    cell.append(line);
    return;
  }
  // 最後一種不准拿掉：空清單伺服器整份拒收（整份做法都存不進去）。
  const edit = next => {
    if (!next.length) {
      zoneNote(`${wallNames[wall]}「${zoneName(key)}」：清單至少要留一種；不想讓電腦挑，就把做法改成「指定貼一種」或「不貼，保留原牆」`, "notice");
      renderZoneTable(wall);
      return;
    }
    changeAllowed(wall, key, next);
  };
  cell.append(allowedChips(wall, key, allowed, edit), presetRow(wall, key, zone, allowed, edit));
  // 勾選表收在「改清單」後面（每一區都攤開二十幾格，整張表讀不下去）；點開過的那幾區重畫後照樣開著。
  // 只加減勾的這一種：存著的清單原樣帶回（順序不動、材料庫找不到的代號也不偷丟）。
  const details = document.createElement("details"); details.id = `zone-allowed-box-${wall}-${key}`;
  details.open = zoneAllowedOpen.has(`${wall}/${key}`);
  const summary = document.createElement("summary"); summary.id = `zone-allowed-summary-${wall}-${key}`;
  const words = () => details.open ? "收起清單" : `改清單（現在 ${allowed.length} 種，勾要讓電腦挑的）`;
  details.ontoggle = () => {
    if (details.open) zoneAllowedOpen.add(`${wall}/${key}`); else zoneAllowedOpen.delete(`${wall}/${key}`);
    summary.textContent = words();
  };
  summary.textContent = words();
  const list = allowedChecklist(wall, key, allowed, (id, input) => {
    edit(input.checked ? [...allowed.filter(choice => choice !== id), id] : allowed.filter(choice => choice !== id));
  });
  details.append(summary, list);
  cell.append(details);
  const missing = allowed.filter(id => id !== ZONE_NONE && !libraryEntry(id));
  if (missing.length) {
    const line = document.createElement("p"); line.className = "notice"; line.id = `zone-allowed-missing-${wall}-${key}`;
    line.textContent = `清單裡有材料庫找不到的：${missing.join("、")}（照原樣留著，開跑時伺服器會點名；要拿掉就按它旁邊的 ×）`;
    cell.append(line);
  }
  if (allowedProblem(allowed)) cell.append(problemBox(wall, key, allowed, edit));
}
function originalSchemeRow() {
  const row = document.createElement("div"); row.className = "fields";
  const wrap = document.createElement("label"); wrap.textContent = "原方案（算「改善多少」的原牆、也給拆不出原牆的區用）";
  const select = document.createElement("select"); select.id = "original-scheme";
  const none = document.createElement("option"); none.value = ""; none.textContent = "未指定（拆掉貼上去的層當原牆）"; select.append(none);
  const self = $("save-id").value;
  for (const option of $("scheme-list").options) {
    if (option.value === self) continue;
    const item = document.createElement("option"); item.value = option.value; item.textContent = option.value; select.append(item);
  }
  const chosen = zoneDesign?.original_scheme_id ?? "";
  if (chosen && ![...select.options].some(option => option.value === chosen)) {
    const missing = document.createElement("option"); missing.value = chosen; missing.textContent = `${chosen}（不在清單裡）`; select.append(missing);
  }
  select.value = chosen;
  select.onchange = () => { designForEdit().original_scheme_id = select.value || null; markStale(); };
  wrap.append(select); row.append(wrap);
  return row;
}
// 「顯示全部材料」（#1176）：一張表一個開關，打開後選單與勾選表都分組列出不是這一面用的那幾組；只記在這一頁。
function showAllRow(wall) {
  const row = document.createElement("p"); row.className = "fields";
  const wrap = document.createElement("label");
  const box = document.createElement("input"); box.type = "checkbox"; box.id = "zone-show-all"; box.checked = zoneShowAll;
  box.onchange = () => { zoneShowAll = box.checked; renderZoneTable(wall); };
  wrap.append(box, document.createTextNode("顯示全部材料（門、窗、整片天花構造、另一間示範房的這些平常不列；要選它們才打開）"));
  row.append(wrap);
  return row;
}
function renderZoneTable(wall) {
  const area = $("elevation-zones");
  if (!area) return;
  area.replaceChildren();
  if (!wall || !scheme) return;
  area.append(originalSchemeRow());
  area.append(depthRow(wall));
  area.append(limitsSummary());
  if (wallLimitLines(wall).length) area.append(limitLinesBlock(wallLimitLines(wall), `limit-wall-${wall}`));
  for (const text of [zoneDesignNote, zoneLibraryProblem]) {
    if (!text) continue;
    const line = document.createElement("p"); line.className = "notice"; line.textContent = text; area.append(line);
  }
  const table = document.createElement("table"); table.id = `zone-table-${wall}`; table.className = "zone-table";
  const head = document.createElement("tr");
  for (const text of ["區", "大小（公尺）", "做法", "材料", "深度上限（公尺）"]) { const cell = document.createElement("th"); cell.textContent = text; head.append(cell); }
  table.append(head);
  for (const zone of zoneList(wall)) {
    const key = zoneKey(zone), setting = zoneSetting(wall, key);
    const row = document.createElement("tr"); row.id = `zone-row-${wall}-${key}`;
    row.className = `zone-mode-${setting.mode}${zoneIsRed(wall, key) ? " zone-red" : ""}`;
    const name = document.createElement("td"); name.textContent = zoneName(key);
    const size = document.createElement("td");
    size.textContent = zone.area === "rectangle"
      ? `${Number((zone.u_m[1] - zone.u_m[0]).toFixed(3))} × ${Number((zone.v_m[1] - zone.v_m[0]).toFixed(3))}` : "整面";
    const mode = document.createElement("td"); mode.append(modeSelect(wall, key, setting));
    const material = document.createElement("td"); material.id = `zone-material-${wall}-${key}`;
    if (setting.mode === "fixed") fixedCell(wall, key, zone, material);
    else if (setting.mode === "keep_original") {
      const line = document.createElement("p"); line.id = `zone-keep-${wall}-${key}`;
      line.textContent = `${originalWords(originalMaterial(zone.material))}，不貼東西`;
      material.append(line, numbersDetail(zone.material, `zone-numbers-${wall}-${key}`));
    }
    else searchCell(wall, key, zone, setting, material);
    const depth = document.createElement("td"); depth.id = `zone-depth-${wall}-${key}`;
    zoneDepthCell(wall, key, setting, depth);
    const lines = zoneLimitLines(wall, key);
    if (lines.length) material.append(limitLinesBlock(lines, `limit-${wall}-${key}`));
    row.append(name, size, mode, material, depth); table.append(row);
  }
  area.append(table, showAllRow(wall));
  const toSearch = searchButtonRow();
  if (toSearch) area.append(toSearch);
  const note = document.createElement("p"); note.id = "zone-note"; note.setAttribute("aria-live", "polite");
  note.textContent = zoneNoteText; note.className = zoneNoteKind; area.append(note);
}
window.renderZoneTable = renderZoneTable;
