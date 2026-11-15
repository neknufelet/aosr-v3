/* 每一區的做法（#1105，#1082 第 1-3 段）：立面圖下面一張表，這一面每一區（畫的、自動補的、整面一區）選
   「指定貼一種」「不貼，保留原牆」「交給電腦挑」（設計檔裡叫 fixed／keep_original／search）；交給電腦挑的區列出電腦
   可以從哪幾種裡挑。材料選單只列這一面能貼的（#1176，分組由伺服器 /api/materials 的 placement 現算），其餘收在
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
const ZONE_MODE_WORDS = {fixed: "指定貼一種", keep_original: "不貼，保留原牆", search: "交給電腦挑"};
const ZONE_NONE = "none";

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
  zoneDesignNote = state.note; zoneNote("", ""); zoneLimits = null;
}
// 載入範本：新的方案，沒有設計設定。
function resetZoneDesign() { zoneDesignGeneration += 1; zoneDesign = null; zoneDesignStored = false; zoneDesignBroken = false; zoneDesignNote = ""; zoneNote("", ""); zoneLimits = null; }
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
// 「拿去搜尋」：搜尋頁讀的是存好的方案與做法，先照畫面存一份；兩樣都存成才跳過去，沒存成留在這頁照講。
async function sendToSearch() {
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
  const red = limitsRedCount();
  button.disabled = red > 0;
  const note = document.createElement("span"); note.id = "zone-to-search-note";
  note.className = red ? "limit-red" : "note";
  note.textContent = red ? `有 ${red} 條超過限制（標紅的區），改好之前不能拿去搜尋` : "會先照畫面儲存，再打開第三階段搜尋頁";
  row.append(button, " ", note);
  return row;
}
// 切完一面：自動補的代號每次重編，做法跟著 /api/zone-cut 回的 inherited（每一塊沿用了切之前哪一區的材料）走。
function remapZoneDesign(wall, inherited) {
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
  const entry = libraryLabel(material);
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

// 交伺服器換這一區的材料；成功才原地換進方案（同切牆面：等回覆時整頁鎖住）。
async function changeZoneMaterial(wall, key, choice) {
  if (inputBusy) { zoneNote("上一個改動還在處理，請稍等再選", "notice"); return false; }
  if (measuredDrafts[wall]) { zoneNote(`${wallNames[wall]}有還沒套用的實測阻抗草稿：先套用或取消`, "notice"); return false; }
  const draft = wallDocument();
  inputBusy = true;
  document.querySelector("main").inert = true;
  let data;
  try {
    let response;
    ({response, data} = await reply(fetch("/api/zone-material", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scheme: draft, wall, zone: key, ...choice, original_scheme_id: zoneDesign?.original_scheme_id ?? null})})));
    if (!response.ok) { zoneNote(`沒有換：${data.error}`, "notice"); return false; }
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
  return true;
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
    return false;
  }
}
async function changeZoneMode(wall, key, mode) {
  const before = zoneSetting(wall, key);
  if (mode !== "fixed" && !await zoneMaterialChanged(wall, key, {choice: "original"})) { renderZoneTable(wall); return; }
  const next = {...before, mode};
  if (mode !== "search") delete next.allowed;
  setZoneSetting(wall, key, next);
  if (mode === "fixed") zoneNote(`${wallNames[wall]}「${zoneName(key)}」：指定貼一種，在右邊選要貼哪一種；不選就照方案現在的材料`, "ok");
  await afterZoneChange();
}
async function chooseFixed(wall, key, materialId) {
  if (!await zoneMaterialChanged(wall, key, {choice: "fixed", material_id: materialId})) { renderZoneTable(wall); return; }
  setZoneSetting(wall, key, {...zoneSetting(wall, key), mode: "fixed"});
  await afterZoneChange();
}
// 准用清單：不限定＝搜尋照預設換全部可調材料與不貼；限定就只換勾的（起點要是可調材料或不貼，伺服器開跑時再核一次）。
function changeAllowed(wall, key, allowed) {
  const next = {...zoneSetting(wall, key), mode: "search"};
  if (allowed === null) delete next.allowed; else next.allowed = allowed;
  setZoneSetting(wall, key, next);
  markStale();
  renderZoneTable(wall);
}
function allowedProblem(allowed) {
  if (!allowed) return "";
  const optimizable = new Set(libraryEntries("optimizable").map(entry => entry.id));
  return allowed.some(id => id === ZONE_NONE || optimizable.has(id)) ? ""
    : "只放了固定成品，電腦沒得挑：請再勾一種可調材料或「不貼」";
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
  if (away && !zoneShowAll) select.append(fixedOption(current, `${current.label}（現在用的；${groupWords(current)}）`));
  if (zoneShowAll) for (const [title, entries] of hiddenGroups(wall, fixed)) select.append(optionGroup(`不列的：${title}`, entries));
  select.value = current?.id ?? "";
  select.onchange = () => { if (select.value) action(() => chooseFixed(wall, key, select.value)); };
  cell.append(select);
  if (away) {
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
function searchCell(wall, key, zone, setting, cell) {
  // 寫方案裡這一區現在真的放什麼（#1128）：網頁改成交給搜尋時放原牆，但搜尋存成的方案（或它的另存）放的是搜尋選的材料。
  const start = document.createElement("p"); start.className = "note"; start.id = `zone-search-now-${wall}-${key}`;
  start.textContent = zone.material.kind === "mounted_on_wall"
    ? `現在：${plainNow(zone.material)}`
    : `現在：${plainNow(zone.material)}，電腦挑了要貼的就貼在它上面`;
  const limit = document.createElement("label");
  const box = document.createElement("input"); box.type = "checkbox"; box.id = `zone-limit-${wall}-${key}`;
  box.checked = Array.isArray(setting.allowed);
  // 打勾時先勾滿這一面能用的可調材料與不貼，再讓人拿掉不要的。
  box.onchange = () => {
    if (box.checked) zoneAllowedOpen.add(`${wall}/${key}`);
    const usable = libraryEntries("optimizable").filter(entry => usableHere(entry, wall)).map(entry => entry.id);
    changeAllowed(wall, key, box.checked ? [...usable, ZONE_NONE] : null);
  };
  limit.append(box, document.createTextNode("只讓電腦從我勾的幾種裡挑（不勾＝全部可調材料與不貼都可以）"));
  cell.append(start, limit);
  cell.append(numbersDetail(zone.material, `zone-numbers-${wall}-${key}`));
  if (!box.checked) return;
  // 清單收在一行摘要後面（每一區都攤開二十幾格，整張表讀不下去）；點開過的那幾區重畫後照樣開著。
  const details = document.createElement("details"); details.id = `zone-allowed-box-${wall}-${key}`;
  details.open = zoneAllowedOpen.has(`${wall}/${key}`);
  details.ontoggle = () => { if (details.open) zoneAllowedOpen.add(`${wall}/${key}`); else zoneAllowedOpen.delete(`${wall}/${key}`); };
  const summary = document.createElement("summary"); summary.id = `zone-allowed-summary-${wall}-${key}`;
  // 只加減勾的這一種：存著的清單原樣帶回（順序不動、材料庫找不到的代號也不偷丟）。
  // 最後一種不准拿掉：空清單伺服器整份拒收（整份做法都存不進去）；不想限定就取消上面那一格。
  const list = allowedChecklist(wall, key, setting.allowed, (id, input) => {
    const next = input.checked ? [...setting.allowed.filter(choice => choice !== id), id] : setting.allowed.filter(choice => choice !== id);
    if (!next.length) {
      input.checked = true;
      zoneNote(`${wallNames[wall]}「${zoneName(key)}」：清單至少要留一種；不想限定就取消「只讓電腦從我勾的幾種裡挑」那一格`, "notice");
      return;
    }
    changeAllowed(wall, key, next);
  });
  const name = id => id === ZONE_NONE ? "不貼" : libraryEntry(id) ? shortLabel(libraryEntry(id)) : `${id}（材料庫找不到）`;
  const missing = setting.allowed.filter(id => id !== ZONE_NONE && !libraryEntry(id));
  const names = setting.allowed.map(name);
  summary.textContent = `電腦從 ${names.length} 種裡挑：${names.join("、") || "（沒有）"}（點開改）`;
  details.append(summary, list);
  cell.append(details);
  if (missing.length) {
    const line = document.createElement("p"); line.className = "notice"; line.id = `zone-allowed-missing-${wall}-${key}`;
    line.textContent = `清單裡有材料庫找不到的：${missing.join("、")}（照原樣留著，開跑時伺服器會點名；要拿掉：取消上面那一格再重新勾）`;
    cell.append(line);
  }
  const problem = allowedProblem(setting.allowed);
  if (problem) { const line = document.createElement("p"); line.className = "notice"; line.id = `zone-allowed-problem-${wall}-${key}`; line.textContent = problem; cell.append(line); }
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
