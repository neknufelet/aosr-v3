/* 每一區的做法（#1105，#1082 第 1-3 段）：立面圖下面一張表，這一面每一區（畫的、自動補的、整面一區）選
   「指定材料組（固定）」「不貼，保留原牆」「交給搜尋」；交給搜尋的區可以限定准用哪幾組材料。
   做法存在設計設定（/api/designs，跟方案同名、跟著方案存）；方案那一區的材料由伺服器換（/api/zone-material）：
   固定材料貼在原牆上、不貼與交給搜尋放回原牆。材料選單讀材料庫（/api/materials），不另做一份。
   設計設定整份原樣帶著走（原方案、之後的段加的欄位都不丟），只改這張表管的那幾格。 */
let zoneDesign = null;
let zoneDesignStored = false;
let zoneDesignNote = "";
// 伺服器上那一份設計設定讀不到（壞檔或連不上）：這一頁不准存回去，免得一份空的蓋掉原本每一區的做法。
let zoneDesignBroken = false;
let zoneLibrary = null;
let zoneLibraryProblem = "";
let zoneNoteText = "";
let zoneNoteKind = "";
const zoneAllowedOpen = new Set();
const ZONE_MODE_WORDS = {fixed: "指定材料組（固定）", keep_original: "不貼，保留原牆", search: "交給搜尋"};
const ZONE_NONE = "none";

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
async function loadZoneDesign(name) {
  zoneDesign = null; zoneDesignStored = false; zoneDesignBroken = false; zoneDesignNote = ""; zoneNote("", "");
  try {
    const data = await api(`/api/designs/${encodeURIComponent(name)}`);
    zoneDesign = data.design;
    zoneDesignStored = Object.keys(data.design.zones || {}).length > 0 || data.design.original_scheme_id != null;
    if (data.problems.length) zoneDesignNote = `設計設定有幾條跟方案對不上（下次儲存時拿掉）：${data.problems.join("；")}`;
  } catch (error) {
    zoneDesignBroken = true;
    zoneDesignNote = `每一區的做法讀不到（${errorText(error)}）：這一頁改的做法不會存回去，免得蓋掉原本存的；先處理好再重新打開方案`;
  }
}
// 載入範本：新的方案，沒有設計設定。
function resetZoneDesign() { zoneDesign = null; zoneDesignStored = false; zoneDesignBroken = false; zoneDesignNote = ""; zoneNote("", ""); }
// 存方案之後接著存設計設定（整份換上）；對不上方案的區先拿掉（伺服器會整份拒收），原方案指到自己就清掉。
// 從來沒設過任何一區、伺服器上也沒有這一份，就不寫（舊方案存檔不多出一個檔）。
async function saveZoneDesign(name) {
  if (zoneDesignBroken) throw new Error("原本的設計設定讀不到，這次沒存每一區的做法（免得蓋掉原本存的）");
  const touched = zoneDesign && (Object.keys(zoneDesign.zones || {}).length || zoneDesign.original_scheme_id != null);
  if (!touched && !zoneDesignStored) return;
  const design = structuredClone(designForEdit());
  design.scheme_id = name;
  for (const wall of Object.keys(design.zones)) {
    const keys = new Set(zoneList(wall).map(zoneKey));
    for (const key of Object.keys(design.zones[wall])) if (!keys.has(key)) delete design.zones[wall][key];
    if (!Object.keys(design.zones[wall]).length) delete design.zones[wall];
  }
  if (design.original_scheme_id === name) design.original_scheme_id = null;
  await api(`/api/designs/${encodeURIComponent(name)}`, "PUT", design);
  zoneDesign = design; zoneDesignStored = true; zoneDesignNote = "";
}
// 切完一面：自動補的代號每次重編，做法跟著 /api/zone-cut 回的 inherited（每一塊沿用了切之前哪一區的材料）走。
function remapZoneDesign(wall, inherited) {
  const old = zoneDesign?.zones?.[wall];
  if (!old) return;
  const next = {};
  for (const [key, source] of Object.entries(inherited)) if (old[source]) next[key] = structuredClone(old[source]);
  if (Object.keys(next).length) zoneDesign.zones[wall] = next; else delete zoneDesign.zones[wall];
}

async function loadZoneLibrary() {
  try {
    const data = await api("/api/materials");
    zoneLibrary = data.materials.map(row => ({...row.entry, basis: row.basis}));
    zoneLibraryProblem = data.user_file_problem ? `使用者材料庫讀不進來，選單只列內建材料：${data.user_file_problem}` : "";
  } catch (error) {
    zoneLibrary = [];
    zoneLibraryProblem = `材料庫讀不到：${errorText(error)}`;
  }
}
function libraryEntries(kind) { return (zoneLibrary || []).filter(entry => entry.kind === kind); }
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
function originalMaterial(material) { return material.kind === "mounted_on_wall" ? material.wall : material; }

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
  if (mode === "fixed") zoneNote(`${wallNames[wall]}「${zoneName(key)}」：固定，在右邊選材料庫的一組；不選就照方案現在的材料`, "ok");
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
  if (!allowed.length) return "准用清單至少要勾一種";
  const optimizable = new Set(libraryEntries("optimizable").map(entry => entry.id));
  return allowed.some(id => id === ZONE_NONE || optimizable.has(id)) ? ""
    : "准用清單只有固定材料：搜尋要一個起點，請再勾一種可調材料或「不貼」";
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
function fixedCell(wall, key, zone, cell) {
  const select = document.createElement("select"); select.id = `zone-fixed-${wall}-${key}`;
  select.setAttribute("aria-label", `${wallNames[wall]} ${zoneName(key)} 的固定材料`);
  const current = libraryLabel(zone.material);
  const now = document.createElement("option"); now.value = ""; now.textContent = `方案現在的材料：${materialWords(zone.material)}`;
  if (!current) select.append(now);
  for (const entry of libraryEntries("fixed")) {
    const option = document.createElement("option"); option.value = entry.id; option.textContent = `${entry.label}（${entry.basis}）`;
    select.append(option);
  }
  select.value = current?.id ?? "";
  select.onchange = () => { if (select.value) action(() => chooseFixed(wall, key, select.value)); };
  cell.append(select);
}
function searchCell(wall, key, zone, setting, cell) {
  const start = document.createElement("p"); start.className = "note";
  start.textContent = `方案裡先放原牆：${wallText(originalMaterial(zone.material))}`;
  const limit = document.createElement("label");
  const box = document.createElement("input"); box.type = "checkbox"; box.id = `zone-limit-${wall}-${key}`;
  box.checked = Array.isArray(setting.allowed);
  // 打勾時先勾滿「預設會換的」（全部可調材料與不貼），再讓人拿掉不要的。
  box.onchange = () => { if (box.checked) zoneAllowedOpen.add(`${wall}/${key}`); changeAllowed(wall, key, box.checked ? [...libraryEntries("optimizable").map(entry => entry.id), ZONE_NONE] : null); };
  limit.append(box, document.createTextNode("限定准用哪幾組材料（不勾＝全部可調材料與不貼都可以換）"));
  cell.append(start, limit);
  if (!box.checked) return;
  // 清單收在一行摘要後面（每一區都攤開二十幾格，整張表讀不下去）；點開過的那幾區重畫後照樣開著。
  const details = document.createElement("details"); details.id = `zone-allowed-box-${wall}-${key}`;
  details.open = zoneAllowedOpen.has(`${wall}/${key}`);
  details.ontoggle = () => { if (details.open) zoneAllowedOpen.add(`${wall}/${key}`); else zoneAllowedOpen.delete(`${wall}/${key}`); };
  const summary = document.createElement("summary"); summary.id = `zone-allowed-summary-${wall}-${key}`;
  const list = document.createElement("div"); list.className = "zone-allowed"; list.id = `zone-allowed-${wall}-${key}`;
  const choices = [...libraryEntries("optimizable").map(entry => [entry.id, `${entry.label}（可調）`]),
                   ...libraryEntries("fixed").map(entry => [entry.id, `${entry.label}（固定，整塊選或不選）`]), [ZONE_NONE, "不貼（保留原牆）"]];
  for (const [id, text] of choices) {
    const wrap = document.createElement("label");
    const input = document.createElement("input"); input.type = "checkbox"; input.id = `zone-allow-${wall}-${key}-${id}`;
    input.checked = setting.allowed.includes(id);
    input.onchange = () => changeAllowed(wall, key, choices.map(([choice]) => choice).filter(choice =>
      choice === id ? input.checked : setting.allowed.includes(choice)));
    wrap.append(input, document.createTextNode(text)); list.append(wrap);
  }
  const names = choices.filter(([id]) => setting.allowed.includes(id)).map(([, text]) => text.replace(/（.*$/, ""));
  summary.textContent = `准用 ${names.length} 種：${names.join("、") || "（沒有）"}（點開改）`;
  details.append(summary, list);
  cell.append(details);
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
function renderZoneTable(wall) {
  const area = $("elevation-zones");
  if (!area) return;
  area.replaceChildren();
  if (!wall || !scheme) return;
  area.append(originalSchemeRow());
  for (const text of [zoneDesignNote, zoneLibraryProblem]) {
    if (!text) continue;
    const line = document.createElement("p"); line.className = "notice"; line.textContent = text; area.append(line);
  }
  const table = document.createElement("table"); table.id = `zone-table-${wall}`; table.className = "zone-table";
  const head = document.createElement("tr");
  for (const text of ["區", "大小（公尺）", "做法", "材料"]) { const cell = document.createElement("th"); cell.textContent = text; head.append(cell); }
  table.append(head);
  for (const zone of zoneList(wall)) {
    const key = zoneKey(zone), setting = zoneSetting(wall, key);
    const row = document.createElement("tr"); row.id = `zone-row-${wall}-${key}`; row.className = `zone-mode-${setting.mode}`;
    const name = document.createElement("td"); name.textContent = zoneName(key);
    const size = document.createElement("td");
    size.textContent = zone.area === "rectangle"
      ? `${Number((zone.u_m[1] - zone.u_m[0]).toFixed(3))} × ${Number((zone.v_m[1] - zone.v_m[0]).toFixed(3))}` : "整面";
    const mode = document.createElement("td"); mode.append(modeSelect(wall, key, setting));
    const material = document.createElement("td"); material.id = `zone-material-${wall}-${key}`;
    if (setting.mode === "fixed") fixedCell(wall, key, zone, material);
    else if (setting.mode === "keep_original") material.textContent = `原牆：${wallText(originalMaterial(zone.material))}`;
    else searchCell(wall, key, zone, setting, material);
    row.append(name, size, mode, material); table.append(row);
  }
  area.append(table);
  const note = document.createElement("p"); note.id = "zone-note"; note.setAttribute("aria-live", "polite");
  note.textContent = zoneNoteText; note.className = zoneNoteKind; area.append(note);
}
window.renderZoneTable = renderZoneTable;
