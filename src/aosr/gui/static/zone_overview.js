/* 六面分區圖與共用編輯框。選區、材料與範圍先存在草稿；按套用才成批改方案。
   限制判決與材料曲線可能晚到。重畫只讀資料，保留正在輸入的值、捲動位置與選區。 */
(() => {
const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_WIDTH = 898;          // 左邊那張圖的寬（px，樣稿）
const WALL_GAP = 26;            // 同一排兩面牆中間的空
const WALL_ROW_HEIGHT = 190;    // 一面牆最高畫多高（px）
const TILE_DEPTH = 190, TILE_WIDTH = 140;  // 地板、天花那兩張小圖最大的深與寬（px）
const WALL_ORDER = ["front", "back", "left", "right"];
const VPR_WORDS = "鋼板共振吸音";
const SYMMETRY_TITLE = "左右對稱";
let overviewOpen = false;
let cutting = null;
let cutGeneration = 0;
let overviewReturnY = 0;        // 打開這一頁之前輸入頁捲到哪裡（回來時捲回去）
let followed = [];              // 由左右對稱代選的區
let selected = [];              // 選了哪幾區（「牆/區代號」，照點的先後）
const familyOpen = new Map();   // 本頁按面別、做法記住使用者展開的材料組。
let draft = null;               // 面板上現在點的：{mode, materialId, ticks, depthText, depthBad}；還沒點是 null（後三格見 untouched）
let draftTouched = false;       // 面板上的選項是使用者自己點的（再加選別區時不重算）
let ruleDraft = false;
let keptRule = null;            // 這次打開方案後定下來的「其他牆面」做法鍵（第一次推出來的，或使用者最後在下拉設的）；換方案才重推
let keptRuleZone = null;        // 剛照規則寫過的一區：下次重畫時規則就是它現在的做法（改規則、改清單之後做法鍵會換）
let seenGeneration = -1;        // 上次畫的時候是第幾份做法（換方案就把選取清掉）
let changing = 0;               // 做法寫完、平面圖檢查還沒回來的改動有幾個（寫在 #zone-page 的 data-changing，考卷等畫面穩用）
let unsaved = false;            // 這一頁改過、還沒存進檔案（標題列那一句；存好了由 zone_modes.js 叫 window.zonePageSaved 拿掉）
let pageSaid = "";              // 輸入頁訊息列最近一句警示（這一頁開著時照抄一份，見 window.zonePageSay）
let keptSaid = "";              // 「拿去搜尋」被還在路上的改動擋下的那一句：留到使用者下一次按東西為止（見 window.zonePageKeep）
let view = null;                // 最近一次算出來的：{roles, room, faces, byId, inferred, list}
let pairing = null;             // 最近一次整理好的左右對稱配對（readPairing）
let symmetryDetail = false;     // 「配對細節」開著（重畫後照樣開著）
const openDetails = new Set();  // 哪幾個摺疊開著（重畫後照樣開著）
const curves = new Map();       // 材料代號 → 吸音曲線的回覆（還在算是 "loading"）

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, typeof value === "number" ? Number(value.toFixed(1)) : value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function button(className, text, id, onclick) {
  const item = el("button", className, text); item.type = "button"; item.id = id; item.onclick = onclick;
  return item;
}
// 摺疊：開著的記在這個檔裡，重畫後照樣開著。
function fold(id, summary) {
  const details = el("details"); details.id = id; details.open = openDetails.has(id);
  details.append(el("summary", null, summary));
  details.firstElementChild.addEventListener("click", event => {
    if (event.defaultPrevented) return;
    if (details.open) openDetails.delete(id); else openDetails.add(id);
  });
  details.ontoggle = () => { if (!details.isConnected) return; if (details.open) openDetails.add(id); else openDetails.delete(id); };
  return details;
}
function shownMetres(value, digits) { return Number(value.toFixed(digits)); }
// 一句話裡第一次出現 VPR 時後面接中文（「VPR（鋼板共振吸音）」）；那個 VPR 後面本來就接著括號說明就不重複。
function vprWords(text) {
  const at = text.indexOf("VPR");
  if (at < 0 || text[at + 3] === "（") return text;
  return `${text.slice(0, at + 3)}（${VPR_WORDS}）${text.slice(at + 3)}`;
}
window.zoneVprWords = vprWords;

// 哪一面是前、後、左、右牆：照伺服器排的牆名認；定不出來（還是座標名）就照座標順序排，圖上照樣寫座標名。
function overviewRoles() {
  const find = word => Object.keys(wallNames).find(wall => wallNames[wall] === word);
  const named = {front: find("前牆"), back: find("後牆"), left: find("左牆"), right: find("右牆")};
  return Object.values(named).every(Boolean) ? named : {front: "x0", back: "xL", left: "y0", right: "yL"};
}
// 一面的每一區（照號碼排）：做法、材料、號碼與名字、畫面上的框（公尺，a 往右、b 往上）、面積、做法鍵（zone_rules.js）、
// 圖上畫哪一種顏色。mode／allowed／fixed 三格是交給推法用的：沒寫清單的由程式最佳化照伺服器的意思（全部可調＋不貼）比；
// 指定貼的是哪一種只看貼上去的（沒貼東西的一律算原牆原樣，跟「不貼」同一個做法）。
// shown：search 由程式最佳化、fixed 指定貼一種（真的有貼東西）、still 保留原樣（不貼，或指定貼但其實沒貼；地毯也算）、
// opening 門窗（沒被改成別的做法時畫斜線）、unknown 裸構造與原牆的關係看不出來。
function faceZones(wall) {
  const {views, wallSize, screenBox} = window.elevationGeometry;
  const size = wallSize(wall), labels = zoneLabels(wall);
  return zoneList(wall).map(zone => {
    const key = zoneKey(zone), setting = zoneSetting(wall, key), label = labels.get(key);
    const entry = zoneLibraryMatch(zone.material, wall, key), special = zoneSpecialName(entry);
    const box = !size ? null : zone.area === "rectangle" ? screenBox(views[wall], size[0], size[1], zone) : [0, size[0], 0, size[1]];
    const pasted = zone.material.kind === "mounted_on_wall";
    const opening = special === "門" || special === "窗";
    const construction = constructionAt(wall, key);
    const fixedId = construction?.state === "picked" ? construction.id : pasted ? entry?.id : null;
    const item = {construction, id: `${wall}/${key}`, wall, key, zone, setting, label, entry, special, opening, box, pasted,
                  area: box ? (box[1] - box[0]) * (box[3] - box[2]) : 0, mode: setting.mode,
                  allowed: setting.allowed ?? serverDefaultAllowed(), ranges: setting.mode === "search" ? setting.ranges ?? null : null,
                  fixed: zoneFixedIdentity(zone.material, fixedId)};
    item.effectiveAllowed = (zoneLimitsFresh() && zoneLimits.candidates?.[wall]?.[key]?.allowed) || item.allowed;
    item.method = zoneMethodKey(item);
    const mode = zoneMethodMode(item.method);
    item.shown = mode === "unknown" || mode === "search" ? mode : mode === "fixed" && !special ? "fixed" : opening ? "opening" : "still";
    return item;
  }).sort((left, right) => left.label.number - right.label.number);
}
// 第 2 段右欄唯讀清單（wall_section.js，#1265 單元 3）照同一份分組：沒貼東西的「指定貼一種」算保留原樣、門窗另認。
window.zoneFaceZones = faceZones;
// 第 2 段右欄每一區那一句「跟誰一對」（wall_section.js，#1265 單元 3 審查真錯 5）：用這一頁同一份配對（readPairing，伺服器的 pairs），
// 不另讀幾何鏡像（partners 只配畫的框、不管做法）。回區代號（牆/區）→ {partner: 那一區} 或 {missed: true}；沒算出配對是空的。
window.zonePairMarks = () => {
  const found = readPairing(new Map(Object.keys(wallNames).flatMap(face => faceZones(face)).map(item => [item.id, item])));
  return found.state === "paired" ? found.marks : new Map();
};
window.zonePartnerWords = item => partnerWords(item);
// 考卷用：照指定大小與配對記號（"pair" 成對、"missed" 沒配成、null 沒有）畫一區，量字與記號排得開不開（#1338）。不改現在的配對。
window.zoneShapeFor = (item, width, height, mark) => {
  const saved = pairing;
  const partner = {...item, wall: item.wall === "y0" ? "yL" : "y0", id: `mirror/${item.id}`};
  pairing = mark ? {marks: new Map([[item.id, mark === "missed" ? {missed: true} : {partner, shape: "hexagon"}]])} : null;
  try { return zoneShape(item, 0, 0, width, height); } finally { pairing = saved; }
};
// 交給推法的那幾區：四面牆照前、後、左、右，每面照號碼，扣掉門窗。
function ruleZones(faces, roles) { return WALL_ORDER.flatMap(role => faces[roles[role]].filter(item => !item.opening)); }
// 六面全部由程式最佳化的區（清單卡管的）：四面牆照前後左右，再來天花、地板。
function searchZones(faces, roles) {
  return [...WALL_ORDER.map(role => roles[role]), "ceiling", "floor"].flatMap(wall => faces[wall]).filter(item => item.mode === "search");
}
function numberBadge(number) { return el("span", "zv-num", String(number)); }
// 「前牆 ① 左聲道喇叭後面那塊」：牆名、圈起來的號碼、位置名（整面一區不必號碼）。
function zoneWhere(item) {
  const line = el("div", "zv-where-line");
  if (item.key === "whole_wall") { line.append(`${wallNames[item.wall]} 整面`); return line; }
  line.append(`${wallNames[item.wall]} `, numberBadge(item.label.number), ` ${item.label.name}`);
  return line;
}
function zoneWhereText(item) {
  return item.key === "whole_wall" ? `${wallNames[item.wall]}整面` : `${wallNames[item.wall]} ${item.label.numberText} ${item.label.name}`;
}
// 好幾區寫成一行：同一面牆的號碼排在一起（「前牆 ①②、左牆 ③」）。
// marked：正在改的那幾區（區的 id）——它們的號碼圈塗黑，讀的人不必猜是哪一個。
function zonesBrief(items, marked = new Set()) {
  const pieces = [];
  const badgeOf = item => { const badge = numberBadge(item.label.number); if (marked.has(item.id)) { badge.classList.add("zv-now"); badge.dataset.zone = item.id; } return badge; };
  for (const wall of new Set(items.map(item => item.wall))) {
    const here = items.filter(item => item.wall === wall);
    pieces.push(here[0].key === "whole_wall" ? [`${wallNames[wall]}整面`] : [`${wallNames[wall]} `, ...here.map(badgeOf)]);
  }
  const line = el("span");
  pieces.forEach((piece, index) => { if (index) line.append("、"); line.append(...piece); });
  return line;
}
// 貼上去的那一種怎麼叫：材料庫認得寫它的名字，認不到寫那幾層。
function pastedWords(item) { return item.construction?.label ?? item.entry?.label ?? layersText(item.zone.material); }
function methodWords(group) {
  if (group.mode === "unknown") return uncertainWords(group.zones[0].construction);
  return group.mode === "fixed" ? vprWords(`${ZONE_MODE_WORDS.fixed}：${pastedWords(group.zones[0])}`) : ZONE_MODE_WORDS[group.mode];
}
function shownOf(mode) { return {search: "search", fixed: "fixed", keep_original: "still", unknown: "unknown"}[mode]; }
function hoverWords(item) {
  const mode = zoneMethodMode(item.method);
  if (mode === "unknown") return `${zoneWhereText(item)}：${uncertainWords(item.construction)}`;
  // 這一區的材料是「鋼板直接貼在棉上」的構造（#1208）：滑過去的說明接那一句（圖上寫不下字的區也看得到）。
  const reference = zoneReferenceOnly(item.zone.material, item.wall, item.key) ? `；${zoneReference.sentence}` : "";
  const behind = mode === "search" && mapBacking(item) ? `，${mapBacking(item)}` : "";
  return `${zoneWhereText(item)}：${mode === "fixed" ? `${ZONE_MODE_WORDS.fixed}：${pastedWords(item)}` : ZONE_MODE_WORDS[mode]}${behind}${reference}`;
}
// 由程式最佳化而背後換成材料庫一筆的區（#1330）：圖上寫「背後 ○○」（短名，同右欄那一格）；沒換背後回 ""。
// 那一筆從材料庫刪掉了就寫代號，不寫成原牆。
function mapBacking(item) {
  const id = zoneSetting(item.wall, item.key).backing_entry;
  if (!id) return "";
  const entry = libraryEntry(id);
  return `背後 ${entry ? backingName(entry) : id}`;
}

/* ───── 左右對稱：伺服器算好的配對整理成畫面要的樣子（不在這裡算誰跟誰一對） ───── */
const PAIR_SHAPES = ["hexagon", "diamond", "triangle", "square", "pentagon", "octagon"];
function symmetryIsOn() { return zoneDesign?.left_right_symmetry !== "off"; }
// state：off 關著、pending 還在等伺服器、unavailable 算不了、no-axis 定不出對稱軸、none 沒有由程式最佳化的區、paired 配對算出來了。
// marks：區的 id → {partner, shape}（成對）或 {missed: true}（沒配成）。回覆只收最新那一次問的（跟現在的切法、做法同一份；
// 還在路上就先寫計算中，不拿上一次的來畫），所以伺服器回的片代號（牆@區）一定對得到現在的區。幾對、幾區各自調都用伺服器回的數。
function readPairing(byId) {
  const on = symmetryIsOn(), found = zoneLimits?.symmetry;
  const base = {on, state: "pending", message: "", marks: new Map(), pairs: [], missed: [], axis: null, alone: 0, searched: 0};
  if (!on) return {...base, state: "off"};
  if (!zoneLimitsFresh()) return base;
  if (!found) return {...base, state: "unavailable", message: `配對沒算成：${zoneLimits.message ?? "限制檢查沒有做成"}`};
  if (!found.available) return {...base, state: "unavailable", message: found.message};
  if (found.reason) return {...base, state: "no-axis", message: found.reason};
  // 判決點名的區一定在現在的切法裡：換區代號的每一條路（切牆、清掉再復原、改房間尺寸、換方案）重畫之前都先標判決在路上
  // （staleZoneLimits），上面 zoneLimitsFresh 就擋掉了。以前這裡另有一道「點名的區找不到就當還在等」（第二趟追加第 1 條），
  // 換方案那一條也標了之後走不到，拿掉（#1265 單元 3 之二收尾）。
  const at = surface => { if (!surface) return null; const [wall, key] = surface.split("@"); return byId.get(`${wall}/${key ?? "whole_wall"}`) ?? null; };
  const pairs = found.pairs.map((pair, index) => ({first: at(pair.first), second: at(pair.second), shape: PAIR_SHAPES[index % PAIR_SHAPES.length]}));
  const missed = found.missed.map(entry => ({...entry, firstItem: at(entry.first), secondItem: at(entry.second), otherItem: at(entry.other)}));
  const marks = new Map();
  for (const pair of pairs) { marks.set(pair.first.id, {partner: pair.second, shape: pair.shape}); marks.set(pair.second.id, {partner: pair.first, shape: pair.shape}); }
  for (const entry of missed) for (const item of [entry.firstItem, entry.secondItem]) if (item) marks.set(item.id, {missed: true});
  return {...base, state: found.searched.length ? "paired" : "none", marks, pairs, missed, axis: found.axis, alone: found.alone, searched: found.searched.length};
}
// 成對的記號：幾種不帶順序的圖形（沒有 1、2 或 A、B），同一對兩區畫同一個。
function pairGlyph(shape, cx, cy, r) {
  const [sides, turn, squeeze] = {hexagon: [6, 0, 1], diamond: [4, 0, 0.78], triangle: [3, 0, 1], square: [4, 45, 1], pentagon: [5, 0, 1], octagon: [8, 22.5, 1]}[shape];
  const points = Array.from({length: sides}, (_, index) => {
    const angle = (turn - 90 + index * 360 / sides) * Math.PI / 180;
    return `${(cx + r * squeeze * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`;
  });
  return svg("polygon", {class: "zv-pair-mark", "data-shape": shape, points: points.join(" ")});
}
function glyphIcon(shape) {
  const icon = svg("svg", {class: "zv-sym-icon", width: 18, height: 18, viewBox: "0 0 18 18", "aria-hidden": "true"});
  icon.append(pairGlyph(shape, 9, 9, 8));
  return icon;
}
function partnerWords(item) { return item.key === "whole_wall" ? `和${wallNames[item.wall]}整面一對` : `和${wallNames[item.wall]}${item.label.numberText}一對`; }
// 夠大的區：號碼挪到左上角、中間放記號與一句話；小的區只在號碼旁放記號（那一句照樣在，只是不顯示）。
function roomyZone(width, height) { return width >= 76 && height >= 96; }
// 配對記號的中心（drawWall 同面一對拉線用；每次重畫重記）。
const glyphAt = new Map();
// 夠大的由程式最佳化區上面幾行字怎麼排：「由程式最佳化」平常在中間偏上；區窄（左上角號碼圈會壓到整句）就挪到號碼圈底下；
// 背後換過再多一行。記號與那一句跟著往下挪（shift）；放不下背後那一行就不寫（滑過去的說明照樣有）。
function searchLayout(item, x, y, width, height) {
  const cy = y + height / 2, mark = pairing?.marks.get(item.id);
  const fits = (top, extra) => {
    const shift = top - (cy - 22) + extra, bottom = !mark ? top + extra + 4 : mark.missed ? cy + 26 + shift : cy + 40 + shift;
    return bottom <= y + height - 2 ? {top, shift} : null;
  };
  const wide = measuredWidth(ZONE_MODE_WORDS.search, 13) / 2 <= width / 2 - 30;
  const top = wide ? cy - 22 : Math.max(cy - 22, y + 44);
  const behind = item.shown === "search" ? mapBacking(item) : "";
  if (behind) { const laid = fits(top, 16); if (laid) return {...laid, behind}; }
  // 連「由程式最佳化」都排不開：寬的照原位；窄的不寫那一句（底色已經看得出，滑過去的說明照樣有），免得壓到號碼圈。
  return {...(fits(top, 0) ?? {top: wide ? cy - 22 : null, shift: 0}), behind: ""};
}
// 一區的配對記號掛到圖上那一區：成對的掛圖形與「和○牆○一對」，沒配成的掛「沒配成一對」。回這一區有沒有掛。
// shift：夠大的區上面多寫了幾行（見 searchLayout），記號與那一句整組往下挪這麼多。
function pairDecor(item, group, x, y, width, height, shift = 0) {
  const mark = pairing?.marks.get(item.id);
  if (!mark) return false;
  const roomy = roomyZone(width, height), cx = x + width / 2, cy = y + height / 2 + (roomy ? shift : 0);
  if (mark.missed) {
    group.setAttribute("data-missed", "true");
    // 放得下一條字的區（夠大的在中間偏下、窄的在號碼底下）掛整條「沒配成一對」；再小的只掛一個小記號。
    const band = roomy || (width >= 60 && height >= 52), top = roomy ? cy + 4 : cy + 14;
    if (band) group.append(svg("rect", {class: "zv-missed-bg", x: x + 1.5, y: top, width: width - 3, height: 22}));
    group.append(svg("text", {class: `zv-pair-words zv-missed-words${band ? "" : " zv-hide"}`, x: cx, y: top + 16}, "沒配成一對"));
    if (!band) group.append(svg("rect", {class: "zv-missed-dot", x: cx - 5, y: Math.min(cy + 13, y + height - 9), width: 10, height: 5}));
    return true;
  }
  group.setAttribute("data-pair", mark.partner.id);
  group.setAttribute("data-shape", mark.shape);
  glyphAt.set(item.id, cy + 6);
  if (roomy) group.append(pairGlyph(mark.shape, cx, cy + 6, 11), svg("text", {class: "zv-pair-words", x: cx, y: cy + 36}, partnerWords(mark.partner)));
  else group.append(pairGlyph(mark.shape, cx, Math.min(cy + 19, y + height - 7), 6), svg("text", {class: "zv-pair-words zv-hide", x: cx, y: cy}, partnerWords(mark.partner)));
  return true;
}

/* ───── 左邊：圖 ───── */
// 字大約多寬（px）：全形一個字高、半形半個多一點；只拿來決定圖上的字放不放得下。
function textWidth(text, size) { return [...text].reduce((sum, char) => sum + (char.charCodeAt(0) > 0x2e7f ? 1 : 0.58), 0) * size; }
// 放不下就從尾巴砍、補「…」（滑過去的說明照樣有整句）。
// 量字用頁面實際的字型（畫布 measureText）；量不到才用 textWidth 估。
let wordRuler = null;
function measuredWidth(text, size) {
  try {
    wordRuler ??= document.createElement("canvas").getContext("2d");
    wordRuler.font = `${size}px ${getComputedStyle(document.getElementById("zone-page") ?? document.body).fontFamily}`;
    return wordRuler.measureText(text).width;
  } catch { return textWidth(text, size); }
}
function fitWords(text, room, size) {
  if (measuredWidth(text, size) <= room) return text;
  const chars = [...text];
  while (chars.length > 1 && measuredWidth(chars.join("") + "…", size) > room) chars.pop();
  return chars.join("") + "…";
}
// 指定貼的那一塊上面寫材料名：照括號與「＋」斷成幾行，第一行大字（放不下就小一號），後面幾行從頭照順序放到放不下為止；
// 第一行都放不下就不寫（滑過去的說明照樣有整個名字）。只寫在另外指定的區上：跟著規則的那一大片不重複寫同一個名字。
function fixedLabel(item, x, y, width, height) {
  const entry = item.construction ?? item.entry;
  const group = svg("g", {class: "zv-fixed-words"});
  if (!entry || !item.named) return group;
  const named = entry.label.split(/[（）]|(?=＋)/).map(part => part.trim()).filter(Boolean);
  // 第一行就是「VPR」而名字自己沒接著說明：底下加一行小字說它是什麼（同右欄的「VPR（鋼板共振吸音）」）。
  if (named[0] === "VPR" && vprWords(entry.label) !== entry.label) named.splice(1, 0, VPR_WORDS);
  const room = Math.max(0, Math.floor((height - 28) / 16));
  // 名字最多用 nameRoom 行：第一行大字（放不下就小一號、再不行照空白折），後面幾行從頭照順序放到放不下為止。
  const laid = nameRoom => {
    const lines = [...named];
    let big = [20, 15, 13].find(size => textWidth(lines[0], size) <= width - 4);
    if (!big) {
      // 第一行整句放不下：照空白折成幾行。
      const wrapped = [];
      for (const word of lines[0].split(" ")) {
        const joined = wrapped.length ? `${wrapped.at(-1)} ${word}` : word;
        if (wrapped.length && textWidth(joined, 15) <= width - 10) wrapped[wrapped.length - 1] = joined; else wrapped.push(word);
      }
      // 折完要整句都寫得下（每一行夠寬、行數夠高）才寫：寫一半的名字比不寫更容易看錯。
      if (wrapped.length > 1 && wrapped.length <= nameRoom && wrapped.every((line, index) => textWidth(line, index ? 13 : 15) <= width - 10)) { lines.splice(0, 1, ...wrapped); big = 15; }
    }
    if (!big || nameRoom < 1) return null;
    const fits = [lines[0]];
    for (const line of lines.slice(1)) { if (fits.length >= nameRoom || textWidth(line, 13) > width - 10) break; fits.push(line); }
    return {big, fits};
  };
  // 這一區貼的是「鋼板直接貼在棉上」的構造（#1208）：材料字下面多兩行（伺服器給的字），面板關著也看得到。
  // 這兩行用 12 px（這一頁最小的字級；1.1 公尺寬的一塊板寫得下）。行數不夠時這兩行先留、名字後面的幾行讓出來；連名字第一行都擠不下就不留這兩行（滑過去的說明照樣有那一句）。
  const marks = zoneReferenceOnly(item.zone.material, item.wall, item.key) ? zoneReference.map_words : [];
  const marked = marks.length && marks.every(line => textWidth(line, 12) <= width - 4) ? laid(room - marks.length) : null;
  const found = marked ?? laid(room);
  if (!found) return group;
  const {big, fits} = found, noted = marked ? marks : [];
  const top = y + height / 2 - (fits.length + noted.length - 1) * 9;
  fits.forEach((line, index) => group.append(svg("text", {class: index ? "zv-fixed-sub" : "zv-fixed-label", "font-size": index ? 13 : big,
    x: x + width / 2, y: top + (index ? 2 + index * 19 : 0)}, line)));
  noted.forEach((line, index) => group.append(svg("text", {class: "zv-fixed-sub zv-ref-map", "font-size": 12,
    x: x + width / 2, y: top + 6 + (fits.length + index) * 19 - (index ? 2 : 0)}, line)));
  return group;
}
function badge(cx, cy, number, small) {
  const group = svg("g", {class: `zv-badge${small ? " small" : ""}`});
  group.append(svg("circle", {cx, cy, r: small ? 7.3 : 10}), svg("text", {x: cx, y: cy + 4}, String(number)));
  return group;
}
// 圖上一區的外殼：點一下選、再點一下取消；選到的加一圈白框。
function zoneGroup(item, extra, rect) {
  const chosen = selected.includes(item.id);
  const group = svg("g", {class: `zv-zone ${extra}${item.shown}${chosen ? " selected" : ""}`, "data-wall": item.wall, "data-zone": item.key,
                          "data-shown": item.shown, "data-selected": String(chosen)});
  group.setAttribute("data-apart", String(!!item.apart));
  group.append(svg("rect", {class: "zv-fill", ...rect}), svg("title", {}, hoverWords(item)));
  group.addEventListener("click", () => toggleSelected(item.id));
  return group;
}
// 另外指定的區（做法跟「其他牆面」不同）：一圈虛線內框。顏色一樣、名字寫不下的也看得出來；圖例有講。
function apartFrame(rect) {
  const inset = Math.min(4, rect.width / 4, rect.height / 4);
  return svg("rect", {class: "zv-apart", x: rect.x + inset, y: rect.y + inset, width: rect.width - 2 * inset, height: rect.height - 2 * inset});
}
function selectedRing(rect) {
  const inset = Math.min(3.5, rect.width / 4, rect.height / 4);
  return svg("rect", {class: "zv-ring", x: rect.x + inset, y: rect.y + inset, width: rect.width - 2 * inset, height: rect.height - 2 * inset});
}
// 一區：底色照狀態；門窗加斜線與名字；指定貼的寫材料名、號碼放左上角；其餘號碼放正中間。
function zoneShape(item, x, y, width, height) {
  const group = zoneGroup(item, "", {x, y, width, height});
  const cx = x + width / 2, cy = y + height / 2, small = Math.min(width, height) < 22;
  // 選到的那一圈白框先畫、號碼與字後畫（#1199）：窄條區的號碼圈比區還寬，白框後畫會把號碼蓋掉。
  const ring = selected.includes(item.id) ? [selectedRing({x, y, width, height})] : [];
  let shift = 0;
  if (item.shown === "opening") {
    group.append(svg("rect", {class: "zv-hatch", x, y, width, height}), ...ring, svg("rect", {class: "zv-tagbg", x: cx - 29, y: cy - 15, width: 58, height: 30, rx: 3}),
      badge(cx - 13, cy, item.label.number, false), svg("text", {class: "zv-open-label", x: cx + 12, y: cy + 6}, item.special));
  } else if (item.shown === "fixed" && width >= 44 && height >= 44) {
    group.append(...ring, badge(x + 16, y + 16, item.label.number, false), fixedLabel(item, x, y, width, height));
  } else if (roomyZone(width, height) && (pairing?.marks.has(item.id) || (item.shown === "search" && mapBacking(item)))) {
    // 夠大的由程式最佳化區有配對記號或換過背後：號碼挪左上角、中間寫「由程式最佳化」，背後換過的底下再寫「背後 ○○」。
    const lay = searchLayout(item, x, y, width, height);
    group.append(...ring, badge(x + 16, y + 16, item.label.number, false));
    if (lay.top !== null) group.append(svg("text", {class: "zv-search-label", x: cx, y: lay.top}, ZONE_MODE_WORDS.search));
    if (lay.behind) group.append(svg("text", {class: "zv-backing-label", x: cx, y: lay.top + 16}, fitWords(lay.behind, width - 8, 12)));
    shift = lay.shift;
  } else group.append(...ring, badge(cx, cy, item.label.number, small));
  pairDecor(item, group, x, y, width, height, shift);
  if (item.apart) group.append(apartFrame({x, y, width, height}));
  return group;
}
// 標題旁的小平面圖：方向同地板圖，這面牆那一邊描粗。
function miniPlan(side, x, y, width, height) {
  const [x1, y1, x2, y2] = {top: [x, y, x + width, y], bottom: [x, y + height, x + width, y + height],
                            left: [x, y, x, y + height], right: [x + width, y, x + width, y + height]}[side];
  return [svg("rect", {class: "zv-mini", x, y, width, height}), svg("line", {class: "zv-mini-side", x1, y1, x2, y2})];
}
// 地板線下面的小記號：喇叭、主位落在這面牆的哪裡（前牆寫哪一支喇叭；兩側牆寫喇叭與主位；後牆不寫）。
function wallMarks(role, wall, ox, floorY, scale) {
  const {views, wallSize, flip, project, markerItems} = window.elevationGeometry;
  if (role === "back") return [];
  const [width, height] = wallSize(wall);
  const placed = [];
  for (const item of markerItems()) {
    if (item.kind === "seat" && role === "front") continue;
    const [a] = flip(views[wall], width, height, ...project(wall, item.point));
    const x = ox + a * scale;
    const words = item.kind === "seat" ? "主位" : role === "front" ? item.name.replace("聲道", "") : "喇叭";
    if (placed.some(other => other.words === words && Math.abs(other.x - x) < 2)) continue;
    placed.push({x, words, kind: item.kind});
  }
  // 字平常寫在記號右邊；太靠右緣會壓到那一頭的「地板」兩個字，就改寫在左邊。
  const crowded = x => x + 11 + textWidth(words(x), 11.5) > ox + width * scale - 34;
  const words = x => placed.find(item => item.x === x).words;
  return placed.flatMap(({x, kind}) => [
    ...(kind === "speaker" ? [svg("rect", {class: "zv-spk", x: x - 6, y: floorY + 6, width: 12, height: 9, rx: 1.5})]
      : [svg("circle", {class: "zv-seat", cx: x, cy: floorY + 10.5, r: 5}), svg("circle", {class: "zv-seat-dot", cx: x, cy: floorY + 10.5, r: 1.6})]),
    svg("text", {class: "zv-mark-label", x: crowded(x) ? x - 11 : x + 11, y: floorY + 15, "text-anchor": crowded(x) ? "end" : "start"}, words(x))]);
}
function drawWall(map, role, wall, items, ox, oy, scale, miniSize) {
  const {views, wallSize} = window.elevationGeometry;
  const [width, height] = wallSize(wall);
  const face = svg("g", {class: "zv-face", "data-face": wall});
  for (const item of items) {
    const [a0, a1, b0, b1] = item.box;
    face.append(zoneShape(item, ox + a0 * scale, oy + (height - b1) * scale, (a1 - a0) * scale, (b1 - b0) * scale));
  }
  const floorY = oy + height * scale;
  // 同一面裡成對的兩區（夠大、有地方放記號的）中間拉一條線。
  for (const pair of pairing?.pairs ?? []) {
    if (pair.first.wall !== wall || pair.second.wall !== wall) continue;
    const spots = [pair.first, pair.second].map(item => { const [a0, a1, b0, b1] = item.box;
      return {x: ox + (a0 + a1) / 2 * scale, y: glyphAt.get(item.id) ?? oy + (height - (b0 + b1) / 2) * scale + 6, roomy: roomyZone((a1 - a0) * scale, (b1 - b0) * scale)}; })
      .sort((left, right) => left.x - right.x);
    if (spots.every(spot => spot.roomy)) face.append(svg("line", {class: "zv-pair-link", x1: spots[0].x + 14, y1: spots[0].y, x2: spots[1].x - 14, y2: spots[1].y}));
  }
  // 對稱軸在前後牆上是一條直線（伺服器給橫向落在幾公尺）。
  if (pairing?.axis?.end_walls.includes(wall)) {
    const [a] = window.elevationGeometry.flip(views[wall], width, height, pairing.axis.across_m, 0);
    const axis = svg("g", {class: "zv-sym-axis"});  // 線與字包成一組（一條直線自己沒有寬度）
    axis.append(svg("line", {x1: ox + a * scale, x2: ox + a * scale, y1: oy - 5, y2: floorY + 8}),
      svg("text", {class: "zv-sym-axis-label", x: ox + a * scale + 6, y: oy + 15}, "對稱軸"));  // 字放牆內上緣（下緣外是喇叭、主位的字）
    face.append(axis);
  }
  face.append(svg("rect", {class: "zv-outline", x: ox, y: oy, width: width * scale, height: height * scale}),
    svg("line", {class: "zv-floorline", x1: ox - 1, x2: ox + width * scale + 1, y1: floorY + 1.5, y2: floorY + 1.5}),
    svg("text", {class: "zv-quiet", x: ox + width * scale, y: floorY + 18, "text-anchor": "end"}, wallNames[views[wall].bottom]),
    ...miniPlan({front: "top", back: "bottom", left: "left", right: "right"}[role], ox + 2, oy - 5.5 - miniSize[1], ...miniSize),
    ...wallMarks(role, wall, ox, floorY, scale));
  const title = svg("text", {class: "zv-face-name big", id: `zone-face-title-${wall}`, x: ox + miniSize[0] + 10, y: oy - 8}, wallNames[wall]);
  face.append(title);
  map.append(face);
  // 「切區 ▸」與「選整面」放哪（#1199、#1262）：這面牆名字那一行的最右邊（疊在圖上的真按鈕，見 faceTools）。
  return {wall, right: ox + width * scale, top: oy - 27, titleRow: true};
}
// 地板圖與天花圖（同一個方向：上邊前牆、左邊左牆，從上往下看）。point(x, y)：房間座標 → 小圖上的 px。
function tilePlan(roles, room, px, py, scale) {
  const depthIsX = roles.front[0] === "x";
  const depth = depthIsX ? room.Lx : room.Ly, across = depthIsX ? room.Ly : room.Lx;
  const frontAtZero = roles.front[1] === "0", leftAtZero = roles.left[1] === "0";
  const point = (x, y) => {
    const along = depthIsX ? x : y, sideways = depthIsX ? y : x;
    return [px + (leftAtZero ? sideways : across - sideways) * scale, py + (frontAtZero ? along : depth - along) * scale];
  };
  return {point, x: px, y: py, width: across * scale, height: depth * scale};
}
function tileRect(plan, zone, room) {
  const [u0, u1] = zone.area === "rectangle" ? zone.u_m : [0, room.Lx], [v0, v1] = zone.area === "rectangle" ? zone.v_m : [0, room.Ly];
  const [xa, ya] = plan.point(u0, v0), [xb, yb] = plan.point(u1, v1);
  return {x: Math.min(xa, xb), y: Math.min(ya, yb), width: Math.abs(xb - xa), height: Math.abs(yb - ya)};
}
function drawTile(map, wall, items, plan, roles, room, faces, heading, note) {
  const {markerItems} = window.elevationGeometry;
  const face = svg("g", {class: "zv-face", "data-face": wall});
  const px = plan.x, py = plan.y;
  for (const item of items) {
    const rect = tileRect(plan, item.zone, room);
    const group = zoneGroup(item, "tile ", rect);
    if (item.special === "地毯") {
      group.append(svg("rect", {class: "zv-rug", ...rect}),
        svg("text", {class: "zv-tiny", x: rect.x + rect.width / 2, y: rect.y + rect.height - 6, "text-anchor": "middle"}, "地毯"));
    }
    if (selected.includes(item.id)) group.append(selectedRing(rect));
    // 地板、天花沒有號碼與做法字，背後換過的在上緣寫「背後 ○○」（放得下才寫，滑過去的說明照樣有）。
    const behind = item.shown === "search" ? mapBacking(item) : "";
    if (behind && rect.width >= 60 && rect.height >= 40) {
      group.append(svg("text", {class: "zv-backing-label", x: rect.x + rect.width / 2, y: rect.y + 18}, fitWords(behind, rect.width - 8, 12)));
    }
    pairDecor(item, group, rect.x, rect.y, rect.width, rect.height);
    face.append(group);
  }
  if (pairing?.axis) {
    const [[x1, y1], [x2, y2]] = pairing.axis.line_m.map(([x, y]) => plan.point(x, y));
    face.append(svg("line", {class: "zv-sym-axis", x1, y1, x2, y2}));
  }
  if (wall === "floor") {
    for (const item of markerItems()) {
      const [x, y] = plan.point(item.point[0], item.point[1]);
      if (item.kind === "speaker") face.append(svg("rect", {class: "zv-spk", x: x - 6, y: y - 4.5, width: 12, height: 9, rx: 1.5}));
      else face.append(svg("circle", {class: "zv-seat", cx: x, cy: y, r: 5.5}), svg("circle", {class: "zv-seat-dot", cx: x, cy: y, r: 1.8}),
        svg("text", {class: "zv-tiny", x, y: y + 17, "text-anchor": "middle"}, "主位"));
    }
    // 門、窗在邊上的位置（對照用）：牆上那一區的橫向範圍投到地板圖的那一邊。
    for (const role of WALL_ORDER) {
      const side = roles[role];
      for (const item of faces[side].filter(zone => zone.opening && zone.zone.area === "rectangle")) {
        const [u0, u1] = item.zone.u_m, fixedAt = side[1] === "0" ? 0 : side[0] === "x" ? room.Lx : room.Ly;
        const [x1, y1] = side[0] === "x" ? plan.point(fixedAt, u0) : plan.point(u0, fixedAt);
        const [x2, y2] = side[0] === "x" ? plan.point(fixedAt, u1) : plan.point(u1, fixedAt);
        face.append(svg("line", {class: "zv-edge-open", x1, y1, x2, y2}));
        const outside = {front: [0, -5, "middle"], back: [0, 14, "middle"], left: [-6, 4, "end"], right: [6, 4, "start"]}[role];
        face.append(svg("text", {class: "zv-tiny", x: (x1 + x2) / 2 + outside[0], y: (role === "left" || role === "right" ? Math.min(y1, y2) + 10 : (y1 + y2) / 2 + outside[1]),
          "text-anchor": outside[2]}, item.special));
      }
    }
  }
  if (note) {
    face.append(svg("text", {class: "zv-still-word", x: px + plan.width / 2, y: py + plan.height * 0.3, "text-anchor": "middle"}, note[0]),
      svg("text", {class: "zv-tiny", x: px + plan.width / 2, y: py + plan.height * 0.3 + 18, "text-anchor": "middle"}, note[1]));
  }
  face.append(svg("rect", {class: "zv-outline", x: px, y: py, width: plan.width, height: plan.height}),
    svg("text", {class: "zv-edge-name", x: px + plan.width / 2, y: py + 12, "text-anchor": "middle"}, wallNames[roles.front]),
    svg("text", {class: "zv-edge-name", x: px + plan.width / 2, y: py + plan.height - 5, "text-anchor": "middle"}, wallNames[roles.back]),
    svg("text", {class: "zv-edge-name", x: px + 5, y: py + plan.height / 2 + 4}, wallNames[roles.left]),
    svg("text", {class: "zv-edge-name", x: px + plan.width - 5, y: py + plan.height / 2 + 4, "text-anchor": "end"}, wallNames[roles.right]));
  const title = svg("text", {class: "zv-face-name", id: `zone-face-title-${wall}`, x: px, y: py - 8}, wallNames[wall]);
  title.append(svg("tspan", {class: "zv-quiet", dx: 10}, heading));
  face.append(title);
  map.append(face);
  // 小圖窄，名字那一行放不下鈕：「切區 ▸」放小圖下緣外的右邊（#1262），讓過後牆門、窗那一行字（下緣外 14 px 的基線）。
  return {wall, right: px + plan.width, top: py + plan.height + 22, titleRow: false};
}
function drawMap(faces, roles, room) {
  const {wallSize} = window.elevationGeometry;
  const frontWidth = wallSize(roles.front)[0], sideWidth = wallSize(roles.left)[0];
  const scale = Math.min((MAP_WIDTH - TILE_WIDTH - 3 * WALL_GAP) / (frontWidth + sideWidth), WALL_ROW_HEIGHT / room.Lz, 80);
  const tileScale = Math.min(TILE_DEPTH / wallSize(roles.left)[0], TILE_WIDTH / frontWidth, 30);
  const rowHeight = room.Lz * scale;
  const row1 = 30, row2 = row1 + rowHeight + 56;
  const col1 = 2, col2 = col1 + frontWidth * scale + WALL_GAP;
  const tileWidth = frontWidth * tileScale, tileDepth = sideWidth * tileScale;
  const tileY2 = row1 + tileDepth + 83;
  const height = Math.max(row2 + rowHeight + 32, tileY2 + tileDepth + 55);  // 小圖底下留後牆門窗那一行字、再一顆「切區 ▸」的高
  const map = svg("svg", {id: "zone-map", class: "zv-map", width: MAP_WIDTH, height, viewBox: `0 0 ${MAP_WIDTH} ${height}`,
                          role: "img", "aria-label": "四面牆、地板、天花"});
  const defs = svg("defs");
  const pattern = svg("pattern", {id: "zv-hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"});
  pattern.append(svg("line", {x1: 0, y1: 0, x2: 0, y2: 7, stroke: getComputedStyle(document.getElementById("zone-page")).getPropertyValue("--still-mark").trim() || "#7d8a8c", "stroke-width": 1.6}));
  defs.append(pattern); map.append(defs);
  const miniSize = [15, 15 * sideWidth / frontWidth];
  // 樣稿的排法：上排前牆、右牆，下排後牆、左牆（前後同寬排同一欄）。
  const places = [];
  for (const [role, ox, oy] of [["front", col1, row1], ["right", col2, row1], ["back", col1, row2], ["left", col2, row2]]) {
    places.push(drawWall(map, role, roles[role], faces[roles[role]], ox, oy, scale, miniSize));
  }
  // 兩張小圖靠右排，左邊留給圖例與「左右對稱」那一塊。
  const tileX = MAP_WIDTH - tileWidth - 18;
  const allStill = faces.ceiling.every(item => item.shown === "still");
  places.push(drawTile(map, "floor", faces.floor, tilePlan(roles, room, tileX, row1, tileScale), roles, room, faces, "從上往下看", null),
    drawTile(map, "ceiling", faces.ceiling, tilePlan(roles, room, tileX, tileY2, tileScale), roles, room, faces, "方向同地板圖",
      allStill && faces.ceiling.length === 1 ? ["保留原樣", originalFace("ceiling")] : null));
  return {map, legendTop: height, places, underWidth: MAP_WIDTH};
}
function legend() {
  const box = el("div", "zv-legend");
  const entries = [["search", ZONE_MODE_WORDS.search], ["fixed", ZONE_MODE_WORDS.fixed], ["still", "保留原樣"], ["hatch", "門、窗"]];
  if ([...view.byId.values()].some(item => item.shown === "unknown")) entries.push(["unknown", "看不出"]);
  for (const [className, words] of entries) {
    const item = el("span"); item.append(el("i", `zv-sw ${className}`), words); box.append(item);
  }
  const apart = el("span"); apart.id = "zone-legend-apart"; apart.append(el("i", "zv-sw apart"), "虛線框：另外指定（跟其他牆面不同）"); box.append(apart);
  return box;
}

/* ───── 選取與改動 ───── */
function chosenItems() { return selected.map(id => view.byId.get(id)).filter(Boolean); }
// 小框裡由程式最佳化那一塊的草稿（#1199），套用才寫進設計檔，取消就丟：
// ticks：使用者在小框裡動過的勾（材料代號 → add 只補、only 只留可貼的區、remove 拿掉）；沒動過的代號不在裡面，套用時那一種各區照目前的；
// depthText：「最深」那一格打的字（公分；沒動過是 null）；depthBad：打的不成數字；
// ranges：範圍格子打的字（材料代號 → {porous: [從, 到], cavity: [從, 到]}，棉厚是下拉的值、空腔是公分，沒動過的那一格是 null）；
// rangeRefused：套用前伺服器判範圍紅的那幾句（改了任何一格就清掉）。
function untouched() { return {modeChanged: false, materialChanged: false, ticks: {}, history: [], expanded: [], query: null, notice: "", order: null, depthText: null, depthBad: false, ranges: {}, rangeRefused: ""}; }
// 面板一打開先點在哪一項：選到的區是同一種做法就點那一項（指定貼的：貼的都是同一筆才帶出那一筆），不一樣就都不點。
function startingDraft() {
  const items = chosenItems();
  const modes = new Set(items.map(item => zoneMethodMode(item.method)));
  if (!items.length || modes.size !== 1) return null;
  const mode = [...modes][0], sameMethod = new Set(items.map(item => item.method)).size === 1;
  if (mode === "unknown") return null;
  return {...untouched(), mode, materialId: mode === "fixed" && sameMethod ? libraryLabel(items[0].zone.material, items[0].wall, items[0].key)?.id ?? "" : ""};
}
let draftRevision = 0;
let libraryRead = 0;
function markDraftEdited() {
  draftRevision++;
  if (draft.libraryRetry) {
    delete draft.libraryRetry;
    $("zone-library-retry")?.parentElement.remove();
  }
  draftTouched = true;
}
function select(ids, partners = []) {
  draftRevision++;
  followed = partners;
  selected = ids; draftTouched = false; ruleDraft = false;
  openDetails.delete("zone-panel-zones"); openDetails.delete("zone-panel-ref-why");
  draft = startingDraft();
  renderZoneOverview();
}
// 換成選這幾區：面板上已經自己點過選項就留著（只換選了哪幾區），沒點過就照新選的那幾區重新起頭。
async function pick(next, toggled = null) {
  if (symmetryIsOn() && zoneLimitsReceived !== zoneLimitsGeneration) {
    const generation = zoneDesignGeneration, beforeCut = cutGeneration;
    await settledZoneLimits();
    if (!overviewOpen || zoneDesignGeneration !== generation || cutGeneration !== beforeCut) return;
  }
  const partners = symmetryIsOn() ? zoneLimits?.symmetry?.partners : null;
  const partner = id => partners?.[id.replace("/", "@")]?.replace("@", "/");
  if (toggled) {
    if (!view.byId.has(toggled)) return;
    next = selected.includes(toggled) ? selected.filter(id => id !== toggled && id !== partner(toggled)) : [...selected, toggled];
  }
  const added = [];
  for (const id of next) {
    const other = partner(id);
    if (other && view.byId.has(other) && !next.includes(other) && !added.includes(other)) added.push(other);
  }
  next = [...next, ...added];
  const followers = toggled ? followed.filter(id => next.includes(id)).concat(added) : added;
  if (next.length === selected.length && next.every(id => selected.includes(id)) && sameList(followed, followers)) return;
  if (draftTouched && next.length) {
    draftRevision++;
    selected = next; followed = followers;
    const message = "換了選區，剛才清單上還沒套用的改動放掉了；要的話再做一次，會套到全部選中的區";
    const discarded = draft.notice === message || Object.keys(draft.ticks).length || Object.keys(draft.ranges).length;
    const inheritDepth = draft.depthText !== null && !draft.modeChanged && !discarded;
    const common = inheritDepth ? startingDraft() : draft;
    draft = {...untouched(), mode: common?.mode ?? null, modeChanged: draft.modeChanged, materialChanged: draft.materialChanged,
             materialId: common?.materialId ?? "", depthText: draft.depthText, depthBad: draft.depthBad};
    if (discarded) draft.notice = message;
    renderZoneOverview(); return;
  }
  select(next, followers);
}
function toggleSelected(id) { pick([], id); }
// 一面牆的「整面」（#1199）：那一面全部的區扣掉門、窗（跟推「其他牆面」時扣的一樣），照號碼排。
function wholeFace(wall) { return view.faces[wall].filter(item => !item.opening).map(item => item.id); }
// 每一面一組鈕（疊在圖上，靠右對齊 place.right）：「切區 ▸」在本頁立面圖畫框；
// 四面牆再加「選整面」（#1199），一次選那一面全部的區。
// 整面都是門窗的牆沒有「選整面」，「切區 ▸」照樣有（門窗也要能重切）。
function byFace(zones) {
  return [...new Set(zones.map(item => item.wall))].map(wall => {
    const here = zones.filter(item => item.wall === wall).sort((left, right) => left.label.number - right.label.number);
    const face = WALL_ORDER.some(role => view.roles[role] === wall) ? wholeFace(wall) : [];
    return {wall, zones: here, whole: face.length > 1 && face.length === here.length && here.every(item => face.includes(item.id))};
  });
}
function faceTools(places) {
  return places.map(place => {
    const tools = el("div", "zv-face-tools"); tools.id = `zone-tools-${place.wall}`; tools.dataset.wall = place.wall;
    if (place.titleRow) tools.dataset.titleRow = "true";
    const cut = button("zv-whole", "切區 ▸", `zone-cut-${place.wall}`, () => cutFace(place.wall));
    cut.setAttribute("aria-label", `${wallNames[place.wall]}切區：在這一頁的立面圖畫框`);
    tools.append(cut);
    if (place.titleRow && wholeFace(place.wall).length) {
      const whole = button("zv-whole", "選整面", `zone-whole-${place.wall}`, () => pick(wholeFace(place.wall)));
      whole.setAttribute("aria-label", `${wallNames[place.wall]}選整面`);
      tools.append(whole);
    }
    tools.style.left = `${place.right.toFixed(1)}px`; tools.style.top = `${place.top.toFixed(1)}px`;
    return tools;
  });
}
// 整塊搬，讓原有的拖曳、數字格與延後重畫監聽仍共用同一個立面圖。
function cutFace(wall) {
  const page = $("zone-page"), map = page.querySelector(".zv-mapwrap");
  const panel = $("elevation-panel"), home = document.createComment("立面圖原位");
  const toolbar = panel.querySelector(".walls-row");
  const title = el("strong", "zv-cut-title", `${wallNames[wall]}　切區`);
  let pendingAtPress = null;
  const done = button("zv-cut-done", "切好了", "zone-finish-cut", event => {
    if (event.detail && pendingAtPress?.canFinish === false) return;
    finishCut();
  });
  // 等完整點擊才讓數字格失焦，避免滑鼠還沒放開，拒收的回覆就先到了。
  done.onmousedown = event => {
    pendingAtPress = cutting.pending;
    if (document.activeElement?.matches("#elevation-selected input")) event.preventDefault();
  };
  cutGeneration++;
  cutting = {wall, map, home, title, done, revision: editRevision, pending: false, finishRequested: false};
  panel.before(home);
  map.replaceWith(panel);
  toolbar.prepend(title);
  toolbar.append($("elevation-clear-slot"), done);
  // 舊選取顯示到切完，再按新座標核對；這段時間不能把草稿套到已重編的代號。
  for (const region of page.querySelectorAll(".zv-side, .zv-foot")) {
    region.inert = true;
    region.setAttribute("aria-disabled", "true");
  }
  window.showElevationWall(wall);
  done.focus({preventScroll: true});
}

function restoreElevation() {
  const {map, home, title, done, revision} = cutting;
  if (window.elevationDrawing()) renderElevation();  // 中途離開：撤掉指標監聽，放開不會送出
  const panel = $("elevation-panel");
  panel.replaceWith(map);
  home.replaceWith(panel);
  $("elevation-wall-line").append($("elevation-clear-slot"));
  title.remove(); done.remove();
  if (revision !== editRevision) unsaved = true;
  renderElevation();
  cutting = null;
}

// 送出中保留完成鈕。從它逐層往外鎖住旁支，解鎖時保留切區原本就鎖住的右欄與底列。
window.lockZoneCut = () => {
  const page = $("zone-page"), editor = cutting;
  if (!editor) {
    page.inert = true;
    return () => { page.inert = false; };
  }
  const request = {};
  editor.pending = request;
  const locked = [];
  for (let branch = editor.done; branch !== page; branch = branch.parentElement) {
    for (const sibling of branch.parentElement.children) {
      if (sibling === branch || sibling.inert) continue;
      sibling.inert = true;
      locked.push(sibling);
    }
  }
  return canFinish => {
    request.canFinish = canFinish;
    for (const element of locked) element.inert = false;
    editor.pending = false;
    if (cutting !== editor) return;
    if (canFinish && editor.finishRequested) finishCut();
    else editor.done.focus({preventScroll: true});
    editor.finishRequested = false;
  };
};

function finishCut() {
  const field = document.activeElement;
  if (field?.matches("#elevation-selected input") && field.value !== field.defaultValue) {
    cutting.done.focus({preventScroll: true});  // 失焦會送出數字；格子本身拒收時留下原因
    if (!cutting.pending) return;
  }
  if (cutting.pending) { cutting.finishRequested = true; return; }
  const invalid = $("elevation-selected").querySelector('[aria-invalid="true"]');
  if (invalid) { invalid.focus(); return; }
  if (inputBusy) return;
  const wall = cutting.wall;
  restoreElevation();
  renderZoneOverview();
  $(`zone-cut-${wall}`)?.focus({preventScroll: true});
}

document.addEventListener("keydown", event => {
  if (cutting && event.key === "Escape") {
    event.preventDefault();
    if (!event.repeat) finishCut();
  }
});
// 牆名那一行讓出右邊那組鈕（#1262）：「面向它站…」壓到鈕就換短句，還壓到就不寫。量畫出來的字（字型各台不同，估的不準）。
// 這一面由程式最佳化時寫哪一份清單：照畫面上那一份，這一面貼不了的拿掉（材料庫找不到的代號原樣留著）；一種都不剩就只放不貼。
// 「鋼板直接貼在棉上」的構造不跟著帶（#1208，carriedList）：拿掉它之後只剩不貼或空了，改用預設的三種玻璃棉＋不貼。
function listForWall(wall, list) {
  if (!serverDefaultAllowed()) return null;
  if (!list) return startingAllowed(wall);
  const fit = carriedList(list, id => fitsHere(id, wall) || (id !== ZONE_NONE && !libraryEntry(id)));
  if (!fit) return defaultAllowed(wall);
  return fit.length ? fit : [ZONE_NONE];
}
// 最多區用的那一份清單（清單卡第一張列的）；現在沒有哪一區由程式最佳化就是 null（新的區照「三種玻璃棉＋不貼」起）。
function shownList() { return view.list.common?.zones[0].allowed ?? null; }
// 一區自己的清單（小框的勾從這裡開始）：已經由程式最佳化的就是它存的那一份，原樣、不照這一面濾；
// 還不是的，是改成由程式最佳化時會放的那一份（最多區用的那一份照這一面能貼的濾過）。材料庫沒讀全、列不出來時回 null。
function ownList(item) { return item.mode === "search" ? item.allowed : listForWall(item.wall, shownList()); }
// 一區照小框動過的勾加減之後的清單：拿掉的從它自己的清單拿掉；勾上的，它還沒有、這一面又貼得了才加在最後面。
// 沒動過的代號原封不動（這一面貼不了的、材料庫找不到的照樣留著），順序也不動。
function tickedList(item, ticks) {
  const own = ownList(item);
  if (!own) return null;
  const actionFor = id => ticks[id]?.[item.id];
  const next = own.filter(id => actionFor(id) !== "remove" && (actionFor(id) !== "only" || fitsHere(id, item.wall)));
  for (const id of Object.keys(ticks)) {
    const action = actionFor(id);
    if (action && action !== "remove" && !next.includes(id) && fitsHere(id, item.wall)) next.push(id);
  }
  return next;
}
// 把好幾區一次改成同一個做法。method：{mode, materialId（指定貼哪一筆；空的＝照現在貼的不換）, allowed（由程式最佳化的清單：
// 規則那條路用，照每一面能貼的濾過再寫）, lists（小框那條路用：區 → 那一區要寫的清單；undefined＝那一區的清單不碰）,
// depth（最深：undefined 不碰、null 拿掉、數字寫公尺）, ranges（小框那條路用，#1199：區 → 那一區的可調範圍；null 拿掉、
// 沒給或 undefined 不碰）}。
// 方案那幾區的材料先交伺服器一批換（指定貼：貼上那一筆；其餘兩種：放回原樣），全部成了才改做法；不成就一區都不動、照講。
// 等伺服器回那一批的時候整頁鎖住（inert：點圖、按套用都沒作用），所以同時只會有一批在路上。
// written：做法寫完之後、第一次重畫之前要同步做的事（這一頁自己的變數在這裡改；平面圖檢查回來之後不准再寫）。
async function applyMethod(items, method, written = () => {}) {
  const generation = zoneDesignGeneration;
  const drafted = items.find(item => measuredDrafts[item.wall]);
  if (drafted) { zoneNote(`${wallNames[drafted.wall]}有還沒套用的實測阻抗草稿：先套用或取消`, "notice"); return false; }
  const preserve = method.mode === null;
  // 背後（#1330）：由程式最佳化的區背後要換成材料庫一筆就送 backing（method.backing 沒給＝各區照存的）；背後換過的區不管改成什麼，
  // 原牆都照設計設定記的那一份找（backing_wall）。
  const behind = item => method.mode === "search" ? (method.backing !== undefined ? method.backing : zoneSetting(item.wall, item.key).backing_entry ?? "") : "";
  const changes = items.filter(() => !preserve && (method.mode !== "fixed" || method.materialId)).map(item => ({...(method.mode === "fixed"
    ? {wall: item.wall, zone: item.key, choice: "fixed", material_id: method.materialId}
    : behind(item) ? {wall: item.wall, zone: item.key, choice: "backing", material_id: behind(item),
      ...(behind(item) === zoneSetting(item.wall, item.key).backing_entry ? {keep_backing: true} : {})} : {wall: item.wall, zone: item.key, choice: "original"}),
  ...(zoneSetting(item.wall, item.key).backing_wall ? {backing_wall: zoneSetting(item.wall, item.key).backing_wall} : {})}));
  let results = [];
  if (changes.length) {
    const document_ = wallDocument();
    inputBusy = true;
    document.querySelector("main").inert = true; $("zone-page").inert = true;
    try {
      const {response, data} = await reply(fetch("/api/zone-material/batch", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({scheme: document_, original_scheme_id: zoneDesign?.original_scheme_id ?? null,
          changes: changes.map(change => ({...change, picked_entry: zoneSetting(change.wall, change.zone).picked_entry ?? null}))})}));
      if (generation !== zoneDesignGeneration) return false;
      if (!response.ok) {
        const where = data.wall && data.zone && view.byId.get(`${data.wall}/${data.zone}`);
        zoneNote(where && data.reason ? `沒有改：${zoneWhereText(where)}：${data.reason}；一區都沒改` : `沒有改：${data.error}`, "notice");
        return false;
      }
      if (!scheme.scene.surface_zones_by_wall) {
        scheme.scene.surface_zones_by_wall = document_.scene.surface_zones_by_wall;
        delete scheme.scene.impedance_pa_s_per_m_by_wall;
      }
      results = data.changes;
      changes.forEach((change, index) => {
        scheme.scene.surface_zones_by_wall[change.wall].find(zone => zoneKey(zone) === change.zone).material = results[index].material;
      });
    } catch (error) {
      if (generation === zoneDesignGeneration) zoneNote(`沒有改：${errorText(error)}`, "notice");
      return false;
    } finally {
      document.querySelector("main").inert = false; $("zone-page").inert = false;
      inputBusy = false;
    }
  }
  // 選的跟那一區原牆一模一樣（伺服器回 same_as_original，做法變成不貼）：回話照舊做法表當年那一句（sameAsOriginalWords）講，不說成改成指定貼一種
  // （#1265 單元 3 之二第 E 條：回話跟存下來的要對得上）。
  const same = items.filter((item, index) => method.mode === "fixed" && results[index]?.same_as_original);
  // 「選的跟原本一樣」那一區自己寫為什麼（第 2 段右欄那一區底下，#1265 單元 3 之二第二趟）：這一次選到一樣的記下選的那一筆，
  // 其餘這一次改到的區拿掉（再改做法、選別的就不寫）。
  items.forEach((item, index) => {
    if (preserve) {
      if (zoneMethodMode(item.method) !== "keep_original" && method.depth !== undefined) {
        const next = {...item.setting};
        if (method.depth === null) delete next.max_depth_m; else next.max_depth_m = method.depth;
        setZoneSetting(item.wall, item.key, next);
      }
      return;
    }
    if (method.mode === "keep_original" && results[index]?.origin === "unknown") return;
    setZoneSetting(item.wall, item.key, settingAfter(item, method, results[index]));
    if (same.includes(item)) rememberSameAsOriginal(item.wall, item.key, libraryEntry(method.materialId).label);
    else zoneSameAsOriginal.delete(item.id);
  });
  const what = method.mode === "fixed" && method.materialId ? vprWords(`${ZONE_MODE_WORDS.fixed}：${libraryEntry(method.materialId).label}`) : ZONE_MODE_WORDS[method.mode];
  const pickedNotes = new Set(results.filter(change => change.picked_entry || change.backing_entry ||
    (method.mode === "keep_original" && change.origin === "original_scheme")).map(change => change.note));
  const unrestored = results.flatMap((change, index) => {
    if (changes[index].choice !== "original" || change.origin !== "unknown") return [];
    const item = items[index];
    const material = item.entry ? `「${item.entry.label}」` : "目前材料";
    const backing = method.mode === "search" ? `。把${material}當作最佳化時背後的${faceWord(item.wall)}` : "";
    return [`${zoneWhereText(item)}：${change.note}${backing}`];
  });
  const changed = items.length - same.length - (method.mode === "keep_original" ? unrestored.length : 0);
  zoneNote([...(preserve ? ["已套用最深設定，各區保留目前的做法"] : changed ? [`${changed} 區改成「${what}」`] : []), ...same.map(item => sameAsOriginalWords(item.wall, item.key)), ...pickedNotes, ...unrestored].join(" "), unrestored.length ? "notice" : "ok");
  unsaved = true;
  written();
  changing += 1; showChanging();
  try { await afterZoneChange(); } finally { changing -= 1; showChanging(); }
  // 原牆不明是正常狀態（沒選原方案就會有）：照現在材料不動、上面那句點名，照樣算套用成功，「套用並拿去搜尋」一次就送（#1334）。
  return true;
}
// 一區照 method 改過之後的那一份（寫進去之前還要過 setZoneSetting 的收法）；套用與套用前的範圍檢查都用這一支。
// change.same_as_original：指定貼的那一筆跟這一區原本的面一模一樣（伺服器放回原樣），做法跟著記成不貼。
function settingAfter(item, method, change) {
  const kept = zoneSetting(item.wall, item.key).allowed;
  const next = {...zoneSetting(item.wall, item.key), mode: method.mode};
  delete next.allowed;
  if (change) {
    if (change.picked_entry) next.picked_entry = change.picked_entry; else delete next.picked_entry;
    // 背後（#1330）：伺服器回了就照記（哪一筆、換之前的原牆），沒回就是背後放原牆了，兩格一起拿掉。
    if (change.backing_entry) { next.backing_entry = change.backing_entry; next.backing_wall = change.backing_wall; }
    else { delete next.backing_entry; delete next.backing_wall; }
  }
  if (method.mode === "fixed" && change?.same_as_original) next.mode = "keep_original";
  if (next.mode !== "search") { delete next.backing_entry; delete next.backing_wall; }  // 背後只給由程式最佳化的區
  if (method.mode === "search" && !method.lists) next.allowed = listForWall(item.wall, method.allowed);
  if (method.mode === "search" && method.lists) {
    const list = method.lists.get(item.id);
    if (list) next.allowed = list; else if (list === undefined && item.mode === "search" && kept) next.allowed = kept;
  }
  // 「最深」（小框那一格）：有改才寫——清空就拿掉那一格，填了就寫公尺；沒改的各區照目前的。
  // 指定貼的那一筆跟原本一樣、做法變成不貼時不寫（不貼不加層：舊表那一格寫「不加層」、小框也不畫，寫進去就看不到也清不掉）。
  if (next.mode !== "keep_original") {
    if (method.depth === null) delete next.max_depth_m; else if (method.depth !== undefined) next.max_depth_m = method.depth;
  }
  // 可調範圍（#1199）：小框改了範圍才有這一格；清單換掉、做法換掉時多出來的那幾種由 setZoneSetting 拿掉。
  const ranges = method.ranges?.get(item.id);
  if (ranges === null) delete next.ranges; else if (ranges !== undefined) next.ranges = ranges;
  return next;
}
function showChanging() { $("zone-page").dataset.changing = String(changing); }
// 「其他牆面」現在的做法（改回、面板上「＝其他牆面」那一項用）；規則貼的不是材料庫裡的一筆就回 null（沒有代號可以照貼）。
function ruleMethod() {
  const rule = view.inferred.rule;
  if (!rule || rule.mode === "unknown") return null;
  if (rule.mode !== "fixed") return {mode: rule.mode, allowed: rule.mode === "search" ? rule.zones[0].allowed : null};
  const entry = libraryLabel(rule.zones[0].zone.material, rule.zones[0].wall, rule.zones[0].key);
  return entry ? {mode: "fixed", materialId: entry.id} : null;
}
// 小框裡每一種材料現在的樣子：選中的每一區照草稿加減之後的清單（lists），加上每一種「選中的區裡幾區有」（count）。
// ids：出現在任何一區清單裡的代號，照最先出現的順序。材料庫沒讀全、有一區列不出來時回 null。
async function applyRule(method) {
  const followers = view.inferred.rule.zones;
  // 規則換成剛寫進去的那一種（那幾區現在的做法鍵）：做法一寫完就定下來，接著的每一次重畫都照它。
  if (!await applyMethod(followers, method, () => { ruleDraft = false; keptRuleZone = followers[0].id; })) renderZoneOverview();
}
async function resetSpecial() {
  const method = ruleMethod();
  if (!method) { zoneNote("其他牆面現在貼的不是材料庫裡的一筆，沒辦法照著改回去：先在「其他牆面」選一種", "notice"); return; }
  // 其他牆面的清單裡有「鋼板直接貼在棉上」的構造（#1208）：那一筆帶不過來，這幾區的清單跟規則永遠差一種、併不回去。
  // 不改、照講（伺服器給的那一句），不寫「改好了」。
  const held = method.mode === "search" ? (method.allowed ?? []).find(referenceOnly) : null;
  if (held) { zoneNote(zoneReference.reset_blocked[held], "notice"); return; }
  await applyMethod(view.inferred.special.flatMap(group => group.zones), method);
}
function panelTicks(items) {
  const lists = items.map(item => tickedList(item, draft.ticks));
  if (lists.some(list => !list)) return null;
  return ticksForLists(items, lists);
}
function ticksForLists(items, lists, coveredByZone = []) {
  const ids = [...new Set(lists.flat())];
  const count = id => lists.filter(list => list.includes(id)).length;
  // 「幾區有」只算貼得了這一種的區（清單裡本來就有的也算）：那一面貼不了的區不算進分母，不然永遠湊不滿。
  const able = id => items.filter((item, index) => lists[index].includes(id) || (fitsHere(id, item.wall) && !coveredByZone[index]?.[id])).length;
  const state = id => !count(id) ? "none" : count(id) === able(id) ? "all" : "some";
  return {lists, ids, count, able, state, coveredByZone};
}
// 「最深」那一格打開時顯示什麼（公分）：選到的區都一樣就帶出來，不一樣、或都沒有，就是空的。
function shownDepth(items) {
  const depths = items.map(item => zoneSetting(item.wall, item.key).max_depth_m ?? null);
  return depths.every(depth => depth === depths[0]) && depths[0] !== null ? centimetresBox(depths[0]) : "";  // 數字格：不帶逗號
}
// 「最深」那一格空白是什麼意思（底下那一行與不收時那一句都用它）：選到的區目前都一樣，清空就是不限；各區不同，空白就是沒改。
function blankDepthWords(items) {
  const depths = items.map(item => zoneSetting(item.wall, item.key).max_depth_m ?? null);
  return depths.every(depth => depth === depths[0]) ? "不限" : "各區照目前的";
}
// 「最深」那一格要寫什麼：格子的值跟打開時顯示的一樣（沒動過、或打了又改回去）undefined（不寫，各區照目前的）、
// 清空 null（拿掉）、正數回公尺（記到公釐；不到 1 公釐會變成 0，存檔時伺服器不收，這裡就擋）、其餘 NaN（不收）。
function draftDepth(items) {
  if (draft.depthText === null) return undefined;
  if (!draft.depthBad && draft.depthText.trim() === shownDepth(items)) return undefined;
  return depthBoxValue(draft.depthText, draft.depthBad);  // 讀不出來、空白、公分數：跟輸入頁兩格同一支判
}
/* ───── 這一區自己的可調範圍（#1199 範圍第 2 支） ───── */
const RANGE_WORDS = {porous: "棉厚", cavity: "空腔"};
// 一種可調材料在小框裡有哪幾格：棉厚（產品厚度不只一個才有，兩個下拉列的就是那幾個）、空腔（有空腔而且真的能調才有；
// 有沒有空腔同 adjustWords 的判法：空腔起點沒有＝沒有空腔這一格）。
function rangeParts(entry) {
  const thick = entry.porous_thickness, cavity = entry.cavity;
  return {porous: thick.adjustable && (thick.choices_m?.length ?? 0) > 1 ? thick.choices_m : null,
          cavity: Boolean(cavity?.initial_m) && cavity.adjustable && cavity.range_m[1] > cavity.range_m[0]};
}
// 材料頁那一份（公尺）：棉厚最薄到最厚、空腔下上（沒有空腔是 null）。
function pageRange(entry) { return {porous: entry.porous_thickness.range_m, cavity: entry.cavity?.initial_m ? entry.cavity.range_m : null}; }
// 一區那一種現在實際用的範圍：寫了就是它（沒寫空腔照材料頁），沒寫就是材料頁那一份。
function ownRange(item, entry) {
  const page = pageRange(entry), own = item.ranges?.[entry.id];
  return {porous: own?.porous_m ?? page.porous, cavity: own?.cavity_m ?? page.cavity};
}
// 格子裡的字：棉厚是下拉的值（產品厚度的原數，公尺，存檔逐位照它），空腔寫公分。
function rangeText(part, metres) { return part === "porous" ? String(metres) : centimetresBox(metres); }  // 空腔是數字格：不帶逗號
// 小框打開時每一對格子帶出什麼：會拿到這一種的區（套用後清單裡有它的）現在用的都一樣，就是那一對（values 是原數、texts 是格子裡的字）；
// 不一樣是 null（兩格空著）。
function shownRange(items, lists, entry) {
  const having = items.filter((item, index) => lists[index].includes(entry.id)).map(item => ownRange(item, entry));
  const pair = part => {
    const same = having.length && having.every(range => JSON.stringify(range[part]) === JSON.stringify(having[0][part]));
    return same && having[0][part] ? {values: having[0][part], texts: having[0][part].map(metres => rangeText(part, metres))} : null;
  };
  return {porous: pair("porous"), cavity: pair("cavity")};
}
function rangeDraft(id) { return draft.ranges[id] ?? {porous: [null, null], cavity: [null, null]}; }
function readEnd(part, text) {
  if (part === "porous") return Number(text);
  const trimmed = text.trim();
  return trimmed === "" ? NaN : Math.round(Number(trimmed) * 10) / 1000;
}
// 一對格子套用後的值（公尺）：兩格都跟打開時一樣＝沒改（回 {}）；改了的那一格換成打的、沒改的那一格留原本的原數（不經公分來回）。
// 各區不同時兩格都要填。這裡只看格子本身的形狀（不成數字、負的、從大於到），回 {problem}；超出材料頁、收窄後放不下交伺服器判。
function draftPair(entry, part, shown) {
  const typed = rangeDraft(entry.id)[part], texts = shown?.texts ?? ["", ""];
  const moved = typed.map((text, end) => text !== null && text.trim() !== texts[end]);
  if (!moved.some(Boolean)) return {};
  const name = `${shortLabel(entry)}的${RANGE_WORDS[part]}`;
  if (!shown && !moved.every(Boolean)) return {problem: `${name}：各區不同，要改就「從」「到」兩格都填`};
  const pair = typed.map((text, end) => moved[end] ? readEnd(part, text) : shown.values[end]);
  if (pair.some(value => !Number.isFinite(value) || value < 0)) return {problem: `${name}要填 0 以上的數字（公分）`};
  if (pair[0] > pair[1]) return {problem: `${name}：「從」要不大於「到」`};
  return {pair};
}
// 小框裡改了範圍的那幾種（只看勾著的可調材料）：{edits: Map 代號 → {entry, porous?, cavity?}}，或第一個形狀不對的 {problem}。
function rangeEdits(items, ticks) {
  const edits = new Map();
  for (const id of Object.keys(draft.ranges)) {
    const entry = libraryEntry(id);
    if (entry?.kind !== "optimizable" || ticks.state(id) === "none") continue;
    const shown = shownRange(items, ticks.lists, entry), parts = rangeParts(entry), found = {};
    for (const part of ["porous", "cavity"].filter(name => parts[name])) {
      const result = draftPair(entry, part, shown[part]);
      if (result.problem) return {problem: result.problem};
      if (result.pair) found[part] = result.pair;
    }
    if (found.porous || found.cavity) edits.set(id, {entry, ...found});
  }
  return {edits};
}
function samePair(left, right) { return Boolean(left) && Boolean(right) && left[0] === right[0] && left[1] === right[1]; }
// 每一區套用後的範圍：只寫套用後清單裡有那一種的區；改回跟材料頁一樣的那一種不寫，一種都不剩是 null（整格拿掉）。
// 回 Map（區 → 那一份或 null）；一格都沒改回 undefined（各區的範圍一個字都不碰）。
function plannedRanges(items, lists, edits) {
  if (!edits.size) return undefined;
  return new Map(items.map((item, index) => {
    const next = {...(item.ranges ?? {})};
    for (const [id, edit] of edits) {
      if (!lists[index].includes(id)) continue;
      const own = ownRange(item, edit.entry), page = pageRange(edit.entry);
      const porous = edit.porous ?? own.porous, cavity = edit.cavity ?? own.cavity;
      const sameCavity = !cavity || samePair(cavity, page.cavity);
      if (samePair(porous, page.porous) && sameCavity) delete next[id];
      else next[id] = {porous_m: [...porous], ...(sameCavity ? {} : {cavity_m: [...cavity]})};
    }
    return [item.id, Object.keys(next).length ? next : null];
  }));
}
// 改了範圍才走這一步：把套用後的那一份（清單、最深、範圍都照小框，只動選中的區）交伺服器的限制檢查判一次；選中的區有範圍的紅
// （超出材料頁、收窄後放不下）就不套用，照伺服器的原句寫在小框、套用鈕停用（改了任何一格才再試）。判法只在伺服器一份。
// 等回覆時整頁鎖住（同 applyMethod 換材料那一步）。回 true＝可以套用。
// 套用前範圍檢查要的方案與設計設定照草稿的背後（#1330 審查第 7 條）：換回原牆的那幾區放回記的原牆、拿掉兩格；
// 換成別一筆的放那一筆（原牆照記的，沒記過就是現在的材料）。沒動背後的不碰。
function plannedBacking(items, method, design) {
  const document_ = wallDocument();
  if (method.mode !== "search" || method.backing === undefined) return document_;
  for (const item of items) {
    const saved = zoneSetting(item.wall, item.key), setting = design.zones[item.wall][item.key];
    if (method.backing === (saved.backing_entry ?? "")) continue;
    const zone = document_.scene.surface_zones_by_wall?.[item.wall]?.find(part => zoneKey(part) === item.key);
    if (!zone) continue;
    if (!method.backing) {
      if (saved.backing_wall) zone.material = saved.backing_wall;
      delete setting.backing_entry; delete setting.backing_wall;
    } else {
      setting.backing_entry = method.backing; setting.backing_wall = saved.backing_wall ?? zone.material;
      zone.material = fixedSurface(libraryEntry(method.backing) ?? {}) ?? zone.material;
    }
  }
  return document_;
}
async function rangesPass(items, method) {
  const generation = zoneDesignGeneration;
  const design = structuredClone(zoneDesign ?? freshZoneDesign());
  design.zones ??= {};
  for (const item of items) {
    design.zones[item.wall] ??= {};
    design.zones[item.wall][item.key] = settledSetting(settingAfter(item, method), zoneSetting(item.wall, item.key).allowed);
  }
  const scheme_ = plannedBacking(items, method, design);
  inputBusy = true;
  document.querySelector("main").inert = true; $("zone-page").inert = true;
  let data;
  try {
    ({data} = await reply(fetch("/api/zone-limits", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scheme: scheme_, design})})));
  } catch (error) {
    if (generation === zoneDesignGeneration) zoneNote(`範圍沒檢查成，這次沒有套用：${errorText(error)}`, "notice");
    return false;
  } finally {
    document.querySelector("main").inert = false; $("zone-page").inert = false;
    inputBusy = false;
  }
  if (generation !== zoneDesignGeneration) return false;
  if (data.status !== "checked") { zoneNote(`範圍沒檢查成，這次沒有套用：${data.message ?? data.error ?? "限制檢查失敗"}`, "notice"); return false; }
  const who = item => items.length > 1 ? `${zoneWhereText(item)}：` : "";
  const red = [...(data.design ?? []).filter(line => line.level === "red").map(line => line.text),
    ...items.flatMap(item => (data.zones?.[item.wall]?.[item.key] ?? [])
      .filter(line => line.level === "red" && line.code.startsWith("range_")).map(line => `${who(item)}${line.text}`))];
  if (!red.length) return true;
  draft.rangeRefused = red.join("；");
  renderZoneOverview();
  return false;
}
// 小框現在能不能套用：指定貼與由程式最佳化都檢查最深；由程式最佳化另看每個選中區套用後的清單。
function draftProblem(items) {
  if (!draft || draft.mode === "keep_original") return "";
  const ticks = draft.mode === "search" && panelTicks(items);
  if (ticks) {
    const who = index => items.length > 1 ? `${zoneWhereText(items[index])}：` : "";
    const empty = ticks.lists.findIndex(list => !list.length);
    if (empty >= 0) return `${who(empty)}至少勾一種材料，或勾「也可以不貼」；不想讓程式選，就改選上面的「${ZONE_MODE_WORDS.fixed}」或「${KEEP_WORDS}」`;
    const stuck = ticks.lists.findIndex(list => allowedProblem(list));
    if (stuck >= 0) return `${who(stuck)}${allowedProblem(ticks.lists[stuck])}程式要從一個能調的東西開始：勾一種可調的材料，或勾「也可以不貼」`;
    const ranged = rangeEdits(items, ticks);
    if (ranged.problem) return ranged.problem;
  }
  if (Number.isNaN(draftDepth(items))) return depthRefusedWords("「最深」", `＝${blankDepthWords(items)}`);
  if (draft.mode === "search") {
    const shallow = items.find(item => backingDepth(item) > 0 && Math.round((totalDepth(item, draftDepth(items)) - backingDepth(item)) * 1e9) <= 0);
    if (shallow) return `${items.length > 1 ? zoneWhereText(shallow) + "：" : ""}最深沒有比背後的${backingName(libraryEntry(zoneBacking(shallow)) ?? {label: "那一筆"})}深，前面放不了東西；把最深加大，或背後換回${originalFace(shallow.wall)}`;
    const fixed = items.find((item, index) => zoneBacking(item) && fixedInFront(items, index).length);
    if (fixed) return `${items.length > 1 ? zoneWhereText(fixed) + "：" : ""}背後換過的區只能讓程式調可調材料或不貼；固定成品會整塊換掉背後那一筆，先取消勾選`;
  }
  return draft.mode === "search" ? draft.rangeRefused : "";
}
async function applyDraft() {
  const items = chosenItems();
  if (draft.mode !== "search") {
    const method = {mode: draft.mode, materialId: draft.materialChanged ? draft.materialId : ""};
    if (method.mode === "fixed" || method.mode === null) method.depth = draftDepth(items);
    return applyMethod(items, method, () => select([]));
  }
  // 由程式最佳化：只寫選中的區。每一區寫它自己的清單照動過的勾加減之後的那一份；勾都沒動、那一區本來就是由程式最佳化，
  // 清單就不碰（只改最深時清單一個字都不動）。沒選中的區一律不寫。
  const ticks = draft.ticks, moved = Object.keys(ticks).length > 0;
  const lists = new Map(items.map(item => [item.id, moved || item.mode !== "search" ? tickedList(item, ticks) : undefined]));
  // 規則本身是由程式最佳化、跟著規則的區全都選中了：清單換了做法鍵跟著換，規則還是那幾區。
  const rule = view.inferred.rule;
  const ruled = moved && rule?.mode === "search" && rule.zones.every(item => selected.includes(item.id)) ? rule.zones[0].id : null;
  // 可調範圍（#1199）：改了才寫、只寫選中的區；先交伺服器判一次，紅就不套用（rangesPass）。
  const ticked = panelTicks(items);
  const ranges = ticked ? plannedRanges(items, ticked.lists, rangeEdits(items, ticked).edits ?? new Map()) : undefined;
  const method = {mode: "search", lists, depth: draftDepth(items), ranges, backing: draft.backing};
  if (ranges && !await rangesPass(items, method)) return false;
  return applyMethod(items, method, () => {
    if (ruled) keptRuleZone = ruled;
    if (moved && lists.get(items[0].id)) zoneLastAllowed = [...lists.get(items[0].id)];
    select([]);
  });
}
/* ───── 右邊：面板 ───── */
// 選的那一筆的構造：房間這側 → 每一層 → 原本的面。薄的（薄板、開孔板、布）畫成一條線，下面一句說是什麼。
function structureStrip(entry, backingText) {
  const spec = inputDefaults.material_form;
  const strip = el("div", "zv-layers");
  strip.append(el("div", "zv-l-room", "房間這側"));
  const thin = [];
  for (const layer of fixedLayers(entry)) {
    const name = spec[layer.kind]?.label ?? layer.kind;
    const thick = typeof layer.thickness_m === "number" ? ` ${shownNumber(layer.thickness_m * 1000)} mm` : "";
    if (layer.kind === "porous" || layer.kind === "air") {
      const box = el("div", layer.kind === "air" ? "zv-l-air" : "zv-l-foam"); box.append(el("span", null, `${name}${thick}`)); strip.append(box);
    } else { strip.append(el("div", "zv-l-thin")); thin.push(`${name}${thick}`); }
  }
  const back = el("div", "zv-l-wall"); back.append(el("span", null, backingText)); strip.append(back);
  return [strip, ...(thin.length ? [el("div", "zv-cap", `黑線：${thin.join("、")}`)] : [])];
}
async function loadCurve(entry) {
  curves.set(entry.id, "loading");
  const {basis: _basis, placement: _placement, ...document_} = entry;
  if (entry.kind === "construction") document_.placement = entry.placement.group;
  let result;
  try {
    const {response, data} = await reply(fetch("/api/material-curve", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(document_)}));
    result = response.ok ? data : {status: "failed", message: `畫不出曲線：${data.error ?? response.status}`};
  } catch (error) { result = {status: "failed", message: `畫不出曲線：${errorText(error)}`}; }
  curves.set(entry.id, result);
  if (overviewOpen && draft?.mode === "fixed" && draft.materialId === entry.id) renderZoneOverview();
  for (const plot of document.querySelectorAll("[data-detail-curve]")) {
    if (plot.dataset.detailCurve === entry.id) plot.replaceChildren(...curveBlock(entry, `zone-panel-curve-${entry.id}`));
  }
  for (const chart of document.querySelectorAll(".zv-spark[data-curve]")) {
    if (chart.dataset.curve === entry.id) renderSpark(chart, result);
  }
}
function curveBlock(entry, scope = "zone-panel") {
  const state = curves.get(entry.id);
  if (!state) { loadCurve(entry); return [el("div", "zv-cap", "計算吸音曲線中…")]; }
  if (state === "loading") return [el("div", "zv-cap", "計算吸音曲線中…")];
  if (state.status !== "drawn") return [el("div", "zv-cap", state.message ?? "這一筆畫不出吸音曲線")];
  const width = 304, height = 82, left = 30, bottom = 20, top = 6, low = 20, high = 8000;
  const x = hz => left + (Math.log10(hz) - Math.log10(low)) / (Math.log10(high) - Math.log10(low)) * (width - left - 8);
  const y = value => top + (1 - value) * (height - top - bottom);
  const chart = svg("svg", {class: "zv-curve-chart", width, height, viewBox: `0 0 ${width} ${height}`});
  // 「鋼板直接貼在棉上」的構造（#1208，伺服器在曲線回覆裡給界線與字）：界線以下淡藍底，圖下面一行圖例與「為什麼」。
  const reference = state.reference_only ?? null;
  if (reference && reference.band_hz > low) {
    const band = svg("rect", {class: "zv-ref-band", x: left, y: top, width: x(Math.min(reference.band_hz, high)) - left, height: height - top - bottom});
    band.dataset.hz = String(reference.band_hz); chart.append(band);
  }
  for (const value of [0, 0.5, 1]) {
    chart.append(svg("line", {class: "zv-grid", x1: left, x2: width - 8, y1: y(value), y2: y(value)}),
      svg("text", {class: "zv-tick", x: left - 5, y: y(value) + 3.5, "text-anchor": "end"}, String(value)));
  }
  for (const hz of [50, 100, 200, 500, 1000, 2000, 5000]) {
    chart.append(svg("line", {class: "zv-grid", x1: x(hz), x2: x(hz), y1: top, y2: height - bottom}),
      svg("text", {class: "zv-tick", x: x(hz), y: height - 8, "text-anchor": "middle"}, String(hz)));
  }
  const points = state.frequencies_hz.map((hz, index) => [hz, state.absorption[index]]).filter(([hz]) => hz >= low && hz <= high);
  chart.append(svg("polyline", {class: "zv-curve", points: points.map(([hz, value]) => `${x(hz).toFixed(1)},${y(Math.min(Math.max(value, 0), 1)).toFixed(1)}`).join(" ")}));
  const preview = entry.backing === "zone_wall" && state.arrangement ? [el("div", "zv-cap", state.arrangement)] : [];
  if (!reference) return [chart, el("div", "zv-axis", "頻率（赫茲）"), ...preview];
  const note = el("div", "ref-note"); note.id = scope + "-ref-note";
  const more = fold(scope + "-ref-why", "為什麼"); more.append(el("p", null, reference.why));
  note.append(el("span", "ref-sw"), "藍底：", el("span", "ref-say", reference.sentence), more);
  return [chart, el("div", "zv-axis", "頻率（赫茲）"), ...preview, note];
}
function materialShow(entry, wall) {
  const show = el("div", "zv-show"); show.id = "zone-panel-show";
  // 整個名字寫一次（會折行）：下拉收起來的那一格只有一行，名字長的會被截掉。
  const name = el("div", "zv-fullname", vprWords(entry.label)); name.id = "zone-panel-name"; show.append(name);
  if (fixedLayers(entry).length) {
    const backing = replacesZone(entry) ? (fixedSurface(entry).backing === "air" ? "空氣" : "硬牆") : null;
    const head = el("h4", null, "構造 "); head.append(el("span", null, [fixedDepthWords(entry), backing ? `背後是${backing}` : `貼在${originalFace(wall)}上`].filter(Boolean).join("，")));
    show.append(head, ...structureStrip(entry, backing ?? originalFace(wall)));
  }
  const head = el("h4", null, "吸音率 "); head.append(el("span", null, "各方向來的平均；模型算的"));
  show.append(head, ...curveBlock(entry));
  return show;
}
function caseDepth(item, edited) {
  const total = totalDepth(item, edited);
  return Number.isFinite(total) ? Math.max(Math.round((total - backingDepth(item)) * 1e9) / 1e9, 0) : total;  // 十進位相減，同伺服器 minus_backing
}
// 「最深」三層取最小（從原牆往房間量，含背後換上去的那一筆）；caseDepth 是扣掉背後之後前面能用的。
function totalDepth(item, edited) {
  const local = edited === undefined ? item.setting.max_depth_m : edited;
  return Math.min(local ?? Infinity, zoneDesign?.room_max_depth_m ?? Infinity, zoneDesign?.wall_max_depth_m?.[item.wall] ?? Infinity);
}
/* ───── 背後換成材料庫一筆（#1330：例如 VPR 固定當背牆，程式只調前面的層） ───── */
// 背後下拉列哪幾筆：能整塊當牆的固定材料、選中的每一面都用得到；多孔吸音不列（前面的棉本來就由程式調）。
function backingEntries(walls) {
  return assignableEntries().filter(entry => fixedSurface(entry) && entry.placement.family !== "porous" && walls.every(wall => usableHere(entry, wall)));
}
// 一區背後是哪一筆（"" 是原牆）：小框動過照小框，沒動照存的。
function zoneBacking(item) { return draft?.mode === "search" && draft.backing !== undefined ? draft.backing : zoneSetting(item.wall, item.key).backing_entry ?? ""; }
// 選中的幾區背後都一樣就回那一筆（"" 是原牆），不一樣回 undefined。
function panelBacking(items) {
  const ids = new Set(items.map(zoneBacking));
  return ids.size === 1 ? [...ids][0] : undefined;
}
// 背後換過的區程式實際會挑的固定成品（同伺服器 fixed_in_front）：畫好的候選列是伺服器判過的實際候選就照它；
// 還沒畫好照勾的清單，拿掉被同一份清單裡勾的可調材料包含的那幾款。
function fixedInFront(items, index) {
  const painted = $("zone-panel-form")?.dataset.candidates === "checked" ? $("zone-panel-form").candidateRows?.[index]?.allowed : null;
  const list = painted ?? panelTicks(items)?.lists?.[index] ?? [];
  const covered = new Set(list.flatMap(id => libraryEntry(id)?.kind === "optimizable" ? libraryEntry(id).preset_members ?? [] : []));
  return list.filter(id => id !== ZONE_NONE && libraryEntry(id)?.kind !== "optimizable" && !covered.has(id));
}
// 背後那一筆的總深（同伺服器 fixed_material_depth_m）：沒重選照方案那一區現在的材料量（材料庫那一筆存檔後改過也照存的），
// 重選了照材料庫那一筆；實測表照它的底構造量。
function backingDepth(item) {
  const id = zoneBacking(item);
  if (!id) return 0;
  const material = id === zoneSetting(item.wall, item.key).backing_entry ? item.zone.material : fixedSurface(libraryEntry(id) ?? {});
  const layers = material?.kind === "assembly" ? material.layers : material?.base_assembly?.layers ?? [];
  return layers.reduce((sum, layer) => sum + (typeof layer.thickness_m === "number" ? layer.thickness_m : 0), 0);
}
// 背後那一筆在標籤、剖面條裡的短名：名字第一段（「VPR 鋼板 1 mm＋…」叫 VPR）；完整名字在下拉裡。
// 背後那一筆的短名：名字的第一個詞（例如「VPR」）；第一個詞只是數字（「0.8 mm 圓孔面板」）就取到「＋」或「（」之前。
function backingName(entry) {
  const word = entry.label.split(/[ ＋（]/)[0];
  return (word && !/^[\d.]+$/.test(word) ? word : entry.label.split(/[＋（]/)[0].trim()) || entry.label;
}
function backingRow(items, walls) {
  const box = el("div", "zv-backing"); box.id = "zone-panel-backing";
  const chosen = panelBacking(items), entry = chosen ? libraryEntry(chosen) : null;
  const label = el("label", "zv-extra-name", entry ? "背後（固定，不調）" : "背後"); label.htmlFor = "zone-panel-backing-pick";
  const pick = el("select", "zv-backing-pick"); pick.id = "zone-panel-backing-pick";
  if (chosen === undefined) { const mixed = el("option", null, "各區不同（不改）"); mixed.value = "*"; pick.append(mixed); }
  const own = el("option", null, walls.length === 1 ? originalFace(walls[0]) : "原本的面"); own.value = ""; pick.append(own);
  const listed = backingEntries(walls);
  if (entry && !listed.includes(entry)) listed.push(entry);
  const labels = {resonant: "共振吸音", custom: "我組的", surface: "門窗與牆面"};
  for (const family of ["custom", "resonant", "surface"]) {
    const members = listed.filter(item => (labels[item.placement.family] ? item.placement.family : "surface") === family);
    if (!members.length) continue;
    const group = el("optgroup"); group.label = labels[family];
    for (const item of members) { const option = el("option", null, vprWords(item.label)); option.value = item.id; group.append(option); }
    pick.append(group);
  }
  pick.value = chosen === undefined ? "*" : chosen;
  pick.onchange = () => { draft.backing = pick.value === "*" ? undefined : pick.value; markDraftEdited(); renderZoneOverview(); };
  box.append(label, pick);
  if (entry) {
    const strip = el("div", "zv-layers zv-backing-strip"); strip.id = "zone-panel-backing-strip";
    const tuned = el("div", "zv-l-air"); tuned.append(el("span", null, "程式調"));
    const depth = fixedDepthWords(entry).replace("總厚約 ", "").replace("（不含薄板）", "");
    const fixed = el("div", "zv-l-foam"); fixed.append(el("span", null, `${backingName(entry)}${depth ? " " + depth : ""}`));
    const wall = el("div", "zv-l-wall"); wall.append(el("span", null, "牆"));
    strip.append(tuned, fixed, wall);
    box.append(strip);
  }
  return box;
}
// 逐區原勾選留在設計檔。顯示與開跑取同一支伺服器判斷；晚到的回覆不能蓋過新草稿。
let candidatePreview = null;
function previewCandidates(items, raw) {
  const edited = draftDepth(items);
  if (!raw.ids.some(id => libraryEntry(id)?.preset_members?.length) && items.every(item => !Number.isFinite(caseDepth(item, edited)))) {
    candidatePreview = null;
    return {rows: raw.lists.map(allowed => ({allowed, covered: {}}))};
  }
  const ranges = plannedRanges(items, raw.lists, rangeEdits(items, raw).edits ?? new Map());
  const selections = items.map((item, index) => ({
    allowed: raw.lists[index], ranges: ranges ? ranges.get(item.id) : item.ranges,
    cap: caseDepth(item, edited)
  }));
  // 已存的清單與範圍沒有動，直接讀這一版分區檢查已經算好的覆蓋結果。
  if (zoneLimitsFresh() && zoneLimits.status === "checked" && edited === undefined && draft.backing === undefined &&
      !Object.keys(draft.ranges).length && !Object.keys(draft.ticks).length &&
      items.every((item, i) => zoneLimits.candidates?.[item.wall]?.[item.key] &&
        item.mode === "search" && item.allowed && sameList(item.allowed, raw.lists[i]))) {
    candidatePreview = null;
    return {rows: items.map(item => zoneLimits.candidates[item.wall][item.key])};
  }
  const key = JSON.stringify(selections);
  if (candidatePreview?.key !== key || candidatePreview.library !== zoneLibrary) {
    const preview = {key, library: zoneLibrary};
    candidatePreview = preview;
    api("/api/zone-candidates", "POST", selections).then(rows => { preview.rows = rows; })
      .catch(error => { preview.problem = errorText(error); })
      .finally(() => {
        if (candidatePreview !== preview || !overviewOpen || draft?.mode !== "search") return;
        const painted = $("zone-panel-form")?.candidateRows;
        // 畫著的表單還標 pending（暫定列畫出來之後又改了「最深」，換成這一次判決，displayedPending 沒跟過來）：照樣重畫，標記才換成 checked。
        const stale = $("zone-panel-form")?.dataset.candidates === "pending";
        if (preview.problem || preview.displayedPending || stale || !painted || JSON.stringify(painted) !== JSON.stringify(preview.rows)) renderZoneOverview();
      });
  }
  return candidatePreview;
}
function coveredWhy(family) { return `上面可調的 ${shortLabel(family)} 已經包含它；只想用成品，先取消上面的 ${shortLabel(family)}`; }
// 勾旁邊那一句：第三種狀態寫貼得了的區裡幾區有；勾著、但選中的區裡有那一面貼不了的（tickedList 不會加），照實講有幾區不會加。
function someWords(id, ticks, items) {
  const all = items.length, fitting = items.filter(item => fitsHere(id, item.wall)).length;
  const unable = all - fitting;
  if (ticks.state(id) === "some") {
    const group = unable && fitting === ticks.able(id) ? `貼得了這一種的 ${fitting} 區` : `選中的 ${all} 區`;
    return [el("small", "zv-some", `${group}裡 ${ticks.count(id)} 區有；不動它，各區照目前的${unable ? `；${unable} 區那一面貼不了這一種，不會加` : ""}`)];
  }
  return ticks.state(id) === "all" && unable ? [el("small", "zv-some", `選中的 ${all} 區裡 ${unable} 區那一面貼不了這一種，不會加`)] : [];
}
// 勾著的一種可調材料底下那一列（#1199）：棉厚兩個下拉（那份材料的產品厚度）、空腔兩格（公分；材料沒有空腔就沒有）。
// 帶出會拿到這一種的區現在用的（都一樣才帶，各區不同就空著、旁邊寫一句）。選、打字只記草稿、就地更新套用鈕（sync），不重畫。
function rangeRow(entry, items, ticks, sync) {
  const parts = rangeParts(entry);
  if (!parts.porous && !parts.cavity) return [];
  const shown = shownRange(items, ticks.lists, entry), typed = rangeDraft(entry.id), name = shortLabel(entry);
  const row = el("div", "zv-mrange"); row.id = `zone-panel-range-${entry.id}`; row.dataset.choice = entry.id;
  const edit = (part, end, text) => {
    const next = structuredClone(rangeDraft(entry.id)); next[part][end] = text;
    draft.ranges = {...draft.ranges, [entry.id]: next}; draft.rangeRefused = ""; markDraftEdited(); sync();
  };
  const cell = (part, end) => {
    const id = `zone-panel-range-${entry.id}-${part}-${end ? "high" : "low"}`, value = typed[part][end] ?? shown[part]?.texts[end] ?? "";
    const words = `${name}${RANGE_WORDS[part]}${end ? "到" : "從"}（公分）`;
    if (part === "porous") {
      const pick = el("select", "zv-rpick"); pick.id = id; pick.setAttribute("aria-label", words);
      if (!shown.porous) { const blank = el("option", null, ""); blank.value = ""; pick.append(blank); }
      for (const metres of parts.porous) { const option = el("option", null, centimetres(metres)); option.value = String(metres); pick.append(option); }
      pick.value = value;
      pick.onchange = () => edit(part, end, pick.value);
      return pick;
    }
    const input = el("input", "zv-rbox"); input.type = "number"; input.min = "0"; input.step = "any"; input.id = id; input.setAttribute("aria-label", words);
    input.value = value;
    if (typed[part][end] != null) input.dataset.typed = "1";  // 畫的是打到一半的字：重畫後放回焦點時不整格反白（renderZoneOverview）
    input.oninput = () => edit(part, end, input.validity.badInput ? "?" : input.value);
    return input;
  };
  for (const part of ["porous", "cavity"].filter(key => parts[key])) {
    const group = el("span", "zv-rpart"); group.id = `zone-panel-range-${entry.id}-${part}`;
    group.append(el("span", "zv-rname", RANGE_WORDS[part]), "從", cell(part, 0), "到", cell(part, 1), el("span", "zv-unit", "cm"));
    if (!shown[part]) { const mixed = el("small", "zv-mixed", "各區不同；不動它，各區照目前的"); mixed.id = `${group.id}-mixed`; group.append(mixed); }
    if (!shown[part]) {
      const values = items.flatMap((item, index) => ticks.lists[index].includes(entry.id)
        ? [`${item.label.numberText} ${cmPair(ownRange(item, entry)[part])}`] : []);
      group.append(el("small", "zv-current-ranges", values.join("、") + " cm"));
    }
    row.append(group);
  }
  return [row];
}
// 「這一區最深」（設計檔每一區的深度上限，公尺；這裡填公分）：選到的區目前都一樣就帶出來，不一樣就空著。
// 打字只記進草稿、就地更新套用鈕（sync），不重畫。空格子裡不放提示字（瀏覽器畫提示字的灰量不到對比），空白是什麼意思寫在底下那一行：
// 格子的值跟打開時一樣就是沒改（draftDepth），所以各區不同時空白＝各區照目前的。
// 選中的區現在貼的是「不是這一面用的」那一筆（「顯示全部材料」選的，例如前牆貼門）：照 awayWords 那一句寫出來；
// 直接取代整面的（地板本身就是木地板那一筆）是那一面本身，不寫。
function awayLines(items) {
  return items.flatMap(item => {
    const current = libraryLabel(item.zone.material, item.wall, item.key);
    if (item.zone.material.kind !== "mounted_on_wall" || !current || usableHere(current, item.wall)) return [];
    const line = el("p", "zv-away", awayWords(current, item.wall)); line.id = `zone-panel-away-${item.wall}-${item.key}`;
    return [line];
  });
}
function deepestRow(items, walls, sync) {
  const row = el("div", "zv-extra zv-deepest"); row.id = "zone-panel-deepest";
  const behind = draft.mode === "search" ? panelBacking(items) : "";
  const backing = behind ? libraryEntry(behind) : null;
  const name = el("label", "zv-extra-name", backing ? `最深（含 ${backingName(backing)}）` : "最深（含空腔）"); name.htmlFor = "zone-panel-depth";
  const input = el("input", "zv-depth"); input.type = "number"; input.min = "0"; input.step = "any"; input.id = "zone-panel-depth";
  input.value = draft.depthText ?? shownDepth(items);
  if (draft.depthText !== null) input.dataset.typed = "1";  // 畫的是打到一半的字：重畫後放回焦點時不整格反白（renderZoneOverview）
  // 打的字讀不出來（數字格讀成空字串、另標 badInput）又遇到整個小框重畫（例如勾「顯示全部材料」）：新格子放不回那幾個字。
  // 那一次照樣不收、套用照擋，格子旁照講（#1265 單元 3 之二第二趟追加設計洞）：不丟掉，免得重畫正好落在打到一半時（「.」還沒打 5）
  // 默默換成別的值；不講的話空白的格子看起來像沒填卻報錯。重打任何字就照新打的判、這一句收掉。
  // 讀不出來時數字格的值一定是空字串（draft.depthText 跟著是 ""），只看 depthBad 就夠（收尾保險行）。
  const lost = draft.depthBad
    ? el("small", "zv-depth-lost", `剛才打的還不成數字，小框重畫後格子顯示不出來：這一次不收；重打一次，或按退格鍵、刪除鍵清成空白（空白＝${blankDepthWords(items)}）`)
    : null;
  if (lost) { lost.id = "zone-panel-depth-lost"; input.setAttribute("aria-invalid", "true"); }
  const typed = () => {
    draft.depthText = input.value; draft.depthBad = input.validity.badInput; draft.rangeRefused = ""; markDraftEdited();
    lost?.remove(); input.removeAttribute("aria-invalid");
    sync();
  };
  input.oninput = typed;
  // 出口（#1265 單元 3 之二收尾追加第 6 條）：重畫後的格子是空的，對它按退格、刪除值沒變，瀏覽器不發 input、只發
  // beforeinput（空格子按剪下連 beforeinput 都不發，收尾第二趟照審查實測拿掉「或剪下」）。那一句還掛著時，這一下就是人說「要空白」：照空白收（值本來就是空的，不換成別的）。
  // 不另判格子是不是空的（收尾植錯判走不到、拿掉）：那一句還掛著＝重畫之後還沒發過 input（typed 第一件事就拿掉它），格子一定是重畫時放的空字串。
  input.addEventListener("beforeinput", event => {
    if (lost?.isConnected && event.inputType.startsWith("delete")) typed();
  });
  const from = walls.length === 1 ? originalFace(walls[0]) : "原本的面";
  input.title = `從${from}往房間算，空白＝${blankDepthWords(items)}`;
  row.append(name, input, el("span", "zv-unit", "cm"));
  // 背後換過：旁邊一小字講前面那幾層最多能用多深（最深扣掉背後那一筆），打字時跟著改；最深不限就不寫。
  if (backing) {
    const room = el("span", "zv-front-room"); room.id = "zone-panel-front-room";
    const front = () => {
      const value = Math.min(...items.map(item => caseDepth(item, draftDepth(items))));
      room.textContent = Number.isFinite(value) ? `前面最多 ${centimetres(value)} cm` : "";
    };
    front(); row.append(room);
    input.addEventListener("input", front);
  }
  input.placeholder = `空白＝${blankDepthWords(items)}`;
  input.style.width = `${input.placeholder.length + 2.5}em`;
  if (lost) row.append(lost);
  return row;
}
// 草稿的順序獨立於各區存檔順序：跨區第一次出現的先列，剛加入的接在後面。
function candidateIds(ticks) {
  const order = new Set([...(draft.order ?? []), ...ticks.ids]);
  return [...order].filter(id => id !== ZONE_NONE && ticks.count(id));
}
function candidateName(id) {
  if (id !== ZONE_NONE) return vprWords(choiceName(id));
  const behind = draft?.mode === "search" ? panelBacking(chosenItems()) : "";
  return behind && libraryEntry(behind) ? `也可以不貼（只有 ${backingName(libraryEntry(behind))}）` : "也可以不貼";
}
function zoneNumbers(items) {
  const walls = [...new Set(items.map(item => item.wall))];
  return walls.map(wall => (walls.length > 1 ? wallNames[wall] : "") +
    items.filter(item => item.wall === wall).map(item => item.label.numberText).join("")).join("、");
}
function candidateAction(items, shown, id, operation) {
  const ticks = panelTicks(items);
  if (operation !== "remove" && referenceOnly(id)) return;
  const before = {ticks: {...draft.ticks}, removedRanges: {}, order: candidateIds(ticks),
                  expanded: draft.expanded, notice: draft.notice};
  const name = candidateName(id), only = operation === "only", add = operation !== "remove";
  // 「加到」保留草稿裡已有的；先前已拿掉的不能從存檔復活，只在可貼的區補回。
  const actions = {...draft.ticks[id]};
  items.forEach((item, index) => {
    if (!only && shown.coveredByZone[index]?.[id]) return;
    const previouslyCleared = ["remove", "only"].includes(actions[item.id]);
    actions[item.id] = operation === "add" && previouslyCleared ? "only" : operation;
  });
  draft.ticks = {...draft.ticks, [id]: actions};
  const cleared = only ? before.order.filter(other => other !== id) : [];
  for (const other of cleared) draft.ticks[other] = Object.fromEntries(items.map(item => [item.id, "remove"]));
  const after = panelTicks(items);
  for (const other of before.order) {
    if (!after.count(other) && other in draft.ranges) {
      before.removedRanges[other] = draft.ranges[other];
      delete draft.ranges[other];
    }
  }
  draft.expanded = draft.expanded.filter(other => after.count(other));
  const changed = items.filter((item, i) => !sameList(after.lists[i], ticks.lists[i]));
  const words = [];
  if (only) {
    words.push(`已只留 ${name}（${zoneNumbers(items)}）`);
    if (cleared.length) words.push("清掉 " + cleared.map(other => `${candidateName(other)}（${ticks.count(other)} 區有）`).join("、"));
    if (changed.length) words.push(`改了 ${zoneNumbers(changed)}`);
    items.forEach((item, index) => {
      if (!after.lists[index].includes(id)) {
        const who = item.label.numberText;
        words.push(`${wallNames[item.wall]}${who}那一面貼不了 ${name}，${who}${after.lists[index].length ? "只剩「也可以不貼」" : "清單空了"}`);
      }
    });
  } else {
    words.push(`${zoneNumbers(changed)} ${add ? "加上" : "拿掉"} ${name}${add ? "" : `（${ticks.count(id)} 區有）`}`);
  }
  if (id !== ZONE_NONE) words.push("「也可以不貼」沒動");
  draft.history.push(before);
  draft.order = [...before.order.filter(other => after.count(other)), ...after.ids.filter(other => !before.order.includes(other))];
  draft.notice = words.join("；");
  draft.rangeRefused = "";
  markDraftEdited();
  renderZoneOverview();
}
function undoCandidate() {
  markDraftEdited();
  const {removedRanges, ...before} = draft.history.pop();
  for (const id of Object.keys(draft.ranges)) if (!before.order.includes(id)) delete draft.ranges[id];
  Object.assign(draft.ranges, removedRanges);
  Object.assign(draft, before);
  draft.rangeRefused = "";
  renderZoneOverview();
}
function replacesZone(entry) {
  return entry.kind === "construction" ? entry.backing !== "zone_wall" :
    ["surface_itself", "door_window", "ceiling_build"].includes(entry.placement.group);
}
function fixedDescription(entry) {
  if (replacesZone(entry)) return "整塊換掉這一區";
  const depth = fixedDepthWords(entry);
  return depth ? `整塊貼、固定 ${depth.replace("總厚約 ", "")}` : "整塊貼、固定厚度未提供";
}
function cmPair(pair) { return pair.map(centimetres).join("–"); }
function parameterRows(entry, pairs) {
  const parts = rangeParts(entry);
  return ["porous", "cavity"].map(part => {
    const name = part === "porous" ? "棉厚" : "背後空腔";
    const metres = part === "porous" ? entry.porous.thickness_m : entry.cavity?.initial_m;
    const value = parts[part] ? (pairs[part] ? cmPair(pairs[part]) + " cm" : "各區不同")
      : metres ? centimetres(metres) + " cm 不調" : "沒有（直接貼牆）";
    const row = el("div", "zv-parameter");
    row.append(el("span", null, name), " ", el("span", null, value));
    return row;
  });
}
function thicknessBounds(entry, pairs) {
  const parts = rangeParts(entry);
  const porous = parts.porous ? pairs.porous : [entry.porous.thickness_m, entry.porous.thickness_m];
  const cavity = parts.cavity ? pairs.cavity : [entry.cavity?.initial_m ?? 0, entry.cavity?.initial_m ?? 0];
  if (!porous || !cavity) return null;
  return porous.map((value, index) => value + cavity[index]);
}
function rowThickness(entry, pairs) {
  const total = thicknessBounds(entry, pairs);
  if (!total) return "厚度各區不同";
  return (total[0] === total[1] ? centimetres(total[0]) : cmPair(total)) + " cm";
}
function describeRange(line, items, ticks, entry) {
  const shown = shownRange(items, ticks.lists, entry), page = pageRange(entry), parts = rangeParts(entry);
  let narrowed = false;
  const pairs = {};
  for (const part of ["porous", "cavity"].filter(key => parts[key])) {
    const pair = draftPair(entry, part, shown[part]).pair ?? shown[part]?.values;
    pairs[part] = pair;
    narrowed ||= pair ? !samePair(pair, page[part]) : items.some((item, i) => ticks.lists[i].includes(entry.id) && !samePair(ownRange(item, entry)[part], page[part]));
  }
  const edited = draftDepth(items);
  const selecting = items.filter((item, i) => ticks.lists[i].includes(entry.id));
  const applicable = selecting.length ? selecting : items.filter(item => fitsHere(entry.id, item.wall));
  const zoneCaps = applicable.map(item => caseDepth(item, edited));
  const mixed = new Set(zoneCaps).size > 1;
  const zoneLimit = Math.min(...zoneCaps);
  if (!selecting.length) Object.assign(pairs, page);
  const limit = Math.min(zoneLimit, entry.max_total_depth_m ?? Infinity);
  let cap = "";
  if (!mixed && applicable.length && applicable.every(item => {
    const own = ownRange(item, entry);
    const thick = parts.porous ? (pairs.porous ?? own.porous)[1] : entry.porous.thickness_m;
    const cavity = parts.cavity ? (pairs.cavity ?? own.cavity)[1] : (entry.cavity?.initial_m ?? 0);
    return limit < thick + cavity - 1e-9;
  })) {
    const source = limit === zoneLimit ? (items.length === 1 ? "這一區最深" : "這幾區最深") : "材料本身";
    cap = `${entry.cavity?.initial_m ? "兩個加起來" : "總厚"}最多 ${centimetres(limit)} cm（${source}）`;
  }
  const summary = line.closest(".zv-candidate")?.querySelector(".zv-description");
  if (summary) summary.textContent = rowThickness(entry, pairs);
  line.replaceChildren(...parameterRows(entry, pairs));
  if (cap) line.append(el("div", null, cap));
  if (narrowed) line.append(el("strong", "zv-narrow", "改窄了"));
  const axis = mixed ? el("div", "zv-depth-axis", "最深各區不同") : depthAxis(thicknessBounds(entry, pairs), limit);
  if (mixed) axis.classList.add("zv-zone-caps");
  if (mixed) applicable.forEach((item, i) => {
    axis.append(el("span", null, `${zoneWhereText(item)}：${Number.isFinite(zoneCaps[i]) ? centimetres(zoneCaps[i]) + " cm" : "不限"}`));
  });
  if (line.nextElementSibling?.matches(".zv-depth-axis")) line.nextElementSibling.remove();
  if (axis) line.after(axis);
}
function depthAxis(bounds, limit = Infinity) {
  if (!bounds) return null;
  const low = bounds[0] * 100, high = Math.min(bounds[1], limit) * 100;
  if (!(high >= low)) return null;
  const end = Math.max(50, Math.ceil(bounds[1] * 10) * 10);
  const axis = el("div", "zv-depth-axis");
  const rail = el("span", "zv-depth-track"), band = el("i");
  band.style.left = (100 * low / end) + "%";
  band.style.width = (100 * (high - low) / end) + "%";
  rail.append(band);
  const rowDepth = cmPair([bounds[0], high / 100]);
  axis.append("0", rail, end + " cm", el("span", null, "總厚 " + rowDepth + " cm"));
  axis.setAttribute("aria-label", "總厚 " + rowDepth + " cm");
  return axis;
}
function fixedThickness(entry) {
  return fixedDepthWords(entry).replace("總厚約 ", "") || "厚度未提供";
}
// 一筆材料自己的層序。沒有分層資料就留空，不能從材料名稱猜厚度。
function candidateCutaway(entry) {
  const strip = el("span", "zv-cutaway");
  strip.setAttribute("aria-hidden", "true");
  const layers = entry.kind === "optimizable" ? [entry.porous, ...(entry.cavity?.initial_m ? [{kind: "air"}] : [])] : fixedLayers(entry);
  for (const layer of layers) strip.append(el("i", `zv-slab ${layer.kind}`));
  return strip;
}
function candidateRestriction(entry, covered, listed) {
  if (!entry) return "材料庫找不到這一筆";
  if (entry.backing === "zone_wall") return "背後要接這一區的原牆，不能整塊當候選";
  if (referenceOnly(entry.id)) return listed ? zoneReference.listed_why : zoneReference.search_why;
  return covered ? coveredWhy(covered) : "";
}
function restrictedReason(node, entry, listed) {
  const reason = candidateRestriction(entry, null, listed);
  if (!reason) return false;
  if (referenceOnly(entry.id)) node.dataset.reference = listed ? "listed" : "off";
  node.append(el("small", "zv-covered-why", reason));
  return true;
}
function materialRow({entry, name, description, role, state, id, activate}) {
  const row = button("zv-material", "", id, activate);
  row.value = entry?.id ?? "";
  row.dataset.materialName = name;
  const windowEntry = entry && ["door_window", "surface_itself"].includes(entry.placement.group) && (entry.label.includes("窗") || /(?:^|_)window(?:_|$)/.test(entry.id) ||
    (entry.placement.group === "door_window" && ["glass_single_3mm", "glass_double_gap30"].includes(entry.id)));
  if (windowEntry && /玻璃|glass/i.test(entry.id + entry.label)) row.dataset.searchAlias = "玻璃窗";
  if (entry) row.dataset.entryKind = entry.kind;
  row.setAttribute("role", role);
  row.setAttribute("aria-checked", state);
  const mark = el("span", "zv-mark");
  mark.setAttribute("aria-hidden", "true");
  const text = el("span", "zv-material-text");
  text.append(el("strong", null, name));
  if (entry?.kind === "optimizable" && Object.values(rangeParts(entry)).some(Boolean)) text.append(el("span", "zv-kind", "可調"));
  if (description && role === "radio") text.append(el("small", "zv-description", description));
  const chart = svg("svg", {class: "zv-spark", viewBox: "0 0 72 26", "aria-hidden": true});
  if (entry) {
    chart.dataset.curve = entry.id;
    renderSpark(chart, curves.get(entry.id));
  }
  if (role === "radio") row.append(mark, entry ? candidateCutaway(entry) : el("span"), text, chart);
  else {
    row.append(mark, text);
    if (description) row.append(el("small", "zv-description", description));
    if (entry) row.append(chart);
  }
  return row;
}

function renderSpark(chart, state) {
  if (state?.status !== "drawn") return;
  const points = state.frequencies_hz.flatMap((hz, i) => hz < 20 || hz > 8000 ? [] :
    [`${1 + 70 * Math.log(hz / 20) / Math.log(400)},${25 - 24 * Math.min(1, Math.max(0, state.absorption[i]))}`]);
  chart.replaceChildren(svg("polyline", {points: points.join(" "), fill: "none", stroke: "currentColor", "stroke-width": 1.6}));
}
// 縮圖依序取，讓套用與存檔有連線可用；滑出畫面或被重畫的縮圖不再排隊。
const pendingSparks = new Set();
let drawingSpark = false;
async function drawVisibleSparks() {
  if (drawingSpark) return;
  drawingSpark = true;
  try {
    for (const chart of pendingSparks) {
      pendingSparks.delete(chart);
      if (!chart.isConnected) continue;
      visibleCurves.unobserve(chart);
      const entry = libraryEntry(chart.dataset.curve);
      if (!entry) continue;
      if (curves.has(entry.id)) renderSpark(chart, curves.get(entry.id));
      else await loadCurve(entry);
    }
  } finally { drawingSpark = false; }
}
const visibleCurves = new IntersectionObserver(entries => {
  for (const {target, isIntersecting} of entries) {
    if (isIntersecting && target.isConnected) pendingSparks.add(target);
    else pendingSparks.delete(target);
  }
  void drawVisibleSparks();
});

function visibleRadios(container) {
  return [...container.querySelectorAll('[role="radio"]:not([hidden])')].filter(row => !row.closest("details:not([open])"));
}
function radioTabStop(container) {
  const visible = visibleRadios(container);
  const stop = visible.find(row => row.getAttribute("aria-checked") === "true") ?? visible[0];
  for (const row of container.querySelectorAll('[role="radio"]')) row.tabIndex = row === stop ? 0 : -1;
}

function materialFamilies(container, walls, rows) {
  const face = walls.every(wall => wall === "floor") ? "floor" : walls.every(wall => wall === "ceiling") ? "ceiling" :
    walls.every(wall => wall !== "floor" && wall !== "ceiling") ? "wall" : "mixed";
  const labels = {porous: "多孔吸音", resonant: "共振吸音", custom: "我組的",
    surface: {floor: "地板", ceiling: "天花構造", wall: "門、窗、一般牆面", mixed: "門窗與一般面"}[face]};
  const order = face === "floor" || face === "ceiling"
    ? ["surface", "porous", "resonant", "custom"] : ["porous", "resonant", "surface", "custom"];
  for (const family of order) {
    const members = rows.filter(([entry]) => entry.placement.family === family);
    const group = el("details", "zv-material-family"), label = labels[family];
    group.dataset.materialGroup = family;
    group.setAttribute("aria-label", label);
    const summary = el("summary", null, label);
    summary.append(el("span", "zv-family-count", String(members.length)));
    const key = [container.id, face, family].join("/");
    group.open = members.length > 0 && (familyOpen.get(key) ?? true);
    group.dataset.expanded = String(group.open);
    summary.onclick = event => {
      if (!members.length) { event.preventDefault(); return; }
      if (draft.query?.trim()) draft.queryFolds = {...draft.queryFolds, [family]: !group.open};
      familyOpen.set(key, !group.open);
      group.dataset.expanded = String(!group.open);
    };
    group.append(summary, ...members.map(([, row]) => row));
    container.append(group);
  }
}

function materialFilter(container) {
  const input = el("input", "zv-find");
  input.type = "search"; input.placeholder = "找材料"; input.setAttribute("aria-label", "找材料");
  input.id = "zone-panel-picker-search"; input.value = draft.query ?? "";
  const empty = el("p", "zv-list-hint");
  empty.setAttribute("role", "status");
  container.append(empty);
  const update = () => {
    if (draft.query !== input.value) draft.queryFolds = {};
    draft.query = input.value;
    const query = input.value.trim().toLowerCase();
    let matches = 0;
    for (const row of container.querySelectorAll("[data-material-name]")) {
      const holder = row.closest(".zv-candidate,.zv-available") ?? row;
      const family = row.closest("[data-material-group]")?.getAttribute("aria-label") ?? "";
      holder.hidden = !row.dataset.materialName.toLowerCase().includes(query) && !family.includes(query) && !(row.dataset.searchAlias ?? "").includes(query);
      if (!holder.hidden) {
        matches++;
        const label = row.querySelector("strong"), name = row.dataset.materialName;
        const at = name.toLowerCase().indexOf(query);
        if (query && at >= 0) label.replaceChildren(name.slice(0, at), el("mark", null, name.slice(at, at + query.length)), name.slice(at + query.length));
        else label.textContent = name;
      }
    }
    for (const group of container.querySelectorAll("[data-material-group]")) {
      const count = group.querySelectorAll(":scope > :not(summary):not([hidden])").length;
      const matched = count > 0;
      group.querySelector(".zv-family-count").textContent = String(count);
      group.hidden = Boolean(query && !matched);
      group.open = matched && (query ? (draft.queryFolds?.[group.dataset.materialGroup] ?? true) : group.dataset.expanded === "true");
    }
    if (container.getAttribute("role") === "radiogroup") radioTabStop(container);
    empty.hidden = !query || matches > 0;
    empty.textContent = empty.hidden ? "" : `沒有符合「${input.value.trim()}」的材料`;
  };
  input.oninput = update;
  input.onkeydown = event => {
    if (event.isComposing) return;
    if (event.key === "Escape") { input.value = ""; update(); }
    if (event.key === "Enter" && input.value.trim()) {
      event.preventDefault();
      const match = [...container.querySelectorAll(".zv-material:enabled")]
        .find(row => !row.closest("[hidden], details:not([open])") && !row.hasAttribute("data-remove-only") && row.getAttribute("aria-checked") !== "true");
      if (match) {
        input.value = ""; update();
        match.click();
      }
    }
  };
  update();
  return input;
}

function candidateRow(id, items, ticks, members, sync) {
  const entry = libraryEntry(id), none = id === ZONE_NONE, name = candidateName(id);
  const count = ticks.count(id), state = ticks.state(id);
  const listed = count || none;
  const row = el("div", listed ? "zv-candidate" : "zv-available");
  if (listed) row.id = none ? "zone-panel-none-row" : `zone-panel-row-${id}`;
  if (count) row.dataset.choice = id;
  if (listed) row.dataset.kind = entry?.kind ?? (none ? "none" : "missing");
  row.dataset.state = state;
  const check = materialRow({entry, name, role: "checkbox", state: state === "some" ? "mixed" : String(!!count),
    id: `zone-panel-pick-${id}`, activate: null,
    description: entry?.kind === "optimizable" ? rowThickness(entry, pageRange(entry)) :
      entry ? fixedThickness(entry) : none ? "" : "材料庫找不到，照原樣留著，開跑時伺服器會點名"});
  if (state === "some") check.querySelector(".zv-material-text").append(el("span", "zv-owned", `${count} 區已有`));
  const removeOnly = count && !none && candidateRestriction(entry, null, true);
  if (removeOnly) check.dataset.removeOnly = "";
  check.onclick = () => {
    candidateAction(items, ticks, id, state === "all" || removeOnly ? "remove" : "add");
    $(check.id)?.focus({preventScroll: true});
  };
  row.append(check);
  if (members.length) {
    const details = fold(`zone-panel-members-${id}`, `含 ${members.length} 款`);
    details.className = "zv-members";
    for (const member of members) {
      const item = el("div", "zv-member"); item.dataset.member = member.id;
      item.append(el("span", null, vprWords(member.label)), el("span", "zv-member-depth", fixedThickness(member)));
      details.append(item);
    }
    const depth = check.querySelector(".zv-description");
    depth.id = `zone-panel-thickness-${id}`;
    check.setAttribute("aria-describedby", depth.id);
    row.append(details, depth, check.querySelector(".zv-spark"));
  }
  const options = fold(`zone-panel-options-${id}`, entry ? "詳情" : "選項");
  options.classList.add("zv-row-options");
  // 滑到「選項」可直接選動作；點一下固定展開，觸控和鍵盤仍可直接開關。
  const toggle = options.firstElementChild;
  // 只把使用者點開的明細帶過重畫。滑入的暫時預覽不能變成「本來已開」，
  // 否則候選回覆剛好重畫後，第一下點擊反而會把明細收掉。
  options.ontoggle = null;
  toggle.onpointerenter = event => {
    if (event.pointerType === "mouse") options.open = true;
  };
  options.addEventListener("click", event => {
    if (toggle.contains(event.target)) {
      if (!event.detail || !options.open || openDetails.has(options.id)) return;
      event.preventDefault();  // 固定滑入的預覽，保留展開。
    }
    openDetails.add(options.id);
  }, true);
  row.onpointerleave = () => {
    if (!openDetails.has(options.id) && !options.contains(document.activeElement)) options.open = false;
  };
  row.append(options);
  if (count) {
    const remove = button("zv-remove", "×", `zone-panel-remove-${id}`, () => candidateAction(items, ticks, id, "remove"));
    remove.setAttribute("aria-label", `拿掉 ${name}`); options.append(remove);
  }
  const restricted = entry && restrictedReason(count ? row : check, entry, Boolean(count));
  if (!count && restricted) check.disabled = true;
  if (entry?.kind === "optimizable") {
    const line = el("div", "zv-description zv-range");
    options.append(line);
    line.dataset.rangeEntry = entry.id;
    describeRange(line, items, ticks, entry);
  } else if (entry) {
    const thickness = fixedLayers(entry).reduce((sum, layer) => sum + (layer.thickness_m ?? 0), 0);
    if (thickness) options.append(depthAxis([thickness, thickness]));
    options.append(el("div", "zv-cap", fixedDescription(entry)));
  }
  if (entry) {
    const plot = el("div", "zv-detail-curve");
    options.append(plot);
    // 大曲線只在展開時取；縮圖與明細共用同一份吸音資料。
    const showCurve = () => {
      if (plot.dataset.detailCurve) return;
      plot.dataset.detailCurve = entry.id;
      plot.replaceChildren(...curveBlock(entry, `zone-panel-curve-${id}`));
    };
    toggle.addEventListener("click", () => { if (!options.open) showCurve(); });
    options.addEventListener("toggle", () => { if (options.open) showCurve(); });
    if (options.open) showCurve();
  }
  if (!listed) return row;
  if (items.length > 1) {
    const dots = el("div", "zv-dots");
    items.forEach((item, index) => {
      if (!fitsHere(id, item.wall)) return;
      const has = ticks.lists[index].includes(id), dot = numberBadge(item.label.number);
      dot.classList.add(has ? "has" : "absent");
      dot.title = `${zoneWhereText(item)}：${has ? "有" : "沒有"}`;
      dot.setAttribute("aria-label", dot.title); dots.append(dot);
    });
    dots.append(...someWords(id, ticks, items)); options.append(dots);
  }
  const actions = el("div", "zv-candidate-actions");
  const missing = ticks.able(id) - ticks.count(id);
  if (missing && !restricted) actions.append(button("zv-mini", none && !ticks.count(id) ? "加上" : `加到 ${missing} 區`,
    `zone-panel-addto-${id}`, () => candidateAction(items, ticks, id, "add")));
  if (entry?.kind === "optimizable" && Object.values(rangeParts(entry)).some(Boolean)) {
    const open = draft.expanded.includes(id);
    const toggle = button("zv-link", "改範圍", `zone-panel-range-toggle-${id}`, () => {
      draft.expanded = open ? draft.expanded.filter(other => other !== id) : [...draft.expanded, id];
      renderZoneOverview();
    });
    toggle.setAttribute("aria-expanded", String(open)); actions.append(toggle);
  }
  if (entry && !restricted && items.some(item => fitsHere(id, item.wall))) actions.append(button("zv-mini", "清掉其他候選", `zone-panel-only-${id}`,
    () => candidateAction(items, ticks, id, "only")));
  options.append(actions);
  if (entry?.kind === "optimizable" && draft.expanded.includes(id)) {
    const fields = rangeRow(entry, items, ticks, sync);
    if (items.length > 1) row.append(el("small", null, `填了，${ticks.count(id)} 區都改成這個`));
    options.append(...fields);
  }
  if (entry?.kind === "construction" && !restricted && catalogKind(entry) !== "fixed") {
    const layers = fold(`zone-panel-layers-${id}`, "看每一層");
    layers.append(el("p", null, "每一層現在還不能調，之後的版本會開"));
    for (const layer of fixedLayers(entry)) {
      const depth = layer.thickness_m == null ? "厚度未提供" : `${centimetres(layer.thickness_m)} cm`;
      layers.append(el("div", null, `${inputDefaults.material_form[layer.kind]?.label ?? layer.kind} · ${depth}`));
    }
    options.append(layers);
  }
  return row;
}
function pickerFits(entry, walls, ticks) {
  return walls.every(wall => usableHere(entry, wall)) ||
    ((ticks.count(entry.id) || entry.id in draft.ticks) && walls.some(wall => usableHere(entry, wall)));
}
function startingListNotes(items, walls, ticks, covered) {
  const groups = new Map();
  for (const [index, item] of items.entries()) {
    const id = item.construction?.id ?? item.entry?.id;
    if (item.mode === "search" || zoneMethodMode(item.method) !== "fixed" || ticks.lists[index].includes(id)) continue;
    const entry = libraryEntry(id);
    let why = candidateRestriction(entry, covered.get(id), false);
    if (!why && !usableHere(entry, item.wall)) {
      why = `${wallNames[item.wall]}那一面貼不了這一筆`;
    } else if (!why && !pickerFits(entry, walls, ticks)) {
      const names = walls.filter(wall => !usableHere(entry, wall)).map(wall => wallNames[wall]);
      why = `選中的另一面（${names.join("、")}）貼不了這一筆`;
    }
    const message = `原本指定貼的 ${vprWords(pastedWords(item))} 沒放進清單，${why || "要用可勾選這筆材料"}`;
    if (!groups.has(message)) groups.set(message, []);
    groups.get(message).push(item);
  }
  return [...groups].map(([message, zones]) => el("p", "zv-start-note", `${zoneNumbers(zones)} ${message}`));
}
function searchForm(items, walls, sync) {
  const raw = panelTicks(items);
  if (!raw) {
    const unknown = el("p", "zv-list-hint", "程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）");
    unknown.id = "zone-panel-search-from";
    return [backingRow(items, walls), deepestRow(items, walls, sync), unknown];
  }
  const form = el("section", "zv-form");
  const preview = previewCandidates(items, raw);
  // 已勾選的全都不可能是家族成員時，候選數和清單已經確定。可先操作這些列，
  // 其餘可能被家族涵蓋的款式等逐區判決回來再列，避免讀取舊方案時短暫顯示成空清單。
  const pendingMembers = new Set(preview.rows ? [] : raw.ids.flatMap(id => libraryEntry(id)?.preset_members ?? []));
  const independent = items.every(item => !Number.isFinite(caseDepth(item, draftDepth(items)))) && raw.lists.every(list => list.every(id => !pendingMembers.has(id)));
  if (!preview.rows && (!independent || preview.problem)) {
    form.append(backingRow(items, walls), deepestRow(items, walls, sync), el("p", "zv-list-hint", preview.problem ?? "正在核對候選"));
    if (preview.problem) form.append(button("zv-link", "重試", "zone-candidates-retry", () => { candidatePreview = null; renderZoneOverview(); }));
    return [form];
  }
  form.id = "zone-panel-form";
  if (!preview.rows) preview.displayedPending = true;
  // 先照勾選畫暫定列讓人可以先操作，伺服器逐區判決回來會再畫一次；這個標記講現在畫的是哪一版（考卷讀列之前等它變 checked）。
  form.dataset.candidates = preview.rows ? "checked" : "pending";
  const selections = preview.rows ?? raw.lists.map(allowed => ({allowed, covered: {}}));
  form.candidateRows = selections;
  const perZone = selections.map(row => row.covered);
  const ticks = ticksForLists(items, selections.map(row => row.allowed), perZone);
  // 完整材料表仍可編輯原勾選（包括超深項的範圍），只看候選才照可搜尋的清單列。
  const editable = ticksForLists(items, raw.lists.map((list, i) => list.filter(id => !perZone[i][id])), perZone);
  const rowTicks = draft.candidatesOnly ? ticks : editable;
  const covered = new Map();
  for (const row of selections) for (const [id, family] of Object.entries(row.covered)) {
    if (items.every((item, i) => (!fitsHere(id, item.wall) && !raw.lists[i].includes(id)) || perZone[i][id]))
      covered.set(id, libraryEntry(family));
  }
  const list = el("div"); list.id = "zone-panel-list";
  const ids = candidateIds(ticks);
  const count = ids.length + Number(Boolean(ticks.count(ZONE_NONE)));
  form.append(backingRow(items, walls), deepestRow(items, walls, sync));
  const rows = [];
  const unavailable = ids.filter(id => !libraryEntry(id)).map(id => candidateRow(id, items, ticks, [], sync));
  if (!libraryEntries("optimizable").some(entry => pickerFits(entry, walls, ticks))) {
    form.append(el("p", "zv-form-note", "這裡沒有可調的材料：程式只能從「不貼」開始，可以再勾固定成品讓它比"));
  }
  if (draft.notice) {
    const notice = el("div", "zv-notice", draft.notice); notice.id = "zone-panel-notice"; notice.setAttribute("role", "status");
    if (draft.history.length) notice.append(button("zv-link", "復原", "zone-panel-undo", undoCandidate));
    form.append(notice);
  }
  const others = el("div"); others.id = "zone-panel-picker";
  const catalog = [...libraryEntries("optimizable"), ...assignableEntries().filter(entry => fixedCandidate(entry) || ticks.count(entry.id))];
  for (const entry of catalog) {
    if (pendingMembers.has(entry.id) || covered.has(entry.id) || (!ticks.count(entry.id) && (draft.candidatesOnly || !pickerFits(entry, walls, ticks)))) continue;
    const members = ticks.count(entry.id) ? (entry.preset_members ?? []).filter(id => perZone.some(row => row[id] === entry.id)).map(libraryEntry).filter(Boolean) : [];
    rows.push([entry, candidateRow(entry.id, items, rowTicks, members, sync)]);
  }
  materialFamilies(others, walls, rows);
  if (!draft.candidatesOnly || ticks.count(ZONE_NONE)) list.append(candidateRow(ZONE_NONE, items, ticks, [], sync));
  list.append(others, ...unavailable);
  const toggle = button("zv-only-candidates", `只看候選（${count}）`, "zone-panel-candidates-only", () => {
    draft.candidatesOnly = !draft.candidatesOnly;
    renderZoneOverview();
    $("zone-panel-candidates-only")?.focus({preventScroll: true});
  });
  toggle.setAttribute("aria-pressed", String(Boolean(draft.candidatesOnly)));
  const tools = el("div", "zv-filter-tools");
  tools.append(materialFilter(list), toggle);
  const counts = el("small", null, ticks.ids.some(id => ticks.state(id) === "some")
    ? `選中的 ${items.length} 區清單不一樣：沒動的勾，各區照目前的。每一區各自最佳化。`
    : `程式從下面勾的 ${pickableChoices(ticks.ids).length} 種裡選，每一區各自最佳化。`);
  counts.id = "zone-panel-search-from";
  const info = fold("zone-panel-candidate-info", "候選明細");
  const scope = el("span", null, items.length > 1 ? "範圍只改這幾區，不改材料頁" : "範圍只改這一區，不改材料頁");
  scope.id = "zone-panel-range-note";
  const link = el("a", "zv-link", "材料頁 ▸");
  link.id = "zone-panel-materials"; link.href = "/materials"; link.target = "_blank"; link.rel = "noopener";
  scope.append(link);
  const omitted = catalog.filter(entry => !ticks.count(entry.id) && !pickerFits(entry, walls, ticks)).length;
  info.append(counts, scope, el("small", "zv-picker-foot", `${walls.length === 1 ? "只能貼在別面的" : "選中的面有一面貼不了的"} ${omitted} 筆不列`));
  form.append(...startingListNotes(items, walls, ticks, covered));
  form.append(tools, list, info);
  appendFixedCandidateNote(form, info, ticks.ids);
  return [form];
}
// 抽屜放在分區頁外，限制與曲線重畫時保留設計器正在編輯的草稿。
function openMaterialBuilder() {
  if ($("zone-builder")) return;
  const items = chosenItems(), mode = draft.mode, generation = zoneDesignGeneration;
  const revision = draftRevision;
  const ids = selected.join(",");
  const label = [...new Set(items.map(item => item.wall))].map(wall => {
    const here = items.filter(item => item.wall === wall);
    return wallNames[wall] + " " + (here[0].key === "whole_wall" ? "整面" : here.map(item => item.label.numberText).join(""));
  }).join("、");
  const query = new URLSearchParams({embed: "zones", zones: ids, depth: $("zone-panel-depth")?.value ?? "",
    mode, label, placement: items.every(item => item.wall === "floor") ? "floor" : items.every(item => item.wall === "ceiling") ? "ceiling_build" : "wall"});
  const dialog = el("dialog"); dialog.id = "zone-builder"; dialog.setAttribute("aria-label", "組一個新的材料");
  const bar = el("div", "zv-drawer-head");
  const status = el("span", null, label); status.setAttribute("role", "status");
  const frame = el("iframe"); frame.title = "材料設計器"; frame.src = "/materials/designer?" + query;
  let savedId = null, loading = false;
  const stillHere = () => revision === draftRevision && generation === zoneDesignGeneration && selected.join(",") === ids && draft?.mode === mode;
  const refresh = async () => {
    if (loading || !stillHere()) return;
    loading = true;
    const read = ++libraryRead;
    try {
      const state = await fetchZoneLibrary();
      if (generation !== zoneDesignGeneration || read !== libraryRead) { if (dialog.open) dialog.close(); return; }
      if (state.problem) throw new Error(state.problem);
      applyZoneLibrary(state);
      renderZoneOverview();  // 材料庫補齊後，重新算目前各區的預設候選。
      if (!stillHere()) { if (dialog.open) dialog.close(); return; }
      const currentItems = chosenItems();
      const entry = libraryEntry(savedId);
      if (!entry) throw new Error("材料庫沒有這一筆");
      if (mode === "search") {
        if (!currentItems.every(item => fitsHere(savedId, item.wall)) || referenceOnly(savedId)) throw new Error("這筆材料不能加入所選區的候選");
        draft.query = "";
        candidateAction(currentItems, panelTicks(currentItems), savedId, "add");
      } else {
        if (!assignableEntries().includes(entry)) throw new Error("這筆材料只能由程式最佳化");
        draft.modeChanged = true; draft.materialChanged = true; draft.materialId = savedId; draft.query = ""; markDraftEdited();
        renderZoneOverview();
      }
      if (dialog.open) dialog.close();
    } catch (error) {
      const message = `已存入材料庫，尚未加入：${errorText(error)}`;
      if (dialog.open) status.textContent = message;
      else if (stillHere()) { draft.libraryRetry = {message, refresh}; renderZoneOverview(); }
    } finally { loading = false; }
  };
  const closeDrawer = () => {
    if (loading || frame.contentWindow?.designerSaving?.()) return;
    if (savedId) { dialog.close(); void refresh(); return; }
    if (!frame.contentWindow?.hasUnsavedDraft?.()) { dialog.close(); return; }
    bar.hidden = true; confirm.hidden = false; frame.inert = true; resume.focus();
  };
  frame.addEventListener("load", () => {
    frame.contentDocument?.addEventListener("keydown", event => {
      if (event.key === "Escape") { event.preventDefault(); if (!event.isComposing) closeDrawer(); }
    });
  });
  const close = button("zv-btn", "關閉", "zone-builder-close", closeDrawer);
  const confirm = el("div", "zv-drawer-head"); confirm.hidden = true;
  const resume = button("zv-btn", "繼續組", "zone-builder-resume", () => {
    confirm.hidden = true; bar.hidden = false; frame.inert = false; close.focus();
  });
  confirm.append(el("strong", null, "放棄這個新構造？"), resume,
    button("zv-btn", "放棄", "zone-builder-discard", () => dialog.close()));
  bar.append(status, close);
  dialog.append(bar, confirm, frame);
  document.body.append(dialog);
  dialog.addEventListener("cancel", event => {
    event.preventDefault();
    if (!confirm.hidden) resume.click(); else closeDrawer();
  });
  const receive = event => {
    if (event.origin !== location.origin || event.source !== frame.contentWindow ||
        event.data?.type !== "aosr-material-saved" || typeof event.data.id !== "string" || savedId) return;
    savedId = event.data.id;
    frame.inert = true;
    void refresh();
  };
  window.addEventListener("message", receive);
  dialog.addEventListener("close", () => {
    window.removeEventListener("message", receive);
    dialog.remove();
    $("zone-panel-build")?.focus({preventScroll: true});
  }, {once: true});
  dialog.showModal();
}

function nowMark(items, mode) {
  const here = items.filter(item => zoneMethodMode(item.method) === mode);
  if (!here.length) return [];
  const tag = el("span", "zv-now-tag", "目前"); tag.id = `zone-panel-now-${mode}`; tag.textContent = "目前" + ZONE_MODE_WORDS[mode]; tag.dataset.zones = here.map(item => item.id).join(" ");
  const what = el("span", "zv-now-what");
  if (here.length < items.length) what.append(zonesBrief(here));
  else if (mode === "fixed") what.append(new Set(here.map(item => item.method)).size === 1 ? vprWords(pastedWords(here[0])) : "各區貼的不一樣");
  const line = el("div", "zv-now"); line.append(tag, what);
  return [line];
}
function panelCard() {
  const items = chosenItems(), rule = view.inferred.rule;
  const card = el("div", "zv-edit"); card.id = "zone-panel";
  card.dataset.zones = selected.join(" "); card.tabIndex = -1;
  const walls = [...new Set(items.map(item => item.wall))];
  // 選到的都在同一面牆：剛好是那一面整面（兩區以上）就只寫一行，一區一行的收在摺疊裡；不是整面就給一條「改選整面」。
  const face = walls.length === 1 && view.roles && WALL_ORDER.some(role => view.roles[role] === walls[0]) ? wholeFace(walls[0]) : [];
  const isWhole = face.length > 1 && face.length === items.length && face.every(id => selected.includes(id));
  card.dataset.mode = draft?.mode ?? "";
  const header = el("div", "zv-panel-head");
  const top = el("div", "zv-ehead"), title = el("h2");
  const automatic = items.filter(item => followed.includes(item.id));
  title.append(zonesBrief(items.filter(item => !followed.includes(item.id))));
  if (isWhole) title.append(" 整面");
  if (automatic.length) {
    const partners = el("small");
    partners.append("＋", zonesBrief(automatic), " 跟著選");
    title.append(partners);
  }
  top.append(title, button("zv-link", "不選", "zone-panel-clear", () => select([])));
  const modesBar = el("div", "zv-modes");
  header.append(top, modesBar);
  const content = el("div", "zv-panel-body");
  if (items.length > 1 || (face.length > 1 && !isWhole)) {
    const each = fold("zone-panel-zones", isWhole ? "整面選區" : "選區明細");
    if (face.length && !isWhole) each.append(button("zv-link", `改選${wallNames[walls[0]]}整面（${face.length} 區）`, "zone-panel-whole", () => pick(face)));
    for (const item of items) {
      const chip = el("div", "zv-chip"); chip.dataset.zone = item.id;
      chip.append(zoneWhere(item), button("zv-link zv-x", "不選這區", `zone-panel-drop-${item.wall}-${item.key}`, () => toggleSelected(item.id)));
      each.append(chip);
    }
    content.append(each);
  }
  const modes = ["fixed", "keep_original", "search"];
  const counts = modes.map(mode => items.filter(item => zoneMethodMode(item.method) === mode).length);
  if (counts.filter(Boolean).length > 1) {
    const labels = ["指定貼", KEEP_WORDS, ZONE_MODE_WORDS.search];
    content.append(el("p", "zv-method-counts", "現在：" + counts.flatMap((count, index) => count ? [`${count} 區${labels[index]}`] : []).join("、")));
  }
  card.append(header, content);
  const ruled = new Set(ruleZones(view.faces, view.roles).map(item => item.id));
  const sameAs = rule && items.every(item => ruled.has(item.id)) ? rule.mode : null;
  const canKeepMaterial = items.every(item => item.pasted || zoneMethodMode(item.method) === "fixed");
  const choose = (mode, materialId = "") => { draft = {...untouched(), ...draft, query: draft?.mode === mode ? draft.query : "", mode, materialId, modeChanged: true, materialChanged: Boolean(materialId), rangeRefused: ""}; markDraftEdited(); renderZoneOverview(); };
  // 套用鈕與它上面那一句「為什麼不能套用」：在小框裡打字（最深那一格）時就地更新這兩樣，不重畫整頁（重畫會換掉正在打的那一格）。
  const apply = button("zv-btn zv-main-btn", `套用到 ${items.length} 區`, "zone-panel-apply", () => action(applyDraft));
  const problem = el("p", "zv-problem zv-panel-problem"); problem.id = "zone-panel-problem"; problem.setAttribute("role", "alert");
  const sync = () => {
    const why = draftProblem(items);
    const ticks = draft?.mode === "search" && panelTicks(items);
    if (ticks && content.isConnected && !why) {
      const preview = previewCandidates(items, ticks);
      if (preview.rows && JSON.stringify(preview.rows) !== JSON.stringify(content.querySelector("#zone-panel-form")?.candidateRows)) {
        renderZoneOverview(); return;
      }
    }
    if (ticks) for (const line of content.querySelectorAll("[data-range-entry]")) {
      describeRange(line, items, ticks, libraryEntry(line.dataset.rangeEntry));
    }
    problem.textContent = why; problem.hidden = !why;
    apply.disabled = !draft || (draft.mode === "fixed" && !draft.materialId && !canKeepMaterial) || Boolean(why);
    if (content.isConnected) $("zone-result")?.replaceWith(resultBar(apply.disabled));
  };
  for (const [mode, label] of [["keep_original", "保留原樣"], ["fixed", "指定貼"], ["search", ZONE_MODE_WORDS.search]]) {
    const row = el("label", "zv-segment"); row.id = `zone-panel-option-${mode}`;
    const radio = el("input"); radio.type = "radio"; radio.name = "zone-panel-mode"; radio.id = `zone-panel-mode-${mode}`; radio.checked = draft?.mode === mode;
    // 點「指定貼一種」而規則本身就是指定貼：先放規則那一筆（選它＝跟其他牆面一樣），要別的再換。
    radio.onchange = () => choose(mode, mode === "fixed" ? (draft?.mode === "fixed" ? draft.materialId : (sameAs === "fixed" ? ruleMethod()?.materialId ?? "" : "")) : "");
    const words = el("span", "zv-opt-words", label);
    const ruleMaterial = sameAs === "fixed" ? ruleMethod()?.materialId ?? null : null;
    // 下拉停在「照現在貼的，不換」而選到的本來就都跟著規則，也算同一筆（規則貼的不是材料庫裡的成品時只有這條路）。
    const keepsRule = Boolean(rule) && draft?.materialId === "" && items.every(item => item.method === rule.key);
    // 由程式最佳化也要連清單都跟規則同一份才算：選中的區套用後的清單（照自己的清單加減）有一區跟規則的不一樣就不標。
    const ruleList = sameAs === "search" ? rule.zones[0].allowed : null;
    const listsAsRule = !ruleList || !draft || (panelTicks(items)?.lists.every(list => sameList(list, ruleList)) ?? true);
    const sameHere = sameAs === mode && listsAsRule && (mode !== "fixed" || draft?.mode !== "fixed" || (ruleMaterial !== null && draft.materialId === ruleMaterial) || keepsRule);
    if (sameHere && mode !== "keep_original") { const same = el("span", "zv-same", "＝其他牆面"); same.id = "zone-panel-same"; same.dataset.mode = mode; words.append(same); }
    row.append(radio, words);
    modesBar.append(row);
    for (const mark of nowMark(items, mode)) {
      if (mode === "keep_original") top.insertBefore(mark, top.lastElementChild);
      else content.append(mark);
    }
    if (draft?.mode !== mode) continue;
    if (mode === "fixed") {
      const material = el("div"); material.id = "zone-panel-material"; material.setAttribute("role", "radiogroup");
      material.setAttribute("aria-label", "指定貼哪一種");
      material.onkeydown = event => {
        const step = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[event.key];
        if (!step || event.isComposing) return;
        const rows = visibleRadios(material);
        const index = rows.indexOf(event.target);
        if (index < 0) return;
        event.preventDefault();
        const next = rows[(index + step + rows.length) % rows.length];
        next.focus(); next.click();
      };
      material.addEventListener("toggle", () => radioTabStop(material), true);
      const option = entry => materialRow({entry, name: entry ? vprWords(entry.label) + (entry.id === ruleMaterial && !sameHere ? "（＝其他牆面）" : "") : "照現在貼的，不換",
        description: entry && fixedDescription(entry), role: "radio", state: String(draft.materialId === (entry?.id ?? "")),
        id: `zone-panel-material-${entry?.id ?? ""}`, activate: () => choose("fixed", entry?.id ?? "")});
      if (canKeepMaterial) material.append(option(null));
      const rows = assignableEntries().filter(entry => walls.every(wall => usableHere(entry, wall)) || entry.id === draft.materialId);
      materialFamilies(material, walls, rows.map(entry => [entry, option(entry)]));
      content.append(deepestRow(items, walls, sync), materialFilter(material), ...awayLines(items), material);
      const entry = libraryEntry(draft.materialId);
      if (entry) {
        const details = fold("zone-panel-preview", "材料詳情");
        details.append(materialShow(entry, walls[0])); content.append(details);
      }
    }
    if (mode === "search" && draft?.mode === "search") content.append(...searchForm(items, walls, sync));
  }
  if (draft && draft.mode === null) content.append(deepestRow(items, walls, sync));
  const foot = el("div", "zv-editfoot");
  sync();
  if (["search", "fixed"].includes(draft?.mode)) {
    card.append(button("zv-build", "＋ 組一個新的", "zone-panel-build", openMaterialBuilder));
  }
  if (draft?.libraryRetry) {
    const {message, refresh} = draft.libraryRetry;
    const notice = el("div", "zv-notice", message); notice.setAttribute("role", "status");
    notice.append(button("zv-link", "重試", "zone-library-retry", refresh)); content.prepend(notice);
  }
  content.append(problem);
  foot.append(apply, button("zv-btn", "取消", "zone-panel-cancel", () => select([])));
  card.append(foot);
  return card;
}

function ruleRow() {
  const {inferred, roles} = view, rule = inferred.rule;
  const mode = ruleDraft ? "fixed" : rule?.mode ?? "keep_original";
  const row = el("div", "zv-group-rule"); row.id = "zone-rule";
  const head = el("b", null, "沒另外指定的牆區");
  row.append(head);
  if (!rule) return row;
  row.dataset.mode = rule.mode; row.dataset.key = rule.key;
  const pick = el("select", `zv-pick ${shownOf(mode)}`); pick.id = "zone-rule-mode"; pick.setAttribute("aria-label", "其他牆面的做法");
  for (const [value, words] of Object.entries(ZONE_MODE_WORDS)) { const option = el("option", null, words); option.value = value; pick.append(option); }
  if (mode === "unknown") {
    const option = el("option", null, methodWords(rule)); option.value = mode; option.disabled = true; pick.append(option);
  }
  pick.value = mode;
  pick.onchange = () => {
    if (pick.value === rule.mode) { ruleDraft = false; renderZoneOverview(); }
    else if (pick.value === "fixed") { ruleDraft = true; renderZoneOverview(); }
    else action(() => applyRule({mode: pick.value, allowed: shownList()}));
  };
  row.append(pick);
  if (mode === "fixed") {
    const material = el("select", "zv-pick zv-pick-material"); material.id = "zone-rule-material"; material.setAttribute("aria-label", "其他牆面指定貼哪一種");
    const current = ruleDraft ? null : libraryLabel(rule.zones[0].zone.material, rule.zones[0].wall, rule.zones[0].key);
    if (!current) { const now = el("option", null, ruleDraft ? "選要貼哪一種" : `現在：${pastedWords(rule.zones[0])}`); now.value = ""; material.append(now); }
    const walls = WALL_ORDER.map(role => roles[role]);
    for (const entry of assignableEntries().filter(item => walls.every(wall => usableHere(item, wall)) || item === current)) {
      const option = el("option", null, vprWords(entry.label)); option.value = entry.id; material.append(option);
    }
    material.value = current?.id ?? "";
    material.onchange = () => { if (material.value) action(() => applyRule({mode: "fixed", materialId: material.value})); };
    row.append(material);
    // 其他牆面已經指定貼一種、跟著它的區裡有任何一區是「鋼板直接貼在棉上」的構造（#1208，貼上去的那一筆或原牆本身）：
    // 跟著規則的區圖上不寫名字，那兩行沒地方放，提醒寫在這張卡上（面板關著也看得到；伺服器給的那一行）。選到一半不寫。
    if (!ruleDraft && rule.zones.some(item => zoneReferenceOnly(item.zone.material, item.wall, item.key))) {
      const mark = el("p", "zv-rule-ref", zoneReference.map_line); mark.id = "zone-rule-reference"; row.append(mark);
    }
  }
  return row;
}
function specialRow() {
  const {inferred} = view;
  const zones = inferred.special.flatMap(group => group.zones);
  const row = el("div", "zv-group-exceptions"); row.id = "zone-special";
  row.dataset.count = String(zones.length);
  row.append(el("h2", null, `另外指定的 ${zones.length} 區`));
  inferred.special.forEach((group, index) => {
    const block = el("div", "zv-group"); block.id = `zone-special-group-${index}`;
    block.dataset.key = group.key; block.dataset.zones = group.zones.map(item => item.id).join(" ");
    const body = el("div");
    const where = el("div", "zv-where");
    for (const part of byFace(group.zones)) {
      if (!part.whole) { for (const item of part.zones) where.append(zoneWhere(item)); continue; }
      const line = el("div", "zv-where-line"); line.dataset.whole = part.wall;
      line.append(`${wallNames[part.wall]}整面 `, ...part.zones.map(item => numberBadge(item.label.number)));
      where.append(line);
    }
    body.append(where, el("div", "zv-what", methodWords(group)));
    block.append(el("i", `zv-bar ${shownOf(group.mode)}`), body,
      button("zv-btn", "改", `zone-special-edit-${index}`, () => (symmetryIsOn() ? pick : select)(group.zones.map(item => item.id))));
    row.append(block);
  });
  if (zones.length) {
    const acts = el("div", "zv-acts");
    acts.append(button("zv-btn", `這 ${zones.length} 區改回跟其他牆面一樣`, "zone-special-reset", () => action(resetSpecial)));
    row.append(acts);
  }
  return row;
}
function listName(id) { return id === ZONE_NONE ? "也可以不貼（保留原樣）" : vprWords(choiceName(id)); }
// 這張清單卡管的就是「其他牆面」那一組（規則本身是由程式最佳化）：清單同一份、而且跟著規則的區全在這一張
// （#1199：規則那一組被各區自己的範圍拆成幾張時，哪一張都不是整組）。
function ruleCard(group) {
  const rule = view.inferred.rule;
  return rule?.mode === "search" && rule.key === group.key && rule.zones.every(item => group.zones.includes(item));
}
// 一段範圍寫成字（公分）：兩端一樣寫一個數。
function spanWords(pair) { return pair[0] === pair[1] ? `${centimetres(pair[0])}` : `${centimetres(pair[0])}～${centimetres(pair[1])}`; }
// 「棉 5～10 cm、空腔 0～15 cm」（#1199）：材料沒有空腔就只寫棉。第三階段頁的預覽寫法同這一句（gradient.js）。
function rangeWords(range) { return [`棉 ${spanWords(range.porous)} cm`, ...(range.cavity ? [`空腔 ${spanWords(range.cavity)} cm`] : [])].join("、"); }
function zoneRangeDescription(setting, entry, who = "這一區") {
  return `${who}自己的範圍：${rangeWords(ownRange(setting, entry))}（材料頁是${rangeWords(pageRange(entry))}）`;
}
window.zoneRangeDescription = zoneRangeDescription;
// 一張清單卡管哪幾區：這一組就是規則那一組（規則本身是由程式最佳化）就寫「沒另外指定的牆區」，其餘的區點名。
// alone：全部由程式最佳化的區只有這一組（一張卡）。
function listWho(group, prefix) {
  const {inferred} = view;
  const line = el("div", "zv-who"); line.id = `${prefix}-who`;
  line.append(`這組有 ${group.zones.length} 區：`);
  const byRule = ruleCard(group) ? new Set(inferred.rule.zones.map(item => item.id)) : new Set();
  const pieces = byRule.size ? [["沒另外指定的牆區"]] : [];
  for (const part of byFace(group.zones.filter(zone => !byRule.has(zone.id)))) {
    if (part.whole) { pieces.push([`${wallNames[part.wall]}整面 `, ...part.zones.map(item => numberBadge(item.label.number))]); continue; }
    for (const item of part.zones) {
      pieces.push(item.key === "whole_wall" ? [`${wallNames[item.wall]}整面`] : [`${wallNames[item.wall]} `, numberBadge(item.label.number), ` ${item.label.name}`]);
    }
  }
  pieces.forEach((piece, index) => { if (index) line.append("、"); line.append(...piece); });
  return line;
}
function searchGroupRow(group, index) {
  const row = el("div", "zv-search-group");
  const prefix = index ? `zone-group-${index + 1}` : "zone-group";
  row.id = prefix;
  const ids = group.zones.map(item => item.id), first = group.zones[0];
  row.dataset.zones = ids.join(" ");
  row.dataset.count = String(ids.length);
  const who = listWho(group, prefix);
  const depth = shownDepth(group.zones);
  const summary = el("div", "zv-group-summary", depth ? `最深 ${depth} cm` : blankDepthWords(group.zones) === "不限" ? "最深不限" : "最深各區不同");
  const narrowed = Object.keys(first.ranges ?? {}).filter(id => {
    const entry = libraryEntry(id);
    return entry?.kind === "optimizable" && Object.keys(rangeParts(entry)).some(part =>
      rangeParts(entry)[part] && !samePair(ownRange(first, entry)[part], pageRange(entry)[part]));
  });
  if (narrowed.length) summary.append(el("span", null, ` · ${narrowed.length} 筆範圍改窄了`));
  const choices = el("ul", "zv-group-choices");
  choices.id = prefix + "-choices";
  const allowed = (first.allowed ?? []).filter(id => group.zones.some(zone => zone.effectiveAllowed?.includes(id)));
  for (const id of [...allowed.filter(id => id !== ZONE_NONE), ...allowed.filter(id => id === ZONE_NONE)]) {
    const entry = libraryEntry(id), item = el("li", null, listName(id));
    item.dataset.choice = id;
    if (entry?.kind === "optimizable") {
      const range = el("small", "zv-range", first.ranges?.[id] ? zoneRangeDescription(first, entry, ids.length > 1 ? "這幾區" : "這一區") : adjustWords(entry));
      if (first.ranges?.[id]) range.dataset.ranged = "true";
      item.append(range);
    }
    if (referenceOnly(id)) restrictedReason(item, entry, true);
    choices.append(item);
  }
  if (!first.allowed?.length) choices.append(el("li", null, "程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）"));
  const editing = sameList(ids, selected);
  const selectGroup = editing ? el("span", "zv-editing", "正在改") : button("zv-link", "選這幾區", prefix + "-select", async () => {
    await pick(ids);
    $("zone-panel")?.focus({preventScroll: true});
  });
  selectGroup.id = prefix + "-select";
  row.append(who, selectGroup, summary, choices);
  const refused = referenceRefusals().filter(line => ids.includes(`${line.zones[0].wall}/${line.zones[0].zone}`));
  const problem = first.allowed && allowedProblem(first.allowed);
  for (const [suffix, words] of [["reference", refused.map(line => line.text).join("；")],
    ["problem", problem && problem + "加可調材料或「也可以不貼」，或" + (editing ? "在選區改成「指定貼」。" : "選這幾區改成「指定貼」。")]]) {
    if (!words) continue;
    const line = el("p", "zv-problem", words);
    line.id = prefix + "-" + suffix; line.setAttribute("role", "alert"); row.append(line);
  }
  return row;
}
function groupDetails() {
  const groups = fold("zone-groups", "依做法與候選分組");
  groups.append(ruleRow(), specialRow(), ...view.list.groups.map(searchGroupRow));
  return groups;
}
function detailTable() {
  const {faces, roles} = view;
  const wrap = el("div", "zv-more");
  const details = fold("zone-details", "每一區的尺寸與代號");
  const table = el("table");
  for (const wall of [...WALL_ORDER.map(role => roles[role]), "ceiling", "floor"]) {
    for (const item of faces[wall]) {
      const row = el("tr"); row.dataset.zone = item.id;
      const flat = wall === "floor" || wall === "ceiling";
      const size = !item.box ? "" : flat ? `${shownMetres(item.box[1] - item.box[0], 2)} × ${shownMetres(item.box[3] - item.box[2], 2)} 公尺`
        : `寬 ${shownMetres(item.box[1] - item.box[0], 2)} × 高 ${shownMetres(item.box[3] - item.box[2], 2)} 公尺`;
      const what = item.shown === "unknown" ? uncertainWords(item.construction) : item.shown === "search" ? ZONE_MODE_WORDS.search : item.shown === "fixed" ? ZONE_MODE_WORDS.fixed : "保留原樣";
      row.append(el("td", null, zoneWhereText(item)), el("td", null, item.key === "whole_wall" ? "" : `代號 ${item.key}`), el("td", null, size), el("td", null, what));
      table.append(row);
    }
  }
  details.append(table); wrap.append(details);
  return wrap;
}

/* ───── 最底下：結果列 ───── */
/* ───── 左右對稱那一塊（圖例底下）與配對細節 ───── */
let symmetryEditGeneration = null;
function setSymmetry(on) {
  if (on === symmetryIsOn()) return;
  followed = [];
  const design = designForEdit();
  if (on) delete design.left_right_symmetry; else design.left_right_symmetry = "off";
  unsaved = true; symmetryEditGeneration = zoneDesignGeneration; symmetryDetail = false;
  // 先畫後問（同 afterZoneChange）：上一版判決是另一個開關問的，鏡子、配對都對不上（#1265 單元 3 之二收尾追加第 4 條）。
  staleZoneLimits();
  renderInputBars();
  renderElevation();
  refreshZoneLimits();
  renderZoneOverview();
}
window.zoneSymmetry = {isOn: symmetryIsOn, set: setSymmetry};
// 句子裡的 {first}／{second}／{other} 換成那一區的牆名與號碼。
function withZones(text, entry) {
  const line = el("span");
  const items = {"{first}": entry.firstItem, "{second}": entry.secondItem, "{other}": entry.otherItem};
  for (const piece of text.split(/(\{first\}|\{second\}|\{other\})/)) {
    if (items[piece]) line.append(zonesBrief([items[piece]])); else if (piece) line.append(piece);
  }
  return line;
}
function symmetryBox() {
  const box = el("div", "zv-sym"); box.id = "zone-symmetry"; box.dataset.state = pairing.state;
  const head = el("div", "zv-sym-head");
  const title = el("span", "zv-sym-title", SYMMETRY_TITLE); title.id = "zone-symmetry-title";
  const toggle = el("span", "zv-seg"); toggle.setAttribute("role", "group"); toggle.setAttribute("aria-labelledby", "zone-symmetry-title");
  for (const [on, words, id] of [[true, "開", "zone-symmetry-on"], [false, "關", "zone-symmetry-off"]]) {
    const item = button("zv-seg-item", words, id, () => setSymmetry(on));
    item.setAttribute("aria-pressed", String(pairing.on === on));
    toggle.append(item);
  }
  head.append(title, toggle);
  const body = el("div", "zv-sym-body");
  if (pairing.state === "pending") body.append(el("div", "zv-sym-quiet", "配對計算中…"));
  else if (pairing.state === "unavailable") body.append(el("div", null, pairing.message));
  else if (pairing.state === "none") body.append(el("small", null, "沒有由程式最佳化的區可配對"));
  else if (pairing.state === "off") body.append(el("small", null, "每一區各自調"));
  else if (pairing.state === "no-axis") body.append(el("div", null, `${pairing.message}，所以現在每一區各自調。`));
  else if (pairing.state === "paired") {
    const pairs = el("div", "zv-sym-row"); pairs.id = "zone-symmetry-pairs";
    pairs.append(el("span", "zv-sym-count", pairing.pairs.length ? `配成 ${pairing.pairs.length} 對` : "沒有配成的對"));
    for (const pair of pairing.pairs) {
      const item = el("span", "zv-sym-pair"); item.dataset.zones = `${pair.first.id} ${pair.second.id}`;
      item.append(glyphIcon(pair.shape), zonesBrief([pair.first]), " ＋ ", zonesBrief([pair.second]));
      pairs.append(item);
    }

    // 「配對細節」放在第一列右邊：沒配成的列再多，它也不會被擠到底下那一條後面（1440×900 不捲動就點得到）。
    if (pairing.missed.length) {
      const more = button("zv-link", symmetryDetail ? "配對細節 ▾" : "配對細節 ▸", "zone-symmetry-details", () => { symmetryDetail = !symmetryDetail; renderZoneOverview(); });
      more.setAttribute("aria-expanded", String(symmetryDetail));
      pairs.append(more);
    }
    body.append(pairs);
    if (pairing.missed.length) {
      const missed = el("div", "zv-sym-row"); missed.id = "zone-symmetry-missed";
      for (const entry of pairing.missed) {
        const item = el("span", "zv-sym-missed"); item.dataset.zones = [entry.firstItem, entry.secondItem].filter(Boolean).map(zone => zone.id).join(" ");
        // 為什麼沒配成、要怎麼辦，直接寫在這一列（伺服器給的那一句）；差幾公分與切法在配對細節。
        item.append(el("span", "zv-sym-tag", "沒配成一對"), zonesBrief([entry.firstItem, entry.secondItem].filter(Boolean)), el("span", "zv-sym-sum", entry.summary.replace("回「房間方案輸入」", "按那一面的「切區」，")));
        missed.append(item);
      }
      body.append(missed);
    }

  }
  if (pairing.axis && !pairing.axis.centred) {
    const left = pairing.axis.left, right = zoneLimits.symmetry.projections[left][0];
    body.prepend(el("div", null, `鏡子不在正中，左牆 ${zoneList(left).length} 區、右牆 ${zoneList(right).length} 區，各自切區。`));
  }
  box.append(head, body);
  return box;
}
// 配對細節的小圖：那一面牆、第二區現在的框（橘）、第一區對過來的框（斜線），對稱軸在這一面上就畫出來。
function missFigure(entry) {
  const {views, wallSize, screenBox, flip} = window.elevationGeometry;
  const {wall} = entry.figure, [width, height] = wallSize(wall);
  const scale = Math.min(300 / width, 150 / height), left = 6, top = 20;
  const figure = svg("svg", {class: "zv-sym-figure", width: width * scale + 12, height: height * scale + 28,
                             viewBox: `0 0 ${width * scale + 12} ${height * scale + 28}`, role: "img", "aria-label": `${wallNames[wall]}上這兩區的位置`});
  const rectOf = (u_m, v_m) => { const [a0, a1, b0, b1] = screenBox(views[wall], width, height, {u_m, v_m});
    return {x: left + a0 * scale, y: top + (height - b1) * scale, width: (a1 - a0) * scale, height: (b1 - b0) * scale}; };
  figure.append(svg("rect", {class: "zv-fig-wall", x: left, y: top, width: width * scale, height: height * scale}),
    svg("rect", {class: "zv-fig-now", ...rectOf(entry.figure.second_u_m, entry.figure.second_v_m)}),
    svg("rect", {class: "zv-fig-hatch", ...rectOf(entry.figure.mirrored_u_m, entry.figure.mirrored_v_m)}),
    svg("rect", {class: "zv-fig-over", ...rectOf(entry.figure.mirrored_u_m, entry.figure.mirrored_v_m)}));
  if (entry.firstItem.wall === wall) figure.append(svg("rect", {class: "zv-fig-now", ...rectOf(entry.firstItem.zone.u_m ?? [0, width], entry.firstItem.zone.v_m ?? [0, height])}));
  if (pairing.axis?.end_walls.includes(wall)) {
    const [a] = flip(views[wall], width, height, pairing.axis.across_m, 0);
    const axis = svg("g", {class: "zv-sym-axis"});
    axis.append(svg("line", {x1: left + a * scale, x2: left + a * scale, y1: top - 4, y2: top + height * scale + 4}),
      svg("text", {class: "zv-sym-axis-label", x: left + a * scale, y: 13, "text-anchor": "middle"}, "對稱軸"));
    figure.append(axis);
  }
  return figure;
}
function symmetryDetailCard() {
  const card = el("div", "zv-card zv-sym-detail"); card.id = "zone-symmetry-detail";
  const head = el("div", "zv-sym-detail-head");
  head.append(el("h2", null, "配對細節：為什麼沒配成一對"), button("zv-link", "✕ 關閉", "zone-symmetry-detail-close", () => { symmetryDetail = false; renderZoneOverview(); }));
  card.append(head);
  for (const entry of pairing.missed) {
    const part = el("div", "zv-sym-why"); part.dataset.kind = entry.kind;
    const name = el("div", "zv-sym-why-name"); name.append(zonesBrief([entry.firstItem, entry.secondItem].filter(Boolean)), `：${entry.reason}`);
    part.append(name);
    if (entry.figure) {
      const key = el("div", "zv-sym-quiet");
      key.append("橘色：", zonesBrief([entry.firstItem, entry.secondItem]), " 現在的位置；斜線：", zonesBrief([entry.firstItem]), " 對過來的位置");
      part.append(missFigure(entry), key);
    }
    part.append(el("p", null).appendChild(withZones(entry.detail, entry)).parentElement);
    if (entry.cuts.length) {
      part.append(el("div", "zv-sym-how", "怎麼改就配得上（兩種挑一種，按那一面的「切區」）："));
      const list = el("ul");
      for (const cut of entry.cuts) { const row = el("li"); row.append(withZones(cut, entry)); list.append(row); }
      part.append(list);
    }
    card.append(part);
  }
  const foot = el("div", "zv-sym-quiet", `程式不會自己改你切的分區，也不會自己改每一區的做法。${pairing.axis?.words ?? ""}`); foot.id = "zone-symmetry-axis-words";
  card.append(foot);
  return card;
}

function hasPendingDraft() {
  if (!draftTouched || !draft || !selected.length) return false;
  const items = chosenItems();
  if (items.some(item => zoneMethodMode(item.method) !== draft.mode)) return true;
  if (draft.mode === "keep_original") return false;
  // 背後（#1330）跟存的不一樣才算沒套用：改成 VPR 又改回原牆＝沒改（#1340 審查）。
  if (draft.mode === "search" && draft.backing !== undefined && items.some(item => (zoneSetting(item.wall, item.key).backing_entry ?? "") !== draft.backing)) return true;
  if (draftDepth(items) !== undefined) return true;
  if (draft.mode === "fixed") return Boolean(draft.materialId && items.some(item =>
    libraryLabel(item.zone.material, item.wall, item.key)?.id !== draft.materialId));
  const ticks = panelTicks(items);
  if (!ticks) return false;
  const ranges = rangeEdits(items, ticks);
  return Boolean(ranges.problem || ranges.edits?.size || items.some((item, index) => !sameList(ownList(item), ticks.lists[index])));
}
function resultBar(cannotApply) {
  const {faces, roles} = view;
  const all = Object.values(faces).flat();
  const of = shown => all.filter(item => item.shown === shown);
  const area = items => items.reduce((sum, item) => sum + item.area, 0).toFixed(1);
  const bar = el("div", "zv-foot"); bar.id = "zone-result";
  const part = (shown, words, items) => {
    const item = el("span", "zv-part"); item.id = `zone-result-${shown}`; item.dataset.count = String(items.length);
    item.dataset.zones = items.map(zone => zone.id).join(" ");
    // 由程式最佳化那一項照單位數寫（#1206）：成對的兩區算一個單位。數字是伺服器回的。
    const units = shown === "search" && pairing?.state === "paired" && pairing.pairs.length
      ? `：${[`${pairing.pairs.length} 對一起調`, ...(pairing.alone ? [`${pairing.alone} 區各自調`] : [])].join("、")}` : "";
    // 面積寫小字（樣稿）：由程式最佳化那一項多了幾對幾區，整條才放得進一行。
    item.append(el("span", `zv-sw ${shown}`), `${words} `, el("b", null, `${items.length} 區`), units, el("span", "zv-area", `（約 ${area(items)} 平方公尺）`));
    return item;
  };
  bar.append(part("search", ZONE_MODE_WORDS.search, of("search")), el("span", "zv-sep"), part("fixed", "指定貼", of("fixed")), el("span", "zv-sep"));
  if (of("unknown").length) bar.append(part("unknown", "看不出", of("unknown")), el("span", "zv-sep"));
  const walls = WALL_ORDER.flatMap(role => faces[roles[role]]);
  const still = [];
  const stillWalls = walls.filter(item => item.shown === "still").length;
  if (stillWalls) still.push(`牆面 ${stillWalls} 區`);
  for (const wall of ["ceiling", "floor"]) {
    const count = faces[wall].filter(item => item.shown === "still").length;
    if (count) still.push(count === faces[wall].length ? wallNames[wall] : `${wallNames[wall]} ${count} 區`);
  }
  const doors = walls.filter(item => item.shown === "opening" && item.special === "門").length;
  const windows = walls.filter(item => item.shown === "opening" && item.special === "窗").length;
  if (doors) still.push(`門 ${doors} 扇`);
  if (windows) still.push(`窗 ${windows} 扇`);
  const rest = el("span", "zv-part"); rest.id = "zone-result-still";
  rest.append(el("span", "zv-sw still"), `保留原樣：${still.length ? still.join("、") : "沒有"}`);
  // 「拿去搜尋」照輸入頁頁底那一顆（zone_modes.js 的 renderSearchAction）的擋法：沒有區由程式最佳化、有紅框（清單只放了固定成品）、有超過限制的區，都按不下去。
  const pending = hasPendingDraft();
  const go = button("zv-go", pending ? "套用並拿去搜尋" : "拿去搜尋", "zone-page-search", () => action(async () => {
    if (!pending) { await sendToSearch(); return; }
    const generation = zoneDesignGeneration;
    // 外層鎖涵蓋套用、重查、存檔與跳頁，各步驟仍自行管理內層的輸入鎖。
    document.body.inert = true;
    try {
      if (!await applyDraft()) return;
      if (generation !== zoneDesignGeneration || !overviewOpen) {
        searchNote("頁面已切換，這次沒有拿去搜尋", "notice"); return;
      }
      await sendToSearch();
    } finally { document.body.inert = false; }
  }));
  // 按不下去時旁邊一句講為什麼（細節在上面那一行紅字）。
  const remaining = all.some(item => pending && draft.mode !== null && selected.includes(item.id) ? draft.mode === "search" : item.shown === "search");
  const why = pending && cannotApply ? "先完成這幾區的設定" : !remaining ? "還沒有由程式最佳化的區" : pending ? "" : stuckZoneNames().length || referenceRefusals().length ? "清單有紅框要先處理" : limitsRedCount() > 0 ? "有超過限制的區要先改" : "";
  go.disabled = Boolean(why);
  // 原因那一句收在按鈕正下方（小字一行），不跟前面三項搶同一行的寬度。
  const box = el("span", "zv-gobox"); box.append(go);
  if (why) { const line = el("span", "zv-why", why); line.id = "zone-page-search-why"; box.append(line); }
  bar.append(rest, box);
  return bar;
}
// 結果列上面那一行：剛才那一步的結果（zone_modes.js 的 zoneNote），以及「拿去搜尋」為什麼按不下去。
function noteLines() {
  const wrap = el("div", "zv-notes");
  const red = limitsRedCount(), stuck = stuckZoneNames();
  // 清單裡有「鋼板直接貼在棉上」的構造（#1208）：開跑會拒收的那一句（伺服器給的）也寫在這一行。
  const blocked = [...(stuck.length ? [stuckWords(stuck)] : []), ...referenceRefusals().map(line => line.text),
    ...(red ? [`有 ${red} 條超過限制，改好之前不能拿去搜尋：${(zoneLimits?.refusals ?? []).join("；")}`] : [])];
  if (blocked.length) { const line = el("p", "limit-red", blocked.join("；")); line.id = "zone-page-blocked"; wrap.append(line); }
  for (const text of [zoneDesignNote, zoneLibraryProblem].filter(Boolean)) wrap.append(el("p", "notice", text));
  const saidText = saidWords();
  if (saidText) { const said = el("p", "notice", saidText); said.id = "zone-page-said"; said.setAttribute("role", "alert"); wrap.append(said); }
  const note = el("p", zoneNoteKind, zoneNoteText); note.id = "zone-page-note"; note.setAttribute("aria-live", "polite");
  wrap.append(note);
  return wrap;
}

// 那一行寫什麼：留著的那一句在前，訊息列後來換的警示（跟它不同句才寫）接在後面。
function saidWords() { return [keptSaid, pageSaid === keptSaid ? "" : pageSaid].filter(Boolean).join("；"); }

// 右欄的草稿離開這一頁照樣留著（回來接著改），但沒套用就不進存檔（#1340：選了背後沒按套用就回輸入頁、按儲存，存檔裡什麼都沒有）：
// 回輸入頁時在「設定每一區貼什麼」旁邊講一句；再打開這一頁就收掉。
function pendingNote(show) {
  let note = $("zone-pending-note");
  if (!note && show) {
    note = el("small", "notice", "分區頁右欄還有沒套用的改動，儲存不會帶上：回去按「套用到」");
    note.id = "zone-pending-note"; note.setAttribute("role", "status");
    $("zone-page-open")?.after(note);
  }
  if (note) note.hidden = !show;
}
function closeOverview() {
  pendingNote(hasPendingDraft());
  if (cutting) restoreElevation();
  overviewOpen = false;
  familyOpen.clear();
  renderZoneOverview();
  // 這一頁開著時輸入頁是藏起來的：那時重畫的平面圖，字量不到寬度、避不開彼此。回來看得到了，照同一份再畫一次。
  drawShownPlan();
  // 捲回打開這一頁之前的地方（從舊表某一列按「改這一區 ▸」進來的，改完回到那一列）。
  window.scrollTo(0, overviewReturnY);
}
// 這一頁沒有存檔鍵：改動當場套到畫面，但要存檔才進檔案。改過、還沒存的時候標題列講一句怎麼存（兩條路）。
function unsavedLine() {
  const line = el("span", "zv-unsaved", "改的還沒存進檔案：回「房間方案輸入」按「儲存」，或按底下的「拿去搜尋」（會先存再跳到搜尋頁，不會自己開始算；到那一頁按「開始搜尋」才會算）");
  line.id = "zone-page-unsaved"; line.setAttribute("role", "status");
  return line;
}
function pageHead() {
  const head = el("div", "zv-head");
  head.append(el("h1", null, "這間房貼什麼"));
  return {head, back: button("zv-link", "◂ 回房間方案輸入", "zone-page-close", closeOverview)};
}
// 整頁先在旁邊組好，組得出來才一次換上去：組到一半丟例外（程式的錯）時畫面留在上一次的樣子、最上面加一句，
// 沒有上一次的樣子就只放標題與「回房間方案輸入」。例外照樣丟給瀏覽器的錯誤紀錄（不吞掉）。
function renderZoneOverview() {
  glyphAt.clear();
  const page = $("zone-page");
  if (!page) return;
  page.hidden = !overviewOpen;
  document.querySelector("main").hidden = overviewOpen;
  document.body.classList.toggle("zone-page-open", overviewOpen);
  // 整份文件也掛一個記號：底列黏在視窗下緣，瀏覽器自己把東西捲進畫面時要讓出底列那一段（zone_overview.css）。
  document.documentElement.classList.toggle("zone-page-open", overviewOpen);
  if (!overviewOpen || cutting) return;  // 切完才換掉選取快照與右欄草稿
  const scrolled = window.scrollY;
  const mapScroll = page.querySelector(".zv-mapwrap")?.scrollTop ?? 0;
  let parts;
  try { parts = overviewParts(); } catch (error) {
    if (!page.querySelector("#zone-page-close")) { const {head, back} = pageHead(); head.append(back); page.replaceChildren(head); }
    if (!page.querySelector("#zone-page-broken")) {
      const line = el("p", "notice", "這一頁畫不出來（程式出錯）：畫面停在上一次的樣子，剛才那一步可能沒顯示出來。先按「回房間方案輸入」，重新整理網頁再試。");
      line.id = "zone-page-broken"; line.setAttribute("role", "alert");
      page.firstElementChild.after(line);
    }
    setTimeout(() => { throw error; });
    return;
  }
  // 正在「最深」或範圍那幾格打字時別的事觸發重畫（晚到的限制檢查）：字記在草稿裡不會掉，焦點放回那一格。
  const focused = document.activeElement?.id ?? "";
  const typing = focused === "zone-panel-depth" || focused === "zone-panel-picker-search" || focused.startsWith("zone-panel-range-") ? focused : null;
  const prior = page.querySelector("#zone-panel");
  const scrollZone = prior?.dataset.zones, scrollMode = prior?.dataset.mode;
  const panelScroll = prior?.querySelector(".zv-panel-body")?.scrollTop ?? 0;
  visibleCurves.disconnect(); pendingSparks.clear();
  // 背景檢查回來時仍是同一組區，保留正在操作的面板或清單卡，只換這次算出的內容。
  const cards = page.querySelectorAll("#zone-panel, .zv-search-group");
  page.replaceChildren(...parts);
  for (const card of cards) {
    const next = $(card.id);
    if (!next || card.dataset.zones !== next.dataset.zones || card.dataset.mode !== next.dataset.mode) continue;
    card.replaceChildren(...next.childNodes);
    next.replaceWith(card);
  }
  page.querySelectorAll(".zv-spark[data-curve]").forEach(chart => visibleCurves.observe(chart));
  const mapPane = page.querySelector(".zv-mapwrap");
  if (mapPane) mapPane.scrollTop = mapScroll;
  const scrollBody = page.querySelector(".zv-panel-body");
  if (scrollBody && scrollZone === selected.join(" ") && scrollMode === draft?.mode) scrollBody.scrollTop = panelScroll;
  const again = typing || ["zone-cut-", "zone-panel-mode-", "zone-panel-material-"].some(prefix => focused.startsWith(prefix)) ? $(focused) : null;
  again?.focus({preventScroll: true});
  // 放回去的是新畫的一格，原本的反白帶不過來（數字格讀不到選取範圍）。那一格畫的是原值（草稿裡沒有它的字，沒有 typed 記號）
  // ＝使用者進了格子還沒打字：整格反白，跟 Tab 進去時一樣，接著打的字換掉舊值——不然黏在舊值後面，12.5 打 20 變成 12.520（#1231）。
  // 畫的是打到一半的字就不動：游標在字尾，接著打的字接在後面。
  if (again instanceof HTMLInputElement && again.type === "number" && !again.dataset.typed) again.select();
  window.scrollTo(0, scrolled);
}
function overviewParts() {
  const {head, back} = pageHead();
  const {wallSize} = window.elevationGeometry;
  if (!scheme || !inputDefaults || Object.keys(wallNames).some(wall => !wallSize(wall))) {
    head.append(el("p", null, "先把房間的長、寬、高都填好正數，才畫得出這間房。"), back);
    return [head];
  }
  // 換了方案（打開別份、從範本開新的）：上一份選的區、面板上點的都不算數。
  if (seenGeneration !== zoneDesignGeneration) { seenGeneration = zoneDesignGeneration; selected = []; draft = null; draftTouched = false; ruleDraft = false; keptRule = null; keptRuleZone = null; unsaved = symmetryEditGeneration === zoneDesignGeneration; keptSaid = ""; }
  const roles = overviewRoles();
  const room = {Lx: wallSize("floor")[0], Ly: wallSize("floor")[1], Lz: wallSize("x0")[1]};
  const depth = roles.front[0] === "x" ? room.Lx : room.Ly, across = roles.front[0] === "x" ? room.Ly : room.Lx;
  head.append(el("p", null, `你的聆聽室，深 ${shownMetres(depth, 2)} × 寬 ${shownMetres(across, 2)} × 高 ${shownMetres(room.Lz, 2)} 公尺`), ...(unsaved ? [unsavedLine()] : []), back);
  const faces = Object.fromEntries(Object.keys(wallNames).map(wall => [wall, faceZones(wall)]));
  const byId = new Map(Object.values(faces).flat().map(item => [item.id, item]));
  // 「其他牆面」不自己翻：定下來的那一種還有區照著，就一直是它（見 inferOtherWalls）。
  const ruled = ruleZones(faces, roles);
  if (keptRuleZone) { keptRule = ruled.find(item => item.id === keptRuleZone)?.method ?? keptRule; keptRuleZone = null; }
  const inferred = inferOtherWalls(ruled, keptRule);
  keptRule = inferred.rule?.key ?? null;
  const earlier = view?.byId;
  view = {roles, room, faces, byId, inferred, list: zoneSearchList(searchZones(faces, roles), inferred.rule?.key)};
  // 選的區那一面重切過（#1262）：代號不在了，或同一個代號換成另一塊長方形（自動補滿的區重新編號），就不再算選著、講一句——
  // 不然小框還沒套用的改動會套到另一塊上。同一份方案裡代號只會因為切面而消失（換方案上面那一行已先清掉選取）。
  // 比方案裡的座標（不比畫面座標：只改房間尺寸時整面一區、鏡像面的畫面座標會跟著變）。一區都不剩，面板與草稿也收起來。
  const shape = item => JSON.stringify([item.zone.area, item.zone.u_m ?? null, item.zone.v_m ?? null]);
  const moved = selected.filter(id => !byId.has(id) || (earlier?.has(id) && shape(earlier.get(id)) !== shape(byId.get(id))));
  selected = selected.filter(id => !moved.includes(id));
  if (moved.length) {
    const lost = !selected.length && draftTouched;  // 草稿只在一區都不剩時才丟
    keptSaid = `剛才選的區有 ${moved.length} 區那一面重切過，已經不是原來那一塊，不再算選著${lost ? "；小框還沒套用的改動放掉了" : ""}，要的話重新選`;
  }
  if (!selected.length) { draft = null; draftTouched = false; }
  // 圖上只在另外指定的區（與天花、地板）寫材料名。
  const apart = new Set(inferred.special.flatMap(group => group.zones.map(item => item.id)));
  for (const item of byId.values()) { item.apart = apart.has(item.id); item.named = item.wall === "floor" || item.wall === "ceiling" || item.apart; }
  const body = el("div", "zv-main");
  const mapWrap = el("div", "zv-mapwrap");
  pairing = readPairing(byId);
  if (!pairing.missed.length) symmetryDetail = false;
  const {map, legendTop, places, underWidth} = drawMap(faces, roles, room);
  const under = el("div", "zv-under"); under.style.top = `${legendTop}px`; under.style.width = `${underWidth}px`;
  under.append(legend(), symmetryBox());
  mapWrap.append(map, under, ...faceTools(places));
  const side = el("div", "zv-side");
  if (symmetryDetail) under.append(symmetryDetailCard());
  under.append(groupDetails(), detailTable());
  if (selected.length) side.append(panelCard());
  else {
    const slot = el("p", "zv-empty", "尚未選區"); slot.id = "zone-slot"; side.append(slot);
  }
  body.append(mapWrap, side);
  return [head, body, noteLines(), resultBar(side.querySelector("#zone-panel-apply")?.disabled)];
}
// 同一區隨尺寸移動，讓上次選取的幾何跟上。若先前已經真的重切，保留原快照交 #1262 判斷。
window.zoneOverviewResized = (wall, zone, next) => {
  const prior = view?.byId.get(`${wall}/${zone.id}`);
  if (prior && prior.zone.area === zone.area && sameList(prior.zone.u_m, zone.u_m) && sameList(prior.zone.v_m, zone.v_m))
    prior.zone = {...zone, ...next};
};
window.renderZoneOverview = renderZoneOverview;
window.openZoneOverview = id => {
  if (!overviewOpen) overviewReturnY = window.scrollY;
  overviewOpen = true; pageSaid = ""; pendingNote(false);
  renderZoneOverview();
  if (id && view?.byId.has(id)) pick([id]);
  window.scrollTo(0, 0);
};
// 輸入頁訊息列每寫一句（app.js 的 say）就叫一次：警示照抄到這一頁，換成成功的一句就拿掉。
// 這一頁打開時從空的開始（上面那一行）：打開之前輸入頁講過的話不帶進來。
// 方案與每一區的做法都存進檔案了（輸入頁按「儲存」、另存，或「拿去搜尋」先存的那一步）：那一句拿掉。
window.zonePageHasEdits = () => unsaved && (seenGeneration === zoneDesignGeneration || symmetryEditGeneration === zoneDesignGeneration);
window.zonePageSaved = () => { symmetryEditGeneration = null; unsaved = false; renderZoneOverview(); };
window.zonePageSay = (text, kind) => {
  const next = kind === "notice" ? text : "";
  if (next === pageSaid) return;
  pageSaid = next;
  if (cutting) $("zone-page").querySelector(".zv-notes").replaceWith(noteLines());
  else renderZoneOverview();
};
// 「拿去搜尋」被還在路上的一批改動擋下（zone_modes.js 的 sendToSearch，#1188 第 6 條）：那一批落地後平面圖檢查通過，
// 訊息列換成成功的一句、上面那一支會把警示拿掉——被擋的那一次什麼都沒存、沒跳頁，那一句不能跟著不見。
// 留到使用者下一次真的按到東西（滑鼠或鍵盤按出來的 click；頁面程式自己發的不算）為止。整頁還鎖著的時候按的不算：
// 那一下什麼都按不到（點擊落在頁面底層），那一批也還在路上，這時清掉、落地後就看不到了。換方案也清（重畫時看做法換過幾次）；
// 打開這一頁不必另外清：用按鈕打開（立面圖那一顆、方案那一節那一顆）時那一下 click 已經先清過；
// 帶著方案名的連結進來（app.js 的 openZonesFromLink，#1198）是整頁重新載入，本來就是空的。
// 按下去那一刻只就地改那一行、不重畫整頁：重畫會把正在被按的那一顆換掉。
function forgetKept(event) {
  if (!event.isTrusted || !keptSaid || inputBusy) return;
  keptSaid = "";
  const line = $("zone-page-said");
  if (line && saidWords()) line.textContent = saidWords(); else line?.remove();
}
window.zonePageKeep = text => { keptSaid = text; renderZoneOverview(); };
document.addEventListener("click", forgetKept, true);
// 入口兩顆，同一件事：立面圖那一段最上面的（切區時就在手邊），與方案那一節的（打開方案後第一屏就看得到，#1198）。
// 進這一頁的鈕只留第 2 段右欄那一顆（#1265；第 1 段頂上那一顆拿掉了，頂條導覽「這間房貼什麼」照舊）。
window.addEventListener("DOMContentLoaded", () => { $("zone-page-open").onclick = () => window.openZoneOverview(); });
})();
