/* 「這間房貼什麼」（#1176 第 C 支，照第五版樣稿）：六面攤開在左邊（四面牆站在房間裡看、地板天花從上往下看），顏色看狀態，
   點圖選區（再點別塊加選、再點一次取消）；右邊一欄：選了區就是那幾區的面板（三個做法），沒選就是提示；再來「其他牆面」
   那條規則、「另外指定的 N 區」、天花地板門窗一行，再來由程式最佳化的區從哪幾種裡選的清單卡（清單一樣的區一張）；最底下一條結果列。
   規則是從各區現算的（zone_rules.js），設計檔沒有「跟著規則」這一格：改規則＝把新做法寫到每一個沒另外指定的區。

   住在方案輸入頁裡當一個整頁的畫面（#zone-page）：打開時把原本那一頁藏起來，方案、每一區的做法、材料庫、區的號碼與名字
   都用那一頁已經讀好的同一份（zone_modes.js、elevation.js）。畫面是（方案、做法、材料庫）加上這一頁自己的幾個變數
   （選了哪幾區、面板上剛點的選項、哪幾個摺疊開著）算出來的：**畫的時候只讀**，不補寫沒列到的區、不碰設計設定——
   打開看一看再存，設計檔與方案檔一個字都不會變。重畫掛在做法表重畫的同一條路上（renderZoneTable 開頭），開檔、切牆、
   換材料、改做法之後都會跟著重畫。

   晚到的回應：限制檢查（/api/zone-limits）每次改動後才回來，回來時 zone_limits.js 的 showZoneLimits 會整塊重畫
   （renderElevation → renderZoneTable → 這一頁）；吸音曲線（/api/material-curve）回來也重畫。選取與面板上剛點的選項
   住在這個檔的變數裡、不住在畫面元件上，所以重畫不會清掉；只有換了方案、或那一區不在了（重切）才拿掉。

   改動一律一次送出：方案那幾區的材料交伺服器一批換（POST /api/zone-material/batch，全有或全無），成了才改每一區的做法。
   做法寫完之後還有一步平面圖檢查（/api/plan，慢、會斷）：這一頁自己要改的變數（規則定成哪一種、選取收起來）一律在
   那一步**之前**同步寫完，等它回來之後什麼都不再寫——等待中使用者又點的、又改的，不會被晚到的那一步蓋掉。

   出錯要在這一頁看得到：輸入頁的訊息列（#messages）在這一頁開著時是藏起來的，所以它寫的每一句警示都照抄一份到
   這一頁結果列上面（window.zonePageSay，app.js 的 say 會叫）；組畫面組到一半丟例外時畫面留在上一次的樣子、加一句話，
   「回房間方案輸入」一定按得到。
   圖上的幾何（哪一軸往右、鏡不鏡像、喇叭與主位的投影）用立面圖那一份（window.elevationGeometry），不另寫一套。

   材料指定（#1199 第一支，純改畫面）：四面牆名字旁各一顆「選整面」（扣門窗）；面板每一項旁標出選到的區「目前」是哪一種；
   選了由程式最佳化，底下一個材料框——可調的一種一列攤開、寫調什麼從多少到多少（只是字，範圍是材料頁設的，這裡不能改），
   被勾著的可調材料包含的成品灰掉，固定成品收在摺疊裡寫幾種勾了幾種；框外兩項「也可以不貼」「這一區最深」。
   框裡勾的、最深填的都只是草稿（draft），套用才寫、取消就丟。
   每一區存自己的清單（搜尋那一步本來就逐區照自己的清單），這一頁的三個入口都只改該改的區（#1199 修補 1）：
   小框打開時的勾照選中的區自己存的清單（各區不同就是第三種狀態）；套用只把使用者動過的那幾個勾加減到每一個選中區
   自己的清單上，沒選中的區不碰；右欄清單卡一組（清單一樣的區）一張，在一張上勾只改那一組。 */
(() => {
const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_WIDTH = 954;          // 左邊那張圖的寬（px，樣稿）
const WALL_GAP = 26;            // 同一排兩面牆中間的空
const WALL_ROW_HEIGHT = 224;    // 一面牆最高畫多高（px）
const TILE_DEPTH = 190, TILE_WIDTH = 140;  // 地板、天花那兩張小圖最大的深與寬（px）
const WALL_ORDER = ["front", "back", "left", "right"];
const VPR_WORDS = "鋼板共振吸音";
let overviewOpen = false;
let selected = [];              // 選了哪幾區（「牆/區代號」，照點的先後）
let draft = null;               // 面板上現在點的：{mode, materialId, ticks, depthText, depthBad}；還沒點是 null（後三格見 untouched）
let draftTouched = false;       // 面板上的選項是使用者自己點的（再加選別區時不重算）
let ruleDraft = false;          // 「其他牆面」下拉剛選了指定貼一種、還沒選貼哪一種
let listAdding = null;          // 第幾張清單卡的「加別種材料」開著（從 0 數；都沒開是 null）
let keptRule = null;            // 這次打開方案後定下來的「其他牆面」做法鍵（第一次推出來的，或使用者最後在下拉設的）；換方案才重推
let keptRuleZone = null;        // 剛照規則寫過的一區：下次重畫時規則就是它現在的做法（改規則、改清單之後做法鍵會換）
let seenGeneration = -1;        // 上次畫的時候是第幾份做法（換方案就把選取清掉）
let changing = 0;               // 做法寫完、平面圖檢查還沒回來的改動有幾個（寫在 #zone-page 的 data-changing，考卷等畫面穩用）
let unsaved = false;            // 這一頁改過、還沒存進檔案（標題列那一句；存好了由 zone_modes.js 叫 window.zonePageSaved 拿掉）
let pageSaid = "";              // 輸入頁訊息列最近一句警示（這一頁開著時照抄一份，見 window.zonePageSay）
let keptSaid = "";              // 「拿去搜尋」被還在路上的改動擋下的那一句：留到使用者下一次按東西為止（見 window.zonePageKeep）
let view = null;                // 最近一次算出來的：{roles, room, faces, byId, inferred, list}
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
  details.ontoggle = () => { if (details.open) openDetails.add(id); else openDetails.delete(id); };
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
// opening 門窗（沒被改成別的做法時畫斜線）。
function faceZones(wall) {
  const {views, wallSize, screenBox} = window.elevationGeometry;
  const size = wallSize(wall), labels = zoneLabels(wall);
  return zoneList(wall).map(zone => {
    const key = zoneKey(zone), setting = zoneSetting(wall, key), label = labels.get(key);
    const entry = zoneLibraryMatch(zone.material), special = zoneSpecialName(entry);
    const box = !size ? null : zone.area === "rectangle" ? screenBox(views[wall], size[0], size[1], zone) : [0, size[0], 0, size[1]];
    const pasted = zone.material.kind === "mounted_on_wall";
    const opening = special === "門" || special === "窗";
    const item = {id: `${wall}/${key}`, wall, key, zone, setting, label, entry, special, opening, box, pasted,
                  area: box ? (box[1] - box[0]) * (box[3] - box[2]) : 0, mode: setting.mode,
                  allowed: setting.allowed ?? serverDefaultAllowed(),
                  fixed: zoneFixedIdentity(zone.material, pasted ? libraryLabel(zone.material)?.id ?? null : null)};
    item.method = zoneMethodKey(item);
    const mode = zoneMethodMode(item.method);
    item.shown = mode === "search" ? "search" : mode === "fixed" && !special ? "fixed" : opening ? "opening" : "still";
    return item;
  }).sort((left, right) => left.label.number - right.label.number);
}
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
function pastedWords(item) { return libraryLabel(item.zone.material)?.label ?? layersText(item.zone.material); }
function methodWords(group) {
  return group.mode === "fixed" ? vprWords(`${ZONE_MODE_WORDS.fixed}：${pastedWords(group.zones[0])}`) : ZONE_MODE_WORDS[group.mode];
}
function shownOf(mode) { return {search: "search", fixed: "fixed", keep_original: "still"}[mode]; }
function hoverWords(item) {
  const mode = zoneMethodMode(item.method);
  return `${zoneWhereText(item)}：${mode === "fixed" ? `${ZONE_MODE_WORDS.fixed}：${pastedWords(item)}` : ZONE_MODE_WORDS[mode]}`;
}

/* ───── 左邊：圖 ───── */
// 字大約多寬（px）：全形一個字高、半形半個多一點；只拿來決定圖上的字放不放得下。
function textWidth(text, size) { return [...text].reduce((sum, char) => sum + (char.charCodeAt(0) > 0x2e7f ? 1 : 0.58), 0) * size; }
// 指定貼的那一塊上面寫材料名：照括號與「＋」斷成幾行，第一行大字（放不下就小一號），後面幾行從頭照順序放到放不下為止；
// 第一行都放不下就不寫（滑過去的說明照樣有整個名字）。只寫在另外指定的區上：跟著規則的那一大片不重複寫同一個名字。
function fixedLabel(item, x, y, width, height) {
  const entry = libraryLabel(item.zone.material);
  const group = svg("g", {class: "zv-fixed-words"});
  if (!entry || !item.named) return group;
  const lines = entry.label.split(/[（）]|(?=＋)/).map(part => part.trim()).filter(Boolean);
  // 第一行就是「VPR」而名字自己沒接著說明：底下加一行小字說它是什麼（同右欄的「VPR（鋼板共振吸音）」）。
  if (lines[0] === "VPR" && vprWords(entry.label) !== entry.label) lines.splice(1, 0, VPR_WORDS);
  const room = Math.max(0, Math.floor((height - 44) / 18));
  let big = [20, 15].find(size => textWidth(lines[0], size) <= width - 10);
  if (!big) {
    // 第一行整句放不下：照空白折成幾行。
    const wrapped = [];
    for (const word of lines[0].split(" ")) {
      const joined = wrapped.length ? `${wrapped.at(-1)} ${word}` : word;
      if (wrapped.length && textWidth(joined, 15) <= width - 10) wrapped[wrapped.length - 1] = joined; else wrapped.push(word);
    }
    // 折完要整句都寫得下（每一行夠寬、行數夠高）才寫：寫一半的名字比不寫更容易看錯。
    if (wrapped.length > 1 && wrapped.length <= room && wrapped.every((line, index) => textWidth(line, index ? 13 : 15) <= width - 10)) { lines.splice(0, 1, ...wrapped); big = 15; }
  }
  if (!big || !room) return group;
  const fits = [lines[0]];
  for (const line of lines.slice(1)) { if (fits.length >= room || textWidth(line, 13) > width - 10) break; fits.push(line); }
  const top = y + height / 2 - (fits.length - 1) * 9;
  fits.forEach((line, index) => group.append(svg("text", {class: index ? "zv-fixed-sub" : "zv-fixed-label", "font-size": index ? 13 : big,
    x: x + width / 2, y: top + (index ? 2 + index * 19 : 0)}, line)));
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
  if (item.shown === "opening") {
    group.append(svg("rect", {class: "zv-hatch", x, y, width, height}), ...ring, svg("rect", {class: "zv-tagbg", x: cx - 29, y: cy - 15, width: 58, height: 30, rx: 3}),
      badge(cx - 13, cy, item.label.number, false), svg("text", {class: "zv-open-label", x: cx + 12, y: cy + 6}, item.special));
  } else if (item.shown === "fixed" && width >= 44 && height >= 44) {
    group.append(...ring, badge(x + 16, y + 16, item.label.number, false), fixedLabel(item, x, y, width, height));
  } else group.append(...ring, badge(cx, cy, item.label.number, small));
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
  face.append(svg("rect", {class: "zv-outline", x: ox, y: oy, width: width * scale, height: height * scale}),
    svg("line", {class: "zv-floorline", x1: ox - 1, x2: ox + width * scale + 1, y1: floorY + 1.5, y2: floorY + 1.5}),
    svg("text", {class: "zv-quiet", x: ox + width * scale, y: floorY + 18, "text-anchor": "end"}, wallNames[views[wall].bottom]),
    ...miniPlan({front: "top", back: "bottom", left: "left", right: "right"}[role], ox + 2, oy - 5.5 - miniSize[1], ...miniSize),
    ...wallMarks(role, wall, ox, floorY, scale));
  const title = svg("text", {class: "zv-face-name big", x: ox + miniSize[0] + 10, y: oy - 8}, wallNames[wall]);
  title.append(svg("tspan", {class: "zv-quiet", dx: 10}, `面向它站：左手邊是${wallNames[views[wall].left]}，右手邊是${wallNames[views[wall].right]}`));
  face.append(title);
  map.append(face);
  // 「選整面」那一顆放哪（#1199）：這面牆名字那一行的最右邊（疊在圖上的一顆真的按鈕，見 wholeButtons）。
  return {wall, right: ox + width * scale, top: oy - 27};
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
    face.append(group);
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
  const title = svg("text", {class: "zv-face-name", x: px, y: py - 8}, wallNames[wall]);
  title.append(svg("tspan", {class: "zv-quiet", dx: 10}, heading));
  face.append(title);
  map.append(face);
}
function drawMap(faces, roles, room) {
  const {wallSize} = window.elevationGeometry;
  const frontWidth = wallSize(roles.front)[0], sideWidth = wallSize(roles.left)[0];
  const scale = Math.min((MAP_WIDTH - 4 - WALL_GAP) / (frontWidth + sideWidth), WALL_ROW_HEIGHT / room.Lz, 80);
  const tileScale = Math.min(TILE_DEPTH / wallSize(roles.left)[0], TILE_WIDTH / frontWidth, 30);
  const rowHeight = room.Lz * scale;
  const row1 = 30, row2 = row1 + rowHeight + 56, tileY = row2 + rowHeight + 48;
  const col1 = 2, col2 = col1 + frontWidth * scale + WALL_GAP;
  const tileWidth = frontWidth * tileScale, tileDepth = sideWidth * tileScale;
  const height = Math.max(tileY + tileDepth + 22, 790);
  const map = svg("svg", {id: "zone-map", class: "zv-map", width: MAP_WIDTH, height, viewBox: `0 0 ${MAP_WIDTH} ${height}`,
                          role: "img", "aria-label": "四面牆、地板、天花"});
  const defs = svg("defs");
  const pattern = svg("pattern", {id: "zv-hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"});
  pattern.append(svg("line", {x1: 0, y1: 0, x2: 0, y2: 7, stroke: "#7d8a8c", "stroke-width": 1.6}));
  defs.append(pattern); map.append(defs);
  const miniSize = [15, 15 * sideWidth / frontWidth];
  // 樣稿的排法：上排前牆、右牆，下排後牆、左牆（前後同寬排同一欄）。
  const whole = [];
  for (const [role, ox, oy] of [["front", col1, row1], ["right", col2, row1], ["back", col1, row2], ["left", col2, row2]]) {
    whole.push(drawWall(map, role, roles[role], faces[roles[role]], ox, oy, scale, miniSize));
  }
  const floorX = 440, ceilingX = floorX + tileWidth + 92;
  const allStill = faces.ceiling.every(item => item.shown === "still");
  drawTile(map, "floor", faces.floor, tilePlan(roles, room, floorX, tileY, tileScale), roles, room, faces, "從上往下看", null);
  drawTile(map, "ceiling", faces.ceiling, tilePlan(roles, room, ceilingX, tileY, tileScale), roles, room, faces, "方向同地板圖",
    allStill && faces.ceiling.length === 1 ? ["保留原樣", originalFace("ceiling")] : null);
  return {map, legendTop: tileY - 10, whole};
}
function legend(top) {
  const box = el("div", "zv-legend"); box.style.top = `${top}px`;
  for (const [className, words] of [["search", ZONE_MODE_WORDS.search], ["fixed", ZONE_MODE_WORDS.fixed], ["still", "保留原樣"], ["hatch", "門、窗"]]) {
    const item = el("span"); item.append(el("i", `zv-sw ${className}`), words); box.append(item);
  }
  const apart = el("span"); apart.id = "zone-legend-apart"; apart.append(el("i", "zv-sw apart"), "虛線框：另外指定（跟其他牆面不同）"); box.append(apart);
  return box;
}

/* ───── 選取與改動 ───── */
function chosenItems() { return selected.map(id => view.byId.get(id)).filter(Boolean); }
// 小框裡由程式最佳化那一塊的草稿（#1199），套用才寫進設計檔，取消就丟：
// ticks：使用者在小框裡動過的勾（材料代號 → 勾上 true／拿掉 false）；沒動過的代號不在裡面，套用時那一種各區照目前的；
// depthText：「最深」那一格打的字（公分；沒動過是 null）；depthBad：打的不成數字。
function untouched() { return {ticks: {}, depthText: null, depthBad: false}; }
// 面板一打開先點在哪一項：選到的區是同一種做法就點那一項（指定貼的：貼的都是同一筆才帶出那一筆），不一樣就都不點。
function startingDraft() {
  const items = chosenItems();
  const modes = new Set(items.map(item => zoneMethodMode(item.method)));
  if (!items.length || modes.size !== 1) return null;
  const mode = [...modes][0], sameMethod = new Set(items.map(item => item.method)).size === 1;
  return {...untouched(), mode, materialId: mode === "fixed" && sameMethod ? libraryLabel(items[0].zone.material)?.id ?? "" : ""};
}
function select(ids) {
  selected = ids; draftTouched = false; ruleDraft = false; listAdding = null;
  openDetails.delete("zone-panel-fixed"); openDetails.delete("zone-panel-zones");
  draft = startingDraft();
  renderZoneOverview();
}
// 換成選這幾區：面板上已經自己點過選項就留著（只換選了哪幾區），沒點過就照新選的那幾區重新起頭。
function pick(next) {
  if (draftTouched && next.length) { selected = next; renderZoneOverview(); return; }
  select(next);
}
function toggleSelected(id) { pick(selected.includes(id) ? selected.filter(other => other !== id) : [...selected, id]); }
// 一面牆的「整面」（#1199）：那一面全部的區扣掉門、窗（跟推「其他牆面」時扣的一樣），照號碼排。
function wholeFace(wall) { return view.faces[wall].filter(item => !item.opening).map(item => item.id); }
// 好幾區照面分開（照交進來的先後）：[{wall, zones, whole}]。whole：這幾區剛好是那一面牆的整面（兩區以上）——右欄就寫一行
// 「前牆整面」加號碼，不必一區一行。
function byFace(zones) {
  return [...new Set(zones.map(item => item.wall))].map(wall => {
    const here = zones.filter(item => item.wall === wall).sort((left, right) => left.label.number - right.label.number);
    const face = WALL_ORDER.some(role => view.roles[role] === wall) ? wholeFace(wall) : [];
    return {wall, zones: here, whole: face.length > 1 && face.length === here.length && here.every(item => face.includes(item.id))};
  });
}
// 四面牆名字旁各一顆「選整面」：一次選那一面全部的區（整面都是門窗就沒有這一顆）。
function wholeButtons(places) {
  return places.filter(place => wholeFace(place.wall).length).map(place => {
    const item = button("zv-whole", "選整面", `zone-whole-${place.wall}`, () => pick(wholeFace(place.wall)));
    item.setAttribute("aria-label", `${wallNames[place.wall]}選整面`);
    item.style.left = `${place.right.toFixed(1)}px`; item.style.top = `${place.top.toFixed(1)}px`;
    return item;
  });
}
// 這一面由程式最佳化時寫哪一份清單：照畫面上那一份，這一面貼不了的拿掉（材料庫找不到的代號原樣留著）；一種都不剩就只放不貼。
function listForWall(wall, list) {
  if (!list) return startingAllowed(wall);
  const fit = list.filter(id => fitsHere(id, wall) || (id !== ZONE_NONE && !libraryEntry(id)));
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
  const next = own.filter(id => ticks[id] !== false);
  for (const [id, on] of Object.entries(ticks)) if (on && !next.includes(id) && fitsHere(id, item.wall)) next.push(id);
  return next;
}
// 把好幾區一次改成同一個做法。method：{mode, materialId（指定貼哪一筆；空的＝照現在貼的不換）, allowed（由程式最佳化的清單：
// 規則那條路用，照每一面能貼的濾過再寫）, lists（小框那條路用：區 → 那一區要寫的清單；undefined＝那一區的清單不碰）,
// depth（最深：undefined 不碰、null 拿掉、數字寫公尺）}。
// 方案那幾區的材料先交伺服器一批換（指定貼：貼上那一筆；其餘兩種：放回原樣），全部成了才改做法；不成就一區都不動、照講。
// 等伺服器回那一批的時候整頁鎖住（inert：點圖、按套用都沒作用），所以同時只會有一批在路上。
// written：做法寫完之後、第一次重畫之前要同步做的事（這一頁自己的變數在這裡改；平面圖檢查回來之後不准再寫）。
async function applyMethod(items, method, written = () => {}) {
  const drafted = items.find(item => measuredDrafts[item.wall]);
  if (drafted) { zoneNote(`${wallNames[drafted.wall]}有還沒套用的實測阻抗草稿：先套用或取消`, "notice"); return false; }
  const changes = items.filter(() => method.mode !== "fixed" || method.materialId).map(item => method.mode === "fixed"
    ? {wall: item.wall, zone: item.key, choice: "fixed", material_id: method.materialId} : {wall: item.wall, zone: item.key, choice: "original"});
  let results = [];
  if (changes.length) {
    const document_ = wallDocument();
    inputBusy = true;
    document.querySelector("main").inert = true; $("zone-page").inert = true;
    try {
      const {response, data} = await reply(fetch("/api/zone-material/batch", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({scheme: document_, original_scheme_id: zoneDesign?.original_scheme_id ?? null, changes})}));
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
      zoneNote(`沒有改：${errorText(error)}`, "notice");
      return false;
    } finally {
      document.querySelector("main").inert = false; $("zone-page").inert = false;
      inputBusy = false;
    }
  }
  items.forEach((item, index) => {
    const kept = zoneSetting(item.wall, item.key).allowed;
    const next = {...zoneSetting(item.wall, item.key), mode: method.mode};
    delete next.allowed;
    // 指定貼的那一筆跟這一區原本的面一模一樣：伺服器放回原樣，這裡做法跟著記成不貼（同做法表）。
    if (method.mode === "fixed" && results[index]?.same_as_original) next.mode = "keep_original";
    if (method.mode === "search" && !method.lists) next.allowed = listForWall(item.wall, method.allowed);
    if (method.mode === "search" && method.lists) {
      const list = method.lists.get(item.id);
      if (list) next.allowed = list; else if (list === undefined && item.mode === "search" && kept) next.allowed = kept;
    }
    // 「最深」（小框那一格）：有改才寫——清空就拿掉那一格，填了就寫公尺；沒改的各區照目前的。
    if (method.depth === null) delete next.max_depth_m; else if (method.depth !== undefined) next.max_depth_m = method.depth;
    setZoneSetting(item.wall, item.key, next);
    zoneFormerFixed.delete(item.id); zoneSameAsOriginal.delete(item.id);
  });
  const what = method.mode === "fixed" && method.materialId ? vprWords(`${ZONE_MODE_WORDS.fixed}：${libraryEntry(method.materialId).label}`) : ZONE_MODE_WORDS[method.mode];
  zoneNote(`${items.length} 區改成「${what}」`, "ok");
  unsaved = true;
  written();
  changing += 1; showChanging();
  try { await afterZoneChange(); } finally { changing -= 1; showChanging(); }
  return true;
}
function showChanging() { $("zone-page").dataset.changing = String(changing); }
// 「其他牆面」現在的做法（改回、面板上「＝其他牆面」那一項用）；規則貼的不是材料庫裡的一筆就回 null（沒有代號可以照貼）。
function ruleMethod() {
  const rule = view.inferred.rule;
  if (!rule) return null;
  if (rule.mode !== "fixed") return {mode: rule.mode, allowed: rule.mode === "search" ? rule.zones[0].allowed : null};
  const entry = libraryLabel(rule.zones[0].zone.material);
  return entry ? {mode: "fixed", materialId: entry.id} : null;
}
// 改規則：寫到所有不是另外指定的區（現在跟著規則的那幾區），另外指定的不動。
async function applyRule(method) {
  ruleDraft = false;
  const followers = view.inferred.rule.zones;
  // 規則換成剛寫進去的那一種（那幾區現在的做法鍵）：做法一寫完就定下來，接著的每一次重畫都照它。
  if (!await applyMethod(followers, method, () => { keptRuleZone = followers[0].id; })) renderZoneOverview();
}
async function resetSpecial() {
  const method = ruleMethod();
  if (!method) { zoneNote("其他牆面現在貼的不是材料庫裡的一筆，沒辦法照著改回去：先在「其他牆面」選一種", "notice"); return; }
  await applyMethod(view.inferred.special.flatMap(group => group.zones), method);
}
// 小框裡每一種材料現在的樣子：選中的每一區照草稿加減之後的清單（lists），加上每一種「選中的區裡幾區有」（count）。
// ids：出現在任何一區清單裡的代號，照最先出現的順序。材料庫沒讀全、有一區列不出來時回 null。
function panelTicks(items) {
  const lists = items.map(item => tickedList(item, draft.ticks));
  if (lists.some(list => !list)) return null;
  const ids = [...new Set(lists.flat())];
  const count = id => lists.filter(list => list.includes(id)).length;
  return {lists, ids, count, state: id => count(id) === items.length ? "all" : count(id) ? "some" : "none"};
}
// 「最深」那一格打開時顯示什麼（公分）：選到的區都一樣就帶出來，不一樣、或都沒有，就是空的。
function shownDepth(items) {
  const depths = items.map(item => zoneSetting(item.wall, item.key).max_depth_m ?? null);
  return depths.every(depth => depth === depths[0]) && depths[0] !== null ? String(centimetres(depths[0])) : "";
}
// 「最深」那一格要寫什麼：格子的值跟打開時顯示的一樣（沒動過、或打了又改回去）undefined（不寫，各區照目前的）、
// 清空 null（拿掉）、正數回公尺（記到公釐；不到 1 公釐會變成 0，存檔時伺服器不收，這裡就擋）、其餘 NaN（不收）。
function draftDepth(items) {
  if (draft.depthText === null) return undefined;
  if (draft.depthBad) return NaN;
  const text = draft.depthText.trim();
  if (text === shownDepth(items)) return undefined;
  if (text === "") return null;
  const millimetres = Math.round(Number(text) * 10);
  return Number.isFinite(millimetres) && millimetres > 0 ? millimetres / 1000 : NaN;
}
// 小框選了由程式最佳化、現在這樣套不套得下去：回一句為什麼不行，可以就回空字串。每一個選中的區各自看它套用後的清單。
function draftProblem(items) {
  if (draft?.mode !== "search") return "";
  const ticks = panelTicks(items);
  if (ticks) {
    const who = index => items.length > 1 ? `${zoneWhereText(items[index])}：` : "";
    const empty = ticks.lists.findIndex(list => !list.length);
    if (empty >= 0) return `${who(empty)}至少勾一種材料，或勾「也可以不貼」；不想讓程式選，就改選上面的「${ZONE_MODE_WORDS.fixed}」或「${KEEP_WORDS}」`;
    const stuck = ticks.lists.findIndex(list => allowedProblem(list));
    if (stuck >= 0) return `${who(stuck)}${allowedProblem(ticks.lists[stuck])}程式要從一個能調的東西開始：勾一種可調的材料，或勾「也可以不貼」`;
  }
  if (Number.isNaN(draftDepth(items))) return "「最深」要填大於 0 的數字（公分）；空白表示不限";
  return "";
}
async function applyDraft() {
  const items = chosenItems();
  if (draft.mode !== "search") { await applyMethod(items, {mode: draft.mode, materialId: draft.materialId}, () => select([])); return; }
  // 由程式最佳化：只寫選中的區。每一區寫它自己的清單照動過的勾加減之後的那一份；勾都沒動、那一區本來就是由程式最佳化，
  // 清單就不碰（只改最深時清單一個字都不動）。沒選中的區一律不寫。
  const ticks = draft.ticks, moved = Object.keys(ticks).length > 0;
  const lists = new Map(items.map(item => [item.id, moved || item.mode !== "search" ? tickedList(item, ticks) : undefined]));
  // 規則本身是由程式最佳化、跟著規則的區全都選中了：清單換了做法鍵跟著換，規則還是那幾區。
  const rule = view.inferred.rule;
  const ruled = moved && rule?.mode === "search" && rule.zones.every(item => selected.includes(item.id)) ? rule.zones[0].id : null;
  await applyMethod(items, {mode: "search", lists, depth: draftDepth(items)}, () => {
    if (ruled) keptRuleZone = ruled;
    if (moved && lists.get(items[0].id)) zoneLastAllowed = [...lists.get(items[0].id)];
    select([]);
  });
}
// 在一張清單卡上勾（#1199 修補 1）：只改那一組的區（清單一樣的那幾區），加一種（change.add）或拿掉一種（change.remove），
// 加減在每一區自己的清單上——加的那一種某一面貼不了，那一面的區就不加。跟做法表改清單一樣重判限制。
function writeList(group, change) {
  const lists = group.zones.map(item => change.remove ? item.allowed.filter(id => id !== change.remove)
    : item.allowed.includes(change.add) || !fitsHere(change.add, item.wall) ? item.allowed : [...item.allowed, change.add]);
  group.zones.forEach((item, index) => setZoneSetting(item.wall, item.key, {...zoneSetting(item.wall, item.key), mode: "search", allowed: lists[index]}));
  // 規則本身是由程式最佳化、改的就是規則那一組：清單換了、做法鍵跟著換，規則還是那幾區。
  if (view.inferred.rule?.mode === "search" && view.inferred.rule.key === group.key) keptRuleZone = view.inferred.rule.zones[0].id;
  const list = lists.find(next => change.remove || next.includes(change.add)) ?? lists[0];
  zoneLastAllowed = [...list];
  unsaved = true;
  zoneNote(`${group.zones.length} 區的清單改成 ${list.map(choiceName).join("、")}`, "ok");
  markStale();
  window.renderElevation();
  refreshZoneLimits();
}

/* ───── 右邊：面板 ───── */
// 選的那一筆的構造：房間這側 → 每一層 → 原本的面。薄的（薄板、開孔板、布）畫成一條線，下面一句說是什麼。
function structureStrip(entry, wall) {
  const spec = inputDefaults.material_form;
  const strip = el("div", "zv-layers");
  strip.append(el("div", "zv-l-room", "房間這側"));
  const thin = [];
  for (const layer of entry.material.layers) {
    const name = spec[layer.kind]?.label ?? layer.kind;
    const thick = typeof layer.thickness_m === "number" ? ` ${shownNumber(layer.thickness_m * 1000)} mm` : "";
    if (layer.kind === "porous" || layer.kind === "air") {
      const box = el("div", layer.kind === "air" ? "zv-l-air" : "zv-l-foam"); box.append(el("span", null, `${name}${thick}`)); strip.append(box);
    } else { strip.append(el("div", "zv-l-thin")); thin.push(`${name}${thick}`); }
  }
  const back = el("div", "zv-l-wall"); back.append(el("span", null, originalFace(wall))); strip.append(back);
  return [strip, ...(thin.length ? [el("div", "zv-cap", `黑線：${thin.join("、")}`)] : [])];
}
async function loadCurve(entry) {
  curves.set(entry.id, "loading");
  const {basis: _basis, placement: _placement, ...document_} = entry;
  let result;
  try {
    const {response, data} = await reply(fetch("/api/material-curve", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(document_)}));
    result = response.ok ? data : {status: "failed", message: `畫不出曲線：${data.error ?? response.status}`};
  } catch (error) { result = {status: "failed", message: `畫不出曲線：${errorText(error)}`}; }
  curves.set(entry.id, result);
  renderZoneOverview();
}
function curveBlock(entry) {
  const state = curves.get(entry.id);
  if (!state) { loadCurve(entry); return [el("div", "zv-cap", "計算吸音曲線中…")]; }
  if (state === "loading") return [el("div", "zv-cap", "計算吸音曲線中…")];
  if (state.status !== "drawn") return [el("div", "zv-cap", state.message ?? "這一筆畫不出吸音曲線")];
  const width = 304, height = 82, left = 30, bottom = 20, top = 6, low = 20, high = 8000;
  const x = hz => left + (Math.log10(hz) - Math.log10(low)) / (Math.log10(high) - Math.log10(low)) * (width - left - 8);
  const y = value => top + (1 - value) * (height - top - bottom);
  const chart = svg("svg", {class: "zv-curve-chart", width, height, viewBox: `0 0 ${width} ${height}`});
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
  return [chart, el("div", "zv-axis", "頻率（赫茲）")];
}
function materialShow(entry, wall) {
  const show = el("div", "zv-show"); show.id = "zone-panel-show";
  // 整個名字寫一次（會折行）：下拉收起來的那一格只有一行，名字長的會被截掉。
  const name = el("div", "zv-fullname", vprWords(entry.label)); name.id = "zone-panel-name"; show.append(name);
  if (entry.material.kind === "assembly") {
    const head = el("h4", null, "構造 "); head.append(el("span", null, [fixedDepthWords(entry), `貼在${originalFace(wall)}上`].filter(Boolean).join("，")));
    show.append(head, ...structureStrip(entry, wall));
  }
  const head = el("h4", null, "吸音率 "); head.append(el("span", null, "各方向來的平均；模型算的"));
  show.append(head, ...curveBlock(entry));
  return show;
}
// 清單裡勾著的可調材料已經包含哪幾筆成品（#1199）：成品代號 → 包含它的那一筆可調材料。包含關係是材料庫那一筆的
// preset_members（內建三種玻璃棉才有；自建的可調材料沒有這一格，就不灰）。兩個都放進清單不會報錯，但程式會多一個重複的候選。
function coveredProducts(list) {
  const covered = new Map();
  for (const entry of libraryEntries("optimizable")) if (list.includes(entry.id)) for (const member of entry.preset_members ?? []) covered.set(member, entry);
  return covered;
}
function coveredWhy(family) { return `上面可調的 ${shortLabel(family)} 已經包含它；只想用成品，先取消上面的 ${shortLabel(family)}`; }
// 材料框裡的一列（#1199）：勾選格、名字、底下一行小字（可調的寫調什麼、從多少到多少——只是字，這一支不能改；固定成品寫總厚）。
// covered：這一筆成品已經被哪一筆勾著的可調材料包含（那一筆的 preset_members）——灰掉、勾不了，寫為什麼；
// 清單裡本來就同時存著兩筆的照樣勾著、可以取消（只是顯示，不替使用者改寫）。
// ticks：panelTicks 的結果。選中的區全有＝勾、全沒有＝空、有的有有的沒有＝第三種狀態（indeterminate），旁邊寫幾區有。
function tickBox(id, ticks, toggle) {
  const check = el("input"); check.type = "checkbox"; check.id = `zone-panel-allow-${id}`;
  check.checked = ticks.state(id) === "all"; check.indeterminate = ticks.state(id) === "some";
  check.onchange = () => toggle(id, check.checked);
  return check;
}
function someWords(id, ticks) {
  return ticks.state(id) === "some" ? [el("small", "zv-some", `選中的 ${ticks.lists.length} 區裡 ${ticks.count(id)} 區有；不動它，各區照目前的`)] : [];
}
function materialRow(entry, ticks, covered, toggle) {
  const row = el("label", `zv-mrow${covered ? " zv-covered" : ""}`); row.dataset.choice = entry.id; row.dataset.kind = entry.kind;
  const check = tickBox(entry.id, ticks, toggle);
  check.disabled = Boolean(covered) && ticks.state(entry.id) === "none";
  const words = el("span", "zv-mwords"); words.append(el("span", "zv-mname", vprWords(shortLabel(entry))));
  if (entry.kind === "optimizable") words.append(el("small", "zv-range", adjustWords(entry)));
  else if (fixedDepthWords(entry)) words.append(el("small", null, fixedDepthWords(entry)));
  if (covered) {
    row.dataset.covered = covered.id;
    words.append(el("small", "zv-covered-why", coveredWhy(covered)));
  }
  words.append(...someWords(entry.id, ticks));
  row.append(check, words);
  return row;
}
// 「這一區最深」（設計檔每一區的深度上限，公尺；這裡填公分）：選到的區目前都一樣就帶出來，不一樣就空著。
// 打字只記進草稿、就地更新套用鈕（sync），不重畫。空格子裡不放提示字（瀏覽器畫提示字的灰量不到對比），空白是什麼意思寫在底下那一行：
// 格子的值跟打開時一樣就是沒改（draftDepth），所以各區不同時空白＝各區照目前的。
function deepestRow(items, walls, sync) {
  const row = el("div", "zv-extra zv-deepest"); row.id = "zone-panel-deepest";
  const depths = items.map(item => zoneSetting(item.wall, item.key).max_depth_m ?? null);
  const same = depths.every(depth => depth === depths[0]);
  const name = el("label", "zv-extra-name", items.length > 1 ? "這幾區最深" : "這一區最深"); name.htmlFor = "zone-panel-depth";
  const input = el("input", "zv-depth"); input.type = "number"; input.min = "0"; input.step = "any"; input.id = "zone-panel-depth";
  input.value = draft.depthText ?? shownDepth(items);
  input.oninput = () => { draft.depthText = input.value; draft.depthBad = input.validity.badInput; draftTouched = true; sync(); };
  const from = walls.length === 1 ? originalFace(walls[0]) : "原本的面";
  row.append(name, input, el("span", "zv-unit", "cm"),
    el("small", null, `從${from}往房間算，貼上去的總厚不超過它；空白＝${same ? "不限" : "各區照目前的"}`));
  return row;
}
// 小框選了由程式最佳化之後的那一塊（#1199，照樣稿）：材料框（可調的一種一列攤開、材料庫找不到的代號照列、固定成品收在摺疊裡
// 寫幾種勾了幾種），框外兩項（也可以不貼、最深）。勾與最深都只改草稿，套用才寫，而且只寫選中的區。
function searchForm(items, walls, sync) {
  const ticks = panelTicks(items);
  if (!ticks) return [];
  // 只記被點的那一個代號（勾上或拿掉）；沒點過的代號套用時各區照目前的。
  const toggle = (id, on) => { draft.ticks = {...draft.ticks, [id]: on}; draftTouched = true; renderZoneOverview(); };
  const has = id => ticks.count(id) > 0;
  const fits = entry => walls.every(wall => usableHere(entry, wall)) || has(entry.id);
  const tunable = libraryEntries("optimizable").filter(fits), products = libraryEntries("fixed").filter(fits);
  const coveredBy = coveredProducts(ticks.ids);
  const form = el("div", "zv-form"); form.id = "zone-panel-form";
  const top = el("div", "zv-form-top");
  top.append(button("zv-link", "材料的勾全拿掉", "zone-panel-clear", () => {
    draft.ticks = {...draft.ticks, ...Object.fromEntries(ticks.ids.filter(id => id !== ZONE_NONE).map(id => [id, false]))}; draftTouched = true; renderZoneOverview();
  }), el("span", null, "「也可以不貼」不受影響"));
  form.append(top);
  if (!tunable.length) form.append(el("div", "zv-form-note", "這裡沒有可調的材料：程式只能從「不貼」開始，可以再勾固定成品讓它比"));
  for (const entry of tunable) form.append(materialRow(entry, ticks, null, toggle));
  for (const id of ticks.ids.filter(other => other !== ZONE_NONE && !libraryEntry(other))) {
    const row = el("label", "zv-mrow"); row.dataset.choice = id; row.dataset.kind = "missing";
    const words = el("span", "zv-mwords"); words.append(el("span", "zv-mname", choiceName(id)), el("small", null, "照原樣留著，開跑時伺服器會點名"), ...someWords(id, ticks));
    row.append(tickBox(id, ticks, toggle), words); form.append(row);
  }
  if (tunable.length) {
    const note = el("div", "zv-form-note", "範圍是材料頁設的，這裡只看不改。"); note.id = "zone-panel-range-note";
    const link = el("a", "zv-link", "材料頁 ▸"); link.id = "zone-panel-materials"; link.href = "/materials"; link.target = "_blank"; link.rel = "noopener";
    note.append(link); form.append(note);
  }
  const more = fold("zone-panel-fixed", `固定成品（整塊貼或不貼，不能調） ${products.length} 種，勾了 ${products.filter(entry => has(entry.id)).length} 種`);
  const rows = el("div", "zv-fixed-rows");
  for (const entry of products) rows.append(materialRow(entry, ticks, coveredBy.get(entry.id) ?? null, toggle));
  more.append(rows); form.append(more);
  const none = el("label", "zv-extra zv-mrow"); none.id = "zone-panel-none";
  const words = el("span", "zv-mwords"); words.append(el("span", "zv-mname", "也可以不貼"), el("small", null, "程式算出來不貼比較好的區，就不貼，保留原樣"), ...someWords(ZONE_NONE, ticks));
  none.append(tickBox(ZONE_NONE, ticks, toggle), words);
  return [form, none, deepestRow(items, walls, sync)];
}
// 「目前」小標（#1199）：選到的區現在是這一種做法的，掛在那一項旁邊（講的是現在存著的樣子，面板上點別項它不跟著跑）。
// 選到的全是同一種：指定貼的寫現在貼的是什麼，其餘兩種不必多寫；做法不只一種：每一項各寫是哪幾區。
function nowMark(items, mode) {
  const here = items.filter(item => zoneMethodMode(item.method) === mode);
  if (!here.length) return [];
  const tag = el("span", "zv-now-tag", "目前"); tag.id = `zone-panel-now-${mode}`; tag.dataset.zones = here.map(item => item.id).join(" ");
  const what = el("span", "zv-now-what");
  if (here.length < items.length) what.append(zonesBrief(here));
  else if (mode === "fixed") what.append(new Set(here.map(item => item.method)).size === 1 ? vprWords(pastedWords(here[0])) : "各區貼的不一樣");
  const line = el("div", "zv-now"); line.append(tag, what);
  return [line];
}
function panelCard() {
  const items = chosenItems(), rule = view.inferred.rule;
  const card = el("div", "zv-edit"); card.id = "zone-panel";
  card.dataset.zones = selected.join(" ");
  const walls = [...new Set(items.map(item => item.wall))];
  // 選到的都在同一面牆：剛好是那一面整面（兩區以上）就只寫一行，一區一行的收在摺疊裡；不是整面就給一條「改選整面」。
  const face = walls.length === 1 && view.roles && WALL_ORDER.some(role => view.roles[role] === walls[0]) ? wholeFace(walls[0]) : [];
  const isWhole = face.length > 1 && face.length === items.length && face.every(id => selected.includes(id));
  const top = el("div", "zv-ehead"); top.append(el("h2", null, `已選 ${items.length} 區`));
  if (face.length && !isWhole && !(face.length === 1 && selected.length === 1 && selected[0] === face[0])) {
    top.append(button("zv-link", `改選${wallNames[walls[0]]}整面（${face.length} 區）`, "zone-panel-whole", () => pick(face)));
  }
  card.append(top);
  const chips = items.map(item => {
    const chip = el("div", "zv-chip"); chip.dataset.zone = item.id;
    chip.append(zoneWhere(item), button("zv-link zv-x", "不選這區", `zone-panel-drop-${item.wall}-${item.key}`, () => toggleSelected(item.id)));
    return chip;
  });
  if (isWhole) {
    const line = el("div", "zv-chip zv-face-line"); line.id = "zone-panel-face";
    line.append(`${wallNames[walls[0]]}整面 `, ...[...items].sort((left, right) => left.label.number - right.label.number).map(item => numberBadge(item.label.number)));
    const each = fold("zone-panel-zones", "看每一區、拿掉幾區"); each.classList.add("zv-each");
    each.append(...chips);
    card.append(line, each);
  } else card.append(...chips);
  card.append(el("div", "zv-addmore", "再點圖上別區可以一起改。"));
  // 「＝其他牆面」：選到的都是算進規則的牆區時，標在跟規則同一種做法的那一項旁邊；選它＝這幾區不再另外指定。
  const ruled = new Set(ruleZones(view.faces, view.roles).map(item => item.id));
  const sameAs = rule && items.every(item => ruled.has(item.id)) ? rule.mode : null;
  const allPasted = items.every(item => item.pasted);
  const choose = (mode, materialId = "") => { draft = {...untouched(), ...draft, mode, materialId}; draftTouched = true; renderZoneOverview(); };
  // 套用鈕與它上面那一句「為什麼不能套用」：在小框裡打字（最深那一格）時就地更新這兩樣，不重畫整頁（重畫會換掉正在打的那一格）。
  const apply = button("zv-btn zv-main-btn", `套用到選中的 ${items.length} 區`, "zone-panel-apply", () => action(applyDraft));
  const problem = el("p", "zv-problem zv-panel-problem"); problem.id = "zone-panel-problem"; problem.setAttribute("role", "alert");
  const sync = () => {
    const why = draftProblem(items);
    problem.textContent = why; problem.hidden = !why;
    apply.disabled = !draft || (draft.mode === "fixed" && !draft.materialId && !allPasted) || Boolean(why);
  };
  for (const mode of ["fixed", "keep_original", "search"]) {
    // 一項一列：圓鈕、字（點字也算點圓鈕）、色塊；材料下拉放在字的下面、不包在字的標籤裡（包進去的話點下拉會被當成點圓鈕）。
    const row = el("div", `zv-opt${draft?.mode === mode ? " on" : ""}`); row.id = `zone-panel-option-${mode}`;
    const radio = el("input"); radio.type = "radio"; radio.name = "zone-panel-mode"; radio.id = `zone-panel-mode-${mode}`; radio.checked = draft?.mode === mode;
    // 點「指定貼一種」而規則本身就是指定貼：先放規則那一筆（選它＝跟其他牆面一樣），要別的再換。
    radio.onchange = () => choose(mode, mode === "fixed" ? (draft?.mode === "fixed" ? draft.materialId : (sameAs === "fixed" ? ruleMethod()?.materialId ?? "" : "")) : "");
    const body = el("div", "zv-opt-body");
    const words = el("label", "zv-opt-words", ZONE_MODE_WORDS[mode]); words.htmlFor = radio.id; body.append(words);
    // 指定貼一種要連材料都跟規則同一筆才算「＝其他牆面」：下拉已經選了別筆就不標在這一列，改標在下拉裡規則那一筆後面
    // （列上已經標了就不再加：加了會出現在收起來的那一格，把名字擠到被截掉）。
    const ruleMaterial = sameAs === "fixed" ? ruleMethod()?.materialId ?? null : null;
    // 下拉停在「照現在貼的，不換」而選到的本來就都跟著規則，也算同一筆（規則貼的不是材料庫裡的成品時只有這條路）。
    const keepsRule = Boolean(rule) && draft?.materialId === "" && items.every(item => item.method === rule.key);
    // 由程式最佳化也要連清單都跟規則同一份才算：選中的區套用後的清單（照自己的清單加減）有一區跟規則的不一樣就不標。
    const ruleList = sameAs === "search" ? rule.zones[0].allowed : null;
    const listsAsRule = !ruleList || !draft || (panelTicks(items)?.lists.every(list => sameList(list, ruleList)) ?? true);
    const sameHere = sameAs === mode && listsAsRule && (mode !== "fixed" || draft?.mode !== "fixed" || (ruleMaterial !== null && draft.materialId === ruleMaterial) || keepsRule);
    if (sameHere) { const same = el("span", "zv-same", "＝其他牆面"); same.id = "zone-panel-same"; same.dataset.mode = mode; words.append(same); }
    body.append(...nowMark(items, mode));
    if (mode === "fixed") {
      const material = el("select", "zv-pick zv-pick-material"); material.id = "zone-panel-material"; material.setAttribute("aria-label", "指定貼哪一種");
      const none = el("option", null, allPasted ? "照現在貼的，不換" : "選要貼哪一種"); none.value = ""; material.append(none);
      const current = draft?.mode === "fixed" && draft.materialId ? libraryEntry(draft.materialId) : null;
      for (const entry of libraryEntries("fixed").filter(item => walls.every(wall => usableHere(item, wall)) || item === current)) {
        const option = el("option", null, `${vprWords(entry.label)}${entry.id === ruleMaterial && !sameHere ? "（＝其他牆面）" : ""}`); option.value = entry.id; material.append(option);
      }
      material.value = draft?.mode === "fixed" ? draft.materialId : "";
      material.onchange = () => choose("fixed", material.value);
      body.append(material);
    }
    if (mode === "search" && draft?.mode === "search") {
      // 選到由程式最佳化：這一列底下講從幾種裡選；是哪幾種、各調多少在下面那個材料框（searchForm），全頁只有那一處。
      const ticks = panelTicks(items);
      const mixed = ticks?.ids.some(id => ticks.state(id) === "some");
      const from = el("div", "zv-from", !ticks ? "程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）"
        : mixed ? `選中的 ${items.length} 區清單不一樣：沒動的勾，各區照目前的。每一區各自最佳化。`
        : `程式從下面勾的 ${ticks.ids.length} 種裡選，每一區各自最佳化。`);
      from.id = "zone-panel-search-from"; body.append(from);
    }
    row.append(radio, body, el("span", `zv-sw ${shownOf(mode)}`));
    card.append(row);
    if (mode === "fixed" && draft?.mode === "fixed" && draft.materialId && libraryEntry(draft.materialId)) card.append(materialShow(libraryEntry(draft.materialId), walls[0]));
    if (mode === "search" && draft?.mode === "search") card.append(...searchForm(items, walls, sync));
  }
  const foot = el("div", "zv-editfoot");
  sync();
  card.append(problem);
  const unchanged = el("span", "zv-cancel-note", "取消＝不改"); unchanged.id = "zone-panel-cancel-note";
  foot.append(apply, button("zv-btn", "取消", "zone-panel-cancel", () => select([])), unchanged);
  card.append(foot);
  return card;
}

/* ───── 右邊：規則、另外指定、清單卡 ───── */
// 「其他牆面」那條：下拉三選一，一選就寫到所有沒另外指定的區；選指定貼一種時旁邊多一個材料下拉（列牆面能貼的固定成品），
// 選了貼哪一種才寫。
function ruleRow() {
  const {inferred, roles} = view, rule = inferred.rule;
  const mode = ruleDraft ? "fixed" : rule?.mode ?? "keep_original";
  const row = el("div", `zv-rule ${shownOf(mode)}`); row.id = "zone-rule";
  const head = el("h2", null, "其他牆面"); head.append(el("span", "zv-muted", "沒另外指定的牆區都照這條"));
  row.append(head);
  if (!rule) return row;
  row.dataset.mode = rule.mode; row.dataset.key = rule.key;
  const pick = el("select", `zv-pick ${shownOf(mode)}`); pick.id = "zone-rule-mode"; pick.setAttribute("aria-label", "其他牆面的做法");
  for (const [value, words] of Object.entries(ZONE_MODE_WORDS)) { const option = el("option", null, words); option.value = value; pick.append(option); }
  pick.value = mode;
  pick.onchange = () => {
    if (pick.value === rule.mode) { ruleDraft = false; renderZoneOverview(); }
    else if (pick.value === "fixed") { ruleDraft = true; renderZoneOverview(); }
    else action(() => applyRule({mode: pick.value, allowed: shownList()}));
  };
  row.append(pick);
  if (mode === "fixed") {
    const material = el("select", "zv-pick zv-pick-material"); material.id = "zone-rule-material"; material.setAttribute("aria-label", "其他牆面指定貼哪一種");
    const current = ruleDraft ? null : libraryLabel(rule.zones[0].zone.material);
    if (!current) { const now = el("option", null, ruleDraft ? "選要貼哪一種" : `現在：${pastedWords(rule.zones[0])}`); now.value = ""; material.append(now); }
    const walls = WALL_ORDER.map(role => roles[role]);
    for (const entry of libraryEntries("fixed").filter(item => walls.every(wall => usableHere(item, wall)) || item === current)) {
      const option = el("option", null, vprWords(entry.label)); option.value = entry.id; material.append(option);
    }
    material.value = current?.id ?? "";
    material.onchange = () => { if (material.value) action(() => applyRule({mode: "fixed", materialId: material.value})); };
    row.append(material);
  }
  return row;
}
function specialRow() {
  const {inferred} = view;
  const zones = inferred.special.flatMap(group => group.zones);
  const row = el("div", "zv-rule zv-special"); row.id = "zone-special";
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
      button("zv-btn", "改", `zone-special-edit-${index}`, () => select(group.zones.map(item => item.id))));
    row.append(block);
  });
  if (zones.length) {
    const acts = el("div", "zv-acts");
    acts.append(button("zv-btn", `這 ${zones.length} 區改回跟其他牆面一樣`, "zone-special-reset", () => action(resetSpecial)));
    row.append(acts);
  }
  return row;
}
// 天花、地板、門、窗現在的樣子：一樣的併成一句（「天花、地板、門、窗：保留原樣」）。回 [[名字…], 一句]…。
function othersLines() {
  const {faces, roles} = view;
  const openings = WALL_ORDER.flatMap(role => faces[roles[role]]).filter(item => item.opening);
  const parts = [["天花", faces.ceiling], ["地板", faces.floor], ["門", openings.filter(item => item.special === "門")],
                 ["窗", openings.filter(item => item.special === "窗")]].filter(([, items]) => items.length);
  const words = items => {
    const counts = {search: 0, fixed: 0, still: 0};
    for (const item of items) counts[item.shown === "opening" ? "still" : item.shown] += 1;
    if (counts.still === items.length) return "保留原樣";
    return [counts.search ? `${ZONE_MODE_WORDS.search} ${counts.search} 區` : "", counts.fixed ? `指定貼 ${counts.fixed} 區` : "",
            counts.still ? `保留原樣 ${counts.still} 區` : ""].filter(Boolean).join("、");
  };
  const merged = [];
  for (const [name, items] of parts) {
    const text = words(items), last = merged.at(-1);
    if (last && last.text === text) last.names.push(name); else merged.push({names: [name], text});
  }
  return merged;
}
function othersRow() {
  const row = el("div", "zv-rule still"); row.id = "zone-others";
  for (const {names, text} of othersLines()) {
    const head = el("h2", "zv-others-line", names.join("、")); head.append(el("span", "zv-plain", `：${text}`));
    row.append(head);
  }
  return row;
}
// 面板開著的時候規則那張卡收成三行（樣稿第二張）：只看得到現在的設定，要改先關面板。
function briefRules() {
  const {inferred} = view;
  const card = el("div", "zv-rules zv-brief"); card.id = "zone-rules-brief";
  const line = (shown, name, ...rest) => {
    const row = el("div", `zv-rule ${shown}`); row.append(el("b", null, name), "　", ...rest); card.append(row);
  };
  line(shownOf(inferred.rule?.mode ?? "keep_original"), "其他牆面", inferred.rule ? methodWords(inferred.rule) : "");
  const special = inferred.special.flatMap(group => group.zones);
  // 正在改的是哪幾區：號碼圈塗黑，後面另起一句點名（不用跟在清單尾巴的括號，那會被讀成在改最後一區）。
  const editing = special.filter(item => selected.includes(item.id));
  const now = el("span", "zv-editing");
  if (editing.length) { now.id = "zone-brief-editing"; now.append("正在改：", zonesBrief(editing, new Set(editing.map(item => item.id)))); }
  line("fixed", `另外指定的 ${special.length} 區`, zonesBrief(special, new Set(editing.map(item => item.id))), ...(editing.length ? [now] : []));
  for (const {names, text} of othersLines()) line("still", names.join("、"), text);
  return card;
}
function listName(id) { return id === ZONE_NONE ? "也可以不貼（保留原樣）" : vprWords(choiceName(id)); }
// 一張清單卡管哪幾區：這一組就是規則那一組（規則本身是由程式最佳化）就寫「沒另外指定的牆區」，其餘的區點名。
// alone：全部由程式最佳化的區只有這一組（一張卡）。
function listWho(group, prefix, alone) {
  const {inferred} = view;
  const line = el("div", "zv-who"); line.id = `${prefix}-who`;
  line.append(`${alone ? "現在管" : "這一張管"} ${group.zones.length} 區：`);
  const byRule = inferred.rule?.mode === "search" && inferred.rule.key === group.key ? new Set(inferred.rule.zones.map(item => item.id)) : new Set();
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
// 「加別種材料」：列這一組的區貼得了、還不在清單裡的（可調的在前、固定成品在後），勾了就加進這一組每一區的清單。
function listAddBox(group, allowed, prefix) {
  const box = el("div", "zv-checks zv-add"); box.id = `${prefix}-add-box`;
  const walls = [...new Set(group.zones.map(item => item.wall))];
  const more = [...(allowed.includes(ZONE_NONE) ? [] : [ZONE_NONE]),
    ...["optimizable", "fixed"].flatMap(kind => libraryEntries(kind).filter(entry => walls.some(wall => usableHere(entry, wall)) && !allowed.includes(entry.id)).map(entry => entry.id))];
  // 清單裡勾著的可調材料已經包含的成品：灰掉、勾不了、寫為什麼（跟小框的材料框同一條規則，#1199）。
  const coveredBy = coveredProducts(allowed);
  for (const id of more) {
    const row = el("label", "zv-check"); row.dataset.choice = id;
    const check = el("input"); check.type = "checkbox"; check.id = `${prefix}-add-${id}`;
    check.onchange = () => writeList(group, {add: id});
    const words = el("span", null, listName(id));
    if (id !== ZONE_NONE) words.append(el("small", null, choiceWords(libraryEntry(id))));
    const family = coveredBy.get(id);
    if (family) {
      row.classList.add("zv-covered"); row.dataset.covered = family.id; check.disabled = true;
      words.append(el("small", "zv-covered-why", coveredWhy(family)));
    }
    row.append(check, words); box.append(row);
  }
  if (!more.length) box.append(el("div", "zv-who", "這幾區能貼的都已經在清單裡了"));
  return box;
}
// 清單卡（#1199 修補 1）：由程式最佳化的區照清單分組（zone_rules.js 的 zoneSearchList），清單一樣的一組一張卡。
// 各區清單都一樣（最常見）就是一張，寫「都從這幾種裡選」；不一樣就一組一張，每張寫是哪幾區、准哪些，在一張上勾只改那一組。
// 第一張的代號照舊（zone-list、zone-list-check-…），第二張起加號碼（zone-list-2、zone-list-2-check-…）。
function listCards(compact) {
  const {list} = view;
  if (list.zones.length) return list.groups.map((group, index) => listCard(group, index, compact));
  const card = el("div", "zv-list empty"); card.id = "zone-list"; card.dataset.count = "0";
  const head = el("h2"); head.append(el("i", "zv-sw search"), "現在沒有哪一區由程式最佳化");
  card.append(head);
  return [card];
}
function listCard(group, index, compact) {
  const {list} = view;
  const prefix = index ? `zone-list-${index + 1}` : "zone-list", alone = list.groups.length === 1, size = group.zones.length;
  const card = el("div", "zv-list"); card.id = prefix;
  card.dataset.count = String(size); card.dataset.zones = group.zones.map(item => item.id).join(" ");
  const head = el("h2");
  head.append(el("i", "zv-sw search"), alone ? `由程式最佳化的 ${size} 區` : `由程式最佳化的 ${list.zones.length} 區裡的 ${size} 區`,
    el("span", "zv-muted", alone ? "每一區各自最佳化，都從這幾種裡選" : size > 1 ? `每一區各自最佳化，這 ${size} 區從這幾種裡選` : "這一區從這幾種裡選"));
  card.append(head);
  const allowed = group.zones[0].allowed;
  if (!allowed) {
    card.append(listWho(group, prefix, alone), el("div", "zv-who", "程式從全部可調材料與「不貼」裡選（材料庫沒讀全，列不出是哪幾種）"));
    return card;
  }
  if (compact) {
    const names = el("div", "zv-names"); names.id = `${prefix}-names`;
    const materials = allowed.filter(id => id !== ZONE_NONE).map(id => vprWords(choiceName(id))).join("、");
    names.append(el("span", null, [materials, ...(allowed.includes(ZONE_NONE) ? ["也可以不貼"] : [])].filter(Boolean).join("，")),
      button("zv-link", "改清單", `${prefix}-edit`, () => select([])));
    if (!alone) card.append(listWho(group, prefix, alone));
    card.append(names);
    return card;
  }
  card.append(listWho(group, prefix, alone));
  const checks = el("div", "zv-checks"); checks.id = `${prefix}-checks`;
  for (const id of allowed) {
    const row = el("label", "zv-check"); row.dataset.choice = id;
    const check = el("input"); check.type = "checkbox"; check.checked = true; check.id = `${prefix}-check-${id}`;
    check.onchange = () => {
      if (allowed.length > 1) { writeList(group, {remove: id}); return; }
      zoneNote(`清單至少要留一種；不想讓程式選，就把那幾區改成「${ZONE_MODE_WORDS.fixed}」或「${KEEP_WORDS}」`, "notice");
      renderZoneOverview();
    };
    // 可調的一種一列，名字底下寫調什麼、從多少到多少（跟小框的材料框同一種寫法；範圍是材料頁設的）。
    const words = el("span", "zv-mwords", listName(id)), entry = libraryEntry(id);
    if (entry?.kind === "optimizable") words.append(el("small", "zv-range", adjustWords(entry)));
    row.append(check, words);
    checks.append(row);
  }
  card.append(checks);
  // 清單每一種都是固定成品：程式沒得選（同做法表的紅框），「拿去搜尋」也按不下去。
  if (allowedProblem(allowed)) {
    const problem = el("p", "zv-problem"); problem.id = `${prefix}-problem`; problem.setAttribute("role", "alert");
    problem.textContent = `${allowedProblem(allowed)}程式要從一個能調的東西開始：加一種可調的材料（例如玻璃棉），或加上「也可以不貼」。要的就是這一筆、不必讓程式選：點圖上那一區，改成「${ZONE_MODE_WORDS.fixed}」。`;
    card.append(problem);
  }
  const links = el("div", "zv-links");
  links.append(button("zv-link", listAdding === index ? "加別種材料 ▾" : "加別種材料 ▸", `${prefix}-add`, () => { listAdding = listAdding === index ? null : index; renderZoneOverview(); }));
  card.append(links);
  if (listAdding === index) card.append(listAddBox(group, allowed, prefix));
  return card;
}
// 「每一區的尺寸與代號」：平常收著，點開才看得到代號與尺寸（伺服器的訊息與搜尋頁寫的是代號）。
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
      const what = item.shown === "search" ? ZONE_MODE_WORDS.search : item.shown === "fixed" ? ZONE_MODE_WORDS.fixed : "保留原樣";
      row.append(el("td", null, zoneWhereText(item)), el("td", null, item.key === "whole_wall" ? "" : `代號 ${item.key}`), el("td", null, size), el("td", null, what));
      table.append(row);
    }
  }
  details.append(table); wrap.append(details);
  return wrap;
}

/* ───── 最底下：結果列 ───── */
function resultBar() {
  const {faces, roles} = view;
  const all = Object.values(faces).flat();
  const of = shown => all.filter(item => item.shown === shown);
  const area = items => items.reduce((sum, item) => sum + item.area, 0).toFixed(1);
  const bar = el("div", "zv-foot"); bar.id = "zone-result";
  const part = (shown, words, items) => {
    const item = el("span", "zv-part"); item.id = `zone-result-${shown}`; item.dataset.count = String(items.length);
    item.dataset.zones = items.map(zone => zone.id).join(" ");
    item.append(el("span", `zv-sw ${shown}`), `${words} `, el("b", null, `${items.length} 區`), `（約 ${area(items)} 平方公尺）`);
    return item;
  };
  bar.append(part("search", ZONE_MODE_WORDS.search, of("search")), el("span", "zv-sep"), part("fixed", "指定貼", of("fixed")), el("span", "zv-sep"));
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
  // 「拿去搜尋」照做法表那一顆的擋法：沒有區由程式最佳化、有紅框（清單只放了固定成品）、有超過限制的區，都按不下去。
  const go = button("zv-go", "拿去搜尋", "zone-page-search", () => action(sendToSearch));
  // 按不下去時旁邊一句講為什麼（細節在上面那一行紅字）。
  const why = !of("search").length ? "還沒有由程式最佳化的區" : stuckZoneNames().length ? "清單有紅框要先處理" : limitsRedCount() > 0 ? "有超過限制的區要先改" : "";
  go.disabled = Boolean(why);
  // 原因那一句收在按鈕正下方（小字一行），不跟前面三項搶同一行的寬度。
  const box = el("span", "zv-gobox"); box.append(go);
  if (why) { const line = el("span", "zv-why", why); line.id = "zone-page-search-why"; box.append(line); }
  bar.append(rest, box);
  return bar;
}
// 結果列上面那一行：剛才那一步的結果（同做法表底下那一行），以及「拿去搜尋」為什麼按不下去。
function noteLines() {
  const wrap = el("div", "zv-notes");
  const red = limitsRedCount(), stuck = stuckZoneNames();
  const blocked = [...(stuck.length ? [stuckWords(stuck)] : []), ...(red ? [`有 ${red} 條超過限制，改好之前不能拿去搜尋：${(zoneLimits?.refusals ?? []).join("；")}`] : [])];
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

function closeOverview() {
  overviewOpen = false;
  renderZoneOverview();
  // 這一頁開著時輸入頁是藏起來的：那時重畫的平面圖，字量不到寬度、避不開彼此。回來看得到了，照同一份再畫一次。
  drawShownPlan();
}
// 這一頁沒有存檔鍵：改動當場套到畫面，但要存檔才進檔案。改過、還沒存的時候標題列講一句怎麼存（兩條路）。
function unsavedLine() {
  const line = el("span", "zv-unsaved", "改的還沒存進檔案：回「房間方案輸入」按「儲存」，或按底下的「拿去搜尋」（會先存再跳過去）");
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
  const page = $("zone-page");
  if (!page) return;
  page.hidden = !overviewOpen;
  document.querySelector("main").hidden = overviewOpen;
  document.body.classList.toggle("zone-page-open", overviewOpen);
  // 整份文件也掛一個記號：底列黏在視窗下緣，瀏覽器自己把東西捲進畫面時要讓出底列那一段（zone_overview.css）。
  document.documentElement.classList.toggle("zone-page-open", overviewOpen);
  if (!overviewOpen) return;
  const scrolled = window.scrollY;
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
  // 正在「最深」那一格打字時別的事觸發重畫（晚到的限制檢查）：字記在草稿裡不會掉，焦點放回那一格。
  const typing = document.activeElement?.id === "zone-panel-depth";
  page.replaceChildren(...parts);
  if (typing) $("zone-panel-depth")?.focus();
  window.scrollTo(0, scrolled);
}
function overviewParts() {
  const {head, back} = pageHead();
  const {wallSize} = window.elevationGeometry;
  if (!scheme || !inputDefaults || Object.keys(wallNames).some(wall => !wallSize(wall))) {
    head.append(el("p", null, "先把房間的長、寬、高都填好正數，才畫得出這間房。"), back);
    return [head];
  }
  // 換了方案（打開別份、載入範本）：上一份選的區、面板上點的都不算數。
  if (seenGeneration !== zoneDesignGeneration) { seenGeneration = zoneDesignGeneration; selected = []; draft = null; draftTouched = false; ruleDraft = false; listAdding = null; keptRule = null; keptRuleZone = null; unsaved = false; keptSaid = ""; }
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
  view = {roles, room, faces, byId, inferred, list: zoneSearchList(searchZones(faces, roles), inferred.rule?.key)};
  // 選的區不在了（那一面重切過）就拿掉；一區都不剩，面板也收起來。
  selected = selected.filter(id => byId.has(id));
  if (!selected.length) { draft = null; draftTouched = false; }
  // 圖上只在另外指定的區（與天花、地板）寫材料名。
  const apart = new Set(inferred.special.flatMap(group => group.zones.map(item => item.id)));
  for (const item of byId.values()) { item.apart = apart.has(item.id); item.named = item.wall === "floor" || item.wall === "ceiling" || item.apart; }
  const body = el("div", "zv-main");
  const mapWrap = el("div", "zv-mapwrap");
  const {map, legendTop, whole} = drawMap(faces, roles, room);
  mapWrap.append(map, legend(legendTop), ...wholeButtons(whole));
  const side = el("div", "zv-side");
  if (selected.length) side.append(panelCard(), briefRules(), ...listCards(true));
  else {
    const slot = el("div", "zv-slot"); slot.id = "zone-slot";
    slot.append(el("b", null, "點圖上任何一塊，這裡就出現它的選項。"), el("br"), "牆、門、窗、天花、地板都能點；再點別塊可以一起改；牆名旁「選整面」一次選一整面牆。");
    const rules = el("div", "zv-rules");
    rules.append(ruleRow(), specialRow(), othersRow());
    side.append(slot, rules, ...listCards(false), detailTable());
  }
  body.append(mapWrap, side);
  return [head, body, noteLines(), resultBar()];
}
window.renderZoneOverview = renderZoneOverview;
window.openZoneOverview = () => { overviewOpen = true; pageSaid = ""; renderZoneOverview(); window.scrollTo(0, 0); };
// 輸入頁訊息列每寫一句（app.js 的 say）就叫一次：警示照抄到這一頁，換成成功的一句就拿掉。
// 這一頁打開時從空的開始（上面那一行）：打開之前輸入頁講過的話不帶進來。
// 方案與每一區的做法都存進檔案了（輸入頁按「儲存」、另存，或「拿去搜尋」先存的那一步）：那一句拿掉。
window.zonePageSaved = () => { unsaved = false; renderZoneOverview(); };
window.zonePageSay = (text, kind) => {
  const next = kind === "notice" ? text : "";
  if (next === pageSaid) return;
  pageSaid = next;
  renderZoneOverview();
};
// 「拿去搜尋」被還在路上的一批改動擋下（zone_modes.js 的 sendToSearch，#1188 第 6 條）：那一批落地後平面圖檢查通過，
// 訊息列換成成功的一句、上面那一支會把警示拿掉——被擋的那一次什麼都沒存、沒跳頁，那一句不能跟著不見。
// 留到使用者下一次真的按到東西（滑鼠或鍵盤按出來的 click；頁面程式自己發的不算）為止。整頁還鎖著的時候按的不算：
// 那一下什麼都按不到（點擊落在頁面底層），那一批也還在路上，這時清掉、落地後就看不到了。換方案也清（重畫時看做法換過幾次）；
// 打開這一頁不必另外清：打開只有那一顆按鈕一條路，那一下 click 已經先清過。
// 按下去那一刻只就地改那一行、不重畫整頁：重畫會把正在被按的那一顆換掉。
function forgetKept(event) {
  if (!event.isTrusted || !keptSaid || inputBusy) return;
  keptSaid = "";
  const line = $("zone-page-said");
  if (line && saidWords()) line.textContent = saidWords(); else line?.remove();
}
window.zonePageKeep = text => { keptSaid = text; renderZoneOverview(); };
document.addEventListener("click", forgetKept, true);
window.addEventListener("DOMContentLoaded", () => { $("zone-page-open").onclick = () => window.openZoneOverview(); });
})();
