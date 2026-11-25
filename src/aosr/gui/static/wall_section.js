/* 輸入頁第 2 段「牆上切區」（#1265 單元 3，示意圖 input-page-v3 第 2 段）：選中那一面的牆那一行（尺寸那一句在
   elevation.js 寫）、右欄唯讀清單「前牆這 N 區」、「每一區的位置與大小」摺疊。
   - 牆那一行兩句：原本的牆（只用牆面表單本來就寫出來的字：種類選單上的名字，多層構造寫預設名字或「自訂構造」那句；
     不從阻抗換算成「硬牆」之類的話）、這一面的總厚度上限（這一面另設／照房間預設／都沒設三種）。
     「換原本的牆」「這一面另設」打開那一面唯一的編輯處（room_facts.js 的 toggleEditor，打開哪一塊記在那一塊自己身上）。
   - 右欄清單照做法分組（由程式最佳化、指定貼一種、保留原樣；看不出現在是哪一種的另一組），分法跟「這間房貼什麼」
     那一頁同一份（zone_overview.js 的 faceZones：沒貼東西的「指定貼一種」算保留原樣、門窗算保留原樣）；
     每一區一顆「改這一區 ▸」（zone_modes.js 的 zoneGotoButton，整頁只有這一處）。
   - 摺疊：一區一列，離左、離下（牆面寫離地）、寬 × 高照立面圖畫面上的方向（同選取框那四格），現在貼的照牆面表單的寫法
     （wallText，固定阻抗帶數字與單位），底下接原本那一面的數字。
   - 右欄上方（#1265 單元 3 之二第二趟，舊做法表拆了搬過來）：限制總結「限制都沒超過」（zone-limits-summary）、這一面整面的判決
     （limit-wall-<牆>-<i>）、設計設定讀不到與材料庫有問題那兩句。
   全部從方案（scheme）、每一區的做法（zoneDesign）、限制判決（zoneLimits）現算，畫的時候只讀不寫；跟方案同一條路重畫
   （zone_modes.js 的 renderZoneTable 開頭），開檔、切牆、換材料、改做法、判決回來都會走到。 */
(() => {
const byId = id => document.getElementById(id);
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
const metres = value => Number(value.toFixed(3));
// 右欄清單的組：順序照示意圖（由程式最佳化排最前）；門窗（opening）跟保留原樣同一組。
const GROUPS = [["search", ZONE_MODE_WORDS.search], ["fixed", ZONE_MODE_WORDS.fixed], ["still", "保留原樣"],
                ["unknown", "看不出現在是哪一種"]];
const PAIR_WHERE = "（兩邊要不要貼一樣，在「這間房貼什麼」設）";

// 一種材料在牆面表單上叫什麼（不帶數字；數字在「每一區的位置與大小」摺疊）。
function originalText(material) {
  if (material.kind === "assembly") return matchingPreset(material)?.label ?? "自訂構造（參數與預設不完全相同）";
  return inputDefaults.material_form[material.kind].label;
}
function originalLine(wall) {
  if (!zonedWall(wall)) return originalText(originalMaterial(wallMaterial(wall)));
  const originals = scheme.scene.surface_zones_by_wall[wall].map(zone => originalMaterial(zone.material));
  return originals.every(material => sameMaterial(material, originals[0])) ? originalText(originals[0])
    : "每一區不一樣（看右邊「每一區的位置與大小」）";
}
function depthLine(wall) {
  const own = zoneDesign?.wall_max_depth_m?.[wall], room = zoneDesign?.room_max_depth_m;
  if (own != null) return `貼的東西總厚度最多 ${centimetres(own)} 公分（這一面另設）`;
  if (room != null) return `貼的東西總厚度最多 ${centimetres(room)} 公分（照房間預設）`;
  return "貼的東西總厚度：這一面沒另設，房間預設也沒設上限";
}
// 收起牆那一行打開的一塊（room_facts.js 的 toggleEditor 打開的）：那一塊藏起來、鈕記成收著。
function closeEditor(button, area) {
  if (byId(area).hidden) return;
  byId(area).hidden = true;
  byId(button).setAttribute("aria-expanded", "false");
}
// 牆那一行上一次畫的是哪一面（關掉一塊時 refreshWallLine 照它再畫一次）。
let lineWall = null;
function renderWallLine(wall) {
  lineWall = wall;
  const ready = Boolean(wall && window.elevationGeometry.wallSize(wall));
  // 房間尺寸清空、畫不出這一面時：沒打開的那一行照舊收起來，打開著的那一行照留——裡面那顆鈕是關掉那一塊唯一的地方
  // （#1265 單元 3 之二第二趟決定第 1 條走乙）。關掉之後 room_facts.js 的 toggleEditor 叫 refreshWallLine，那一行跟著收。
  const open = area => Boolean(wall) && !byId(area).hidden;
  const showOriginal = ready || open("wall-original-editor"), showDepth = ready || open("wall-depth-editor");
  byId("elevation-original-line").hidden = !showOriginal;
  byId("elevation-depth-line").hidden = !showDepth;
  if (!showOriginal && !showDepth) return;
  byId("elevation-original").textContent = `原本的牆：${originalLine(wall)}`;
  byId("elevation-depth").textContent = depthLine(wall);
  // 分區的牆：牆面表單不改它（每一區各自存一份），「換原本的牆」那一塊沒有東西可改，鈕收起來、打開的那一塊也關上。
  const editable = !zonedWall(wall);
  byId("edit-wall-original").hidden = !editable;
  if (!editable) closeEditor("edit-wall-original", "wall-original-editor");
}

// 判決在路上（zoneLimitsFresh 不成立）就不寫上一版的上限（#1265 單元 3 之二第二趟追加第 3 條）。
function capWords(wall, key) {
  const cap = zoneLimitsFresh() ? zoneCap(wall, key) : null;
  return cap ? `總厚度最多 ${centimetres(cap.value_m)} 公分（${DEPTH_LEVEL_WORDS[cap.level]}）` : "";
}
// 由程式最佳化的區沒有上限時那一句：只有最新那一次限制檢查回來、而且做成了，才講「沒給上限」；還在路上、沒做成照實講
// （左邊牆那一行照設計設定寫上限，這裡不能說沒有；#1265 單元 3 審查真錯 5）。
function noCapWords() {
  if (!zoneLimitsFresh()) return "總厚度上限還在核對";
  if (zoneLimits.status !== "checked") return "總厚度上限沒核對成（限制檢查沒做成）";
  return "總厚度沒給上限：照材料自己的限制";
}
// 「跟誰一對」：用「這間房貼什麼」那一頁同一份配對（zone_overview.js 的 zonePairMarks），兩頁說法一致；選取框底下那一句
// （elevation.js 的 pairWords）講的是畫框時的幾何鏡像，照舊。
function pairSentence(marks, item) {
  const mark = marks.get(item.id);
  if (mark?.partner) return `${window.zonePartnerWords(mark.partner)}${PAIR_WHERE}。`;
  return mark?.missed ? `沒配成一對${PAIR_WHERE}。` : "";
}
// 每一區那一句：照做法講現在怎樣，接「跟誰一對」。
function zoneSentence(wall, item, group, marks) {
  const {zone, key} = item;
  const paired = pairSentence(marks, item);
  if (group === "unknown") return `${uncertainWords(item.construction)}。${paired}`;
  if (group === "search") {
    const allowed = item.allowed;
    // 從幾種裡挑：跟那一頁小框同一支算（pickableChoices，程式不會挑的那一類不算進 N，另外點名；#1265 單元 3 審查設計洞 2）。
    const pickable = allowed && pickableChoices(allowed), aside = allowed ? allowed.filter(id => !pickable.includes(id)) : [];
    // 括號裡兩段（能挑的、程式不會挑的）有哪段寫哪段，中間才接「；」（全是程式不會挑的那一類時開頭不多一個「；」，第二趟追加）。
    const named = allowed && [pickable.map(choiceName).join("、"), aside.length ? `${aside.map(choiceName).join("、")}程式不會挑` : ""].filter(Boolean).join("；");
    const pick = allowed ? `從 ${pickable.length} 種裡挑（${named}）` : "從全部可調材料與「不貼」裡挑（材料庫沒讀全，列不出是哪幾種）";
    const cap = capWords(wall, key) || noCapWords();
    return `已交給程式挑：${pick}，${cap}。現在存的是「${plainNow(zone.material, wall, key)}」，直接算就照它算。${paired}`;
  }
  if (group === "fixed") {
    const cap = capWords(wall, key);
    return `${plainNow(zone.material, wall, key)}${cap ? `；${cap}` : ""}。${paired}`;
  }
  const words = item.setting.mode === "keep_original" ? keepWords(zone, wall, key) : plainNow(zone.material, wall, key);
  return `${words}。${paired}`;
}
function zoneRow(wall, item, group, words, marks) {
  const red = zoneIsRed(wall, item.key);
  const row = el("div", `zone-item zone-mode-${item.setting.mode}${red ? " zone-red" : ""}`);
  row.id = `wall-zone-${wall}-${item.key}`; row.dataset.zone = `${wall}/${item.key}`;
  const body = el("div");
  const name = el("b", null, item.label.name); name.id = `wall-zone-name-${wall}-${item.key}`;
  const what = el("div", "what", zoneSentence(wall, item, group, marks)); what.id = `wall-zone-what-${wall}-${item.key}`;
  what.append(zoneGotoButton(wall, item.key));
  body.append(name, el("span", `tag tag-${group}`, words), what);
  // 紅的那一區底下寫出它的每一條限制（同一份 zoneLimitLines，原文照伺服器；#1265 單元 3 之二收尾 G）：立面圖那一塊滑過去也有，
  // 這裡不必滑就看得到。跟這一列標紅同一個判法，不紅的區不寫。id limit-<牆>-<區>-<i>（舊做法表那一區底下那幾行的 id），整頁只這一份。
  if (red) body.append(limitLinesBlock(zoneLimitLines(wall, item.key), `limit-${wall}-${item.key}`));
  // 選的跟原本一樣、自己改成不貼的區（#1175）：那一區自己寫為什麼（舊做法表那一列的新家，#1265 單元 3 之二第二趟；
  // 「這間房貼什麼」小框套用時記，zone_modes.js 的 zoneSameAsOriginal）。原本那一面跟記下時一樣才寫（sameAsOriginalLabel 自己比，
  // 收尾追加第 5 條起這裡不另判做法：改做法的三個入口——小框套用、改規則、改回跟其他牆面一樣——都走 zone_overview.js 的
  // applyMethod，它自己重記或拿掉）。
  const same = sameAsOriginalLabel(wall, item.key);
  if (same) {
    const why = el("p", "ok", `選的「${same}」跟原本一樣，當成不貼（做法自己改成「${KEEP_WORDS}」）`); why.id = `zone-same-${wall}-${item.key}`;
    body.append(why);
  }
  const depths = depthTargetLines(wall, item.key);
  if (depths.length) {
    const block = el("div", "zone-depths"); block.id = `wall-zone-depths-${wall}-${item.key}`;
    block.append(...depths); body.append(block);
  }
  row.append(el("span", "zone-num", String(item.label.number)), body);
  return row;
}
function renderList(wall) {
  const title = byId("elevation-list-title"), area = byId("elevation-list-zones");
  area.replaceChildren();
  // 清單上方：限制總結與兩句錯誤提示（設計設定讀不到、材料庫有問題）講的是整份方案，有方案就寫，選不選牆都看得到；
  // 這一面整面的判決接在總結底下（舊做法表上方那幾行，#1265 單元 3 之二第二趟搬來，id 照舊）。
  // 列出來的區（照號碼排）一起交給總述：寫在那幾區底下的，總述不再抄（#1265 單元 3 之二收尾第二趟 A）。
  const shown = Boolean(wall && window.elevationGeometry.wallSize(wall)), lines = shown ? wallLimitLines(wall) : [];
  const items = shown ? window.zoneFaceZones(wall) : [];
  const keys = items.toSorted((left, right) => left.label.number - right.label.number).map(item => item.key);
  byId("elevation-list-notes").replaceChildren(limitsSummary(shown ? wall : null, keys),
                                               ...(lines.length ? [limitLinesBlock(lines, `limit-wall-${wall}`)] : []), ...zoneProblemLines());
  if (!shown) {
    title.textContent = "選一面牆，這裡列出那一面的每一區";
    return;
  }
  const marks = window.zonePairMarks();
  // 交給搜尋的區那一句底下接逐組生效上限（zone_limits.js 的 depthTargetLines）。
  title.textContent = items.length === 1 ? `${wallNames[wall]}整面一區` : `${wallNames[wall]}這 ${items.length} 區`;
  for (const [group, words] of GROUPS) {
    const here = items.filter(item => (item.shown === "opening" ? "still" : item.shown) === group);
    if (!here.length) continue;
    const block = el("div", "zone-group"); block.id = `wall-zones-${group}`; block.dataset.group = group;
    for (const item of here) block.append(zoneRow(wall, item, group, words, marks));
    area.append(block);
  }
}
// 「現在貼的」那一格：貼了東西的先寫貼的（牆面表單的寫法），底下一律接原本那一面叫什麼、數字多少
// （id zone-numbers-<牆>-<區>：舊做法表「數字（給要核對的人）」的新家，#1265 單元 3 之二第二趟；貼了東西的區也寫得出原本那一面）。
function nowCell(wall, key, material) {
  const cell = el("td");
  if (material.kind === "mounted_on_wall") cell.append(el("div", null, wallText(material)));
  const original = el("div", null, `${originalFace(wall)}：${wallText(originalMaterial(material))}`); original.id = `zone-numbers-${wall}-${key}`;
  cell.append(original);
  return cell;
}
// 「每一區的位置與大小」：號碼照立面圖（不寫自動補的代號），四個數照畫面方向（同選取框那四格）。
function renderNumbers(wall) {
  const area = byId("zone-numbers");
  area.replaceChildren();
  const {views, wallSize, screenBox} = window.elevationGeometry;
  const size = wall ? wallSize(wall) : null;
  if (!size) return;
  const [width, height] = size, labels = zoneLabels(wall);
  const zones = zoneList(wall).toSorted((left, right) => labels.get(zoneKey(left)).number - labels.get(zoneKey(right)).number);
  if (zonedWall(wall)) {
    const caption = el("p", "note", `${wallNames[wall]}：分成 ${zones.length} 區（分區在立面圖上改，這張表單不改這一面）`);
    caption.id = `zoned-${wall}`;
    area.append(caption);
  }
  const table = el("table", "zone-sizes"); table.id = `zoned-list-${wall}`;
  const head = el("tr");
  for (const text of ["區", "離左", wall === "floor" || wall === "ceiling" ? "離下緣" : "離地", "寬 × 高", "現在貼的"]) head.append(el("th", null, text));
  table.append(head);
  for (const zone of zones) {
    const key = zoneKey(zone), label = labels.get(key);
    const [a0, a1, b0, b1] = zone.area === "rectangle" ? screenBox(views[wall], width, height, zone) : [0, width, 0, height];
    const row = el("tr"); row.id = `zone-size-${wall}-${key}`;
    for (const text of [`${label.numberText} ${label.name}`, metres(a0), metres(b0), `${metres(a1 - a0)} × ${metres(b1 - b0)}`]) {
      row.append(el("td", null, String(text)));
    }
    row.append(nowCell(wall, key, zone.material));
    table.append(row);
  }
  area.append(table);
}
// 只從 renderZoneTable 叫，而它只在有方案時叫（renderElevation 開頭擋掉沒方案）：
// 這一支底下都不另判沒方案（#1265 單元 3 審查保險行）。
function renderWallSection(wall) {
  renderWallLine(wall);
  renderList(wall);
  renderNumbers(wall);
  byId("elevation-legend").hidden = !(wall && window.elevationGeometry.wallSize(wall));
}
window.renderWallSection = renderWallSection;
window.refreshWallLine = () => renderWallLine(lineWall);
})();
