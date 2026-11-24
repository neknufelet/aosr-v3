/* 輸入頁第 2 段「牆上切區」（#1265 單元 3，示意圖 input-page-v3 第 2 段）：選中那一面的牆那一行（尺寸那一句在
   elevation.js 寫）、右欄唯讀清單「前牆這 N 區」、「每一區的位置與大小」摺疊。
   - 牆那一行兩句：原本的牆（只用牆面表單本來就寫出來的字：種類選單上的名字，多層構造寫預設名字或「自訂構造」那句；
     不從阻抗換算成「硬牆」之類的話）、這一面的總厚度上限（這一面另設／照房間預設／都沒設三種）。
     「換原本的牆」「這一面另設」打開那一面唯一的編輯處（room_facts.js 的 toggleEditor，打開哪一塊記在那一塊自己身上）。
   - 右欄清單照做法分組（由程式最佳化、指定貼一種、保留原樣；看不出現在是哪一種的另一組），分法跟「這間房貼什麼」
     那一頁同一份（zone_overview.js 的 faceZones：沒貼東西的「指定貼一種」算保留原樣、門窗算保留原樣）；
     每一區一顆「改這一區 ▸」（zone_modes.js 的 zoneGotoButton，整頁只有這一處）。
   - 摺疊：一區一列，離左、離下（牆面寫離地）、寬 × 高照立面圖畫面上的方向（同選取框那四格），現在貼的照牆面表單的寫法
     （wallText，固定阻抗帶數字與單位）。
   全部從方案（scheme）、每一區的做法（zoneDesign）、限制判決（zoneLimits）現算，畫的時候只讀不寫；跟做法表同一條路重畫
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
function renderWallLine(wall) {
  const ready = Boolean(wall && window.elevationGeometry.wallSize(wall));
  byId("elevation-original-line").hidden = !ready;
  byId("elevation-depth-line").hidden = !ready;
  if (!ready) return;
  byId("elevation-original").textContent = `原本的牆：${originalLine(wall)}`;
  byId("elevation-depth").textContent = depthLine(wall);
  // 分區的牆：牆面表單不改它（每一區各自存一份），「換原本的牆」那一塊沒有東西可改，鈕收起來、打開的那一塊也關上。
  const editable = !zonedWall(wall);
  byId("edit-wall-original").hidden = !editable;
  if (!editable) closeEditor("edit-wall-original", "wall-original-editor");
}

function capWords(wall, key) {
  const cap = zoneCap(wall, key);
  return cap ? `總厚度最多 ${centimetres(cap.value_m)} 公分（${DEPTH_LEVEL_WORDS[cap.level]}）` : "";
}
// 由程式最佳化的區沒有上限時那一句：只有最新那一次限制檢查回來、而且做成了，才講「沒給上限」；還在路上、沒做成照實講
// （左邊牆那一行照設計設定寫上限，這裡不能說沒有；#1265 單元 3 審查真錯 5）。
function noCapWords() {
  if (!zoneLimits || zoneLimitsReceived !== zoneLimitsGeneration) return "總厚度上限還在核對";
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
    const pick = allowed ? `從 ${pickable.length} 種裡挑（${pickable.map(choiceName).join("、")}${aside.length ? `；${aside.map(choiceName).join("、")}程式不會挑` : ""}）`
      : "從全部可調材料與「不貼」裡挑（材料庫沒讀全，列不出是哪幾種）";
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
  const row = el("div", `zone-item zone-mode-${item.setting.mode}${zoneIsRed(wall, item.key) ? " zone-red" : ""}`);
  row.id = `wall-zone-${wall}-${item.key}`; row.dataset.zone = `${wall}/${item.key}`;
  const body = el("div");
  const name = el("b", null, item.label.name); name.id = `wall-zone-name-${wall}-${item.key}`;
  const what = el("div", "what", zoneSentence(wall, item, group, marks)); what.id = `wall-zone-what-${wall}-${item.key}`;
  what.append(zoneGotoButton(wall, item.key));
  body.append(name, el("span", `tag tag-${group}`, words), what);
  const depths = depthTargetLines(wall, item.key, item.setting);
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
  // 兩句錯誤提示（設計設定讀不到、材料庫有問題）講的是整份方案：有方案就寫在清單上方，選不選牆都看得到。
  byId("elevation-list-notes").replaceChildren(...zoneProblemLines());
  if (!wall || !window.elevationGeometry.wallSize(wall)) {
    title.textContent = "選一面牆，這裡列出那一面的每一區";
    return;
  }
  const items = window.zoneFaceZones(wall), marks = window.zonePairMarks();
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
    for (const text of [`${label.numberText} ${label.name}`, metres(a0), metres(b0), `${metres(a1 - a0)} × ${metres(b1 - b0)}`,
                        wallText(zone.material)]) row.append(el("td", null, String(text)));
    table.append(row);
  }
  area.append(table);
}
// 只從 renderZoneTable 叫，而它只在有方案時叫（renderElevation 開頭擋掉沒方案；做法表的下拉、開關也只在有方案時才畫）：
// 這一支底下都不另判沒方案（#1265 單元 3 審查保險行）。
function renderWallSection(wall) {
  renderWallLine(wall);
  renderList(wall);
  renderNumbers(wall);
  byId("elevation-legend").hidden = !(wall && window.elevationGeometry.wallSize(wall));
}
window.renderWallSection = renderWallSection;
})();
