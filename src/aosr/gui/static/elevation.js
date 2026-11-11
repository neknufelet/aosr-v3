/* 立面圖（#1091）：站在房間裡看那一面，畫框、選框、改尺寸、刪框；每改一次交伺服器補滿其餘（/api/zone-cut）。
   吸附格距、最小邊、自動補滿的代號字首由伺服器給（/api/input-defaults 的 zone_cut），這裡不另寫一份數字。
   只換被切的那一面：方案物件原地換這一面的分區，再跟改房間尺寸一樣重問預覽與圖面——
   別面因為房間改小而對不上時，這一面照樣切得動（整份方案交接會因為別面卡住，哪一面都修不了）。 */
(() => {
let elevationWall = null;
let elevationSelected = null;
let elevationThin = [];
// 站在房間裡看那一面：畫面往右、往上各是方案的哪一軸（反了就鏡像），四邊標的是隔壁哪一面。
// 四面牆：往上是 z；x 起點牆面向 -x、右手邊是 +y；x 終點牆面向 +x、右手邊是 -y（鏡像）；
// y 起點牆面向 -y、右手邊是 -x（鏡像）；y 終點牆面向 +y、右手邊是 +x。
// 地板往下看、天花往上看，都讓 x 往右：地板 y 往上（跟平面圖同向），天花抬頭看 y 往下。
const views = {
  x0: {flipU: false, flipV: false, left: "y0", right: "yL", top: "ceiling", bottom: "floor"},
  xL: {flipU: true, flipV: false, left: "yL", right: "y0", top: "ceiling", bottom: "floor"},
  y0: {flipU: true, flipV: false, left: "xL", right: "x0", top: "ceiling", bottom: "floor"},
  yL: {flipU: false, flipV: false, left: "x0", right: "xL", top: "ceiling", bottom: "floor"},
  floor: {flipU: false, flipV: false, left: "x0", right: "xL", top: "yL", bottom: "y0"},
  ceiling: {flipU: false, flipV: true, left: "x0", right: "xL", top: "y0", bottom: "yL"},
};
const SVG = "http://www.w3.org/2000/svg";
const metres = value => Number(value.toFixed(3));
function cutSpec() { return inputDefaults.zone_cut; }
function node(tag, attributes, text) {
  const element = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
// 牆面寬高照表單上現在填的房間尺寸（還沒交伺服器也算），讀不出正數就不畫。沒改過的格子用方案原值
// （格子裡顯示的是伺服器排過位數的字，4.41988145469 不等於 4.41988145469311），跟存檔時 collect 讀的同一份。
function wallSize(wall) {
  const room = Object.fromEntries(["Lx", "Ly", "Lz"].map(key => [key, $(`room-${key}`) ? numericInput(`room-${key}`) : scheme.scene.room_m[key]]));
  if (!Object.values(room).every(value => typeof value === "number" && Number.isFinite(value) && value > 0)) return null;
  return {floor: [room.Lx, room.Ly], ceiling: [room.Lx, room.Ly], x0: [room.Ly, room.Lz], xL: [room.Ly, room.Lz],
          y0: [room.Lx, room.Lz], yL: [room.Lx, room.Lz]}[wall];
}
// 方案座標（u、v）跟畫面座標（a 從左緣往右、b 從下緣往上，公尺）互換；鏡像兩次就回原值，同一支來回用。
function flip(view, width, height, u, v) {
  return [view.flipU ? width - u : u, view.flipV ? height - v : v];
}
function screenBox(view, width, height, zone) {
  const [a0, b0] = flip(view, width, height, zone.u_m[0], zone.v_m[0]);
  const [a1, b1] = flip(view, width, height, zone.u_m[1], zone.v_m[1]);
  return [Math.min(a0, a1), Math.max(a0, a1), Math.min(b0, b1), Math.max(b0, b1)];
}
function wallBox(view, width, height, a0, a1, b0, b1, id) {
  const [u0, v0] = flip(view, width, height, a0, b0);
  const [u1, v1] = flip(view, width, height, a1, b1);
  return {id, u_m: [Math.min(u0, u1), Math.max(u0, u1)], v_m: [Math.min(v0, v1), Math.max(v0, v1)]};
}
function wallZones(wall) {
  return scheme.scene.surface_zones_by_wall?.[wall] ?? [{area: "whole_wall"}];
}
function drawnBoxes(wall) {
  return wallZones(wall).filter(zone => zone.area === "rectangle" && !zone.id.startsWith(cutSpec().fill_prefix))
    .map(zone => ({id: zone.id, u_m: [...zone.u_m], v_m: [...zone.v_m]}));
}
// 喇叭與座位投影到這一面：x 兩面牆看 (y, z)、y 兩面牆看 (x, z)、地板天花看 (x, y)。
function project(wall, point) {
  return wall[0] === "x" ? [point[1], point[2]] : wall[0] === "y" ? [point[0], point[2]] : [point[0], point[1]];
}
// 拖曳畫的框：先吸到畫面上的格點、夾在牆內；再把離牆邊或別框的邊不到最小邊的邊貼過去（不然補滿會剩一條太窄的縫，
// 牆寬不在格點上時最容易發生）。打數字改的框只做第二步。
function snapEdges(box, others, width, height) {
  // 跟伺服器同一把：差不到「最小邊扣容差」才貼（0.7 − 0.5 在浮點不到 0.2，那是剛好一個最小邊，不貼）。
  const reach = cutSpec().min_side_m - cutSpec().edge_merge_tol_m;
  const snapOne = (value, targets) => {
    let best = value, gap = reach;
    for (const target of targets) {
      const distance = Math.abs(value - target);
      if (distance > 0 && distance < gap) { best = target; gap = distance; }
    }
    return best;
  };
  const uTargets = [0, width, ...others.flatMap(other => other.u_m)];
  const vTargets = [0, height, ...others.flatMap(other => other.v_m)];
  return {id: box.id, u_m: box.u_m.map(value => snapOne(value, uTargets)), v_m: box.v_m.map(value => snapOne(value, vTargets))};
}
function gridSnap(value, limit) {
  const step = cutSpec().snap_m;
  return Math.min(Math.max(Number((Math.round(value / step) * step).toFixed(6)), 0), limit);
}
function boxTooSmall(box) {
  const tolerance = cutSpec().edge_merge_tol_m;
  return Math.min(box.u_m[1] - box.u_m[0], box.v_m[1] - box.v_m[0]) < cutSpec().min_side_m - tolerance;
}
function newBoxId(wall) {
  const used = new Set(wallZones(wall).map(zone => zone.id));
  let n = 1;
  while (used.has(`z${n}`)) n += 1;
  return `z${n}`;
}
function note(text, kind) {
  const line = $("elevation-note");
  line.textContent = text; line.className = kind;
}
// 交伺服器切這一面；成功才原地換掉這一面的分區。等回覆時整頁鎖住（同交接），免得半路又改了別格被畫面蓋掉。
async function cutWall(wall, boxes, done) {
  if (inputBusy) { note("上一個改動還在處理，請稍等再切", "notice"); return; }
  if (measuredDrafts[wall]) { note(`${wallNames[wall]}有還沒套用的實測阻抗草稿：先套用或取消，再切這一面`, "notice"); return; }
  const draft = wallDocument();
  inputBusy = true;
  document.querySelector("main").inert = true;
  let data;
  try {
    let response;
    ({response, data} = await reply(fetch("/api/zone-cut", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scheme: draft, wall, boxes})})));
    if (!response.ok) {
      elevationThin = data.thin || [];
      renderElevation();
      note(`沒有切：${data.error || problemLines(data.problems)}${elevationThin.length ? "（紅色虛線是太窄的那幾條）" : ""}`, "notice");
      return;
    }
    if (!scheme.scene.surface_zones_by_wall) {
      scheme.scene.surface_zones_by_wall = draft.scene.surface_zones_by_wall;
      delete scheme.scene.impedance_pa_s_per_m_by_wall;
    }
    scheme.scene.surface_zones_by_wall[wall] = data.zones;
    remapZoneDesign(wall, data.inherited);
    elevationThin = [];
  } finally {
    document.querySelector("main").inert = false;
    inputBusy = false;
  }
  if (done) done(data);
  renderWalls();
  const filled = data.filled_ids.length;
  note(data.zones[0].area === "whole_wall" ? `${wallNames[wall]}回到整面一區`
    : `${wallNames[wall]}切成 ${data.zones.length} 區：畫的 ${data.zones.length - filled} 區、自動補 ${filled} 區`, "ok");
  markStale();
  action(updateMultiples);
  action(updateInputPreview);
  await refreshPlan();
}
function pointerMetres(svg, event, geometry) {
  const point = svg.createSVGPoint();
  point.x = event.clientX; point.y = event.clientY;
  const local = point.matrixTransform(svg.getScreenCTM().inverse());
  const a = (local.x - geometry.pad) / geometry.scale;
  const b = geometry.height - (local.y - geometry.pad) / geometry.scale;
  return [Math.min(Math.max(a, 0), geometry.width), Math.min(Math.max(b, 0), geometry.height)];
}
function startDrawing(svg, event, geometry, view, wall) {
  const start = pointerMetres(svg, event, geometry);
  const preview = node("rect", {class: "zone-preview", x: 0, y: 0, width: 0, height: 0});
  svg.append(preview);
  svg.setPointerCapture(event.pointerId);
  const corners = end => [Math.min(start[0], end[0]), Math.max(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[1], end[1])];
  const move = moved => {
    const [a0, a1, b0, b1] = corners(pointerMetres(svg, moved, geometry));
    preview.setAttribute("x", geometry.pad + a0 * geometry.scale);
    preview.setAttribute("y", geometry.pad + (geometry.height - b1) * geometry.scale);
    preview.setAttribute("width", (a1 - a0) * geometry.scale);
    preview.setAttribute("height", (b1 - b0) * geometry.scale);
  };
  const finish = ended => {
    svg.removeEventListener("pointermove", move);
    svg.removeEventListener("pointerup", finish);
    preview.remove();
    const [a0, a1, b0, b1] = corners(pointerMetres(svg, ended, geometry));
    // 幾乎沒拖：當作點在空白處，取消選取。
    if (Math.max(a1 - a0, b1 - b0) * geometry.scale < 4) { elevationSelected = null; renderElevation(); return; }
    // 格點量的是畫面上離左緣、離下緣的距離（鏡像的牆也一樣），選取後四格數字才是整數格。
    const [g0, g1] = [a0, a1].map(value => gridSnap(value, geometry.width));
    const [h0, h1] = [b0, b1].map(value => gridSnap(value, geometry.height));
    const others = drawnBoxes(wall);
    const box = snapEdges(wallBox(view, geometry.width, geometry.height, g0, g1, h0, h1, newBoxId(wall)),
                          others, geometry.width, geometry.height);
    if (boxTooSmall(box)) {
      note(`框太小：寬與高都要至少 ${cutSpec().min_side_m * 100} 公分（離牆邊或鄰框不到這麼近會自動貼齊）`, "notice");
      return;
    }
    action(() => cutWall(wall, [...others, box], () => { elevationSelected = box.id; }));
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerup", finish);
}
// 選取的框用畫面方向的四個數字改：只改動到的那一格，另外三格沿用原值（顯示只到公釐，不拿顯示的數字回填）。
function editSelected(wall, view, width, height, field, value) {
  const others = drawnBoxes(wall).filter(box => box.id !== elevationSelected);
  const current = drawnBoxes(wall).find(box => box.id === elevationSelected);
  if (!current) return;
  if (!Number.isFinite(value) || value < 0 || (["width", "height"].includes(field) && value <= 0)) {
    note("請填 0 以上的數字（公尺）；寬與高要大於 0", "notice"); return;
  }
  let [a0, a1, b0, b1] = screenBox(view, width, height, current);
  if (field === "left") [a0, a1] = [value, value + (a1 - a0)];
  if (field === "bottom") [b0, b1] = [value, value + (b1 - b0)];
  if (field === "width") a1 = a0 + value;
  if (field === "height") b1 = b0 + value;
  const box = snapEdges(wallBox(view, width, height, a0, a1, b0, b1, current.id), others, width, height);
  if (boxTooSmall(box)) { note(`框太小：寬與高都要至少 ${cutSpec().min_side_m * 100} 公分`, "notice"); return; }
  action(() => cutWall(wall, [...others, box]));
}
function selectedEditor(wall, view, width, height) {
  const area = $("elevation-selected"); area.replaceChildren();
  const current = drawnBoxes(wall).find(box => box.id === elevationSelected);
  if (!current) {
    elevationSelected = null;
    const hint = document.createElement("p"); hint.className = "note";
    hint.textContent = "在牆上空白處（含標「自動」的格子）按住拖曳畫一個框；點畫好的框可以改尺寸或刪掉。";
    area.append(hint); return;
  }
  const [a0, a1, b0, b1] = screenBox(view, width, height, current);
  const heading = document.createElement("p"); heading.id = "elevation-selected-name";
  heading.textContent = `選取：${current.id}（照畫面方向，公尺）`; area.append(heading);
  const fields = document.createElement("div"); fields.className = "fields";
  for (const [field, label, value] of [["left", "離左緣", a0], ["bottom", "離下緣", b0], ["width", "寬", a1 - a0], ["height", "高", b1 - b0]]) {
    const wrap = document.createElement("label"); wrap.textContent = label;
    const input = document.createElement("input"); input.type = "number"; input.step = "0.01"; input.id = `elevation-${field}`;
    input.value = metres(value);
    input.addEventListener("change", () => editSelected(wall, view, width, height, field, Number(input.value)));
    wrap.append(input); fields.append(wrap);
  }
  const remove = document.createElement("button"); remove.id = "elevation-delete"; remove.textContent = "刪掉這個框";
  remove.onclick = () => action(() => cutWall(wall, drawnBoxes(wall).filter(box => box.id !== current.id), () => { elevationSelected = null; }));
  fields.append(remove); area.append(fields);
}
// 喇叭與主位的座標照表單上現在填的（還沒交伺服器也算，同房間尺寸）；沒改過的格子取方案原值，填不出數字就不畫那一點。
function fieldPoint(prefix, id, fallback) {
  return ["x", "y", "z"].map((axis, index) => $(`${prefix}-${id}-${axis}`) ? numericInput(`${prefix}-${id}-${axis}`) : fallback[index]);
}
function markers(svg, wall, view, geometry) {
  const items = [...Object.entries(scheme.speakers).map(([id, point]) => ({key: `speaker-${id}`, name: speakerName(id), point: fieldPoint("speaker", id, [point.x, point.y, point.z]), kind: "speaker"})),
    ...scheme.receiver_set.points.filter(point => point.role === "primary").map(point => ({key: `receiver-${point.receiver_id}`, name: pointName(point.receiver_id), point: fieldPoint("receiver", point.receiver_id, point.position_m), kind: "seat"}))]
    .filter(item => item.point.every(value => typeof value === "number" && Number.isFinite(value)));
  // 兩支喇叭在 y 兩面牆上會投影到同一點（同 x、同高度）：字往下一行一行排，不疊在一起。
  const labels = [];
  for (const item of items) {
    const [u, v] = project(wall, item.point);
    const [a, b] = flip(view, geometry.width, geometry.height, u, v);
    const x = geometry.pad + a * geometry.scale, y = geometry.pad + (geometry.height - b) * geometry.scale;
    let labelY = y - 8;
    while (labels.some(([lx, ly]) => Math.abs(lx - x) < 80 && Math.abs(ly - labelY) < 16)) labelY += 18;
    labels.push([x, labelY]);
    const group = node("g", {id: `elevation-${item.key}`, class: `elevation-marker ${item.kind}`});
    group.append(item.kind === "speaker" ? node("rect", {x: x - 6, y: y - 6, width: 12, height: 12})
      : node("circle", {cx: x, cy: y, r: 7}), node("text", {x: x + 9, y: labelY}, item.name));
    svg.append(group);
  }
}
function renderElevation() {
  const area = $("elevation");
  if (!area || !scheme || !inputDefaults) return;
  const buttons = $("elevation-walls"); buttons.replaceChildren();
  for (const name of Object.keys(wallNames)) {
    const button = document.createElement("button"); button.id = `elevation-wall-${name}`; button.textContent = wallNames[name];
    button.setAttribute("aria-pressed", String(name === elevationWall));
    button.onclick = () => { if (elevationWall !== name) { elevationWall = name; elevationSelected = null; elevationThin = []; note("", ""); } renderElevation(); };
    buttons.append(button);
  }
  area.replaceChildren();
  $("elevation-selected").replaceChildren();
  if (!elevationWall) { $("elevation-size").textContent = "選一面牆打開立面圖。"; renderZoneTable(null); return; }
  const wall = elevationWall, view = views[wall], size = wallSize(wall);
  if (!size) { $("elevation-size").textContent = "先把房間的長、寬、高都填好正數，才畫得出這一面。"; renderZoneTable(wall); return; }
  const [width, height] = size;
  const scale = Math.min(560 / width, 320 / height), pad = 48;
  const geometry = {width, height, scale, pad};
  const svg = node("svg", {id: "elevation-svg", viewBox: `0 0 ${width * scale + 2 * pad} ${height * scale + 2 * pad}`,
                           "data-wall": wall, role: "img", "aria-label": `${wallNames[wall]}立面圖（站在房間裡看）`});
  svg.append(node("rect", {id: "elevation-frame", x: pad, y: pad, width: width * scale, height: height * scale}));
  const zones = wallZones(wall);
  // 整面一區的牆：牆框本身照那一區的做法上色。
  if (!zones.some(zone => zone.area === "rectangle")) svg.lastChild.setAttribute("class", `zone-mode-${zoneSetting(wall, "whole_wall").mode}`);
  const outside = zones.some(zone => zone.area === "rectangle" && (zone.u_m[1] > width + cutSpec().edge_merge_tol_m || zone.v_m[1] > height + cutSpec().edge_merge_tol_m));
  for (const zone of zones.filter(zone => zone.area === "rectangle")) {
    const auto = zone.id.startsWith(cutSpec().fill_prefix);
    const [a0, a1, b0, b1] = screenBox(view, width, height, zone);
    const mode = zoneSetting(wall, zone.id).mode;
    const group = node("g", {class: `zone ${auto ? "zone-auto" : "zone-drawn"} zone-mode-${mode}${zone.id === elevationSelected ? " selected" : ""}`, "data-zone-id": zone.id});
    group.append(node("rect", {x: pad + a0 * scale, y: pad + (height - b1) * scale, width: (a1 - a0) * scale, height: (b1 - b0) * scale}),
      node("title", {}, `${auto ? "自動 " : ""}${zone.id}：${ZONE_MODE_WORDS[mode]}`));
    // 字放得下才寫（窄條裡「自動」兩字會擠出界）；放不下的滑過去看 title。
    const label = auto ? "自動" : zone.id;
    if ((a1 - a0) * scale >= 14 * label.length + 6 && (b1 - b0) * scale >= 18)
      group.append(node("text", {x: pad + (a0 + a1) / 2 * scale, y: pad + (height - (b0 + b1) / 2) * scale, "text-anchor": "middle"}, label));
    svg.append(group);
  }
  for (const strip of elevationThin) {
    const [a0, a1, b0, b1] = screenBox(view, width, height, strip);
    svg.append(node("rect", {class: "zone-thin", x: pad + a0 * scale, y: pad + (height - b1) * scale, width: (a1 - a0) * scale, height: (b1 - b0) * scale}));
  }
  markers(svg, wall, view, geometry);
  const mid = [pad + width * scale / 2, pad + height * scale / 2];
  for (const [edge, x, y, rotate] of [["left", pad - 14, mid[1], -90], ["right", pad + width * scale + 14, mid[1], 90],
                                       ["top", mid[0], pad - 16, 0], ["bottom", mid[0], pad + height * scale + 26, 0]]) {
    svg.append(node("text", {id: `elevation-edge-${edge}`, class: "elevation-edge", x, y, "text-anchor": "middle",
                             transform: `rotate(${rotate} ${x} ${y})`}, wallNames[view[edge]]));
  }
  svg.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    const zone = event.target.closest?.("[data-zone-id]");
    if (zone && zone.classList.contains("zone-drawn")) {
      elevationSelected = zone.dataset.zoneId; renderElevation(); return;
    }
    event.preventDefault();
    startDrawing(svg, event, geometry, view, wall);
  });
  area.append(svg);
  const count = zones.filter(zone => zone.area === "rectangle").length;
  $("elevation-size").textContent = `${wallNames[wall]}：寬 ${metres(width)} 公尺 × 高 ${metres(height)} 公尺，` +
    (count ? `分成 ${count} 區（實線是畫的框、虛線標「自動」是補滿的）` : "整面一區") +
    `；框邊吸附 ${cutSpec().snap_m * 100} 公分格，最小邊 ${cutSpec().min_side_m * 100} 公分` +
    (outside ? "。這一面有分區超出目前的牆面（房間尺寸改過）：請改框或清掉這一面的切法" : "");
  selectedEditor(wall, view, width, height);
  const clear = document.createElement("button"); clear.id = "elevation-clear"; clear.textContent = "清掉這一面的切法";
  clear.disabled = !count;
  clear.onclick = () => action(() => cutWall(wall, [], () => { elevationSelected = null; }));
  $("elevation-selected").append(clear);
  renderZoneTable(wall);
}
window.renderElevation = renderElevation;
window.addEventListener("DOMContentLoaded", () => {
  for (const id of ["room-fields", "speakers", "receivers"]) $(id).addEventListener("input", () => renderElevation());
});
})();
