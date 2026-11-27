/* 立面圖（#1091）：站在房間裡看那一面，畫框、選框、改尺寸、刪框；每改一次交伺服器補滿其餘（/api/zone-cut）。
   吸附格距、最小邊、自動補滿的代號字首由伺服器給（/api/input-defaults 的 zone_cut），這裡不另寫一份數字。
   只換被切的那一面：方案物件原地換這一面的分區，再跟改房間尺寸一樣重問預覽與圖面——
   別面因為房間改小而對不上時，這一面照樣切得動（整份方案交接會因為別面卡住，哪一面都修不了）。 */
(() => {
let elevationWall = null;
let elevationSelected = null;
let elevationThin = [];
let mirrorWarning = null;
let movingText = "";
let showSelectionBanner = false;
let bannerOwner = null;  // 說明條是哪一份方案上講的；換方案或對稱關掉就清（#1274）
let elevationDrawing = false;  // 正在拖框：限制判決回來先不重畫（#1111），放開滑鼠才畫
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
const movePrefix = "搬到：";
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
function roomDimensions() {
  const room = Object.fromEntries(["Lx", "Ly", "Lz"].map(key => [key, $(`room-${key}`) ? numericInput(`room-${key}`) : scheme.scene.room_m[key]]));
  return Object.values(room).every(value => Number.isFinite(value) && value > 0) ? room : null;
}
function wallSize(wall) {
  const room = roomDimensions();
  if (!room) return null;
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
function markRepeatedNote() {
  const noteLine = $("elevation-note"), top = $("elevation-banner");
  noteLine.toggleAttribute("data-repeated", !top.hidden && top.textContent === noteLine.textContent);
}
function note(text, kind) {
  const line = $("elevation-note");
  line.textContent = text; line.className = kind;
  markRepeatedNote();
}
function banner(text = "", kind = "", button = null) {
  const line = $("elevation-banner");
  line.textContent = text;
  line.className = kind;
  line.hidden = !text;
  bannerOwner = text ? scheme : null;
  if (button) line.append(button);
  markRepeatedNote();
}
function repairWords(wall, before, repair) {
  const size = wallSize(wall), view = views[wall];
  const [a0, a1, b0, b1] = screenBox(view, ...size, repair);
  const vertical = repair.v_m.some((v, i) => Math.abs(v - before.v_m[i]) > cutSpec().edge_merge_tol_m);
  const display = value => metres(value).toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 3, useGrouping: false});
  const fields = vertical ? `離下緣 ${display(b0)}、高 ${display(b1 - b0)}` : `離左緣 ${display(a0)}、寬 ${display(a1 - a0)}`;
  return `改成：${fields}（${repair.action}）`;
}
function repairButton(wall, before, repair) {
  const button = document.createElement("button");
  button.type = "button"; button.textContent = "照這樣改";
  button.onclick = () => {
    if (!window.zoneSymmetry.isOn()) { banner(); return; }  // 對稱已經關了：照這樣改只會改一邊，不送
    const {id, u_m, v_m} = repair, after = {id, u_m, v_m};
    const boxes = drawnBoxes(wall).map(b => b.id === id ? after : b);
    action(() => cutWall(wall, boxes, () => { elevationSelected = id; }, {before, after}, true));
  };
  return button;
}
function resultBanner(warning) {
  const {wall, before, repair, summary, message, kind} = warning;
  banner(summary || message, kind, repair ? repairButton(wall, before, repair) : null);
}
// 尺寸確定時一趟重算六面。回覆只換座標，材料和以區代號記的做法繼續用原件。
const resizeProblems = new Map();
let resizeRequest = null;
let resizePending = null;
window.pendingRoomResize = () => resizePending;

async function applyRoomResize(live, request) {
  const changes = await request;
  if (live !== scheme || request !== resizeRequest) return;
  for (const [wall, change] of Object.entries(changes)) {
    if (change.blocked) {
      resizeProblems.set(wall, {...change, generation: zoneDesignGeneration});
      continue;
    }
    resizeProblems.delete(wall);
    if (!change.zones) continue;
    for (const zone of wallZones(wall)) {
      const next = change.zones[zone.id];
      window.zoneOverviewResized(wall, zone, next);
      Object.assign(zone, next);
    }
  }
  // 每一區的新座標由下面重畫立面圖時一起寫進第 2 段右欄「每一區的位置與大小」（#1265 單元 3：牆面表單不再列分區的牆）。
  // 先畫後問（同 applyCut）：上一版判決對的是舊尺寸；走到這裡 request 還是最新那一次，resizeRoom 的 finally 一定接著重問
  // （#1265 單元 3 之二收尾追加第 4 條順手查到的）。
  staleZoneLimits();
  renderElevation();
  window.renderZoneOverview();
}
function resizeRoom() {
  resizeRequest = null;
  resizePending = null;
  const zones = scheme.scene.surface_zones_by_wall;
  const room = roomDimensions();
  const pending = room && zones && Object.values(zones).some(entries => entries[0].area === "rectangle");
  // 第 2 段的清掉／復原那一格與右欄清單（「改這一區 ▸」）也一起鎖（#1265 單元 3 審查設計洞 3：同時點會跟調尺寸改同一份分區）。
  // 舊做法表（elevation-zones）拆了（#1265 單元 3 之二第二趟），不在這張單上。
  const editors = ["walls", "elevation", "elevation-selected", "elevation-clear-slot", "elevation-list"].map($);
  // 框的拖曳起點、數字欄與牆面材料要等新座標，房間尺寸格仍可繼續改。
  for (const editor of editors) editor.inert = Boolean(pending);
  if (!pending) { refreshZoneLimits(); return; }
  const live = scheme;
  const request = api("/api/zone-resize", "POST", {scene: {room_m: room, surface_zones_by_wall: zones}});
  resizeRequest = request;
  resizePending = applyRoomResize(live, request).catch(error => {
    if (request === resizeRequest && live === scheme) note(`分區尺寸沒有調整：${errorText(error)}`, "notice");
  }).finally(() => {
    if (request !== resizeRequest) return;
    resizePending = null;
    for (const editor of editors) editor.inert = false;
    refreshZoneLimits();
    action(updateMultiples);
    action(updateInputPreview);
  });
}

function resizeProblem(wall) {
  const issue = resizeProblems.get(wall);
  if (!issue || issue.generation !== zoneDesignGeneration || !wallZones(wall).some(zone => zone.id === issue.blocked)) return "";
  const where = zoneCalled(wall, issue.blocked);
  const fix = issue.blocked.startsWith(cutSpec().fill_prefix) ? "調整旁邊畫的框" : "把這一區的框改窄、移開或刪掉";
  return `${where}${issue.reason}，這一面的分區保留原位。請先恢復原尺寸，${fix}，再改房間尺寸。`;
}
// 交伺服器切這一面；成功才原地換掉這一面的分區。等回覆時整頁鎖住（同交接），免得半路又改了別格被畫面蓋掉。
let pendingMirror = null;
window.pendingMirrorCut = () => pendingMirror;

// cut：真的切了之後、等圖面檢查之前要同步做的事（清掉的復原快照在這裡記：applyCut 結束時頁面已解鎖，等 /api/plan 的空檔
// 人可以改別格，等檢查回來才記就會把改完的樣子當成清掉後那一刻，檢查丟錯則整個沒記到；#1265 單元 3 審查設計洞 1）。
async function cutWall(wall, boxes, done, change, repairing = false, cut = null) {
  if (inputBusy) { note("上一個改動還在處理，請稍等再切", "notice"); return false; }
  const mirror = change && window.zoneSymmetry.isOn() ? change : null;
  const unlock = window.lockZoneCut();
  const applied = applyCut(wall, boxes, done, mirror, repairing);
  if (mirror && !pendingMirror) pendingMirror = applied;
  let outcome;
  try {
    outcome = await applied;
    if (outcome) cut?.();
  } finally {
    if (pendingMirror === applied) pendingMirror = null;
    unlock(outcome === "ok");
  }
  if (outcome) await refreshPlan();
  return Boolean(outcome);  // 清掉這一面的切法要知道真的切了沒（沒切就不給復原）
}

function cutZoneLabel(surface, wall) {
  const [face, key] = surface.split("@");
  const label = zoneLabels(face).get(key);
  return `${label.numberText}（${face === wall ? "" : wallNames[face] + " "}${label.name}）`;
}

async function applyCut(wall, boxes, done, mirror, repairing) {
  if (measuredDrafts[wall]) { note(`${wallNames[wall]}有還沒套用的實測阻抗草稿：先套用或取消，再切這一面`, "notice"); return; }
  const draft = wallDocument();
  showSelectionBanner = false; mirrorWarning = null; banner();
  inputBusy = true;
  document.querySelector("main").inert = true;
  let data, changedWalls, message = "", kind = "ok", fields = "", repairBefore = null;
  const payload = {scheme: draft, wall, boxes};
  if (mirror) payload.mirror = {...mirror, protected_walls: Object.keys(measuredDrafts)};
  try {
    let response;
    ({response, data} = await reply(fetch("/api/zone-cut", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)})));
    if (!response.ok) {
      elevationThin = data.thin || [];
      renderElevation();
      note(`沒有切：${data.error || problemLines(data.problems)}${elevationThin.length ? "（紅色虛線是太窄的那幾條）" : ""}`, "notice");
      if (mirror) banner($("elevation-note").textContent, "notice");
      return;
    }
    const changes = mirror ? data.walls : {[wall]: data};
    // 底下先改寫分區、renderWalls 先畫、判決後問：上一版判決（配對、上限）對的是切之前，改寫第一格之前就標好（#1265 單元 3 之二
    // 第二趟追加第 1 條；收尾第二趟 D1 從改寫之後挪上來：中間組字那一段萬一丟錯，分區已經換了、判決卻還算新鮮）。
    // 沒切到任何一面（changes 空的）不標：那一路不重問，標了會一直「核對中」。
    if (Object.keys(changes).length) staleZoneLimits();
    if (Object.keys(changes).length && !scheme.scene.surface_zones_by_wall) {
      scheme.scene.surface_zones_by_wall = draft.scene.surface_zones_by_wall;
      delete scheme.scene.impedance_pa_s_per_m_by_wall;
    }
    const named = surface => zoneCalled(...surface.split("@"));
    if (data.deleted_partner) {
      message = `刪掉 ${cutZoneLabel(`${wall}@${mirror.before.id}`, wall)}，${cutZoneLabel(data.deleted_partner, wall)} 因為左右對稱一起刪了`;
    }
    const first = mirror?.before ? zoneCalled(wall, mirror.before.id) : "";
    for (const [face, cut] of Object.entries(changes)) {
      scheme.scene.surface_zones_by_wall[face] = cut.zones;
      remapZoneDesign(face, cut.inherited);
      resizeProblems.delete(face);
    }
    // 修法的「改前」要用套用切牆之後牆上的那一塊（新畫的框這時才在清單裡）。
    const repair = mirror ? data.repairs?.[0] : null;
    if (repair) {
      repairBefore = drawnBoxes(wall).find(b => b.id === mirror.after?.id);
      fields = repairWords(wall, repairBefore, repair);
    }
    if (data.invalid_scheme) {
      kind = "notice";
      message = data.message;
    } else if (data.message) {
      kind = data.source_edit ? "ok" : "notice";
      message = data.message.replaceAll("{first}", mirror.after ? named(data.first) : first);
      if (data.other) {
        const [face, key] = data.other.split("@");
        message = message.replaceAll("{other}", named(data.other))
          .replaceAll("{other_number}", zoneLabels(face).get(key).numberText)
          .replaceAll("{first_number}", zoneLabels(wall).get(mirror.after.id).numberText);
      }
      if (fields) message += `${message.endsWith("。") ? "" : "。"}${fields}。`;
      if (changes[wall]) message += `（${wallNames[wall]}切成 ${changes[wall].zones.length} 區）`;
    } else if (repairing && (data.centred || data.paired)) {
      const firstLabel = cutZoneLabel(wall + "@" + mirror.after.id, wall);
      message = data.centred ? `改好了：${firstLabel} 自己對稱` : `改好了：兩邊都在，${firstLabel}、${cutZoneLabel(data.paired, wall)}`;
    } else if (data.completed) {
      message = `補好了：${cutZoneLabel(data.completed, wall)}`;
    }
    if (mirror && message) {
      const summary = fields ? `${data.cause}。${fields}` : data.reason;
      mirrorWarning = {owner: scheme, generation: zoneDesignGeneration, wall, before: repairBefore, repair, summary,
        message, kind, blocked: data.blocked};
    }
    if (!Object.keys(changes).length) { renderElevation(); return; }
    changedWalls = changes;
    data = changes[wall] ?? Object.values(changes)[0];
    elevationThin = [];
  } finally {
    document.querySelector("main").inert = false;
    inputBusy = false;
  }
  if (done) done(data);
  renderWalls();
  if (message) note(message, kind);
  else note(Object.entries(changedWalls).map(([face, cut]) => {
    const filled = cut.filled_ids.length;
    return cut.zones[0].area === "whole_wall" ? `${wallNames[face]}回到整面一區`
      : `${wallNames[face]}切成 ${cut.zones.length} 區：畫的 ${cut.zones.length - filled} 區、自動補 ${filled} 區`;
  }).join("；"), "ok");
  if (mirrorWarning) resultBanner(mirrorWarning);
  else if (mirror) banner($("elevation-note").textContent, kind);
  markStale();
  action(updateMultiples);
  action(updateInputPreview);
  return mirrorWarning?.summary ? "notice" : kind;
}
function pointerMetres(svg, event, geometry) {
  const point = svg.createSVGPoint();
  point.x = event.clientX; point.y = event.clientY;
  const local = point.matrixTransform(svg.getScreenCTM().inverse());
  const a = (local.x - geometry.pad) / geometry.scale;
  const b = geometry.height - (local.y - geometry.pad) / geometry.scale;
  return [Math.min(Math.max(a, 0), geometry.width), Math.min(Math.max(b, 0), geometry.height)];
}
// 一次拖曳（畫框、搬框共用）：按下就抓住指標、標「正在拖」，放開才走 complete。瀏覽器沒等放開就收回指標（觸控被取消），
// 或拖到一半立面圖被重畫（焦點留在房間尺寸格時按方向鍵），都拆掉監聽、標回沒在拖、整張重畫（預覽框與拖到半路的框回原位），
// 不留一個等下一次點擊才觸發的收尾（#1168 審查：框卡在半路，之後點一下就被搬走；畫框留著預覽框同樣）。
// 收著的限制判決在拖曳結束（不管走哪條路）之後才畫：收尾要用的這張圖在那之前不能換掉。
let dropDrag = null;
function trackDrag(svg, event, move, complete, cancelled = () => {}) {
  const line = $("elevation-note"), previous = [line.textContent, line.className];
  movingText = ""; showSelectionBanner = false;
  mirrorWarning = null; banner();
  let live = true;
  const stop = () => {
    if (!live) return false;
    live = false; dropDrag = null; movingText = ""; banner();
    if (!line.textContent || line.textContent.startsWith("放開後") || line.textContent.startsWith("鏡像預覽")) note(...previous);
    svg.removeEventListener("pointermove", move);
    svg.removeEventListener("pointerup", finish);
    svg.removeEventListener("lostpointercapture", lost);
    return true;
  };
  const endDrawing = () => { if (!elevationDrawing) return; elevationDrawing = false; zoneLimitsDragEnded(); };
  const finish = ended => {
    if (!stop()) return;
    try { complete(ended); } finally { endDrawing(); }
  };
  const lost = () => { if (stop()) { cancelled(); endDrawing(); renderElevation(); } };
  dropDrag = () => { if (stop()) { cancelled(); endDrawing(); } };  // 重畫的那一支自己會畫
  svg.setPointerCapture(event.pointerId);
  elevationDrawing = true;
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerup", finish);
  svg.addEventListener("lostpointercapture", lost);
}
function symmetryResult() {
  return window.zoneSymmetry.isOn() && zoneLimitsFresh() ? zoneLimits.symmetry : null;
}
function mirrorBox(wall, box) {
  const result = symmetryResult();
  if (!result?.axis) return null;
  const [target, flip] = result.projections[wall];
  if (!flip) return result.axis.centred ? {wall: target, box} : null;
  const edges = flip + "_m", twice = 2 * result.axis.across_m;
  return {wall: target, box: {...box, [edges]: [twice - box[edges][1], twice - box[edges][0]]}};
}
function selfMirrored(wall, box) {
  const mirror = mirrorBox(wall, box);
  return mirror?.wall === wall && ["u_m", "v_m"].every(k => box[k].every((x, i) =>
    Math.abs(x - mirror.box[k][i]) <= symmetryResult().pair_within_m + 1e-9));
}
function followsEdit(wall, id) {
  const result = symmetryResult(), key = `${wall}@${id}`;
  const offsetSide = result?.axis && !result.axis.centred && result.projections[wall][0] !== wall;
  return offsetSide || result?.partners?.[key] || result?.placement_refusals?.[key];
}
function svgBox(view, g, box) {
  const [a0, a1, b0, b1] = screenBox(view, g.width, g.height, box);
  return {x: g.pad + a0 * g.scale, y: g.pad + (g.height - b1) * g.scale, width: (a1 - a0) * g.scale, height: (b1 - b0) * g.scale};
}
// 搬框時提示列已經寫著「搬到：…（放開才切）」：預覽的判決接在後面，不蓋掉它。
function previewNote(text, kind, summary = text) {
  note(movingText ? `${movingText}；${text}` : text, kind);
  banner(summary, kind);
}
function drawMirrorPreview(svg, wall, g, mirror, reason) {
  svg.querySelector(".mirror-preview")?.remove();
  if (!mirror || mirror.wall !== wall) return;
  const bounds = svgBox(views[wall], g, mirror.box);
  const group = node("g", {class: "mirror-preview"});
  group.append(node("rect", bounds));
  svg.append(group);
  if (reason === null) { group.classList.add("pending"); return; }
  if (reason) group.classList.add("refused");
  if (!reason && (bounds.width <= 40 || bounds.height <= 34)) return;
  // 完全超出圖外時，標字仍留在看得到的那一側。
  const x = Math.min(Math.max(bounds.x + bounds.width / 2, 24), svg.viewBox.baseVal.width - 24);
  const y = Math.min(Math.max(reason ? bounds.y + bounds.height - 6 : bounds.y + 18, 16), svg.viewBox.baseVal.height - 5);
  const lines = reason ? ["放不下"] : ["放開後", "這裡跟著出現"];
  lines.forEach((line, i) => group.append(node("text", {x, y: y + i * 15, "text-anchor": "middle"}, line)));
  const axis = svg.querySelector(".mirror-axis text");
  if (reason && axis) {
    const label = group.querySelector("text"), a = axis.getBBox(), b = label.getBBox();
    if (b.x < a.x + a.width && a.x < b.x + b.width && b.y < a.y + a.height && a.y < b.y + b.height) {
      label.setAttribute("y", a.y + a.height + b.height + 2);
    }
  }
}
// 每次拖曳只取一次草稿，一次最多一個請求。等待時的新落點取代舊的；放開、取消或換圖後，舊回覆失效。
function checkPlacement(scheme, wall, boxes, before, after, complete = false) {
  const mirror = {before, after, complete, protected_walls: Object.keys(measuredDrafts)};
  return api("/api/zone-mirror-check", "POST", {scheme, wall, boxes, mirror});
}
function mirrorPreview(svg, wall, g, before = null) {
  let draft, others, latest, checking = false;
  const live = position => position === latest && dropDrag === position.drag && svg.isConnected;
  async function check() {
    checking = true;
    try {
      while (latest && live(latest)) {
        const position = latest;
        const {box, mirror} = position;
        try {
          draft ??= wallDocument();
          others ??= drawnBoxes(wall).filter(peer => peer.id !== box.id);
          const data = await checkPlacement(draft, wall, [...others, box], before, box);
          if (!live(position)) continue;
          let reason = data.other && data.reason ? data.reason.replaceAll("{other}", zoneCalled(...data.other.split("@"))) : data.reason;
          // 伺服器有幾句原因是接在「{first}」後面寫的（「的對面還有實測阻抗草稿」）；預覽沒有那一段，補成「這一塊」。
          if (reason?.startsWith("的")) reason = `這一塊${reason}`;
          drawMirrorPreview(svg, wall, g, data.source_refused ? null : mirror, reason ?? "");
          if (reason) {
            const repair = data.repairs?.[0];
            const shown = repair ? `${data.cause}。${repairWords(wall, box, repair)}` : reason;
            const full = data.source_refused ? `放開後不會切：${shown}` : mirror ? `放開後另一邊不會跟著畫：${shown}` : `放開後，${shown}`;
            position.note = [full, "notice", repair ? `放開後：${shown}` : (data.cause && !mirror ? data.cause : full)];
          } else if (mirror && mirror.wall !== wall) position.note = [`放開後，${wallNames[mirror.wall]}同一個位置跟著出現`, ""];
          else position.note = ["放開後，另一邊跟著出現", ""];
        } catch (error) {
          if (!live(position)) continue;
          drawMirrorPreview(svg, wall, g, mirror, errorText(error));
          position.note = [`鏡像預覽無法確認：${errorText(error)}`, "notice"];
        }
        if (position.note) previewNote(...position.note);
        break;
      }
    } finally { checking = false; }
  }
  return box => {
    if (!symmetryResult()?.axis) return;
    const key = JSON.stringify(box);
    if (latest?.key === key) {
      if (latest.note) previewNote(...latest.note);
      return;
    }
    const line = $("elevation-note");
    if (line.textContent.startsWith("放開後") || line.textContent.startsWith("鏡像預覽")) note("", "");
    banner();
    const checkBox = !selfMirrored(wall, box) && !boxTooSmall(box);
    const mirror = checkBox ? mirrorBox(wall, box) : null;
    drawMirrorPreview(svg, wall, g, mirror, null);
    latest = checkBox ? {key, box, mirror, drag: dropDrag} : null;
    if (latest && !checking) void check();
  };
}
function showSymmetry(svg, wall, g, layout) {
  const result = symmetryResult();
  if (!result?.axis) return;
  const {width, height, scale, pad} = g;
  const [target, flipAxis] = result.projections[wall];
  if (target !== wall) return;
  const vertical = flipAxis === "u";
  const [a, b] = flip(views[wall], width, height, result.axis.across_m, result.axis.across_m);
  const x = pad + a * scale, y = pad + (height - b) * scale;
  const axis = node("g", {class: "mirror-axis"});
  axis.append(node("line", vertical ? {x1: x, x2: x, y1: pad, y2: pad + height * scale}
    : {x1: pad, x2: pad + width * scale, y1: y, y2: y}),
    node("text", {x: vertical ? x + 5 : pad + 5, y: vertical ? pad + 15 : y - 5}, MIRROR_LINE_WORDS));
  svg.append(axis);
  const drawn = new Map(drawnBoxes(wall).map(box => [wall + "@" + box.id, svgBox(views[wall], g, box)]));
  for (const [key, first] of drawn) {
    const partner = result.partners[key], other = drawn.get(partner);
    if (!other || key >= partner) continue;
    const [lo, hi] = [first, other].sort((l, r) => vertical ? l.x - r.x : l.y - r.y);
    const x1 = vertical ? lo.x + lo.width : lo.x + lo.width / 2, y1 = vertical ? lo.y + lo.height / 3 : lo.y + lo.height;
    const x2 = vertical ? hi.x : hi.x + hi.width / 2, y2 = vertical ? hi.y + hi.height / 3 : hi.y;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const labelX = mx + (vertical ? 0 : 9), anchor = vertical ? "middle" : "start";
    const obstacles = layout.flatMap(marker => [marker.mark, marker.label]);
    for (const badge of svg.querySelectorAll(".zone-number circle")) {
      obstacles.push(badgeBox(Number(badge.getAttribute("cx")), Number(badge.getAttribute("cy"))));
    }
    const labelY = [my - 12, my + 24, my - 32, my + 44, my - 52].find(at =>
      at >= pad + LABEL_PX && at <= pad + height * scale &&
      !obstacles.some(obstacle => boxesOverlap(textBox(labelX, at, "左右對稱", anchor), obstacle))) ?? my - 12;
    const bend = vertical ? 2 * (labelY + 6 - my) : 0;
    const group = node("g", {class: "mirror-link"});
    group.append(node("path", {d: `M${x1} ${y1} Q ${mx + (vertical ? 0 : 15)} ${my + bend} ${x2} ${y2}`}),
      node("text", {x: labelX, y: labelY, "text-anchor": anchor}, "左右對稱"));
    svg.append(group);
  }
}
function startDrawing(svg, event, geometry, view, wall) {
  const start = pointerMetres(svg, event, geometry);
  const preview = node("rect", {class: "zone-preview", x: 0, y: 0, width: 0, height: 0});
  // 拖的時候框旁的小字（#1265 單元 3）：吸附幾公分格、最小邊多少，跟著框的左上角走。
  const hint = node("text", {class: "zone-preview-hint", x: 0, y: 0},
    `框邊吸附 ${cutSpec().snap_m * 100} 公分格，最小邊 ${cutSpec().min_side_m * 100} 公分`);
  svg.append(preview, hint);
  const corners = end => [Math.min(start[0], end[0]), Math.max(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[1], end[1])];
  const others = drawnBoxes(wall);
  const previewMirror = mirrorPreview(svg, wall, geometry);
  const snapped = (a0, a1, b0, b1) => {
    // 格點量的是畫面上離左緣、離下緣的距離（鏡像的牆也一樣），選取後四格數字才是整數格。
    const [g0, g1] = [a0, a1].map(value => gridSnap(value, geometry.width));
    const [h0, h1] = [b0, b1].map(value => gridSnap(value, geometry.height));
    return snapEdges(wallBox(view, geometry.width, geometry.height, g0, g1, h0, h1, newBoxId(wall)),
                     others, geometry.width, geometry.height);
  };
  const move = moved => {
    // 畫的就是吸過格點與邊、放開後要交出去的那一塊（#1274：不然鏡像預覽照吸過的框算，跟眼睛看到的對不上）。
    const box = snapped(...corners(pointerMetres(svg, moved, geometry)));
    const bounds = svgBox(view, geometry, box);
    for (const [key, value] of Object.entries(bounds)) preview.setAttribute(key, value);
    hint.setAttribute("x", String(bounds.x));
    hint.setAttribute("y", String(Math.max(bounds.y - 6, 14)));
    previewMirror(box);
  };
  const complete = ended => {
    preview.remove(); hint.remove();
    svg.querySelector(".mirror-preview")?.remove();
    const [a0, a1, b0, b1] = corners(pointerMetres(svg, ended, geometry));
    // 幾乎沒拖：當作點在空白處，取消選取。
    if (Math.max(a1 - a0, b1 - b0) * geometry.scale < 4) { elevationSelected = null; renderElevation(); return; }
    const box = snapped(a0, a1, b0, b1);
    if (boxTooSmall(box)) {
      note(`框太小：寬與高都要至少 ${cutSpec().min_side_m * 100} 公分（離牆邊或鄰框不到這麼近會自動貼齊）`, "notice");
      return;
    }
    action(() => cutWall(wall, [...others, box], () => { elevationSelected = box.id; }, {before: null, after: box}));
  };
  trackDrag(svg, event, move, complete);
}
// 整塊平移要挪多少才貼齊（#1168）：有一邊已經貼著牆邊或別框的邊就不挪；不然挑離最近的那一對，
// 差不到「最小邊扣容差」才貼（同 snapEdges 那一把）。只平移、不改寬高。
function moveShift(edges, targets) {
  const tolerance = cutSpec().edge_merge_tol_m;
  if (edges.some(edge => targets.some(target => Math.abs(edge - target) <= tolerance))) return 0;
  let best = 0, gap = cutSpec().min_side_m - tolerance;
  for (const edge of edges) for (const target of targets) {
    if (Math.abs(target - edge) < gap) { best = target - edge; gap = Math.abs(target - edge); }
  }
  return best;
}
// 按住畫好的框拖著搬（#1168）：寬高不變；左下角吸到畫面格點、夾在牆內，再整塊貼齊最近的牆邊或鄰框。
// 拖的時候框跟著滑鼠走（看到的就是放開後要交出去的位置），放開才交伺服器切；幾乎沒拖就當點選。
// 疊到別框、留下窄條照樣交伺服器判，不切就畫回原位。代號不變，材料與做法照代號沿用。
function startMoving(svg, event, geometry, view, wall, group) {
  const id = group.dataset.zoneId;
  const current = drawnBoxes(wall).find(box => box.id === id);
  const others = drawnBoxes(wall).filter(box => box.id !== id);
  if (!current) return;
  const previewMirror = mirrorPreview(svg, wall, geometry, current);
  const {width, height, scale} = geometry;
  const [a0, a1, b0, b1] = screenBox(view, width, height, current);
  const start = pointerMetres(svg, event, geometry);
  group.classList.add("moving");
  // 提到其餘分區上面（不然拖過去被後畫的「自動」格蓋住，只看得到原位的空洞），喇叭與座位記號照舊在最上層。
  svg.insertBefore(group, svg.querySelector(".elevation-marker"));
  const line = $("elevation-note"), before = [line.textContent, line.className];  // 只是點選時放回原本的提示
  const landing = point => {
    const left = Math.min(gridSnap(a0 + point[0] - start[0], width), width - (a1 - a0));
    const bottom = Math.min(gridSnap(b0 + point[1] - start[1], height), height - (b1 - b0));
    const box = wallBox(view, width, height, left, left + (a1 - a0), bottom, bottom + (b1 - b0), id);
    const clamp = (shift, pair, limit) => Math.min(Math.max(shift, -pair[0]), limit - pair[1]);
    const du = clamp(moveShift(box.u_m, [0, width, ...others.flatMap(other => other.u_m)]), box.u_m, width);
    const dv = clamp(moveShift(box.v_m, [0, height, ...others.flatMap(other => other.v_m)]), box.v_m, height);
    return {id, u_m: box.u_m.map(value => value + du), v_m: box.v_m.map(value => value + dv)};
  };
  const move = moved => {
    const box = landing(pointerMetres(svg, moved, geometry));
    const [c0, , d0] = screenBox(view, width, height, box);
    group.setAttribute("transform", `translate(${(c0 - a0) * scale} ${(b0 - d0) * scale})`);
    movingText = `${movePrefix}離左緣 ${metres(c0)}、離下緣 ${metres(d0)} 公尺（放開才切）`;
    note(movingText, "");
    if (followsEdit(wall, id)) previewMirror(box);
  };
  const complete = ended => {
    elevationSelected = id;
    const box = landing(pointerMetres(svg, ended, geometry));
    const same = [...box.u_m, ...box.v_m].every((value, index) => Math.abs(value - [...current.u_m, ...current.v_m][index]) < 1e-9);
    const dragged = Math.hypot(ended.clientX - event.clientX, ended.clientY - event.clientY) >= 4;
    if (!dragged || same) {
      showSelectionBanner = !dragged;
      if (dragged) note("框沒有搬動（放開的位置吸回原處）", "notice"); else note(...before);
      renderElevation(); return;
    }
    // 切成功或被拒都會整張重畫；沒送出（上一個改動還在處理）或連線失敗時，這裡把拖走的框畫回原位，
    // 提示列還停在「放開才切」的話改說沒有切（沒送出那兩種 cutWall 自己寫了原因，不蓋掉）。
    action(async () => {
      try { await cutWall(wall, [...others, box], null, {before: current, after: box}); } finally {
        if (group.isConnected) {
          if ($("elevation-note").textContent.startsWith(movePrefix)) note("沒有切：框放回原位（連線失敗）", "notice");
          renderElevation();
        }
      }
    });
  };
  trackDrag(svg, event, move, complete, () => note(...before));
}
// 選中後拉右上角；與打數字共用吸邊，放開前只換虛線預覽。
function resizeHandle(svg, wall, box, g) {
  const view = views[wall], bounds = svgBox(view, g, box);
  const handle = node("rect", {class: "mirror-resize", x: bounds.x + bounds.width - 5, y: bounds.y - 5, width: 10, height: 10});
  handle.append(node("title", {}, "拖曳改框的大小"));
  handle.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const [left, , bottom] = screenBox(view, g.width, g.height, box);
    const others = drawnBoxes(wall).filter(other => other.id !== box.id);
    const previewMirror = mirrorPreview(svg, wall, g, box);
    const landing = e => {
      const [a, b] = pointerMetres(svg, e, g);
      return snapEdges(wallBox(view, g.width, g.height, left, Math.max(left, gridSnap(a, g.width)), bottom,
        Math.max(bottom, gridSnap(b, g.height)), box.id), others, g.width, g.height);
    };
    const preview = node("rect", {class: "zone-preview", ...bounds});
    svg.append(preview);
    trackDrag(svg, event, moved => {
      const next = landing(moved);
      for (const [key, value] of Object.entries(svgBox(view, g, next))) preview.setAttribute(key, value);
      if (followsEdit(wall, box.id)) previewMirror(next);
    }, ended => {
      const next = landing(ended);
      preview.remove(); svg.querySelector(".mirror-preview")?.remove();
      if (boxTooSmall(next)) { note(`寬與高都要至少 ${cutSpec().min_side_m * 100} 公分`, "notice"); return; }
      action(() => cutWall(wall, [...others, next], null, {before: box, after: next}));
    });
  });
  svg.append(handle);
}
// 選取的框用畫面方向的四個數字改：只改動到的那一格，另外三格沿用原值（顯示只到公釐，不拿顯示的數字回填）。
function editSelected(wall, view, width, height, field, input) {
  const value = input.valueAsNumber;
  const others = drawnBoxes(wall).filter(box => box.id !== elevationSelected);
  const current = drawnBoxes(wall).find(box => box.id === elevationSelected);
  if (!current) return;
  if (!Number.isFinite(value) || value < 0 || (["width", "height"].includes(field) && value <= 0)) {
    input.setAttribute("aria-invalid", "true");
    note("請填 0 以上的數字（公尺）；寬與高要大於 0", "notice"); return;
  }
  let [a0, a1, b0, b1] = screenBox(view, width, height, current);
  if (field === "left") [a0, a1] = [value, value + (a1 - a0)];
  if (field === "bottom") [b0, b1] = [value, value + (b1 - b0)];
  if (field === "width") a1 = a0 + value;
  if (field === "height") b1 = b0 + value;
  const box = snapEdges(wallBox(view, width, height, a0, a1, b0, b1, current.id), others, width, height);
  if (boxTooSmall(box)) { note(`框太小：寬與高都要至少 ${cutSpec().min_side_m * 100} 公分`, "notice"); return; }
  action(() => cutWall(wall, [...others, box], null, {before: current, after: box}));
}
function axisDistances(axis) {
  const width = wallSize(axis.front)?.[0];
  if (!width) return null;
  const distance = axis.left.endsWith("0") ? axis.across_m : width - axis.across_m;
  return {left: distance.toFixed(2), centre: (width / 2).toFixed(2)};
}
// 這一區跟哪一區左右對稱（畫框時的幾何鏡像，限制判決順便回的 partners）：選取框底下那一句；沒有回空字串。第 2 段右欄
// 清單講「跟誰一對」改用「這間房貼什麼」那一頁同一份配對（wall_section.js，#1265 單元 3 審查真錯 5）。
function pairWords(wall, key) {
  const partner = symmetryResult()?.partners?.[`${wall}@${key}`]?.split("@");
  return partner ? `和${zoneCalled(...partner)}左右對稱` : "";
}
// 開關底下那一句與立面圖上那條線的字（#1265 單元 3 之二第 A 條，示意圖 input-page-v3 第 2 段）：講成鏡子；
// 在不在房間正中照伺服器的判斷（axis.centred），不另算。只改輸入頁，「這間房貼什麼」那一頁的「左右對稱」字樣照舊。
// 輸入頁上自己寫的幾句（還在等、選取框裡那兩句）也一起叫鏡子（第二趟決定第 6 條）；伺服器回來的原因字（跨在對稱軸上…、
// 定不出對稱軸…）照原樣接上，不在這裡改字（下面 startsWith 那一行比的就是伺服器的字）。
const MIRROR_LINE_WORDS = "鏡子：兩支喇叭的中線";
function inputAxisWords(result) {
  // 沒有判決可看（symmetryResult 只給新鮮的）：最新那一次判決回來了、可是沒做成（連不上、房間尺寸讀不出來），沒有鏡子可講，
  // 照實講沒確認成，不寫成還在等（#1265 單元 3 之二收尾追加設計洞）；判決在路上才是「正在確認」。
  if (!result) return zoneLimitsFresh() && zoneLimits.status !== "checked" ? `鏡子在哪沒確認成：${zoneLimits.message}` : "正在確認鏡子在哪…";
  const axis = result.axis;
  if (!axis) return result.reason || result.message || "正在確認鏡子在哪…";
  // 鏡子在哪已經知道，可是房間尺寸格現在讀不出來（例如高度清空、還沒離開那一格）：算不出離左牆多遠。這時沒有在問，
  // 不寫「正在確認」，講尺寸沒填好（#1265 單元 3 之二收尾第二趟 B4 之二；用語同上面那一行「先把房間的長、寬、高都填好正數」）。
  const distances = axisDistances(axis);
  if (!distances) return "先把房間的長、寬、高都填好正數，才算得出鏡子在哪";
  return `鏡子：喇叭中線，離左牆 ${distances.left} 公尺` + (axis.centred ? "" : "；不在正中，左右牆不跟著畫");
}
// 開關底下那一句（renderElevation 畫；限制檢查失敗之後又問一次時 zone_limits.js 只重寫這一句，見 window.refreshAxisWords）。
function refreshAxisControl() {
  const on = window.zoneSymmetry.isOn(), result = symmetryResult();
  const sideOnly = elevationWall && result?.axis && !result.axis.centred && result.projections[elevationWall][0] !== elevationWall;
  const title = $("elevation-panel").closest("#zone-page") ? "左右對稱" : "兩邊一起畫";
  $("elevation-symmetry-title").textContent = sideOnly ? `${title}（只切${wallNames[elevationWall]}）` : title;
  $("elevation-axis-words").textContent = on ? inputAxisWords(result) : "";
  $("elevation-centre-axis").hidden = !sideOnly;
}
function selectedEditor(wall, view, width, height) {
  const area = $("elevation-selected"); area.replaceChildren();
  const current = drawnBoxes(wall).find(box => box.id === elevationSelected);
  if (!current) {
    elevationSelected = null;  // 沒選框時不擺編輯格；第 2 段標題旁只寫「在牆上拖框切一區」（#1265 說明字瘦身）
    return;
  }
  const [a0, a1, b0, b1] = screenBox(view, width, height, current);
  const heading = document.createElement("p"); heading.id = "elevation-selected-name";
  const label = zoneLabels(wall).get(current.id);
  heading.textContent = `選取：${label.numberText} ${label.name}（代號 ${current.id}；照畫面方向，公尺）`; area.append(heading);
  const partner = symmetryResult()?.partners?.[`${wall}@${current.id}`]?.split("@");
  const fields = document.createElement("div"); fields.className = "fields";
  for (const [field, label, value] of [["left", "離左緣", a0], ["bottom", "離下緣", b0], ["width", "寬", a1 - a0], ["height", "高", b1 - b0]]) {
    const wrap = document.createElement("label"); wrap.textContent = label;
    const input = document.createElement("input"); input.type = "number"; input.step = "0.01"; input.id = `elevation-${field}`;
    input.defaultValue = metres(value);  // 畫出來的值：跟它不同才算正在打字（限制判決先不重畫，#1111）
    input.addEventListener("input", () => input.removeAttribute("aria-invalid"));
    input.addEventListener("change", () => editSelected(wall, view, width, height, field, input));
    wrap.append(input); fields.append(wrap);
  }
  const remove = document.createElement("button"); remove.id = "elevation-delete"; remove.textContent = "刪掉這個框";
  remove.onclick = () => action(() => cutWall(wall, drawnBoxes(wall).filter(box => box.id !== current.id), () => { elevationSelected = null; }, {before: current, after: null}));
  fields.append(remove); area.append(fields);
  if (window.zoneSymmetry.isOn()) {
    const status = document.createElement("div"); status.className = "elevation-pair-status";
    area.append(status);
    const result = symmetryResult();
    if (partner) status.textContent = pairWords(wall, current.id);
    if (!result || partner) return;
    if (selfMirrored(wall, current)) { status.textContent = "這一塊跨在鏡子正中，自己對稱"; return; }
    if (!result.available) {
      status.textContent = result.message;
    } else if (!result.axis) {
      status.textContent = `${inputAxisWords(result)}。請先檢查方案，讓主位對齊兩支喇叭的中點。`;
    } else if (result.projections[wall][0] !== wall && !result.axis.centred) {
      const distances = axisDistances(result.axis);
      if (distances) status.textContent = `鏡子離左牆 ${distances.left} 公尺，房間正中是 ${distances.centre} 公尺：主位和兩支喇叭的中點都要在正中，左右牆才會跟著畫`;
    } else if (result.placement_refusals?.[`${wall}@${current.id}`]) {
      const reason = result.placement_refusals[`${wall}@${current.id}`];
      const prefix = reason.startsWith("跨在對稱軸") ? "這一塊沒有另一邊：" : "這一塊翻過去放不下：";
      status.textContent = prefix + reason;
      showSelectedRepair(status, wall, current, prefix);
    } else {
      const complete = document.createElement("button");
      complete.id = "elevation-complete"; complete.textContent = "補畫另一邊";
      complete.onclick = () => action(() => cutWall(wall, drawnBoxes(wall), null, {before: current, after: current, complete: true}));
      status.append("這一塊沒有另一邊", complete);
    }
  }
}
async function showSelectedRepair(status, wall, current, prefix) {
  try {
    const data = await checkPlacement(wallDocument(), wall, drawnBoxes(wall), current, current, true);
    if (!status.isConnected) return;
    const repair = data.repairs?.[0];
    const fields = repair && repairWords(wall, current, repair);
    status.textContent = prefix + (repair ? data.cause : data.reason);
    if (repair) status.append("。" + fields, repairButton(wall, current, repair));
    if (showSelectionBanner && !mirrorWarning && !elevationDrawing) {
      const summary = fields ? `${data.cause}。${fields}` : data.reason;
      banner(summary, "notice", repair ? repairButton(wall, current, repair) : null);
      note(prefix + (fields ? `${data.cause}。${fields}` : data.reason), "notice");
    }
  } catch (error) {
    if (status.isConnected) status.append(`。修法無法確認：${errorText(error)}`);
  }
}
// 喇叭與主位的座標照表單上現在填的（還沒交伺服器也算，同房間尺寸）；沒改過的格子取方案原值，填不出數字就不畫那一點。
function fieldPoint(prefix, id, fallback) {
  return ["x", "y", "z"].map((axis, index) => $(`${prefix}-${id}-${axis}`) ? numericInput(`${prefix}-${id}-${axis}`) : fallback[index]);
}
// 喇叭與主位（照表單上現在填的；填不出數字的那一點不算）：立面圖畫記號、區名（zone_names.js）判蓋住誰，都用這一份。
function markerItems() {
  return [...Object.entries(scheme.speakers).map(([id, point]) => ({key: `speaker-${id}`, name: speakerName(id), point: fieldPoint("speaker", id, [point.x, point.y, point.z]), kind: "speaker"})),
    ...scheme.receiver_set.points.filter(point => point.role === "primary").map(point => ({key: `receiver-${point.receiver_id}`, name: pointName(point.receiver_id), point: fieldPoint("receiver", point.receiver_id, point.position_m), kind: "seat"}))]
    .filter(item => item.point.every(value => typeof value === "number" && Number.isFinite(value)));
}
// 牆名定得出前後左右時（#1134）牆的角色；地板天花沒有。只認伺服器排出來的這四個字，座標名一律回 null（不猜）。
const WALL_ROLES = {"前牆": "front", "後牆": "back", "左牆": "side", "右牆": "side"};
// 這一面每一區的號碼與位置名（#1176）：立面圖、第 2 段右欄與「這間房貼什麼」那一頁共用。畫面座標（鏡像翻好）、四邊隔壁那一面的名字、喇叭與主位的投影
// 交給 zone_names.js 的 nameZones 算；房間尺寸填不出數字時照列表順序編號、名字退回代號（畫不出這一面，也就看不出位置）。
function zoneLabels(wall) {
  const zones = wallZones(wall), size = wallSize(wall);
  const fill = cutSpec().fill_prefix;
  if (!size) {
    return new Map(zones.map((zone, index) => [zone.area === "rectangle" ? zone.id : "whole_wall", {number: index + 1,
      numberText: zoneNumberText(index + 1), name: zone.area !== "rectangle" ? "整面" : zone.id.startsWith(fill) ? `自動 ${zone.id}` : zone.id}]));
  }
  const view = views[wall], [width, height] = size;
  const surface = wall === "floor" || wall === "ceiling" ? wall : "wall";
  const edges = Object.fromEntries(["left", "right", "top", "bottom"].map(edge => [edge, wallNames[view[edge]]]));
  const edgeOf = word => Object.keys(edges).find(edge => edges[edge] === word) ?? null;
  const items = markerItems().map(item => ({...item, at: flip(view, width, height, ...project(wall, item.point))}));
  return nameZones({
    surface, role: surface === "wall" ? WALL_ROLES[wallNames[wall]] ?? null : null, width, height,
    tolerance: cutSpec().edge_merge_tol_m, edges, front: edgeOf("前牆"), left: edgeOf("左牆"),
    zones: zones.map(zone => zone.area === "rectangle"
      ? {key: zone.id, drawn: !zone.id.startsWith(fill), whole: false, box: screenBox(view, width, height, zone), entry: zoneLibraryMatch(zone.material, wall, zone.id)}
      : {key: "whole_wall", drawn: false, whole: true, box: [0, width, 0, height], entry: zoneLibraryMatch(zone.material ?? wallMaterial(wall), wall, "whole_wall")}),
    speakers: items.filter(item => item.kind === "speaker").map(item => ({name: item.name, at: item.at})),
    primary: items.find(item => item.kind === "seat")?.at ?? null,
  });
}
window.zoneLabels = zoneLabels;
// 喇叭與主位先排好位置再畫（#1119）：字的外框用估的寬（14px 字，一字約一個字高），一律留在牆框裡——
// 點右邊放不下就放到點的左邊，免得壓到牆邊直排的牆名；兩個點投影到同一處（y 兩面牆上的兩支喇叭）字往下一行一行排。
// 區名（renderElevation）再避開這些點與字。
const LABEL_PX = 14;
function textBox(x, y, text, anchor) {
  const width = LABEL_PX * text.length;
  const left = anchor === "end" ? x - width : anchor === "middle" ? x - width / 2 : x;
  return {left, right: left + width, top: y - LABEL_PX, bottom: y + 4};
}
function boxesOverlap(a, b) {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}
function markerLayout(wall, view, geometry) {
  const items = markerItems();
  const frameLeft = geometry.pad, frameRight = geometry.pad + geometry.width * geometry.scale;
  // 先放好所有記號，字再避開全部記號與先排好的字（後畫的記號不會蓋到先排的字）。
  const points = items.map(item => {
    const [u, v] = project(wall, item.point);
    const [a, b] = flip(view, geometry.width, geometry.height, u, v);
    const x = geometry.pad + a * geometry.scale, y = geometry.pad + (geometry.height - b) * geometry.scale;
    return {item, x, y, mark: {left: x - 8, right: x + 8, top: y - 8, bottom: y + 8}};
  });
  const placed = [];
  for (const {item, x, y, mark} of points) {
    const width = LABEL_PX * item.name.length;
    const toLeft = x + 9 + width > frameRight && x - 9 - width >= frameLeft;
    const anchor = toLeft ? "end" : "start", labelX = toLeft ? x - 9 : x + 9;
    const blocked = labelY => {
      const box = textBox(labelX, labelY, item.name, anchor);
      return points.some(other => boxesOverlap(box, other.mark)) || placed.some(other => boxesOverlap(box, other.label));
    };
    let labelY = y - 8;
    for (let tries = 0; tries < 12 && blocked(labelY); tries += 1) labelY += 18;
    placed.push({item, x, y, labelX, labelY, anchor, mark, label: textBox(labelX, labelY, item.name, anchor)});
  }
  return placed;
}
function drawMarkers(svg, layout) {
  for (const {item, x, y, labelX, labelY, anchor} of layout) {
    const group = node("g", {id: `elevation-${item.key}`, class: `elevation-marker ${item.kind}`});
    group.append(item.kind === "speaker" ? node("rect", {x: x - 6, y: y - 6, width: 12, height: 12})
      : node("circle", {cx: x, cy: y, r: 7}), node("text", {x: labelX, y: labelY, "text-anchor": anchor}, item.name));
    svg.append(group);
  }
}
// 區號碼（#1176）畫成一個圓：圓心在 (x, y)，半徑 BADGE_PX。
const BADGE_PX = 11;
function badgeBox(x, y) { return {left: x - BADGE_PX, right: x + BADGE_PX, top: y - BADGE_PX, bottom: y + BADGE_PX}; }
// 區號碼的高度：先試正中，跟點或點的字疊到就往上下各挪一行試，一律留在這一區框內；都不行就回 null 不畫
// （同「窄條放不下就不寫」，滑過去看 title 照樣有）。
function zoneLabelY(centerX, centerY, top, bottom, layout) {
  const obstacles = layout.flatMap(placed => [placed.mark, placed.label]);
  for (let step = 0; step <= 8; step += 1) {
    const offset = Math.ceil(step / 2) * 18 * (step % 2 ? -1 : 1);
    const y = centerY + offset;
    const box = badgeBox(centerX, y);
    if (box.top < top || box.bottom > bottom) continue;
    if (!obstacles.some(other => boxesOverlap(box, other))) return y;
  }
  return null;
}
// 一區的號碼圓（data-zone-key、data-number 給考卷對表上的號碼）；框太窄放不下圓就不畫（title 照樣有號碼與名字）。
function zoneBadge(key, label, box, geometry, layout) {
  const {pad, scale, height} = geometry;
  const [a0, a1, b0, b1] = box;
  if ((a1 - a0) * scale < 2 * BADGE_PX + 2 || (b1 - b0) * scale < 2 * BADGE_PX + 2) return null;
  const centerX = pad + (a0 + a1) / 2 * scale, centerY = pad + (height - (b0 + b1) / 2) * scale;
  const y = zoneLabelY(centerX, centerY, pad + (height - b1) * scale, pad + (height - b0) * scale, layout);
  if (y === null) return null;
  const badge = node("g", {class: "zone-number", "data-zone-key": key, "data-number": label.number});
  badge.append(node("circle", {cx: centerX, cy: y, r: BADGE_PX}),
    node("text", {x: centerX, y, "text-anchor": "middle", "dominant-baseline": "central"}, String(label.number)));
  return badge;
}
// 滑過去看到的那一句：號碼、位置名、代號（伺服器的訊息寫代號）、做法、超過的限制。
function zoneHoverText(label, key, mode, limits) {
  const code = key === "whole_wall" ? "" : `（代號 ${key}）`;
  return `${label.numberText} ${label.name}${code}：${ZONE_MODE_WORDS[mode]}${limits}`;
}
// 牆名鈕上的小字（#1265 單元 3）：這一面現在幾區；只有一區寫「整面」；門、窗、地毯這種整塊就是材料庫那一筆的區
// 照區名那一支（zone_names.js 的 zoneSpecialName）點名、其餘寫幾區（「門、窗＋4 區」）。
function tabWords(wall) {
  const zones = zoneList(wall);
  if (zones.length <= 1) return "整面";
  const special = zones.map(zone => zoneSpecialName(zoneLibraryMatch(zone.material, wall, zoneKey(zone))));
  const named = [...new Set(special.filter(Boolean))], rest = special.filter(name => !name).length;
  if (!named.length) return `${zones.length} 區`;
  return rest ? `${named.join("、")}＋${rest} 區` : named.join("、");
}
/* 「清掉這一面的切法」與復原（#1265 單元 3）：只回一步、只在這一頁、做下一個改動之前有效。
   清掉前記下那一面的切法與設計設定裡那一面每一區的設定（深拷貝）；復原就原樣放回去，不再交伺服器重切
   （重切會重新補滿、重編代號，貼的材料也回不來）。還算「下一個改動之前」：方案與做法那一份沒換過（開檔、範本）、
   改動序號沒動（markStale），而且切法與做法跟清掉後那一刻一字不差（左右對稱開關、「這間房貼什麼」那一頁改做法不走 markStale）。 */
let clearedWall = null;
// 兩個呼叫的人都在有方案時才叫（clearUndoable 先比 owner === scheme、清掉鈕只畫在有方案時），不另判沒方案。
function undoPrint() { return JSON.stringify([scheme.scene.surface_zones_by_wall, zoneDesign]); }
function clearUndoable() {
  const saved = clearedWall;
  return Boolean(saved) && saved.owner === scheme && saved.generation === zoneDesignGeneration && saved.revision === editRevision
    && saved.print === undoPrint();
}
// 做了下一個改動：復原那一句收掉（改動從哪裡來都會走到這裡：markStale、renderZoneTable 重畫）。
function refreshClearUndo() {
  if (!clearedWall || clearUndoable()) return;
  clearedWall = null;
  if ($("elevation-cleared")) renderClearSlot(elevationWall);
}
window.refreshClearUndo = refreshClearUndo;
// 存檔、另存新名字（app.js）：方案物件沒換、改動序號也沒動，比對認不出來，由存檔那一路直接收掉（#1265 單元 3 審查真錯 3）。
// 不另判那一格寫著什麼（#1265 單元 3 之二收尾 B 改判，回到第一趟的寫法）：第二趟加的那一道兩家實測都走不到，拿掉。
// 還能復原（clearedWall 在）時一定選過牆：elevationWall 只在 showElevationWall 換成牆名，不會回到沒選。
window.forgetClearUndo = () => {
  if (!clearedWall) return;
  clearedWall = null;
  renderClearSlot(elevationWall);
};
function renderClearSlot(wall) {
  // 叫到這裡時一定選了一面、畫得出這一面：立面圖重畫時沒選牆、尺寸不齊就先清空這一格回去了（renderElevation）；
  // 清掉鈕那一路在切完那一刻同步畫（cut，那時剛畫好這一面）；改動那一路（refreshClearUndo）只在「已清掉…｜復原」那一句還在時才叫，
  // 存檔那一路（forgetClearUndo）只在還能復原時叫。房間高度清空那一刻先重畫立面圖（markStale 先叫 renderInputBars），那一句已經不在。
  // 以前清掉鈕在等圖面檢查回來之後才畫，空檔裡清空高度會被塞回去（#1265 單元 3 之二第二趟追加第 2 條），所以改到 cut 裡。
  const slot = $("elevation-clear-slot"); slot.replaceChildren();
  if (clearedWall?.wall === wall && clearUndoable()) {
    const said = document.createElement("span"); said.id = "elevation-cleared"; said.setAttribute("role", "status");
    said.textContent = `已清掉${wallNames[wall]} ${clearedWall.count} 區｜`;
    const undo = document.createElement("button"); undo.type = "button"; undo.id = "elevation-undo-clear"; undo.className = "lnk";
    undo.textContent = "復原";
    undo.onclick = () => action(undoClear);
    said.append(undo); slot.append(said);
    return;
  }
  const count = wallZones(wall).filter(zone => zone.area === "rectangle").length;
  const clear = document.createElement("button"); clear.type = "button"; clear.id = "elevation-clear"; clear.className = "danger";
  clear.textContent = slot.closest("#zone-page") ? "清掉這面" : "清掉這一面的切法";
  clear.disabled = !count;
  clear.onclick = () => action(async () => {
    const zones = structuredClone(wallZones(wall)), design = structuredClone(zoneDesign?.zones?.[wall] ?? null);
    // 記快照與畫「已清掉…｜復原」都在 cut 裡同步做（切完、等圖面檢查之前；#1265 單元 3 之二第二趟追加第 2 條與審查設計洞 1）：
    // 等 /api/plan 的空檔頁面已經解鎖，人可能換一面、把房間高度清空（立面圖畫不出來，那一格先被清空），檢查回來不能再畫回去；
    // 檢查丟錯也照樣看得到復原。cut 緊接在 applyCut 解鎖之後同一串跑（中間沒有輪到人按東西的空檔），切的時候 main 整塊鎖著、
    // 換牆的鈕與導覽列都在 main 裡面（「這間房貼什麼」那一頁也只能從 main 裡打開），看著的一定還是這一面，不另判。
    const cut = () => {
      clearedWall = {wall, count: zones.length, zones, design, owner: scheme, generation: zoneDesignGeneration,
                     revision: editRevision, print: undoPrint()};
      renderClearSlot(wall);
    };
    await cutWall(wall, [], () => { elevationSelected = null; }, undefined, false, cut);
  });
  slot.append(clear);
}
// 復原鈕只在還能復原時畫（renderClearSlot），之後任何改動都先走 refreshClearUndo 把它收掉；忙的時候整頁鎖著按不到；
// 按下去同一步就重畫（renderWalls → renderElevation），按不到第二下。清掉前那一面沒有做法時，清掉也不會長出來
// （remapZoneDesign 沒有舊的就不寫），不用另外刪。
async function undoClear() {
  const saved = clearedWall;
  clearedWall = null;
  scheme.scene.surface_zones_by_wall[saved.wall] = structuredClone(saved.zones);
  if (saved.design) designForEdit().zones[saved.wall] = structuredClone(saved.design);
  resizeProblems.delete(saved.wall);
  elevationSelected = null; elevationThin = [];
  staleZoneLimits();  // 同 applyCut：先畫後問
  renderWalls();
  note(`復原了：${wallNames[saved.wall]}回到清掉前的 ${saved.count} 區`, "ok");
  markStale();
  action(updateMultiples);
  action(updateInputPreview);
  await refreshPlan();
}
// 打開哪一面的立面圖（牆名那一排鈕、分區頁每一面的「切區 ▸」都走這裡，#1262）。
// 換到別一面：上一面選的框、太窄的縫、訊息（立面圖底下那一行、回話那一行）一起清掉，不留到別面。
function showElevationWall(name) {
  if (elevationWall !== name) { showSelectionBanner = false; mirrorWarning = null; banner(); elevationWall = name; elevationSelected = null; elevationThin = []; note("", ""); zoneNote("", ""); }
  renderElevation();
}
window.showElevationWall = showElevationWall;
// 打開一份方案時第 2 段先看前牆（#1265 單元 5c）：牆名照伺服器依喇叭與座位排的那一套（WALL_ROLES 認的「前牆」）。
// 分不出前後（牆名是座標名）就不換：開頁那一份沒選任何一面，換方案時留著上一份選的那一面。只在打開那一下叫，
// 之後搬喇叭、牆名換了都不搶使用者選的那一面。回選了哪一面，沒換回 null。
window.showFrontWall = () => {
  const front = Object.keys(wallNames).find(wall => WALL_ROLES[wallNames[wall]] === "front") ?? null;
  if (front) showElevationWall(front);
  return front;
};
function renderElevation() {
  if (dropDrag) dropDrag();  // 拖到一半被重畫：那一張圖要換掉了，這次拖曳作廢（見 trackDrag）
  const area = $("elevation");
  if (!area || !scheme || !inputDefaults) return;
  const on = window.zoneSymmetry.isOn();
  for (const state of [true, false]) {
    const button = $(state ? "elevation-symmetry-on" : "elevation-symmetry-off");
    button.setAttribute("aria-pressed", String(on === state));
    button.onclick = () => {
      if (state !== on) { mirrorWarning = null; banner(); note("", ""); }
      window.zoneSymmetry.set(state);
    };
  }
  refreshAxisControl();
  if (mirrorWarning && (mirrorWarning.owner !== scheme || mirrorWarning.generation !== zoneDesignGeneration || !on)) {
    mirrorWarning = null; banner();
    note("", "");
  }
  if (!on) showSelectionBanner = false;
  if (bannerOwner && (!on || bannerOwner !== scheme)) banner();
  if (mirrorWarning) { note(mirrorWarning.message, mirrorWarning.kind); resultBanner(mirrorWarning); }
  const buttons = $("elevation-walls"); buttons.replaceChildren();
  for (const name of Object.keys(wallNames)) {
    const button = document.createElement("button"); button.id = `elevation-wall-${name}`; button.type = "button"; button.className = "wall";
    const words = document.createElement("span"); words.className = "wall-name"; words.textContent = wallNames[name];
    const count = document.createElement("small"); count.id = `elevation-wall-count-${name}`; count.textContent = tabWords(name);
    button.append(words, count);
    button.setAttribute("aria-pressed", String(name === elevationWall));
    button.onclick = () => showElevationWall(name);
    buttons.append(button);
  }
  // 牆面表單（「換原本的牆」打開的那一塊）只露選中那一面；六面的格子都留在頁面上（存檔照樣逐面讀）。
  for (const row of $("walls").children) row.hidden = row.id !== `wall-editor-${elevationWall}`;
  area.replaceChildren($("elevation-banner"));
  $("elevation-selected").replaceChildren();
  $("elevation-clear-slot").replaceChildren();
  const problems = Object.keys(wallNames).map(resizeProblem).filter(Boolean).join(" ");
  if (!elevationWall) { $("elevation-size").textContent = problems || "選一面牆打開立面圖。"; renderZoneTable(null); return; }
  const wall = elevationWall, view = views[wall], size = wallSize(wall);
  if (!size) { $("elevation-size").textContent = "先把房間的長、寬、高都填好正數，才畫得出這一面。"; renderZoneTable(wall); return; }
  const [width, height] = size;
  const [maxWidth, maxHeight] = area.closest("#zone-page") ? [840, 600] : [560, 320];
  const scale = Math.min(maxWidth / width, maxHeight / height), pad = 48;
  const geometry = {width, height, scale, pad};
  // 照算好的大小畫（一個 viewBox 單位一像素，字固定 14 像素；home.css 只在卡片比圖窄時縮）。
  const svg = node("svg", {id: "elevation-svg", viewBox: `0 0 ${width * scale + 2 * pad} ${height * scale + 2 * pad}`,
                           width: width * scale + 2 * pad, height: height * scale + 2 * pad,
                           "data-wall": wall, role: "img", "aria-label": `${wallNames[wall]}立面圖（站在房間裡看）`});
  svg.append(node("rect", {id: "elevation-frame", x: pad, y: pad, width: width * scale, height: height * scale}));
  const zones = wallZones(wall);
  // 整面一區的牆：牆框本身照那一區的做法上色。
  if (!zones.some(zone => zone.area === "rectangle"))
    svg.lastChild.setAttribute("class", `zone-mode-${zoneSetting(wall, "whole_wall").mode}${zoneIsRed(wall, "whole_wall") ? " zone-red" : ""}`);
  if (wallIsRed(wall)) svg.lastChild.classList.add("wall-red");
  const layout = markerLayout(wall, view, geometry);
  const outside = zones.some(zone => zone.area === "rectangle" && (zone.u_m[1] > width + cutSpec().edge_merge_tol_m || zone.v_m[1] > height + cutSpec().edge_merge_tol_m));
  // 每一區畫號碼（#1176，跟下面表的「區」那一欄同一個號碼；不再畫「z1」「自動」），滑過去看號碼、位置名與代號。
  const labels = zoneLabels(wall);
  const limitsOf = key => zoneLimitLines(wall, key).map(line => `\n${line.text}`).join("");
  if (!zones.some(zone => zone.area === "rectangle")) {
    const mode = zoneSetting(wall, "whole_wall").mode;
    svg.lastChild.append(node("title", {}, zoneHoverText(labels.get("whole_wall"), "whole_wall", mode, limitsOf("whole_wall"))));
    const badge = zoneBadge("whole_wall", labels.get("whole_wall"), [0, width, 0, height], geometry, layout);
    if (badge && zoneIsRed(wall, "whole_wall")) badge.classList.add("zone-red");
    if (badge) svg.append(badge);
  }
  for (const zone of zones.filter(zone => zone.area === "rectangle")) {
    const auto = zone.id.startsWith(cutSpec().fill_prefix);
    const box = screenBox(view, width, height, zone), [a0, a1, b0, b1] = box;
    const mode = zoneSetting(wall, zone.id).mode;
    const red = zoneIsRed(wall, zone.id);
    const group = node("g", {class: `zone ${auto ? "zone-auto" : "zone-drawn"} zone-mode-${mode}${red ? " zone-red" : ""}${zone.id === elevationSelected ? " selected" : ""}`, "data-zone-id": zone.id});
    group.append(node("rect", {x: pad + a0 * scale, y: pad + (height - b1) * scale, width: (a1 - a0) * scale, height: (b1 - b0) * scale}),
      node("title", {}, zoneHoverText(labels.get(zone.id), zone.id, mode, limitsOf(zone.id))));
    const badge = zoneBadge(zone.id, labels.get(zone.id), box, geometry, layout);
    if (badge) group.append(badge);
    svg.append(group);
  }
  for (const strip of elevationThin) {
    const [a0, a1, b0, b1] = screenBox(view, width, height, strip);
    svg.append(node("rect", {class: "zone-thin", x: pad + a0 * scale, y: pad + (height - b1) * scale, width: (a1 - a0) * scale, height: (b1 - b0) * scale}));
  }
  if (mirrorWarning?.blocked?.wall === wall) {
    svg.append(node("rect", {class: "mirror-blocked", ...svgBox(view, geometry, mirrorWarning.blocked)}));
  }
  showSymmetry(svg, wall, geometry, layout);
  drawMarkers(svg, layout);
  const selected = drawnBoxes(wall).find(box => box.id === elevationSelected);
  if (on && selected) resizeHandle(svg, wall, selected, geometry);
  const mid = [pad + width * scale / 2, pad + height * scale / 2];
  for (const [edge, x, y, rotate] of [["left", pad - 14, mid[1], -90], ["right", pad + width * scale + 14, mid[1], 90],
                                       ["top", mid[0], pad - 16, 0], ["bottom", mid[0], pad + height * scale + 26, 0]]) {
    svg.append(node("text", {id: `elevation-edge-${edge}`, class: "elevation-edge", x, y, "text-anchor": "middle",
                             transform: `rotate(${rotate} ${x} ${y})`}, wallNames[view[edge]]));
  }
  svg.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    event.preventDefault();
    const zone = event.target.closest?.("[data-zone-id]");
    // 畫好的框：按住拖＝整塊搬，按一下沒拖＝選取（#1168）；空白處與「自動」格：拖曳畫新框。
    if (zone && zone.classList.contains("zone-drawn")) { startMoving(svg, event, geometry, view, wall, zone); return; }
    startDrawing(svg, event, geometry, view, wall);
  });
  area.append(svg);
  // 選中那一面的一行（#1265 單元 3）：牆名、寬高；幾區寫在上面那一排牆名鈕上。
  const name = document.createElement("b"); name.textContent = wallNames[wall];
  $("elevation-size").replaceChildren(name, ` 寬 ${metres(width)} × 高 ${metres(height)} 公尺` +
    (problems ? "。" + problems : outside ? "。這一面有分區超出目前的牆面（房間尺寸改過）：請改框或清掉這一面的切法" : ""));
  selectedEditor(wall, view, width, height);
  renderClearSlot(wall);
  renderZoneTable(wall);
}
window.renderElevation = renderElevation;
// 六面一起看的那一頁（zone_overview.js）畫圖用同一份幾何：哪一軸往右、鏡不鏡像、牆面寬高、喇叭與主位。
window.elevationGeometry = {views, wallSize, screenBox, flip, project, markerItems};
window.elevationDrawing = () => elevationDrawing;  // 限制判決（zone_limits.js）要知道是不是正在拖框
// 只重畫第 2 段牆那一行與右欄（wall_section.js），不動立面圖、不動牆面表單的格子（#1265 單元 3 之二收尾第二趟 B2：
// 「換原本的牆」改構造某一層時，原本那一面跟著變；app.js 那一格的 input 叫這一支，打字的那一格不重建）。
window.refreshWallSection = () => renderWallSection(elevationWall);
// 只重寫鏡子那一句（#1265 單元 3 之二收尾第二趟 B4 之一）：失敗之後重試、請求還在路上，那一句要回到「正在確認」。
// 不重畫整個立面圖：正在打字的格子、拖到一半的框不能換掉（zone_limits.js 判決回來時為這兩件事才延後重畫）。
window.refreshAxisWords = refreshAxisControl;
window.addEventListener("DOMContentLoaded", () => {
  $("elevation-centre-axis").onclick = () => {
    if (inputBusy || !symmetryResult()?.axis) return;
    action(() => holdInput(async () => {
      if (!await editInput("centre_axis", true, null, true)) {
        banner($("messages").textContent, "notice");
        return;
      }
      const next = "已置中。要補上既有的框，請在原牆選取那一塊，按「補畫另一邊」。";
      banner(next, "ok");
      zoneNote(next, "ok");
    }));
  };
  for (const id of ["room-fields", "speakers", "receivers"]) $(id).addEventListener("input", () => {
    showSelectionBanner = false;
    if (mirrorWarning || !$("elevation-banner").hidden) { mirrorWarning = null; banner(); note("", ""); }
    renderElevation();
  });
  // 房間尺寸改定（離開格子）後調分區，再用調好的草稿重判限制。
  $("room-fields").addEventListener("change", resizeRoom);
  for (const id of ["speakers", "receivers"]) $(id).addEventListener("change", () => {
    if (window.zoneSymmetry.isOn()) refreshZoneLimits();
  });
});
})();
