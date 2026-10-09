/* 共用方案圖面：文字與點資料都由伺服器提供。 */
(() => {
function svgNode(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
// 字壓到別的字或別的點就往下挪一格（step，由呼叫端依畫框換算），最多試 24 次。
function separateLabels(labels, text, circles, ownCircle, step) {
  for (let tries = 0; tries < 24; tries++) {
    const box = text.getBBox();
    const crossesLabel = labels.some((other) => {
      const old = other.getBBox();
      return box.x < old.x + old.width && old.x < box.x + box.width &&
        box.y < old.y + old.height && old.y < box.y + box.height;
    });
    const crossesCircle = circles.some((circle) => {
      if (circle === ownCircle) return false;
      const x = Number(circle.getAttribute("cx"));
      const y = Number(circle.getAttribute("cy"));
      const r = Number(circle.getAttribute("r"));
      return box.x < x + r && x - r < box.x + box.width &&
        box.y < y + r && y - r < box.y + box.height;
    });
    if (!crossesLabel && !crossesCircle) break;
    text.setAttribute("y", Number(text.getAttribute("y")) + step);
  }
  labels.push(text);
}
function drawObjects(svg, plan, vertical, x, y, unit, changedKeys, detailTarget) {
  for (const item of [...(plan.furniture || []), ...(plan.cabinets || [])]) {
    const cabinet = item.kind === "cabinet";
    const cloud = item.kind === "ceiling_cloud";
    const table = ["desk", "coffee_table"].includes(item.kind);
    const changed = !cabinet && changedKeys.has(item.key);
    const group = svgNode("g", {[cabinet ? "data-cabinet" : "data-furniture"]: item.id, tabindex: 0});
    const title = svgNode("title", {}); title.textContent = item.detail_text; group.append(title);
    group.append(svgNode("polygon", {
      points: (vertical ? item.side_polygon : item.polygon).map(([u, v]) => `${x(u)},${y(v)}`).join(" "),
      fill: cloud || table ? "none" : cabinet ? "#db6b3a" : "#697e6a", "fill-opacity": "0.22",
      "pointer-events": "all",
      stroke: item.blocked ? "#c62828" : changed ? "#b01b6a" : cabinet ? "#db6b3a" : "#697e6a",
      "stroke-width": (item.blocked || changed ? 2.5 : 1.5) * unit,
      ...(cloud ? {"stroke-dasharray": `${5 * unit} ${4 * unit}`} : {}),
      class: [changed ? "changed-furniture" : "", item.blocked ? "blocked-furniture" : ""].join(" ")}));
    group.addEventListener("click", () => { detailTarget.textContent = item.detail_text; });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") detailTarget.textContent = item.detail_text;
    });
    svg.append(group);
  }
}
function drawingRange(range, plan, vertical) {
  const points = [...(plan.furniture || []), ...(plan.cabinets || [])]
    .flatMap(item => vertical ? item.side_polygon : item.polygon);
  return {u: [Math.min(range.u[0], ...points.map(p => p[0])), Math.max(range.u[1], ...points.map(p => p[0]))],
    v: [Math.min(range.v[0], ...points.map(p => p[1])), Math.max(range.v[1], ...points.map(p => p[1]))]};
}
function draw(svg, range, markers, zoomRange, zoomed, speakers, vertical, detailTarget, scaleRoom, changedKeys, plan) {
  svg.replaceChildren();
  const frame = zoomed ? range : drawingRange(range, plan, vertical);
  const width = frame.u[1] - frame.u[0];
  const height = frame.v[1] - frame.v[0];
  const frameWidth = scaleRoom ? Math.max(scaleRoom.Lx, width) : width;
  const frameHeight = scaleRoom ? Math.max(scaleRoom[vertical ? "Lz" : "Ly"], height) : height;
  const scale = Math.min(520 / frameWidth, 320 / frameHeight);
  // 畫框裁到要畫的那一塊（整間房、或聆聽區）四周各留 40，房間塞滿格子、不在一邊空一大塊；
  // 比較頁兩邊都用同一間比例房，畫框一樣大。畫框寬至少是高的 1.4 倍（範例房平面圖的比例），
  // 窄長的房間或方形的聆聽區置中，不會把格子撐成又高又窄、跟同一排的圖高低不齊。
  const spanU = frameWidth * scale;
  const spanV = frameHeight * scale;
  const boxHeight = spanV + 80;
  const boxWidth = Math.max(spanU + 80, boxHeight * 1.4);
  svg.setAttribute("viewBox", `0 0 ${boxWidth} ${boxHeight}`);
  // 字、點、外框跟著畫框寬度換算（畫框 600 寬時字 16）：格子一樣寬，每張圖的字與點就一樣大。
  const unit = boxWidth / 600;
  const left = (boxWidth - spanU) / 2;
  const x = (u) => left + (u - frame.u[0]) * scale;
  const y = (v) => boxHeight - 40 - (v - frame.v[0]) * scale;
  svg.append(svgNode("rect", {x: x(range.u[0]), y: y(range.v[1]),
    width: (range.u[1] - range.u[0]) * scale, height: (range.v[1] - range.v[0]) * scale,
    fill: "none", stroke: "#334b58", "stroke-width": unit}));
  if (!zoomed) svg.append(svgNode("rect", {x: x(zoomRange.u[0]), y: y(zoomRange.v[1]),
    width: (zoomRange.u[1] - zoomRange.u[0]) * scale,
    height: (zoomRange.v[1] - zoomRange.v[0]) * scale,
    fill: "none", stroke: "#167997", "stroke-width": unit, "stroke-dasharray": `${5 * unit} ${4 * unit}`}));
  if (!zoomed) drawObjects(svg, plan, vertical, x, y, unit, changedKeys, detailTarget);
  if (!zoomed) for (const speaker of speakers) {
    if (speaker.aim) svg.append(svgNode("line", {
      x1: x(speaker.point.x), y1: y(speaker.point[vertical ? "z" : "y"]),
      x2: x(speaker.aim.x), y2: y(speaker.aim[vertical ? "z" : "y"]),
      stroke: "#db6b3a", "stroke-width": unit}));
  }
  if (!zoomed) for (const path of plan.blocked_paths || []) {
    const axis = vertical ? 2 : 1;
    svg.append(svgNode("path", {
      d: `M ${x(path.start_m[0])} ${y(path.start_m[axis])} L ${x(path.end_m[0])} ${y(path.end_m[axis])}`,
      fill: "none", stroke: "#c62828", "stroke-width": 2.5 * unit,
      class: "blocked-direct", "data-speaker": path.speaker_id, "data-receiver": path.receiver_id}));
  }
  const labels = [];
  const captions = [];
  for (const item of markers) {
    if (zoomed && (item.u < range.u[0] || item.u > range.u[1] ||
                   item.v < range.v[0] || item.v > range.v[1])) continue;
    if (!item.drawn && !item.keys.some((key) => changedKeys.has(key))) continue;
    const detail = item.detail_lines.join("\n");
    const group = svgNode("g", {"data-keys": item.keys.join(" "), tabindex: 0});
    const title = svgNode("title", {}); title.textContent = detail; group.append(title);
    const circle = svgNode("circle", {cx: x(item.u), cy: y(item.v), r: 5 * unit,
      fill: item.kind === "receiver" ? "#167997" : "#db6b3a"});
    const changed = item.keys.filter((key) => changedKeys.has(key));
    if (changed.length) group.append(svgNode("circle", {cx: x(item.u), cy: y(item.v),
      r: 9 * unit, fill: "none", stroke: "#b01b6a", "stroke-width": 2 * unit,
      class: "changed-ring", "data-keys": changed.join(" ")}));
    group.append(circle);
    if (item.caption) {
      const caption = svgNode("text", {x: x(item.u) + 8 * unit, y: y(item.v) - 8 * unit,
        "font-size": 16 * unit});
      caption.textContent = item.caption;
      group.append(caption);
      captions.push([caption, circle]);
    }
    svg.append(group);
    group.addEventListener("click", () => { detailTarget.textContent = detail; });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") detailTarget.textContent = detail;
    });
  }
  // 躲字只躲實心點；改動外框不算障礙，不然改過的點字會被推到下方，同一張圖標法不一致。
  const circles = [...svg.querySelectorAll("circle:not(.changed-ring)")];
  for (const [caption, circle] of captions) separateLabels(labels, caption, circles, circle, 16 * unit);
}
function drawPlan(plan, targets, scaleRoom = plan.room, changedKeys = []) {
  targets.detail.textContent = "";
  const changed = new Set(changedKeys);
  const limits = {plan: {u: [0, plan.room.Lx], v: [0, plan.room.Ly]},
    side: {u: [0, plan.room.Lx], v: [0, plan.room.Lz]}};
  draw(document.getElementById(targets.planXY), limits.plan, plan.views.plan,
    plan.listening_zoom.plan, false, plan.speakers, false, targets.detail, scaleRoom, changed, plan);
  draw(document.getElementById(targets.planXZ), limits.side, plan.views.side,
    plan.listening_zoom.side, false, plan.speakers, true, targets.detail, scaleRoom, changed, plan);
  if (targets.zoomXY) draw(document.getElementById(targets.zoomXY), plan.listening_zoom.plan,
    plan.views.zoom_plan, plan.listening_zoom.plan, true, plan.speakers, false,
    targets.detail, null, changed, plan);
  if (targets.zoomXZ) draw(document.getElementById(targets.zoomXZ), plan.listening_zoom.side,
    plan.views.zoom_side, plan.listening_zoom.side, true, plan.speakers, true,
    targets.detail, null, changed, plan);
  targets.legend.replaceChildren();
  for (const item of [...plan.speakers, ...plan.receivers]) {
    const line = document.createElement("li");
    line.dataset.id = item.key;
    line.textContent = `${item.marker}－${item.detail_text}`;
    targets.legend.append(line);
  }
  for (const item of [...(plan.furniture || []), ...(plan.cabinets || [])]) {
    const line = document.createElement("li");
    line.dataset.id = item.key;
    line.textContent = item.detail_text;
    line.tabIndex = 0;
    line.addEventListener("click", () => { targets.detail.textContent = item.detail_text; });
    line.addEventListener("keydown", (event) => {
      if (event.key === "Enter") targets.detail.textContent = item.detail_text;
    });
    targets.legend.append(line);
  }
  for (const text of plan.drawing_notes || []) {
    const line = document.createElement("li"); line.textContent = text; targets.legend.append(line);
  }
}
window.drawPlan = drawPlan;
})();
