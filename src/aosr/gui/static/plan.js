/* 共用方案圖面：文字與點資料都由伺服器提供。 */
(() => {
function svgNode(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
function separateLabels(labels, text, circles, ownCircle, step) {
  for (let step = 0; step < 24; step++) {
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
function draw(svg, range, markers, zoomRange, zoomed, speakers, vertical, detailTarget, scaleRoom, changedKeys) {
  svg.replaceChildren();
  const width = range.u[1] - range.u[0];
  const height = range.v[1] - range.v[0];
  const scale = scaleRoom ? Math.min(520 / scaleRoom.Lx,
    320 / scaleRoom[vertical ? "Lz" : "Ly"]) : Math.min(520 / width, 320 / height);
  // 畫框裁到要畫的那一塊（整間房、或聆聽區）四周各留 40，房間塞滿格子、不在一邊空一大塊；
  // 比較頁兩邊都用同一間比例房，畫框一樣大。畫框寬至少是高的 1.4 倍（範例房平面圖的比例），
  // 窄長的房間或方形的聆聽區置中，不會把格子撐成又高又窄、跟同一排的圖高低不齊。
  const spanU = (scaleRoom ? scaleRoom.Lx : width) * scale;
  const spanV = (scaleRoom ? scaleRoom[vertical ? "Lz" : "Ly"] : height) * scale;
  const boxHeight = spanV + 80;
  const boxWidth = Math.max(spanU + 80, boxHeight * 1.4);
  svg.setAttribute("viewBox", `0 0 ${boxWidth} ${boxHeight}`);
  // 字、點、外框跟著畫框寬度換算（畫框 600 寬時字 16）：格子一樣寬，每張圖的字與點就一樣大。
  const unit = boxWidth / 600;
  const left = (boxWidth - spanU) / 2;
  const x = (u) => left + (u - range.u[0]) * scale;
  const y = (v) => boxHeight - 40 - (v - range.v[0]) * scale;
  svg.append(svgNode("rect", {x: left, y: boxHeight - 40 - height * scale,
    width: width * scale, height: height * scale, fill: "none", stroke: "#334b58", "stroke-width": unit}));
  if (!zoomed) svg.append(svgNode("rect", {x: x(zoomRange.u[0]), y: y(zoomRange.v[1]),
    width: (zoomRange.u[1] - zoomRange.u[0]) * scale,
    height: (zoomRange.v[1] - zoomRange.v[0]) * scale,
    fill: "none", stroke: "#167997", "stroke-width": unit, "stroke-dasharray": `${5 * unit} ${4 * unit}`}));
  if (!zoomed) for (const speaker of speakers) {
    if (speaker.aim) svg.append(svgNode("line", {
      x1: x(speaker.point.x), y1: y(speaker.point[vertical ? "z" : "y"]),
      x2: x(speaker.aim.x), y2: y(speaker.aim[vertical ? "z" : "y"]),
      stroke: "#db6b3a", "stroke-width": unit}));
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
    plan.listening_zoom.plan, false, plan.speakers, false, targets.detail, scaleRoom, changed);
  draw(document.getElementById(targets.planXZ), limits.side, plan.views.side,
    plan.listening_zoom.side, false, plan.speakers, true, targets.detail, scaleRoom, changed);
  if (targets.zoomXY) draw(document.getElementById(targets.zoomXY), plan.listening_zoom.plan,
    plan.views.zoom_plan, plan.listening_zoom.plan, true, plan.speakers, false,
    targets.detail, null, changed);
  if (targets.zoomXZ) draw(document.getElementById(targets.zoomXZ), plan.listening_zoom.side,
    plan.views.zoom_side, plan.listening_zoom.side, true, plan.speakers, true,
    targets.detail, null, changed);
  targets.legend.replaceChildren();
  for (const item of [...plan.speakers, ...plan.receivers]) {
    const line = document.createElement("li");
    line.dataset.id = item.key;
    line.textContent = `${item.marker}－${item.detail_text}`;
    targets.legend.append(line);
  }
}
window.drawPlan = drawPlan;
})();
