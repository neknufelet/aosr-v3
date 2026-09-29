/* 共用方案圖面：文字與點資料都由伺服器提供。 */
(() => {
function svgNode(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
function separateLabels(labels, text, circles, ownCircle) {
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
    text.setAttribute("y", Number(text.getAttribute("y")) + 16);
  }
  labels.push(text);
}
function draw(svg, range, markers, zoomRange, zoomed, speakers, vertical, detailTarget, scaleRoom, changedKeys) {
  svg.replaceChildren();
  const width = range.u[1] - range.u[0];
  const height = range.v[1] - range.v[0];
  const scale = scaleRoom ? Math.min(520 / scaleRoom.Lx,
    320 / scaleRoom[vertical ? "Lz" : "Ly"]) : Math.min(520 / width, 320 / height);
  const x = (u) => 40 + (u - range.u[0]) * scale;
  const y = (v) => 360 - (v - range.v[0]) * scale;
  svg.append(svgNode("rect", {x: 40, y: 360 - height * scale,
    width: width * scale, height: height * scale, fill: "none", stroke: "#334b58"}));
  if (!zoomed) svg.append(svgNode("rect", {x: x(zoomRange.u[0]), y: y(zoomRange.v[1]),
    width: (zoomRange.u[1] - zoomRange.u[0]) * scale,
    height: (zoomRange.v[1] - zoomRange.v[0]) * scale,
    fill: "none", stroke: "#167997", "stroke-dasharray": "5 4"}));
  if (!zoomed) for (const speaker of speakers) {
    if (speaker.aim) svg.append(svgNode("line", {
      x1: x(speaker.point.x), y1: y(speaker.point[vertical ? "z" : "y"]),
      x2: x(speaker.aim.x), y2: y(speaker.aim[vertical ? "z" : "y"]),
      stroke: "#db6b3a"}));
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
    const circle = svgNode("circle", {cx: x(item.u), cy: y(item.v), r: 5,
      fill: item.kind === "receiver" ? "#167997" : "#db6b3a"});
    const changed = item.keys.filter((key) => changedKeys.has(key));
    if (changed.length) group.append(svgNode("circle", {cx: x(item.u), cy: y(item.v),
      r: 9, fill: "none", stroke: "#b01b6a", "stroke-width": 2,
      class: "changed-ring", "data-keys": changed.join(" ")}));
    group.append(circle);
    if (item.caption) {
      const caption = svgNode("text", {x: x(item.u) + 8, y: y(item.v) - 8});
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
  const circles = [...svg.querySelectorAll("circle")];
  for (const [caption, circle] of captions) separateLabels(labels, caption, circles, circle);
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
