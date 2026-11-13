/* 首頁第一屏（#1134）：最近算的那間房（俯視圖）、三個入口、存著的方案每份一句白話。
   字、座標、連到哪都由伺服器排好（/api/home）；這裡只把它畫成 SVG、排成連結。前牆朝上、左牆在左。 */
const HOME_SVG = "http://www.w3.org/2000/svg";
function homeSvg(tag, attrs = {}, text = "") {
  const node = document.createElementNS(HOME_SVG, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== "") node.setAttribute(key, String(value));
  if (text) node.textContent = text;
  return node;
}
function homeElement(tag, text = "", className = "") {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}
// 俯視圖：房間外框、有貼東西的那幾段牆（實測藍、模擬橘）、喇叭、主位、四面牆名寫在牆外側。
function drawHomeRoom(room) {
  const svg = document.getElementById("home-room");
  const pad = 30, longest = 240;
  const scale = longest / Math.max(room.width_m, room.depth_m);
  const width = room.width_m * scale, depth = room.depth_m * scale;
  svg.setAttribute("viewBox", `0 0 ${width + 2 * pad} ${depth + 2 * pad}`);
  svg.replaceChildren();
  const at = (point) => [pad + point.x * scale, pad + point.y * scale];
  svg.append(homeSvg("rect", {x: pad, y: pad, width, height: depth, class: "home-wall"}));
  for (const segment of room.segments) {
    const [x1, y1] = at(segment.from), [x2, y2] = at(segment.to);
    svg.append(homeSvg("line", {x1, y1, x2, y2, class: `home-treated home-${segment.basis}`, "data-wall": segment.wall}));
  }
  const [px, py] = at(room.primary);
  for (const speaker of room.speakers) {
    const [sx, sy] = at(speaker.at);
    svg.append(homeSvg("line", {x1: sx, y1: sy, x2: px, y2: py, class: "home-beam"}));
  }
  for (const speaker of room.speakers) {
    const [sx, sy] = at(speaker.at);
    svg.append(homeSvg("rect", {x: sx - 6, y: sy - 6, width: 12, height: 12, rx: 2, class: "home-speaker"}));
    svg.append(homeSvg("text", {x: sx, y: sy + 20, "text-anchor": "middle", class: "home-small"}, speaker.name));
  }
  svg.append(homeSvg("circle", {cx: px, cy: py, r: 7, class: "home-seat"}));
  svg.append(homeSvg("text", {x: px, y: py + 22, "text-anchor": "middle", class: "home-small"}, "主位"));
  for (const wall of room.walls) {
    const [wx, wy] = at(wall.at);
    const side = wy <= pad + 0.5 ? "top" : wy >= pad + depth - 0.5 ? "bottom" : wx <= pad + 0.5 ? "left" : "right";
    const [x, y, turn] = {top: [wx, pad - 10, 0], bottom: [wx, pad + depth + 20, 0],
                          left: [pad - 10, wy, -90], right: [pad + width + 10, wy, 90]}[side];
    svg.append(homeSvg("text", {x, y, "text-anchor": "middle", class: "home-wall-name", "data-wall": wall.wall,
                                transform: turn ? `rotate(${turn} ${x} ${y})` : ""}, wall.name));
  }
}
function homeLink(id, link) {
  const anchor = document.getElementById(id);
  anchor.href = link.url;
  anchor.hidden = !link.url;
  anchor.querySelector(".home-note").textContent = link.note;
}
// 方案清單一行：名字、白話、「打開到下面表單」（借下面「開舊方案」那一組，不另寫一套）。
function homeSchemeRow(item) {
  const row = homeElement("li");
  row.dataset.scheme = item.name;
  row.append(homeElement("b", item.title), homeElement("span", `：${item.text}`));
  const button = homeElement("button", "打開到下面表單");
  button.type = "button";
  button.onclick = () => {
    document.getElementById("scheme-list").value = item.name;
    document.getElementById("open-scheme").click();
    document.getElementById("scheme-section").scrollIntoView();
  };
  row.append(" ", button);
  return row;
}
function drawHomeSchemes(schemes) {
  const own = schemes.filter((item) => !item.auto), auto = schemes.filter((item) => item.auto);
  document.getElementById("home-schemes").replaceChildren(...own.map(homeSchemeRow));
  document.getElementById("home-auto").hidden = !auto.length;
  document.getElementById("home-auto-count").textContent = `電腦自動存的方案（${auto.length} 份）`;
  document.getElementById("home-auto-schemes").replaceChildren(...auto.map(homeSchemeRow));
}
async function refreshHome() {
  const reply = await fetch("/api/home", {cache: "no-store"});
  const data = await reply.json();
  if (!reply.ok) throw new Error(data.error || "首頁摘要讀不到");
  document.getElementById("home-title").textContent = data.title;
  document.getElementById("home-facts").textContent = data.facts;
  document.getElementById("home-picture").hidden = !data.room;
  document.getElementById("home-unnamed").hidden = !data.room || data.room.named;
  if (data.room) drawHomeRoom(data.room);
  for (const [id, key] of [["home-result", "result"], ["home-materials", "materials"], ["home-layout", "layout"]])
    homeLink(id, data.links[key]);
  drawHomeSchemes(data.schemes);
}
window.addEventListener("DOMContentLoaded", () => {
  refreshHome().catch((error) => {
    document.getElementById("home-facts").textContent = `首頁摘要讀不到：${error.message}`;
  });
});
