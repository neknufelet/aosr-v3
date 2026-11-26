/* 頂條狀態旁那一句檢查結果（#1265 單元 6；示意 draw/homes.md 第 36 列）：平面圖檢查（app.js 的 refreshPlan）與限制檢查
   （zone_limits.js 的 checkZoneLimits）合成一句。兩邊都有回音之前寫「檢查中…」，不留上一次的那一句；都沒問題寫「檢查通過」；
   有問題逐種寫一顆可按的字，按了捲到問題那裡（平面圖問題 → 第 1 段平面圖；超過限制 → 第 2 段第一個標紅的區、選那一面牆）。
   頁底兩顆（「照現在牆上貼的算一次」、「交給程式最佳化 ▸」）被檢查擋下時也走這裡：捲過去、焦點放在那一顆字上（stopAtCheck）。
   判法不在這裡：平面圖照伺服器 /api/plan 的回覆，限制照 /api/zone-limits 的判決，這裡只記狀態、只畫。 */
// 平面圖那一半：pending 在路上、ok 通過、problem 有幾條問題（count）、error 沒檢查成；words 是原因原文（念給讀螢幕的人聽）。
let planVerdict = {state: "pending", count: 0, words: ""};
function planChecking() {
  planVerdict = {state: "pending", count: 0, words: ""};
  renderCheckState();
}
function planChecked(state, count = 0, words = "") {
  planVerdict = {state, count, words};
  renderCheckState();
}
// 限制那一半照 zone_limits.js 的訊號：最新那次判決還沒回來（或剛標過「之前收的不算數」）就是在路上。
function limitsVerdict() {
  if (!zoneLimitsFresh()) return {state: "pending", count: 0, words: ""};
  if (zoneLimits.status !== "checked") return {state: "error", count: 1, words: zoneLimits.message};
  const red = limitsRedCount();
  return red ? {state: "problem", count: red, words: zoneLimits.refusals.join("；")} : {state: "ok", count: 0, words: ""};
}
const CHECK_PARTS = {
  plan: {problem: count => `平面圖 ${count} 條問題`, error: "平面圖沒檢查成", where: "第 1 段平面圖"},
  limits: {problem: count => `${count} 條超過限制`, error: "限制沒檢查成", where: "第 2 段標紅的區"},
};
function checkPartWords(name, verdict) {
  const part = CHECK_PARTS[name];
  return verdict.state === "problem" ? part.problem(verdict.count) : part.error;
}
function renderCheckState() {
  const box = document.getElementById("check-state");
  if (!box) return;
  const verdicts = {plan: planVerdict, limits: limitsVerdict()};
  if (Object.values(verdicts).some(verdict => verdict.state === "pending")) {
    box.dataset.kind = "pending"; box.replaceChildren("檢查中…"); return;
  }
  const bad = Object.entries(verdicts).filter(([, verdict]) => verdict.state !== "ok");
  if (!bad.length) { box.dataset.kind = "ok"; box.replaceChildren("檢查通過"); return; }
  box.dataset.kind = "problem";
  const parts = ["檢查沒過："];
  bad.forEach(([name, verdict], index) => {
    const go = document.createElement("button");
    go.type = "button"; go.id = `check-${name}`; go.className = "check-go";
    go.textContent = checkPartWords(name, verdict);
    go.title = verdict.words;
    go.setAttribute("aria-label", `${go.textContent}：${verdict.words}；按了捲到${CHECK_PARTS[name].where}`);
    go.onclick = () => goToCheck(name);
    if (index) parts.push("；");
    parts.push(go);
  });
  box.replaceChildren(...parts);
}
// 限制判決裡第一個標紅的地方：照牆的順序（wallNames）找第一面有紅區的牆、那一面第一個紅區；沒有紅區、只有整面紅的，回那一面；
// 整份方案層級的（設計設定那幾條）沒有牆可指，回 null（捲到第 2 段開頭）。
function firstRedSpot() {
  for (const wall of Object.keys(wallNames)) {
    const key = Object.keys(zoneLimits?.zones?.[wall] ?? {}).find(name => zoneIsRed(wall, name));
    if (key !== undefined) return {wall, key};
  }
  const wall = Object.keys(wallNames).find(name => wallIsRed(name));
  return wall ? {wall, key: null} : null;
}
// 捲到 node：只算頂條底下、頁底那條上面看得到的那一段（兩條都黏在視窗上，頂條多一行訊息就變高，固定的捲動邊距會被蓋住）。
// start：node 頂端貼在頂條下面一點；center：放在那一段正中，放不下就跟 start 一樣頂端對齊。
function scrollBelowBars(node, block) {
  const top = document.getElementById("input-bars")?.getBoundingClientRect().bottom ?? 0;
  const bottom = document.getElementById("input-footer")?.getBoundingClientRect().top ?? window.innerHeight;
  const box = node.getBoundingClientRect(), gap = 12, room = bottom - top - 2 * gap;
  const lead = block === "center" && box.height < room ? (room - box.height) / 2 : 0;
  window.scrollBy({top: box.top - top - gap - lead, behavior: "instant"});
}
// 按了頂條那一顆字：平面圖問題捲到第 1 段；限制問題先選那一面牆（重畫立面圖）、再捲到那一區（重畫完才找得到新畫的那一塊）。
function goToCheck(name) {
  if (name === "plan") { scrollBelowBars(document.getElementById("scheme-section"), "start"); return; }
  const spot = limitsVerdict().state === "problem" ? firstRedSpot() : null;
  if (!spot) { scrollBelowBars(document.getElementById("wall-section"), "start"); return; }
  window.showElevationWall(spot.wall);
  const svg = document.getElementById("elevation-svg");
  const zone = spot.key === null ? null : svg?.querySelector(`[data-zone-id="${CSS.escape(spot.key)}"]`);
  scrollBelowBars(zone ?? svg ?? document.getElementById("wall-section"), "center");
}
// 頁底兩顆被檢查擋下（app.js 的「照現在牆上貼的算一次」、zone_modes.js 的 sendToSearch）：不送，捲到問題那裡，焦點放在頂條那一顆字
// （它的名字念得出原因）。另一半檢查還在路上時頂條寫「檢查中…」、還沒有那一顆字：等限制判決回來再畫一次；還是沒有就放在整句上。
// 等限制判決的時候頁面已經解鎖（第二趟發現 5）：回來時重看一次，三件都成立才捲、才搶焦點，否則什麼都不做——
// 還在輸入頁（「這間房貼什麼」那一頁開著時輸入頁藏著，那一頁自己有那一句，zone_overview.js）、那個問題還在（那一半不是「檢查通過」；
// 還在路上的不知道好了沒，照樣停）、按下之後使用者沒有自己把焦點移走（焦點還在按的那一顆、或落在頁面本身）。
function stillStopping(name, pressed) {
  if (document.body.classList.contains("zone-page-open")) return false;
  const verdict = name === "plan" ? planVerdict : limitsVerdict();
  if (verdict.state === "ok") return false;
  return document.activeElement === pressed || document.activeElement === document.body || document.activeElement === null;
}
async function stopAtCheck(name) {
  const pressed = document.activeElement;
  if (!stillStopping(name, pressed)) return;
  await settledZoneLimits();
  if (!stillStopping(name, pressed)) return;
  renderCheckState();
  goToCheck(name);
  const go = document.getElementById(`check-${name}`) ?? document.getElementById("check-state");
  go.focus({preventScroll: true});
}
window.renderCheckState = renderCheckState;
