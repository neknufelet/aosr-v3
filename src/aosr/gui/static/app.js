/* 只收表單並畫伺服器給的座標；圖形運算只做 SVG 縮放。 */
let scheme;
let openedId = null;
let runId;
let timer;
// 這一次開算之後表單有沒有被改過；改過的話算完不把連結掛在表單旁邊。
let runEdited = false;
// 最近一次圖面檢查沒過時伺服器寫的問題（已照印在訊息列）；「另存新名字」旁邊照抄同一段。
let planProblems = "";
const compareChoice = {a: null, b: null};
// 顯示名稱表（伺服器 /api/labels 給）：喇叭用聲道代號查、座位用座位代號查；查不到就顯示代號本身。
let labels = {speakers: {}, listening_points: {}};
const $ = (id) => document.getElementById(id);
const wallNames = {floor: "地板", ceiling: "天花", x0: "x 起點牆", xL: "x 終點牆", y0: "y 起點牆", yL: "y 終點牆"};
const coordNames = {x: "x", y: "y", z: "z"};
// 新名字要是伺服器收得下的方案代號（跟伺服器同一個規則）：第一個字是英文字母或數字，後面英文字母、數字、_ 或 -。
// 伺服器仍會再檢查一次；這裡先擋，是為了用「新名字」這一列自己的字眼講，不讓中文名換到一句講「方案代號」的話。
const newNameShape = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const newNameHint = "「新名字」只收英文字母、數字、底線（_）與連字號（-），第一個字要是英文字母或數字；中文、空格和其他符號都不收";
// 阻抗倍數每打一個字就問一次伺服器；回來的順序可能亂，只認最後問的那一次。
let multiplesAsked = 0;

// 瀏覽器自己的錯是英文（連不上時「Failed to fetch」、回的不是 JSON 時「Unexpected token …」）：
// 換成白話，原文放在 cause，只進技術細節。伺服器自己寫的中文訊息照原樣往上交。
async function reply(pending) {
  let response;
  try { response = await pending; } catch (error) {
    throw new Error("連不上本機的網頁伺服器，請確認它還開著", {cause: error});
  }
  try { return {response, data: await response.json()}; } catch (error) {
    throw new Error(`伺服器的回覆讀不懂（狀態碼 ${response.status}）`, {cause: error});
  }
}
async function api(path, method = "GET", body, headers = {}) {
  const {response, data} = await reply(fetch(path, {method, headers: {"Content-Type": "application/json", ...headers},
    body: body === undefined ? undefined : JSON.stringify(body)}));
  if (!response.ok) throw new Error(data.error || problemLines(data.problems));
  return data;
}
function errorText(error) {
  return error instanceof Error ? error.message : String(error);
}
// 問題訊息由伺服器寫好：每一條開頭是表單上那一格的中文名、接白話，同一句只一條（列出它落在哪幾格）；
// 這裡只照印，不自己對欄位路徑（原本的英文路徑在 paths，技術細節才用得到）。
function problemLines(problems) {
  return (problems || []).map((problem) => problem.text).join("\n");
}
function numberField(id, label, value) {
  const wrap = document.createElement("label");
  wrap.textContent = label;
  const input = document.createElement("input");
  input.id = id; input.type = "number"; input.step = "any"; input.value = value;
  wrap.append(input);
  return wrap;
}
function lookUp(table, key, fallback) {
  return key !== undefined && Object.hasOwn(table, key) ? table[key] : fallback;
}
function speakerName(speakerId) {
  const channel = scheme.channel_group.channels.find((item) => item.speaker_id === speakerId);
  return lookUp(labels.speakers, channel?.role, speakerId);
}
function pointName(receiverId) {
  return lookUp(labels.listening_points, receiverId, receiverId);
}
// 每列開頭：中文名，代號用小字放括號（座標格的編號仍用代號）；沒有中文名就只寫代號。
function rowTitle(name, shown) {
  const title = document.createElement("strong");
  title.textContent = shown;
  if (shown !== name) {
    const code = document.createElement("small");
    code.textContent = `（${name}）`;
    title.append(code);
  }
  return title;
}
function rows(target, entries, prefix, naming) {
  $(target).replaceChildren();
  for (const [name, point] of entries) {
    const row = document.createElement("div"); row.className = "row";
    row.append(rowTitle(name, naming(name)));
    for (const axis of Object.keys(coordNames)) {
      const primary = scheme.receiver_set.points.find((item) => item.role === "primary");
      const label = prefix === "receiver" && name === primary.receiver_id && axis === "z" ? "主位 z 座標（公尺）" : coordNames[axis];
      row.append(numberField(`${prefix}-${name}-${axis}`, label, point[axis]));
    }
    $(target).append(row);
  }
}
function renderForm() {
  $("save-id").value = scheme.scheme_id;
  $("source-model").value = scheme.source_model;
  $("room-fields").replaceChildren();
  for (const [key, label] of [["Lx", "長 Lx（公尺）"], ["Ly", "寬 Ly（公尺）"], ["Lz", "高 Lz（公尺）"]])
    $("room-fields").append(numberField(`room-${key}`, label, scheme.scene.room_m[key]));
  $("walls").replaceChildren();
  for (const [name, value] of Object.entries(scheme.scene.impedance_pa_s_per_m_by_wall)) {
    const row = document.createElement("div"); row.className = "row";
    row.append(numberField(`wall-${name}`, `${wallNames[name]}阻抗（帕·秒／公尺）`, value));
    const multiple = document.createElement("span"); multiple.id = `multiple-${name}`; row.append(multiple);
    $("walls").append(row);
  }
  $("scattering").replaceChildren();
  for (const name of Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall))
    $("scattering").append(numberField(`scatter-${name}`, `${wallNames[name]}散射`, scheme.scene.scattering_by_wall?.[name] ?? ""));
  $("use-scattering").checked = scheme.scene.scattering_by_wall !== null && scheme.scene.scattering_by_wall !== undefined;
  rows("speakers", Object.entries(scheme.speakers), "speaker", speakerName);
  rows("receivers", scheme.receiver_set.points.map((point) => [point.receiver_id,
       {x: point.position_m[0], y: point.position_m[1], z: point.position_m[2]}]), "receiver", pointName);
  action(updateMultiples);
}
async function updateMultiples() {
  const asked = ++multiplesAsked;
  const walls = Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall);
  const sent = Object.fromEntries(walls.map((name) => [name, $(`wall-${name}`).value]));
  const checked = await api("/api/validate", "POST", collect());
  if (asked !== multiplesAsked) return;
  // 表單有格子空著或填錯時伺服器一格倍數都不給：數字沒改過的牆留著上一次的倍數（仍是那個數字的倍數），
  // 改過或清空的那一格把旁邊的字清掉，不留舊的「約 ρc 的幾倍」配一格空的或新的數字。
  for (const name of walls) {
    const shown = $(`multiple-${name}`);
    const label = checked.impedance_labels[name];
    if (label !== undefined) { shown.textContent = label; shown.dataset.value = sent[name]; }
    else if (shown.dataset.value !== sent[name]) shown.textContent = "";
  }
}
// 訊息列：成功（存好了、檢查通過、算完了）用 .ok，警示與錯誤用 .notice；兩種都是內文字級。
// 訊息列換了新的一句，「打開」與「另存新名字」旁邊上一次的結果就過時了，一起清掉。
function say(text, kind) {
  $("messages").textContent = text;
  $("messages").className = kind;
  for (const note of ["open-note", "save-as-note"]) sayBeside(note, "", "");
}
// 「打開」與「另存新名字」在頁面最上面，底下的訊息列從那裡看不到：它們的結果（沒打開的原因、空名字、
// 檢查沒過、撞名、存好了）寫在按鈕旁邊，用那一列自己的字眼。
function sayBeside(note, text, kind) {
  $(note).textContent = text;
  $(note).className = kind;
}
function collect() {
  scheme.scheme_id = $("save-id").value;
  scheme.source_model = $("source-model").value;
  const numeric = (id) => {
    const input = $(id);
    return input.value.trim() === "" ? null : Number(input.value);
  };
  for (const key of Object.keys(scheme.scene.room_m)) scheme.scene.room_m[key] = numeric(`room-${key}`);
  for (const name of Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall))
    scheme.scene.impedance_pa_s_per_m_by_wall[name] = numeric(`wall-${name}`);
  scheme.scene.scattering_by_wall = $("use-scattering").checked ? Object.fromEntries(
    Object.keys(scheme.scene.impedance_pa_s_per_m_by_wall).map((name) => [name, numeric(`scatter-${name}`)])) : null;
  for (const [name, point] of Object.entries(scheme.speakers))
    for (const axis of Object.keys(coordNames)) point[axis] = numeric(`speaker-${name}-${axis}`);
  for (const receiver of scheme.receiver_set.points)
    receiver.position_m = Object.keys(coordNames).map((axis) => numeric(`receiver-${receiver.receiver_id}-${axis}`));
  return scheme;
}
async function save() {
  const document = collect();
  if (!await refreshPlan()) return false;
  const headers = openedId === null || openedId !== document.scheme_id ? {"If-None-Match": "*"} : {};
  const saved = await api(`/api/schemes/${encodeURIComponent(document.scheme_id)}`, "PUT", document, headers);
  openedId = document.scheme_id;
  say(saved.message, "ok");
  await loadSchemeList();
  return true;
}
async function loadSchemeList() {
  const data = await api("/api/schemes");
  const list = $("scheme-list");
  const selected = list.value;
  list.replaceChildren();
  for (const name of data.schemes) {
    const option = document.createElement("option");
    option.value = name; option.textContent = name; list.append(option);
  }
  if (data.schemes.includes(selected)) list.value = selected;
}
async function openScheme() {
  const name = $("scheme-list").value;
  if (!name) return;
  let opened;
  try { opened = (await api(`/api/schemes/${encodeURIComponent(name)}`)).scheme; } catch (error) {
    // 存著的那一份現在檢查不過（或讀不出）：表單維持原樣，原因寫在訊息列，也寫在「打開」旁邊。
    say(errorText(error), "notice");
    sayBeside("open-note", `沒有打開「${name}」：${errorText(error)}`, "notice");
    return;
  }
  scheme = opened;
  openedId = name;
  renderForm();
  $("result-link").hidden = true;
  $("result-stale").hidden = true;
  await refreshPlan();
}
async function saveAs() {
  const name = $("save-as-id").value.trim();
  if (!name || !newNameShape.test(name)) {
    sayBeside("save-as-note", name ? `沒有另存：${newNameHint}` :
      "「新名字」這一格還空著：先填新名字，再按「另存新名字」", "notice");
    $("save-as-id").focus();
    return;
  }
  let saved;
  try {
    // 檢查沒過：問題已經寫進下面的訊息列（伺服器寫好的白話），旁邊照抄同一段。
    if (!await refreshPlan()) throw new Error(planProblems);
    // 方案代號欄只顯示現在開著哪一份：伺服器存好了才換成新名字，存失敗就維持原樣。
    saved = await api(`/api/schemes/${encodeURIComponent(name)}`, "PUT",
      {...collect(), scheme_id: name}, {"If-None-Match": "*"});
  } catch (error) {
    say(errorText(error), "notice");
    sayBeside("save-as-note", `沒有另存：${errorText(error)}`, "notice");
    return;
  }
  scheme.scheme_id = name; $("save-id").value = name; openedId = name;
  say(saved.message, "ok");
  sayBeside("save-as-note", `「${name}」：${saved.message}`, "ok");
  await loadSchemeList();
  $("scheme-list").value = name;
}
function markStale() {
  runEdited = true;
  if (!$("result-link").hidden) $("result-stale").hidden = false;
}
function chooseCompare(side, item) {
  compareChoice[side] = item;
  $(side === "a" ? "compare-a" : "compare-b").textContent =
    item ? `${side.toUpperCase()}：${item.scheme_id}（${item.finished_text}）` +
    (["done", "none"].includes(item.run_status) ? "" : `・${item.status_text}`) : `${side.toUpperCase()}：未選`;
  const link = $("compare-link");
  link.hidden = !compareChoice.a || !compareChoice.b;
  if (!link.hidden) link.href = `/compare/${compareChoice.a.run_id}/${compareChoice.b.run_id}`;
}
function finishedCell(cell, text) {
  if (!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/.test(text)) return;
  cell.replaceChildren();
  for (const [index, part] of text.split(" ").entries()) {
    if (index) cell.append(document.createTextNode(" "));
    const span = document.createElement("span");
    span.className = index ? "finished-time" : "finished-date";
    span.textContent = part; cell.append(span);
  }
}
function resultRow(item, archived = false) {
  const row = document.createElement("tr");
  row.dataset.runId = item.run_id;
  const finished = ["done", "none"].includes(item.run_status);
  row.classList.toggle("not-finished", !finished);
  // 「計算版本」格只寫白話；計算指紋與程式提交代號是技術細節，滑鼠停在那一格才出現。
  // 說明跟著欄位走、不按字比：壞檔那一列四格都寫「讀不出」，按字比會把說明掛到每一格。
  const values = archived ? [[item.scheme_id], [item.finished_text],
    [item.calculation_text, item.calculation_detail]] : [[item.scheme_id], [item.finished_text],
    [item.duration_text], [item.calculation_text, item.calculation_detail], [item.registry_text]];
  for (const [index, [value, detail]] of values.entries()) {
    const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    if (index === 1) finishedCell(cell, value);
    if (detail) cell.title = detail;
  }
  // 沒有正常完成的那一筆（失敗、已停止、計算中）：方案代號底下多一行狀態，正常完成的不加字。
  // 不另開一欄：多一欄表格就凸出卡片（實量：表寬 1158、卡片 1100）。
  // 封存動作另有按鈕欄；方案代號保持完整，日期與時刻各自不折、只在兩段之間換行，留出新欄的位置。
  if (!finished) {
    const status = document.createElement("span");
    status.className = "run-status"; status.textContent = item.status_text;
    row.firstElementChild.append(status);
  }
  return row;
}
function rowButton(row, text, work) {
  const cell = document.createElement("td"), button = document.createElement("button");
  button.type = "button"; button.textContent = text;
  button.onclick = () => action(work);
  cell.append(button); row.append(cell);
}
async function moveResult(item, restore = false) {
  const url = restore ? `/api/archive/${item.run_id}/restore` : `/api/results/${item.run_id}/archive`;
  const data = await api(url, "POST", {});
  if (!restore) {
    for (const side of ["a", "b"]) {
      if (compareChoice[side]?.run_id === item.run_id) chooseCompare(side, null);
    }
    if ($("result-link").getAttribute("href") === item.result_url) {
      $("result-link").hidden = true;
      $("result-stale").hidden = true;
    }
  }
  await loadResultList();
  say(data.message, "ok");
}
async function loadArchiveList() {
  const data = await api("/api/archive");
  const archive = $("archived"), list = $("archived-list");
  archive.hidden = !data.results.length;
  archive.querySelector("summary").textContent = `已封存的結果（${data.results.length} 筆）`;
  list.replaceChildren();
  for (const item of data.results) {
    const row = resultRow(item, true);
    rowButton(row, "搬回", () => moveResult(item, true));
    list.append(row);
  }
}
async function loadResultList() {
  const data = await api("/api/results");
  $("server-notice").hidden = !data.server_notice;
  $("server-notice").textContent = data.server_notice || "";
  const list = $("results-list"); list.replaceChildren();
  for (const item of data.results) {
    const row = resultRow(item);
    const cell = document.createElement("td");
    const link = document.createElement("a");
    // 「查看」是連到結果頁的連結，外觀跟同一列的「選為 A／B」按鈕一樣（home.css）。
    link.className = "button-link";
    link.href = item.result_url; link.textContent = "查看"; cell.append(link); row.append(cell);
    for (const side of ["a", "b"]) {
      rowButton(row, `選為 ${side.toUpperCase()}`, () => chooseCompare(side, item));
    }
    rowButton(row, "封存", () => moveResult(item));
    list.append(row);
  }
  await loadArchiveList();
}
async function resumeRuns() {
  // 只接回還在算的；已結束的那一筆不貼，表單開的是範本，貼上去會讓人以為是這份的結果。
  const state = (await api("/api/runs")).running[0];
  if (!state) return;
  runId = state.run_id;
  $("run-state").textContent = state.display_text;
  $("stop").disabled = false;
  clearInterval(timer);
  timer = setInterval(() => action(poll), 1000);
}
async function refreshPlan() {
  const {response, data: plan} = await reply(fetch("/api/plan", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(collect())}));
  if (!response.ok) {
    for (const name of ["plan-xy", "plan-xz", "zoom-xy", "zoom-xz", "plan-legend", "plan-detail"])
      $(name).replaceChildren();
    planProblems = plan.problems ? problemLines(plan.problems) : (plan.error || "圖面檢查失敗");
    say(planProblems, "notice");
    return false;
  }
  drawPlan(plan, {planXY: "plan-xy", planXZ: "plan-xz", zoomXY: "zoom-xy",
    zoomXZ: "zoom-xz", detail: $("plan-detail"), legend: $("plan-legend")});
  say(plan.message, "ok");
  return true;
}
async function poll() {
  const state = await api(`/api/runs/${runId}`);
  // 主畫面只寫白話的進度；計算程式自己印的英文輸出收進可展開的技術細節，出錯時照樣找得到原文。
  $("run-state").textContent = state.display_text;
  $("run-log").hidden = !state.stderr_tail.length;
  $("run-log-text").textContent = state.stderr_tail.join("\n");
  if (state.status !== "running") {
    clearInterval(timer); $("stop").disabled = true;
    if (state.status === "done") {
      // 表單還是算的那一份、開算後也沒改過，連結才掛在表單旁邊；不然會讓人以為是表單上這份的結果。
      if (state.scheme_id === openedId && !runEdited) {
        // 講白話就好：結果檔路徑與 32 位計算代號不印，看結果交給下面的「查看結果頁」連結。
        say(`「${state.scheme_id}」算完了，按下面的「查看結果頁」看結果`, "ok");
        $("result-link").href = state.result_url;
        $("result-link").hidden = false;
        $("result-stale").hidden = true;
      } else {
        say(`「${state.scheme_id}」算完了；表單上現在不是算的那一份（開算後改過或換了方案），結果在下方結果清單`, "notice");
      }
      await loadResultList();
    }
  }
}
async function action(work) {
  // 只印伺服器給的中文訊息，不帶 JS 的英文字首「Error: 」。
  try { await work(); } catch (error) { say(errorText(error), "notice"); }
}
window.addEventListener("DOMContentLoaded", () => action(async () => {
  const example = await api("/api/example"); scheme = example.scheme;
  // 名稱表載不到也照樣畫表單（列名退回代號）；說明寫在喇叭與座位那一區自己的一行，
  // 不寫訊息列（訊息列等一下就被「檢查通過」蓋掉，畫面上只剩代號配綠字）。
  // 那一行只講白話；為什麼載不到（伺服器的話、瀏覽器的英文原文）收在底下摺起來的技術細節。
  try { labels = await api("/api/labels"); } catch (error) {
    const cause = error instanceof Error && error.cause !== undefined ? `；瀏覽器原文：${errorText(error.cause)}` : "";
    $("labels-technical").textContent = `原因：${errorText(error)}${cause}`;
    $("labels-notice").hidden = false;
  }
  $("feature-note").textContent = example.feature_match_note;
  $("rho-c").textContent = example.rho_c_label;
  renderForm();
  // 按鈕先綁：下面任一份清單載入失敗，也不能讓整頁按鈕都沒反應。
  // 表單任一格改了都重問阻抗倍數：別格空著時倍數會清掉，那一格補好就要回來。
  for (const id of ["room-fields", "walls", "scattering", "speakers", "receivers", "source-model", "use-scattering"]) {
    $(id).addEventListener("input", markStale);
    $(id).addEventListener("input", () => action(updateMultiples));
  }
  $("check").onclick = () => action(refreshPlan);
  $("save").onclick = () => action(save);
  $("open-scheme").onclick = () => action(openScheme);
  $("save-as").onclick = () => action(saveAs);
  // 新名字一邊打一邊看：打了收不下的字（中文、空格、/ 之類）就先在旁邊說，不用等按下去才知道。
  $("save-as-id").addEventListener("input", () => {
    const typed = $("save-as-id").value.trim();
    const refused = typed !== "" && !newNameShape.test(typed);
    sayBeside("save-as-note", refused ? newNameHint : "", refused ? "notice" : "");
  });
  $("calculate").onclick = () => action(async () => {
    if (!await save()) return;
    const state = await api("/api/runs", "POST", {scheme_id: scheme.scheme_id});
    // 真的開算了才清記號：存檔被擋（例如正在算的那一份改不得）時，記號要留著。
    runEdited = false;
    runId = state.run_id; $("stop").disabled = false;
    // 上一筆的計時器先停：留著的話它會每秒去跑算完那一段，蓋掉訊息、藏掉舊結果標示。
    clearInterval(timer);
    await poll();
    if (!$("stop").disabled) timer = setInterval(() => action(poll), 1000);
  });
  $("stop").onclick = () => action(async () => { await api(`/api/runs/${runId}/stop`, "POST"); await poll(); });
  await refreshPlan();
  for (const load of [loadSchemeList, loadResultList, resumeRuns]) await action(load);
}));
