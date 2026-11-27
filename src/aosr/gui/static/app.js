/* 只收表單並畫伺服器給的座標；圖形運算只做 SVG 縮放。 */
let scheme;
let openedId = null;
let runId;
let timer;
let otherTimer;
let otherAsking = false;
// 每次換主畫面跟的那筆（接回、改看、開算）就加一；等完回覆時號碼已經不是最新的，那一次就不排計時器（#796 審查）。
let followSeq = 0;
// 這一次開算之後表單有沒有被改過；改過的話算完不把連結掛在表單旁邊。
let runEdited = false;
let editRevision = 0;
// 最近一次圖面檢查沒過時伺服器寫的問題（已照印在訊息列）；「另存新名字」旁邊照抄同一段。
let planProblems = "";
const compareChoice = {a: null, b: null};
// 顯示名稱表（伺服器 /api/labels 給）：喇叭用聲道代號查、座位用座位代號查；查不到就顯示代號本身。
let labels = {speakers: {}, listening_points: {}, speaker_setup: {}};
const $ = (id) => document.getElementById(id);
// 六面牆的名字：先用座標名，伺服器依喇叭與主位排出前後左右後換掉（#1134，useWallNames）；鍵與順序不變。
const wallNames = {floor: "地板", ceiling: "天花", x0: "x 起點牆", xL: "x 終點牆", y0: "y 起點牆", yL: "y 終點牆"};
const coordNames = {x: "x", y: "y", z: "z"};
// 新名字要是伺服器收得下的方案代號（跟伺服器同一個規則）：第一個字是英文字母或數字，後面英文字母、數字、_ 或 -。
// 伺服器仍會再檢查一次；這裡先擋，是為了用「新名字」這一列自己的字眼講，不讓中文名換到一句講「方案代號」的話。
const newNameShape = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const newNameHint = "「新名字」只收英文字母、數字、底線（_）與連字號（-），第一個字要是英文字母或數字；中文、空格和其他符號都不收";
// 阻抗倍數每打一個字就問一次伺服器；回來的順序可能亂，只認最後問的那一次。
let multiplesAsked = 0;
let inputDefaults;
let inputValues = {};
let inputPreview = {furniture_items: [], furniture_controls: {}, mounts: []};
let inputPreviewScheme = null;
let formScheme = null;
let inputBusy = false;
let previewAsked = 0;
// 結果清單最近一次讀到的那幾列（新到舊）：第 1 段右卡結果那一行拿這一份找「這一份」最新算好的那一列（room_facts.js）。
let resultRows = [];
// 開頁範本六面的固定阻抗：打開一份方案時，不是整面固定阻抗的那幾面改回固定阻抗就填這個（openScheme）。
let startupWallDefaults = {};

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
// 問題訊息由伺服器寫好：每條開頭是中文欄名、接白話；直達被擋每對一行，其餘同一句合併欄名；
// 這裡只照印，不自己對欄位路徑（原本的英文路徑在 paths，技術細節才用得到）。
function problemLines(problems) {
  return (problems || []).map((problem) => problem.text).join("\n");
}
function numberField(id, label, value) {
  const wrap = document.createElement("label");
  wrap.textContent = label;
  const input = document.createElement("input");
  input.id = id; input.type = "number"; input.step = "any";
  const shown = inputValues[id];
  input.value = shown?.value === value ? shown.text : value;
  input.dataset.originalValue = JSON.stringify(value);
  input.addEventListener("input", () => { delete input.dataset.originalValue; });
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
    if (prefix === "speaker") {
      const note = document.createElement("span"); note.id = `height-${name}`;
      note.className = "height-problem notice"; note.setAttribute("aria-live", "polite");
      row.append(note);
    }
    $(target).append(row);
  }
}
function renderForm() {
  requireInputMatch();
  const furnitureCount = scheme.furniture?.length ?? 0;
  $("furniture-notice").hidden = furnitureCount === 0;
  // 第八支已能編輯固定角色；其他原件在「改家具」打開的那一塊唯讀列出，存檔仍完整保留。
  $("furniture-notice").textContent = furnitureCount ? `這份方案有 ${furnitureCount} 件家具：按「改家具」修改本頁支援的家具；其餘唯讀列出並原樣保留。平面圖與側面圖照方案檔畫。` : "";
  renderSpeakerSetupNotice();
  $("save-id").value = scheme.scheme_id;
  $("source-model").value = scheme.source_model;
  $("room-fields").replaceChildren();
  for (const [key, label] of [["Lx", "長 Lx（公尺）"], ["Ly", "寬 Ly（公尺）"], ["Lz", "高 Lz（公尺）"]])
    $("room-fields").append(numberField(`room-${key}`, label, scheme.scene.room_m[key]));
  renderWalls();
  $("scattering").replaceChildren();
  for (const name of Object.keys(wallNames))
    $("scattering").append(numberField(`scatter-${name}`, `${wallNames[name]}散射`, scheme.scene.scattering_by_wall?.[name] ?? ""));
  $("use-scattering").checked = scheme.scene.scattering_by_wall !== null && scheme.scene.scattering_by_wall !== undefined;
  rows("speakers", Object.entries(scheme.speakers), "speaker", speakerName);
  rows("receivers", scheme.receiver_set.points.map((point) => [point.receiver_id,
       {x: point.position_m[0], y: point.position_m[1], z: point.position_m[2]}]), "receiver", pointName);
  renderInputSetup();
  roomFollowed = {...scheme.scene.room_m};
  formScheme = scheme;
  renderInputPreview();
  action(updateMultiples);
}
// 喇叭類型、擺法、代表模型與否的中文名（名稱表那一份）；表上缺任何一個就回 null，由呼叫的人整句換備用句。
// 「改喇叭與擺法」裡那句長提示與右卡喇叭那一行的短句（room_facts.js）共用這一份，不各查各的。
function speakerSetupNames(setup) {
  const names = labels.speaker_setup ?? {};
  const keys = [setup.kind, setup.mount, setup.representative ? "representative_model" : "actual_model"];
  return keys.every((key) => Object.hasOwn(names, key)) ? keys.map((key) => names[key]) : null;
}
function renderSpeakerSetupNotice() {
  const setup = scheme.speaker_setup;
  const notice = $("speaker-setup-notice");
  notice.hidden = !setup;
  notice.textContent = "";
  if (!setup) return;
  const names = speakerSetupNames(setup);
  if (!names) {
    // 名稱表沒載到：整句換備用句，不把英文代號逐格代進句型。
    notice.textContent = "這份方案設了喇叭類型與擺法（中文名沒載到）；按「改喇叭與擺法」修改，平面圖與側面圖照方案檔畫箱體。";
    return;
  }
  notice.textContent = `這份方案設了喇叭：${names[0]}、放${names[1]}（${names[2]}）；按「改喇叭與擺法」修改，平面圖與側面圖照方案檔畫箱體。`;
}
// 伺服器回的牆名跟現在不同才換（喇叭搬到靠另一面牆、開了別份方案）；回傳有沒有換，換了由呼叫的人重畫。
function useWallNames(names) {
  if (!names || Object.keys(wallNames).every((wall) => wallNames[wall] === names[wall])) return false;
  for (const wall of Object.keys(wallNames)) wallNames[wall] = names[wall] ?? wallNames[wall];
  return true;
}
// 牆面表單與立面圖重畫時每一格都換成新的。換名的回覆是晚到的（每打一個字問一次）：那時游標若在這兩區的某一格
// （剛點進去、正在打、在區內換格），換掉它，下一個按鍵就落到頁面本身、沒進任何一格（#1145：清空地板阻抗沒清到、
// 填回去沒填進去）。滑鼠按著的時候也一樣：在按下與放開之間重畫，按到的東西被換掉或挪了位置，那一下就沒按到
// （點標籤字焦點落空、窄視窗的按鈕往下移）。所以游標在區內、或滑鼠按著，就先記著不換；離開這兩區、放開滑鼠才換。
const wallAreas = "#walls, #elevation-panel";
let relabelLater = false;
let pressing = false;
function inWallAreas(node) {
  // 焦點離開到不能聚焦的地方（標籤字、空白處）時 relatedTarget 是 null。
  return node instanceof Element && node.closest(wallAreas) !== null;
}
function relabelWallsOnLeave(event) {
  // 在區內換格（Tab、滑鼠點另一格）不算離開：瀏覽器自己換焦點時，這裡跑的當下新的那一格還沒拿到焦點。
  if (relabelLater && !inWallAreas(event.relatedTarget)) relabelWalls();
}
// 改定一格自動檢查（#1265 單元 6）：滑鼠按著時（點別的鈕讓那一格失焦）先不發，放開、那一下點完才發——回來會清掉或重畫
// 平面圖、訊息列冒一行，版面在按下與放開之間跳了，那一下就沒按到。按的那一下自己問過平面圖（儲存、算），這一次就不必再問。
// 按的那一顆問過沒有，比同一把序號 planSeq（refreshPlan 開頭取號）。按的那一顆先鎖頁、等分區縮放才問的時候（設計洞 3），
// 延後的這一次照樣發，但它也先等同一次分區縮放，醒來時已經不是最新那一次（按的那一顆取了更新的號），不送就退出（refreshPlan）。
let autoCheckLater = false, autoCheckAsked = 0;
function autoCheck() {
  if (pressing) { if (!autoCheckLater) autoCheckAsked = planSeq; autoCheckLater = true; return; }
  const asked = autoCheckLater && autoCheckAsked !== planSeq;
  autoCheckLater = false;
  if (!asked) action(() => refreshPlan(undefined, true));
}
// 頁底兩顆（「照現在牆上貼的算一次」、zone_modes.js 的 sendToSearch）按下那一刻、任何 await 之前呼叫（第三趟複查發現 1、2）：
// 記下按的是哪一顆（stopAtCheck 回來時拿它比，使用者等待中把焦點移到別處就不捲、不搶焦點）；按下那一刻那一格失焦、延到放開的
// 那一次自動檢查（含重問限制）當場做掉，不等放開後那個 setTimeout——兩顆之後等到的判決才是按下那一刻的表單問的。
// flushed 是那一次檢查（沒有延後的就是已經好了的空等待）。
function bottomPressed() {
  const pressed = document.activeElement;
  let flushed = Promise.resolve();
  if (autoCheckLater) {
    autoCheckLater = false;
    flushed = action(() => refreshPlan(undefined, true));
  }
  return {pressed, flushed};
}
function relabelWallsAfterPress() {
  pressing = false;
  // 放開之後瀏覽器才發 click（點標籤字這時才把焦點交給那一格）：排到這一下點完，再看游標在不在區內。
  if (relabelLater) setTimeout(() => { if (relabelLater) relabelWalls(); }, 0);
  if (autoCheckLater) setTimeout(() => { if (autoCheckLater) autoCheck(); }, 0);
}
// 帶牆名的幾塊照新名字重畫：牆面表單（連立面圖）重畫自方案，散射那排只換欄名、不動已填的值。
// 重畫會清掉「約 ρc 的幾倍」那幾格、超限總述也是舊名字寫的：兩樣都重問一次（各自有序號丟掉晚到的舊回覆）。
function relabelWalls() {
  if (pressing || inWallAreas(document.activeElement)) { relabelLater = true; return; }
  relabelLater = false;
  renderWalls();
  for (const wall of Object.keys(wallNames)) {
    const input = $(`scatter-${wall}`);
    if (input) input.parentElement.firstChild.textContent = `${wallNames[wall]}散射`;
  }
  refreshZoneLimits();
  action(updateMultiples);
}
function useInputPreview(data) {
  inputPreview = data;
  inputValues = data.field_values;
  inputPreviewScheme = scheme;
  return useWallNames(data.wall_names);
}
function requireInputMatch(withForm = false) {
  if (inputPreviewScheme !== scheme || (withForm && formScheme !== scheme))
    throw new Error("方案與預覽尚未對齊，請稍後再試");
}
function furnitureByRole(role) {
  const entry = inputPreview.furniture_items.find((item) => item.editable && item.role === role);
  return entry ? (scheme.furniture || []).find((item) => item.furniture_id === entry.furniture_id) : undefined;
}
function chooseOptions(select, choices, names, current) {
  select.replaceChildren();
  for (const key of choices) {
    const option = document.createElement("option"); option.value = key;
    option.textContent = lookUp(names, key, "中文名沒載到"); select.append(option);
  }
  // 舊檔可能是放在本頁不能編輯的桌子上；顯示原擺法、不能新選，讀回不偷偷改原設定。
  if (current && !choices.includes(current)) {
    const option = document.createElement("option"); option.value = current; option.disabled = true;
    option.textContent = `${lookUp(names, current, "原擺法")}（原方案）`; select.append(option);
  }
  select.value = current;
}
function renderFurnitureFields(role, item) {
  const area = $(`furniture-${role}-fields`); area.hidden = !item; area.replaceChildren();
  if (!item) return;
  const heading = document.createElement("h4");
  heading.textContent = lookUp(labels.furniture_kinds || {}, item.kind, "家具"); area.append(heading);
  const fields = document.createElement("div"); fields.className = "fields"; area.append(fields);
  for (const key of ["width_m", "depth_m", "height_m"]) {
    let title = lookUp(labels.furniture_fields || {}, key, "尺寸（公尺）");
    if (key === "height_m" && role !== "sofa") title = "板厚（公尺）";
    fields.append(numberField(`furniture-${role}-${key}`, title, item[key]));
  }
  const material = document.createElement("label"); material.textContent = "材質";
  const select = document.createElement("select"); select.id = `furniture-${role}-material`;
  chooseOptions(select, inputDefaults.materials[item.kind], labels.furniture_materials || {}, item.material);
  material.append(select); fields.append(material);
  const positions = document.createElement("div"); positions.className = "fields"; area.append(positions);
  if (role === "cloud") {
    for (const [index, axis] of Object.keys(coordNames).entries())
      positions.append(numberField(`furniture-${role}-${axis}`, `底面中心 ${axis}（公尺）`, item.placement.bottom_center_m[index]));
  } else {
    for (const key of ["forward_m", "left_m", "bottom_height_m"])
      positions.append(numberField(`furniture-${role}-${key}`, lookUp(labels.furniture_fields || {}, key, "位置（公尺）"), item.placement[key]));
  }
  const coordinate = document.createElement("p"); coordinate.id = `furniture-${role}-coordinate`;
  coordinate.setAttribute("aria-live", "polite"); area.append(coordinate);
  const note = document.createElement("p"); note.className = "note";
  note.textContent = inputDefaults.furniture[item.kind].description; area.append(note);
}
function renderInputSetup() {
  if (!inputDefaults) return;
  requireInputMatch();
  for (const role of Object.keys(inputDefaults.roles)) {
    const item = furnitureByRole(role);
    const control = $(role === "table" ? "furniture-table-kind" : `furniture-${role}`);
    if (role === "table") control.value = item?.kind || "none";
    else control.checked = Boolean(item);
    control.disabled = !inputPreview.furniture_controls[role];
    renderFurnitureFields(role, item);
  }
  const setup = scheme.speaker_setup;
  $("use-speaker-setup").checked = Boolean(setup);
  $("speaker-setup-fields").hidden = !setup;
  $("cabinet-fields").replaceChildren();
  if (!setup) return;
  chooseOptions($("speaker-kind"), Object.keys(inputDefaults.representative_speakers), labels.speaker_setup || {}, setup.kind);
  chooseOptions($("speaker-mount"), inputPreview.mounts, labels.speaker_setup || {}, setup.mount);
  $("representative").checked = setup.representative;
  $("representative-state").textContent = lookUp(labels.speaker_setup || {}, setup.representative ? "representative_model" : "actual_model", "中文名沒載到");
  for (const [key, value] of Object.entries(setup.cabinet)) {
    const field = numberField(`cabinet-${key}`, `${lookUp(labels.speaker_setup || {}, key, "箱體尺寸")}（公尺）`, value);
    field.querySelector("input").disabled = setup.representative;
    $("cabinet-fields").append(field);
  }
}
function numericInput(id) {
  const input = $(id);
  if (input.value !== "" && input.dataset.originalValue !== undefined)
    return JSON.parse(input.dataset.originalValue);
  return input.value.trim() === "" ? null : Number(input.value);
}
function collectInputSetup() {
  if (!inputDefaults) return;
  requireInputMatch(true);
  for (const entry of inputPreview.furniture_items) {
    if (!entry.editable) continue; // 沒有格子的原件照原樣帶，不能換成只剩表單三件。
    const role = entry.role;
    const item = (scheme.furniture || []).find((item) => item.furniture_id === entry.furniture_id);
    if (!item) continue;
    for (const key of ["width_m", "depth_m", "height_m"]) item[key] = numericInput(`furniture-${role}-${key}`);
    item.material = $(`furniture-${role}-material`).value;
    if (role === "cloud") item.placement.bottom_center_m = Object.keys(coordNames).map((axis) => numericInput(`furniture-${role}-${axis}`));
    else for (const key of ["forward_m", "left_m", "bottom_height_m"])
      item.placement[key] = numericInput(`furniture-${role}-${key}`);
  }
  if (scheme.speaker_setup) for (const key of Object.keys(scheme.speaker_setup.cabinet))
    scheme.speaker_setup.cabinet[key] = numericInput(`cabinet-${key}`);
}
async function updateInputPreview() {
  if (inputBusy) return;
  requireInputMatch(true);
  const asked = ++previewAsked;
  const sentScheme = scheme;
  const stale = () => asked !== previewAsked || sentScheme !== scheme || inputBusy;
  let data;
  try {
    data = await api("/api/input-preview", "POST", collect());
  } catch (error) {
    // 過期那一次的失敗不報：已經有更新的交接或預覽，舊錯誤蓋掉新訊息只會誤導（複查三）。
    if (stale()) return;
    document.querySelectorAll("[data-open-area]").forEach(node => { node.textContent = ""; });
    throw error;
  }
  if (stale()) return;
  if (useInputPreview(data)) relabelWalls();
  renderInputPreview();
}
function renderInputPreview() {
  requireInputMatch();
  const data = inputPreview;
  document.querySelectorAll("[data-open-area]").forEach(node => { node.textContent = data.open_area_texts?.[node.id] || ""; });
  if (scheme.speaker_setup)
    chooseOptions($("speaker-mount"), data.mounts, labels.speaker_setup || {}, scheme.speaker_setup.mount);
  for (const role of Object.keys(inputDefaults.roles)) {
    const label = $(`furniture-${role}-coordinate`);
    if (label) label.textContent = data.coordinates[role] || "位置尚未換算，請填好欄位後檢查";
  }
  const list = $("readonly-furniture"); list.replaceChildren(); list.hidden = !data.readonly.length;
  for (const item of data.readonly) {
    const row = document.createElement("li"); row.textContent = item.text; list.append(row);
  }
  for (const name of Object.keys(scheme.speakers)) $( `height-${name}`).textContent = data.height_problems[name] || "";
  // 喇叭高度跟擺法推出值不同的提醒寫在「座標數字」摺疊裡那一支的 z 旁邊（#1265）：擺法是在「改喇叭與擺法」改的，
  // 摺疊收著就看不到；有提醒就自己打開，不讓它藏著。
  if (Object.keys(scheme.speakers).some(name => data.height_problems[name])) $("coordinates-fold").open = true;
  $("stand-height-hint").textContent = data.stand_hint;
  // #753：建搜尋才查的三條擺放問題先在這裡提醒，不擋存檔；字由伺服器寫。
  const warnings = $("placement-warnings"); warnings.replaceChildren(); warnings.hidden = !data.placement_warnings.length;
  for (const text of data.placement_warnings) {
    const item = document.createElement("li"); item.textContent = text; warnings.append(item);
  }
  renderRoomFacts();
}
// held：呼叫的人已經用 holdInput 鎖著整頁（從範本開新的：存檔、重讀清單、打開一路鎖，審查第 4 條）；這裡不再上鎖、
// 也不在做完時解鎖，解鎖與解鎖後補問預覽交給那一道鎖。
async function changeInputScheme(load, held = false) {
  if (!held && inputBusy) return false;
  inputBusy = true;
  let failed = false;
  $("input-setup").setAttribute("aria-busy", "true");
  ++previewAsked;
  ++multiplesAsked;
  // 開檔、家具、快捷、喇叭設定與開機共用交接；等待時三樣都留在原版本。
  if (!held) document.querySelector("main").inert = true;
  try {
    const resize = window.pendingRoomResize();
    if (resize) await resize;
    const loading = load();
    // 先收完現有格子，再把勾選與選單畫回原版本；候選值已在請求裡，成功才顯示。
    if (scheme) { renderInputSetup(); renderWalls(); }
    const nextScheme = await loading;
    const nextPreview = await api("/api/input-preview", "POST", nextScheme);
    // 這一段沒有等待點：方案、預覽與格子一起換，舊回覆已由序號作廢。
    // 上一份的限制判決對的是上一份方案：先標判決在路上再畫（#1265 單元 3 之二收尾追加設計洞；打開、換牆面材料、
    // 改構造預設、套實測阻抗這幾個呼叫的人換成功之後都接著 refreshPlan，一定會重問；打開時接回計算進度失敗也照問，第二趟 B3）。
    staleZoneLimits();
    scheme = nextScheme;
    useInputPreview(nextPreview);
    renderForm();
    return true;
  } catch (error) {
    if (scheme && formScheme === scheme) {
      renderInputSetup();
      renderWalls();
      renderInputPreview();
    }
    // 開頁那道門還關著（開頁途中、空畫面）：表單上那份是沒露出來的開機範本，不算「原方案」，不講「保留原方案」（#1265 單元 5c 第二趟）。
    const kept = !scheme ? "，請重新載入頁面" : "inputStart" in document.body.dataset ? "" : "，保留原方案";
    say(`方案與預覽換不過去${kept}：${errorText(error)}`, "notice");
    failed = true;
    return false;
  } finally {
    $("input-setup").removeAttribute("aria-busy");
    if (!held) {
      document.querySelector("main").inert = false;
      inputBusy = false;
      // 失敗時原方案的格子已收進去、等在路上的那次預覽也被作廢：解鎖後替目前這份補問一次，換算座標才不停在舊值。
      // 補問又失敗只換訊息列：「打開」「開一份新的」旁邊接著寫的「沒有打開…」「沒有開新的…」照留（#1188 第 8 條）。
      if (failed && scheme && formScheme === scheme) previewAfterUnlock();
    }
  }
}
// 「儲存」「另存新名字」「計算這個方案」跟換表單（打開、從範本開新的、編輯、切牆、換材料）共用同一道鎖（#1187）：
// 存到伺服器回覆之前整頁鎖住、換不了表單，換表單的時候也存不了。呼叫的人先看 inputBusy、自己講被擋的那一句。
// 「儲存」「另存新名字」的鎖只包到伺服器收下方案與做法為止；之後重讀幾份清單不鎖：訊息列已經寫「存好了」，下一下要按得到。
// 「計算」把重讀清單與開算也包在鎖裡；「拿去搜尋」（zone_modes.js 的 sendToSearch）一路包到離開這一頁（#1188 第 5 條）。
// 鎖頁會讓游標離開原本那一格：解鎖後放回去（那一格還在、游標也沒被放到別處才放）。
// 鎖著的時候晚到的預覽一律作廢（updateInputPreview 的 stale）：解鎖後替目前這份補問一次。
const busyWords = "上一個動作還沒做完";
let refusedBeside = null;
async function holdInput(work) {
  inputBusy = true;
  refusedBeside = null;
  // 「這間房貼什麼」那一頁（#zone-page，zone_overview.js）不在 main 裡：一起鎖，存檔途中那一頁也點不了、套用不了。
  const main = document.querySelector("main"), page = $("zone-page"), focused = document.activeElement;
  main.inert = true; page.inert = true;
  // 做的事自己失敗（伺服器不收這一次存檔）：原因在這裡寫進訊息列——同一句，原本是丟出去給 action 寫，
  // 那會排在下面寫回被擋那一句之後、又把它清掉（#1188 第 7 條）。回 undefined：呼叫的人都當「沒做成」。
  try {
    const resize = window.pendingRoomResize();
    if (resize) await resize;
    return await work();
  } catch (error) { say(errorText(error), "notice"); } finally {
    main.inert = false; page.inert = false;
    inputBusy = false;
    if (focused instanceof HTMLElement && focused.isConnected && document.activeElement === document.body)
      focused.focus({preventScroll: true});
    // 存檔途中被擋下的「打開」「開一份新的」：存檔自己的訊息會清掉旁邊的字，做完再寫回去，不讓那一下看起來像沒按到。
    if (refusedBeside) sayBeside(refusedBeside.note, refusedBeside.text, "notice");
    refusedBeside = null;
    if (scheme && formScheme === scheme) previewAfterUnlock();
  }
}
// 解鎖後補問的那一次預覽不是使用者按出來的：它失敗（表單有一格收不下）只換訊息列那一句，
// 按鈕旁邊剛寫的結果（沒有另存的原因、被擋下的那一句）照留（#1193）。旁邊的字在 say() 前一刻才抄：
// 等回覆的時候使用者自己清掉的就不寫回去。
async function previewAfterUnlock() {
  try { await updateInputPreview(); } catch (error) {
    const beside = ["open-note", "new-scheme-note", "save-as-note"].map(note => [note, $(note).textContent, $(note).className]);
    say(errorText(error), "notice");
    for (const [note, text, kind] of beside) sayBeside(note, text, kind);
  }
}
function refuseWhileBusy(note, text) {
  refusedBeside = {note, text};
  sayBeside(note, text, "notice");
}
// 按下那一刻的方案與每一區做法各抄一份（這裡沒有等待點）；後面每個等待點都用這一份，不再讀畫面。
// samePage：回覆回來時畫面還是不是按下時那一份（鎖著的時候一定是；它是鎖漏掉時的第二道，不是則不動畫面上的名字）。
function pressedCopy(designName) {
  const live = scheme, generation = zoneDesignGeneration;
  const revision = editRevision;
  const taken = structuredClone(collect());
  return {document: taken, sendDesign: prepareZoneDesignSave(designName ?? taken.scheme_id),
          sameEdits: () => revision === editRevision,
          samePage: () => scheme === live && generation === zoneDesignGeneration};
}
async function editInput(actionName, value, wall, held = false) {
  if (!await changeInputScheme(async () => {
    const data = await api("/api/input-edit", "POST", {scheme: wall ? wallDocument() : collect(), action: actionName, value, ...(wall ? {wall} : {})});
    return data.scheme;
  }, held)) return false;
  markStale();
  await refreshPlan();
  return true;
}
async function updateMultiples() {
  const asked = ++multiplesAsked;
  const walls = formWalls().filter(name => wallMaterial(name).kind === "constant_impedance" && !measuredDrafts[name]);
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
// 訊息列換了新的一句，「打開」「開一份新的」與「另存新名字」旁邊上一次的結果就過時了，一起清掉。
// 改了就自動檢查那一次（refreshPlan 的 auto，#1265 單元 6）寫訊息列時 besides 是 false：那是離開一格順手做的，不是按了哪一顆，
// 按鈕旁邊那幾句（沒打開的原因、沒另存、沒開新的）與存檔那一句照留，要等下一次按鈕自己的回話才換。
// 訊息列寫過幾次（自動檢查那一次要知道自己出發之後有沒有別的回話寫進來）。
let saidCount = 0;
function say(text, kind, besides = true) {
  saidCount += 1;
  $("messages").textContent = text;
  delete $("messages").dataset.echo;
  if (besides) $("save-feedback").textContent = "";
  $("messages").className = kind;
  // 「這間房貼什麼」那一頁開著時這一列是藏起來的：警示照抄一份到那一頁（zone_overview.js）。
  window.zonePageSay?.(text, kind);
  // 結果抽屜開著時訊息列被遮住：封存、搬回、刪除的回話照抄一份到抽屜上方（#1265 單元 4）。
  if (!$("results-drawer").hidden) { $("results-note").textContent = text; $("results-note").className = kind; }
  if (besides) for (const note of ["open-note", "new-scheme-note", "save-as-note"]) sayBeside(note, "", "");
}
// 「打開」與「另存新名字」在頁面最上面，底下的訊息列從那裡看不到：它們的結果（沒打開的原因、空名字、
// 檢查沒過、撞名、存好了）寫在按鈕旁邊，用那一列自己的字眼。
function sayBeside(note, text, kind) {
  $(note).textContent = text;
  $(note).className = kind;
}
// 改房間尺寸，喇叭與聆聽點跟著走（老闆 11-26「一看房間就跑出去」）：長、寬照比例移（原本在房間裡的改完還在裡面、
// 相對位置不變；原本就在房外的不動）；高度不跟著比例（耳朵、喇叭高度跟天花板無關），只有新高度放不下時才照比例降。
// 主位周圍那幾個點是量主位附近的，間距不縮：跟著主位平移。比的是上一次收下的尺寸（roomFollowed），空白或打錯不算，改對時從上一次的算。
let roomFollowed = null;
function followRoomResize(event) {
  const key = event.target?.id?.match(/^room-(L[xyz])$/)?.[1];
  if (!key || !roomFollowed) return;
  const before = roomFollowed[key], after = numericInput(event.target.id);
  if (!(Number.isFinite(after) && after > 0) || !(before > 0) || after === before) return;
  roomFollowed[key] = after;
  const axis = {Lx: "x", Ly: "y", Lz: "z"}[key];
  // 格子顯示到公釐，存的是沒四捨五入的值（同伺服器排過位數的格子：dataset.originalValue），縮了再放回去不會一路差下去。
  const exact = value => Math.round(value * 1e9) / 1e9;
  // 原本就在舊房間外面的點不動：常見的是先把喇叭搬到外面、再把房間放大到裝得下它。
  const moved = value => value < 0 || value > before ? value : axis !== "z" || value > after ? exact(value * after / before) : value;
  const set = (id, value) => {
    const input = $(id);
    input.value = String(Math.round(value * 1000) / 1000);
    input.dataset.originalValue = JSON.stringify(value);
  };
  for (const name of Object.keys(scheme.speakers)) {
    const id = `speaker-${name}-${axis}`, value = numericInput(id);
    if (Number.isFinite(value)) set(id, moved(value));
  }
  const points = scheme.receiver_set.points;
  const primary = points.find(point => point.role === "primary");
  const primaryId = primary ? `receiver-${primary.receiver_id}-${axis}` : null;
  const primaryBefore = primaryId ? numericInput(primaryId) : null;
  const shift = Number.isFinite(primaryBefore) ? moved(primaryBefore) - primaryBefore : 0;
  for (const point of points) {
    const id = `receiver-${point.receiver_id}-${axis}`, value = numericInput(id);
    if (!Number.isFinite(value)) continue;
    set(id, point.role === "surrounding" ? exact(value + shift) : moved(value));
  }
}
function collect() {
  requireInputMatch(true);
  scheme.scheme_id = $("save-id").value;
  scheme.source_model = $("source-model").value;
  const numeric = numericInput;
  for (const key of Object.keys(scheme.scene.room_m)) scheme.scene.room_m[key] = numeric(`room-${key}`);
  collectWallMaterials();
  scheme.scene.scattering_by_wall = $("use-scattering").checked ? Object.fromEntries(
    Object.keys(wallNames).map((name) => [name, numeric(`scatter-${name}`)])) : null;
  for (const [name, point] of Object.entries(scheme.speakers))
    for (const axis of Object.keys(coordNames)) point[axis] = numeric(`speaker-${name}-${axis}`);
  for (const receiver of scheme.receiver_set.points)
    receiver.position_m = Object.keys(coordNames).map((axis) => numeric(`receiver-${receiver.receiver_id}-${axis}`));
  collectInputSetup();
  return scheme;
}
// 回傳存成了沒有；鎖在這裡面拿。「拿去搜尋」（zone_modes.js）不叫這一支：它自己拿鎖、叫 storeScheme，鎖要留到離開這一頁。
async function save() {
  const cut = window.pendingMirrorCut();
  if (cut) await cut;
  if (inputBusy) { say(`${busyWords}，這一次沒有儲存；等它做完再按一次`, "notice"); return false; }
  const stored = await holdInput(storeScheme);
  $("save-feedback").textContent = $("messages").textContent;
  $("save-feedback").className = $("messages").className;
  if (!stored) return false;
  await refreshAfterSave(stored.message);
  return true;
}
// 存方案與每一區做法（呼叫的人已經拿著鎖）；存成回方案代號與伺服器那一句，圖面檢查沒過回 null。
async function storeScheme() {
  const taken = pressedCopy();
  const document = taken.document;
  const headers = openedId === null || openedId !== document.scheme_id ? {"If-None-Match": "*"} : {};
  // 圖面檢查連不上也算沒過（#1265 單元 6 第二趟發現 4）：原因寫進訊息列，一樣回 null。
  let passed;
  try { passed = await refreshPlan(document); } catch (error) { say(errorText(error), "notice"); return null; }
  if (!passed) return null;
  const saved = await api(`/api/schemes/${encodeURIComponent(document.scheme_id)}`, "PUT", document, headers);
  if (taken.samePage()) {
    openedId = document.scheme_id;
    $("scheme-archive-notice").hidden = true;
    window.forgetClearUndo();  // 存好了：「清掉這一面的切法」的復原只到下一個改動之前（#1265 單元 3 審查真錯 3）
  }
  say(saved.message, "ok");
  // 每一區的做法跟著方案存（#1105）；方案已經存好，做法沒存成要照講、不算存檔失敗。
  try {
    await taken.sendDesign();
    if (taken.samePage() && taken.sameEdits()) { runEdited = false; dirtyField = ""; }
  } catch (error) {
    say(`${saved.message}；每一區的做法沒存成：${errorText(error)}`, "notice");
  }
  renderInputBars();
  return {id: document.scheme_id, message: saved.message};
}
// 頁底兩顆（「照現在牆上貼的算一次」、zone_modes.js 的「交給程式最佳化 ▸」）共用（第二趟發現 4）：鎖著存檔，存成了做
// after(stored, before 拿到的東西)；存之前平面圖沒過（含連不上：storeScheme 回 null）就在鎖放開之後停在平面圖問題上
// （check_state.js 的 stopAtCheck）。兩顆不各自判；別的失敗（存檔被拒、做的事丟錯）holdInput 已經寫進訊息列，回 undefined。
const PLAN_STOPPED = Symbol("平面圖擋下");
async function storeThen(press, after, before = () => null) {
  const out = await holdInput(async () => {
    const context = before();
    const stored = await storeScheme();
    return stored === null ? PLAN_STOPPED : after(stored, context);
  });
  if (out !== PLAN_STOPPED) return out;
  await stopAtCheck("plan", press.pressed);
  return undefined;
}
// 存好之後重讀方案清單與結果清單（不鎖頁）。
async function refreshAfterSave(message) {
  await loadSchemeList();
  // #755：同名方案改過，清單與封存區那幾筆要馬上標「改之前的方案算的」，不等重新整理。
  // 清單讀不到不算存檔失敗：方案已經存好，後面的「計算」照樣要開。
  try { await loadResultList(); } catch (error) {
    say(`${message}；結果清單更新不了：${errorText(error)}`, "notice");
  }
}
async function loadSchemeList() {
  const data = await api("/api/schemes");
  schemeNames = data.schemes;
  savedTimes = data.saved_at || {};
  // 使用者正要點選單時，晚到的清單不能把焦點所在的按鈕換掉；下次展開就畫新清單。
  if ($("scheme-menu").hidden || !$("home-schemes").childElementCount) drawSchemeMenu();
  // 右卡「改造前」那一格的選單列的是方案清單：清單晚到也要換上。
  renderRoomEditors();
}
// 回有沒有真的打開，讓帶名字的連結知道是否繼續進入分區頁。held 見 changeInputScheme：拿著鎖的呼叫的人（從範本開新的、
// 空畫面搬回）不會走被鎖擋下那一支，沒打開時原因就是 changeInputScheme 剛寫進訊息列的那一句，回來之前沒有等待點。
// quiet：沒打開時不寫「打開」旁邊（開頁那一串 openStartScheme：門還關著，原因收起來、開好一份之後一次寫，5c 第二趟審查第 1 條）。
async function openScheme(name, held = false, quiet = false) {
  if (!name) return false;
  if (!held && inputBusy) {
    refuseWhileBusy("open-note", `${busyWords}，沒有打開「${name}」；等它做完再按一次「打開」`);
    return false;
  }
  // 每一區的做法跟著方案讀（#1105）；材料庫可能在別的分頁改過，一起重讀。三樣都在鎖頁期間讀完，
  // 成功才同步套上：不准在表單已換、做法還沒換的空檔放使用者按「儲存」。
  let zoneState;
  if (!await changeInputScheme(async () => {
    // 方案讀得到才讀做法：做法那一頁要拿方案比對分區，方案壞了它也只會回錯，不多打一個注定失敗的請求。
    const opened = await api(`/api/schemes/${encodeURIComponent(name)}`);
    const [design, library] = await Promise.all([fetchZoneDesign(name), fetchZoneLibrary()]);
    zoneState = {design, library};
    return opened.scheme;
  }, held)) {
    // 存著的那一份現在檢查不過（或讀不出）：表單維持原樣，原因寫在訊息列（changeInputScheme 剛寫的），也寫在「打開」旁邊。
    if (!quiet) sayBeside("open-note", `沒有打開「${name}」：${$("messages").textContent}`, "notice");
    return false;
  }
  for (const wall of Object.keys(measuredDrafts)) delete measuredDrafts[wall];
  applyZoneDesign(zoneState.design);
  applyZoneLibrary(zoneState.library);
  // 牆改回「固定阻抗」時填的預設值跟著打開的這一份（#1265 單元 5b）：整面固定阻抗的那幾面記它自己的值
  // （從第一階段範本開的新方案就是那份範本的 30ρc），別種材料或切了區的面用開頁範本的值（不看上一份開過什麼）。
  fixedWallDefaults = structuredClone(startupWallDefaults);
  for (const wall of formWalls()) {
    const material = wallMaterial(wall);
    if (material.kind === "constant_impedance") fixedWallDefaults[wall] = material.impedance_pa_s_m;
  }
  renderWalls();
  // 第 2 段先看前牆（#1265 單元 5c，elevation.js 的 showFrontWall；分不出前後就照舊）。
  window.showFrontWall();
  openedId = name;
  runEdited = false; dirtyField = "";
  $("scheme-archive-notice").hidden = true;
  $("result-link").hidden = true;
  $("result-stale").hidden = true;
  renderInputBars();
  const running = formRunning();
  // 接回計算進度失敗也照樣檢查（#1265 單元 3 之二收尾第二趟 B3）：上一份的判決在 changeInputScheme 已作廢、要靠這一次 refreshPlan
  // 重問限制。兩邊的錯各自接住、最後一起寫（第三趟 E2）：圖面檢查丟錯不吞掉進度那一句，圖面檢查寫好的那一句也不被進度那一句蓋掉。
  let followFailed = null;
  if (running && running.run_id !== runId) {
    try { await follow(running.run_id); } catch (error) { followFailed = errorText(error); }
  }
  let planned;
  try { planned = await refreshPlan(); } catch (error) {
    throw followFailed === null ? error : new Error(`${followFailed}；${errorText(error)}`);
  }
  // 只有進度讀不到：方案已經打開好了，照樣回 true（?zones= 連結照樣進「這間房貼什麼」，審查設計洞 2），原因寫在訊息列。
  if (followFailed !== null) say(planned ? followFailed : `${followFailed}；${planProblems}`, "notice");
  return true;
}
// 「從範本開一份新的」（#1265 單元 5b，取代「載入範本」）：頂條選單打開的那一塊與完全沒有方案時的空畫面共用這一份表單、
// 這一支。先取名、先存（名字照「另存新名字」同一套規則；存檔帶 If-None-Match: *，撞名伺服器也擋），存成了才用「打開」
// 那條路打開它：畫面上從頭到尾不會有一份沒名字、沒存的範本。擋下來的原因寫在這一塊最底下，表單不動。
// 一次只准一個（第三趟不變式一）：從按下到結束（開好、沒開成、詢問裡按取消）newSchemePending 一直是真、那一塊 inert，
// 只在最外層的 finally 收；這段時間再按（真按點不到，頁面程式按）照鎖頁那句擋下，「取消」、Esc（home.js）也不關那一塊。
// 中間「存了再走」那一段 save() 故意先解鎖再重讀清單，所以光看 inputBusy 擋不住，要看這個旗子。
const refusedNew = `${busyWords}，沒有開新的；等它做完再按一次「開一份新的」`;
let newSchemePending = false;
async function newFromTemplate() {
  const note = "new-scheme-note", name = $("new-scheme-id").value.trim();
  const picked = document.querySelector('input[name="new-template"]:checked');
  const label = picked.closest("label").querySelector("b").textContent;
  if (inputBusy || newSchemePending) { refuseWhileBusy(note, refusedNew); return; }
  const refused = !name ? "「新方案的名字」這一格還空著：先填名字，再按「開一份新的」" :
    !newNameShape.test(name) ? newNameHint.replace("「新名字」", "「新方案的名字」") :
    schemeNames.includes(name) ? `已經有叫「${name}」的方案；換一個名字（要改那一份，從「換一個方案」打開它）` : "";
  if (refused) {
    sayBeside(note, `沒有開新的：${refused}`, "notice");
    $("new-scheme-id").focus();
    return;
  }
  newSchemePending = true;
  $("new-scheme").inert = true;
  // 每一步回的結果（見 newSchemeSteps）；丟錯也是一條沒開成的出口。
  let outcome;
  try { outcome = await newSchemeSteps(name, picked.value, label); }
  catch (error) { outcome = {text: `沒有開新的：${errorText(error)}`}; }
  finally { newSchemePending = false; $("new-scheme").inert = false; }
  if (!outcome) return;
  // 每一條沒開成的出口都在那一塊裡講（不變式三）：等的時候那一塊被別的按鈕（換一個方案、另存新名字…）收起來，
  // 也打開回來，原因不寫在看不見的地方。
  if (document.body.dataset.inputStart !== "empty") { closeBarBoxes(); $("template-box").hidden = false; }
  // 詢問裡按了取消：什麼都沒做，游標回「開一份新的」（關詢問的時候那一塊還 inert，瀏覽器放不回去）。
  if (outcome.cancelled) { $("new-scheme-go").focus(); return; }
  // 被別的動作的鎖擋下：照 refuseWhileBusy 記下來，那一道鎖收尾時再寫回一次（它自己的訊息會清掉旁邊的字）。
  if (outcome.busy) refuseWhileBusy(note, outcome.text);
  else sayBeside(note, outcome.text, "notice");
  $("new-scheme-id").focus();
}
// newFromTemplate 的每一步（呼叫的人拿著 newSchemePending）：開好了回 undefined；詢問裡按了取消回 {cancelled}；
// 沒開成回 {text}（寫在那一塊的那一句），被別的動作的鎖擋下另帶 busy。
async function newSchemeSteps(name, template, label) {
  const note = "new-scheme-note";
  sayBeside(note, "", "");
  // 開著的那一份有改動沒存：跟「換一個方案」同一道詢問；問在存新檔之前，按「取消」就什麼都沒有存。
  const left = await canLeaveForm();
  // 等詢問的時候（「存了再走」重讀清單那段沒鎖）別的動作拿了鎖：照鎖頁那句擋下，不疊鎖（不變式二）。
  // 使用者按了取消就不講話；存檔沒成跟取消分開講（不變式三，leaveProblem 見 home.js 的 canLeaveForm）。
  if (!left && !leaveProblem) return {cancelled: true};
  if (inputBusy) return {text: refusedNew, busy: true};
  if (!left) return {text: `沒有開新的：開著的這一份沒有存成（${leaveProblem}）；改好再按一次「開一份新的」，或在詢問裡選「不存走」`};
  let problem = "", stored = false;
  // 要範本、存檔、重讀清單、打開一路鎖著（審查第 4 條）：中間不留空窗給第二下「開一份新的」「打開」「儲存」「另存」「計算」。
  const opened = await holdInput(async () => {
    try {
      const example = await api(`/api/example?template=${encodeURIComponent(template)}`);
      await api(`/api/schemes/${encodeURIComponent(name)}`, "PUT", {...example.scheme, scheme_id: name}, {"If-None-Match": "*"});
    } catch (error) { problem = errorText(error); return false; }
    stored = true;
    // 清單讀不到不算失敗：新的那一份已經存好，照樣打開。
    try { await loadSchemeList(); } catch { /* 下一次讀清單再補 */ }
    try {
      if (await openScheme(name, true)) return true;
      // 沒打開的原因是 changeInputScheme 剛寫進訊息列的那一句（held 不走被鎖擋下那一支，中間也沒有等待點）。
      problem = $("messages").textContent;
      return false;
    } catch (error) {
      // 已經換成新的那一份、只是之後接進度或圖面檢查丟錯：算打開了，原因留在訊息列。
      if (openedId === name) { say(errorText(error), "notice"); return true; }
      problem = errorText(error);
      return false;
    }
  });
  if (!opened) return {text: stored ? `「${name}」已經存好，但沒有打開：${problem}` : `沒有開新的：${problem || $("messages").textContent}`};
  // 開好了：holdInput 的 finally 剛把鎖著時被擋下的那一句寫回這一塊，那一句已經過時，清掉（不變式四）。
  sayBeside(note, "", "");
  $("new-scheme-id").value = "";
  closeBarBoxes();
  leaveEmptyStart();
  sayBeside("open-note", `已從範本「${label}」開一份新的「${name}」，存好了`, "ok");
  refreshHome();
}
// 空畫面（#1265 單元 5b）：開頁時已存的方案一份都沒有（或清單讀不到）。表單搬進那張卡，頂條與兩段由 CSS 照
// body 的 data-input-start 藏起來（main 不藏：結果抽屜住在裡面，卡片底下「看已封存的方案」要打得開它）。
// 這時還沒開任何方案，inputReady 不設；開了一份（從範本開新的、從封存區搬回）才收掉空畫面、設 inputReady。
// unopened（#1265 單元 5c）：清單裡有方案、但每一份都打不開時，每一份沒打開的原因；寫在卡片上，不當成沒有方案。
// missing：網址帶的名字不在清單裡那一句（不算進份數）。
function enterEmptyStart(listProblem, unopened = [], missing = []) {
  document.body.dataset.inputStart = "empty";
  $("empty-slot").append($("new-scheme"));
  $("empty-start").hidden = false;
  $("empty-list-note").hidden = listProblem === null && !unopened.length && !missing.length;
  $("empty-list-note").textContent = listProblem !== null ?
    `方案清單讀不到（${listProblem}），不知道有沒有存過的方案：先確認本機的網頁伺服器開著，再重新整理這一頁。` +
    "也可以直接從範本開一份新的；名字跟已存的撞到時，伺服器會擋下來、不蓋掉。" :
    missing.map(line => line + "。").join("") + (unopened.length ? `存著的 ${unopened.length} 份方案都打不開：${unopened.join("；")}。` +
    "可以先從範本開一份新的（名字不能跟打不開的那幾份一樣），或修好那幾份的檔案再重新整理這一頁。" : "");
  // 結果抽屜的那一句（drawResultRows）照空畫面的說法寫。
  drawResultRows();
}
function leaveEmptyStart() {
  if (document.body.dataset.inputStart !== "empty") return;
  $("template-box").append($("new-scheme"));
  $("empty-start").hidden = true;
  showInputPage();
}
// 開頁的那道門（#1265 單元 5b 第二趟，審查第 5 條；單元 5c 用同一道）：body 的 data-input-start 開頁就是 "loading"
// （index.html 寫好的），判定完才換——"empty" 是空畫面，拿掉是照常的輸入頁。藏哪些東西只看這一格（home.css 一條規則：
// 第 1、2 段、頁底、頂條按鈕、「這間房貼什麼」），還沒判定時只有「讀取中…」那一行。
// 拿掉的那一刻第 1 段剛看得到：藏著時畫的平面圖量不到字框（字疊在收音點上），照同一份重畫一次（審查第 2 條）。
function showInputPage() {
  delete document.body.dataset.inputStart;
  drawShownPlan();
  markInputReady();
}
function markInputReady() {
  document.body.dataset.inputReady = "true";
  document.dispatchEvent(new Event("input-ready"));
}
async function saveAs() {
  const name = $("save-as-id").value.trim();
  if (!name || !newNameShape.test(name)) {
    sayBeside("save-as-note", name ? `沒有另存：${newNameHint}` :
      "「新名字」這一格還空著：先填新名字，再按「另存新名字」", "notice");
    $("save-as-id").focus();
    return;
  }
  if (inputBusy) {
    sayBeside("save-as-note", `沒有另存：${busyWords}，等它做完再按一次「另存新名字」`, "notice");
    return;
  }
  if (!await holdInput(() => storeSchemeAs(name))) return;
  await loadSchemeList();
  // 首頁第一屏「已存的方案」跟著重讀（同「儲存」；它自己有序號丟掉晚到的舊回覆，讀不到只寫在它那一行）。
  // 另存一定是新名字，結果清單與封存區不會變，不多問那兩趟。
  refreshHome();
}
// 存成新名字（呼叫的人已經拿著鎖）；回存成了沒有。
async function storeSchemeAs(name) {
  let saved, taken;
  try {
    // 先抄、後等：存成新名字的是按下那一刻表單上那一份（#1187）。
    taken = pressedCopy(name);
    // 檢查沒過：問題已經寫進下面的訊息列（伺服器寫好的白話），旁邊照抄同一段。
    if (!await refreshPlan(taken.document)) throw new Error(planProblems);
    // 方案代號欄只顯示現在開著哪一份：伺服器存好了才換成新名字，存失敗就維持原樣。
    saved = await api(`/api/schemes/${encodeURIComponent(name)}`, "PUT",
      {...taken.document, scheme_id: name}, {"If-None-Match": "*"});
  } catch (error) {
    say(errorText(error), "notice");
    sayBeside("save-as-note", `沒有另存：${errorText(error)}`, "notice");
    return false;
  }
  if (taken.samePage()) {
    scheme.scheme_id = name; $("save-id").value = name; openedId = name;
    window.forgetClearUndo();  // 另存成新名字：舊方案清掉前的區不准復原進新方案（#1265 單元 3 審查真錯 3）
    $("result-link").hidden = true; $("result-stale").hidden = true;
    $("scheme-archive-notice").hidden = true;
  }
  say(saved.message, "ok");
  sayBeside("save-as-note", `「${name}」：${saved.message}`, "ok");
  try {
    await taken.sendDesign();
    if (taken.samePage() && taken.sameEdits()) { runEdited = false; dirtyField = ""; }
  } catch (error) {
    say(`${saved.message}；每一區的做法沒存成：${errorText(error)}`, "notice");
    sayBeside("save-as-note", `「${name}」：${saved.message}；每一區的做法沒存成：${errorText(error)}`, "notice");
  }
  if (taken.samePage() && zoneDesignSaved) closeBarBoxes();
  renderInputBars();
  return true;
}
function markStale(event) {
  ++editRevision;
  runEdited = true;
  const field = event?.target;
  dirtyField = ({"room-Lx": "房間長度", "room-Ly": "房間寬度", "room-Lz": "房間高度"})[field?.id] ||
    field?.labels?.[0]?.firstChild?.textContent?.trim() || "";
  renderInputBars();
  if (!$("result-link").hidden) $("result-stale").hidden = false;
  refreshClearUndo();  // 第 2 段「清掉這一面的切法」的復原只到下一個改動之前（elevation.js）
}
// 結果抽屜的比較（#1265 單元 4）：兩格 A、B，勾一筆填進空的那一格（A 先），取消勾選就空出那一格（A 空出來時 B 遞補成 A）；
// 兩格都有了，其餘的勾選格停用（第三筆擋下、不偷換）：要換一筆，先取消勾選其中一筆。比較網址照舊 A 在前。
function chooseCompare(side, item) {
  compareChoice[side] = item;
  drawCompareChoice();
}
// 放掉比較裡的筆（取消勾選、封存、重讀清單時不在了，三條路都走這裡；第三趟 G1）：只改狀態、不畫。
// 先把符合的每一格一次清完，再做唯一一處遞補——A 空了、B 還在，B 補成 A（「先勾的是 A」照字面成立，第二趟 F4）。
// 畫面由呼叫端畫一次（drawCompareChoice 只畫、不改狀態），不會清到一半就被搬動、漏清另一格。
function dropCompare(gone) {
  for (const side of ["a", "b"]) if (compareChoice[side] && gone(compareChoice[side])) compareChoice[side] = null;
  if (!compareChoice.a && compareChoice.b) [compareChoice.a, compareChoice.b] = [compareChoice.b, null];
}
function drawCompareChoice() {
  const {a, b} = compareChoice, both = Boolean(a && b);
  const differs = both && a.registry_text !== b.registry_text;
  for (const row of $("results-list").querySelectorAll("tr")) {
    const side = ["a", "b"].find(key => compareChoice[key]?.run_id === row.dataset.runId);
    const box = row.querySelector("input[type=checkbox]");
    box.checked = Boolean(side);
    box.disabled = both && !side;
    row.querySelector(".compare-mark").textContent = side ? side.toUpperCase() : "";
    row.classList.toggle("registry-differs", Boolean(differs && side));
  }
  const link = $("compare-go");
  if (both) link.href = `/compare/${a.run_id}/${b.run_id}`; else link.removeAttribute("href");
  link.setAttribute("aria-disabled", String(!both));
  $("compare-warning").hidden = !differs;
  $("compare-hint").textContent = both ? "已勾兩筆（A 是先勾的）；要換一筆，先取消勾選其中一筆" :
    a ? `已勾 A：${a.scheme_id}（${a.finished_text}）` +
    (runFinished(a) ? "" : `・${a.status_text}`) + "；再勾一筆就能比較" : "勾兩筆就能比較";
}
function finishedCell(cell, text) {
  if (!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$/.test(text)) return;
  cell.replaceChildren();
  for (const [index, part] of text.split(" ").entries()) {
    if (index) cell.append(document.createTextNode(" "));
    const span = document.createElement("span");
    span.className = index ? "finished-time" : "finished-date";
    span.textContent = part; cell.append(span);
  }
}
// 結果清單那一列算好了沒（done：算完；none：沒有計算紀錄，結果檔本身就是算好的那一份）：整個網頁只在這裡寫一次，
// 結果表、A／B 那一格、頁底條鎖不鎖、第 1 段右卡結果那一行都叫這一支（#1265 單元 2 審查設計洞 1）。
function runFinished(row) {
  return ["done", "none"].includes(row.run_status);
}
function resultRow(item, archived = false) {
  if (!archived) return drawerRow(item);
  const row = document.createElement("tr");
  row.dataset.runId = item.run_id;
  const finished = runFinished(item);
  row.classList.toggle("not-finished", !finished || item.restore_available === false);
  // 「物理」格只寫白話；計算指紋與程式提交代號是技術細節，滑鼠停在那一格才出現。
  // 說明跟著欄位走、不按字比：壞檔那一列四格都寫「讀不出」，按字比會把說明掛到每一格。
  const values = [[item.scheme_id], [item.finished_text], [item.calculation_text, item.calculation_detail]];
  for (const [index, [value, detail]] of values.entries()) {
    const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    if (index === 1) finishedCell(cell, value);
    if (detail) cell.title = detail;
  }
  // 沒有正常完成的那一筆（失敗、已停止、計算中）：方案代號底下多一行狀態，正常完成的不加字。
  // 不另開一欄：多一欄表格就凸出卡片（實量：表寬 1158、卡片 1100）。
  // 封存動作另有按鈕欄；方案代號保持完整，日期與時刻各自不折、只在兩段之間換行，留出新欄的位置。
  runNotes(row.firstElementChild, item, !finished || item.restore_available === false);
  return row;
}
// 沒有正常完成的那一筆（失敗、已停止、計算中）多一行狀態；#755 同名方案在這筆算完之後改過，再多一行（伺服器給的字）。
function runNotes(cell, item, statusLine) {
  if (statusLine) {
    const status = document.createElement("span");
    status.className = "run-status"; status.textContent = item.status_text;
    cell.append(status);
  }
  if (item.scheme_text) {
    const changed = document.createElement("span");
    changed.className = "run-status scheme-changed"; changed.textContent = item.scheme_text;
    cell.append(changed);
  }
}
// 結果抽屜的一列＝一筆結果（一次計算）：勾選格（勾了標 A／B）、方案名字（現在開著的那一份加註）、
// 一行小字「算完時間；評分：評分設定；物理：…」、看、封存。全程時間在結果頁「技術細節」裡，這裡不放；
// 「物理」那一格的判定（跟現在同不同）照留在小字裡，物理身分與程式提交仍只在滑鼠停留的說明。
function drawerRow(item) {
  const row = document.createElement("tr");
  row.dataset.runId = item.run_id;
  const finished = runFinished(item);
  row.classList.toggle("not-finished", !finished);
  const pick = document.createElement("td"), main = document.createElement("td"), acts = document.createElement("td");
  pick.className = "res-pick"; main.className = "res-main"; acts.className = "res-acts";
  const box = document.createElement("input"), mark = document.createElement("span");
  box.type = "checkbox"; mark.className = "compare-mark";
  box.setAttribute("aria-label", `勾來比較：${item.scheme_id}（${item.finished_text}）`);
  box.onchange = () => {
    const side = ["a", "b"].find(key => compareChoice[key]?.run_id === item.run_id);
    if (side) { dropCompare(picked => picked.run_id === item.run_id); drawCompareChoice(); }
    else if (!compareChoice.a) chooseCompare("a", item);
    else if (!compareChoice.b) chooseCompare("b", item);
    else drawCompareChoice();
  };
  pick.append(box, mark);
  main.append(document.createTextNode(item.scheme_id));
  if (item.scheme_id === openedId) {
    const opened = document.createElement("span");
    opened.className = "opened-mark"; opened.textContent = "（現在開著）";
    main.append(opened);
  }
  const line = document.createElement("small"), when = document.createElement("span");
  line.className = "res-line"; when.className = "res-finished"; when.textContent = item.finished_text;
  finishedCell(when, item.finished_text);
  const registry = document.createElement("span"), physics = document.createElement("span");
  registry.className = "res-registry"; registry.textContent = item.registry_text;
  physics.className = "res-physics"; physics.textContent = item.calculation_text;
  if (item.calculation_detail) physics.title = item.calculation_detail;
  line.append(when, document.createTextNode(" 算完；評分："), registry, document.createTextNode("；物理："), physics);
  main.append(line);
  runNotes(main, item, !finished);
  const link = document.createElement("a");
  // 「看」是連到結果頁的連結，外觀跟同一列的「封存」按鈕一樣（home.css）。
  link.className = "button-link"; link.href = item.result_url; link.textContent = "看";
  const archive = document.createElement("button");
  archive.type = "button"; archive.textContent = "封存"; archive.className = "res-archive";
  archive.setAttribute("aria-label", `封存 ${item.scheme_id}（${item.finished_text}）`);
  archive.onclick = () => action(() => moveResult(item));
  acts.append(link, archive);
  row.append(pick, main, acts);
  return row;
}

function rowButton(row, text, work, danger = false) {
  const cell = document.createElement("td"), button = document.createElement("button");
  button.type = "button"; button.textContent = text;
  if (danger) button.className = "archive-danger";
  button.onclick = () => action(work);
  cell.append(button); row.append(cell);
}
async function moveResult(item, restore = false) {
  if (!restore) return archiveScheme(item.scheme_id);
  const data = await api(`/api/archive/${item.run_id}/restore`, "POST", {});
  await refreshMovedLists(data.message);
}
// 回清單重讀成不成：成了回 null，沒成回原因（5c 第三趟，複查第 3 條：空畫面搬回要把原因寫在看得到的卡片上）。
async function refreshMovedLists(message, tone = "ok") {
  say(message, tone);
  try { await loadSchemeList(); await loadResultList(); } catch (error) {
    say(`${message}；清單更新不了，請重新整理：${errorText(error)}`, "notice");
    return errorText(error);
  }
  return null;
}
async function archiveScheme(name) {
  if (!name) return;
  const preview = await api(`/api/schemes/${encodeURIComponent(name)}/archive-preview`);
  if (!window.confirm(`會把「${name}」的設定檔和 ${preview.result_count} 筆結果一起收起來，下拉和清單都看不到，名字會空出來，之後可以從封存區整包搬回。確定封存嗎？`)) return;
  const data = await api(`/api/schemes/${encodeURIComponent(name)}/archive`, "POST", {});
  dropCompare(picked => picked.scheme_id === name || data.result_ids.includes(picked.run_id));
  drawCompareChoice();
  if (data.result_ids.some((id) => $("result-link").getAttribute("href") === `/results/${id}`)) {
    $("result-link").hidden = true;
    $("result-stale").hidden = true;
  }
  const onForm = openedId === name || $("save-id").value === name;
  if (onForm) {
    openedId = null;
    $("scheme-archive-notice").textContent = "這份方案已封存，存檔會用這個名字重新建一份";
    $("scheme-archive-notice").hidden = false;
  }
  await refreshMovedLists(data.message);
}
async function movePackage(item) {
  // 空畫面（還沒開任何方案）搬回一份：直接打開搬回的那一份（#1265 單元 5b）；表單上藏著的開機範本不算開著。
  // 搬回、重讀清單、打開一路鎖著（第二趟追加，跟 newFromTemplate 同一套 held）：中間不留空窗給「開一份新的」，
  // 沒打開時寫 changeInputScheme 剛寫進訊息列的那一句（見 openScheme 上面）。
  if (document.body.dataset.inputStart === "empty") {
    if (inputBusy) { say(`${busyWords}，沒有搬回「${item.scheme_id}」；等它做完再按一次「搬回」`, "notice"); return; }
    let restored = false, problem = "", results = 0, withScheme = true, listProblem = null;
    const opened = await holdInput(async () => {
      const data = await api(`/api/archive/packages/${item.package_id}/restore`, "POST", {});
      restored = true;
      // 包裡有沒有方案檔看搬回的回覆（封存清單的成員路徑），不看重讀後的清單：清單讀不到時 refreshMovedLists 不丟錯。
      // 沒有方案檔（只封存了結果）就不去打開它：打開只會撞上「找不到檔案」（5c 第二趟，審查第 4 條）。
      // 回覆缺 paths 當成包裡有方案檔（照 5c 以前去打開）、缺 result_ids 算 0 筆（5c 第三趟，複查第 2 條）。
      results = data.result_ids?.length ?? 0;
      withScheme = !Array.isArray(data.paths) ||
        data.paths.some(path => path.replaceAll("\\", "/") === `schemes/${item.scheme_id}.json`);
      listProblem = await refreshMovedLists(data.message);
      if (!withScheme) return false;
      try {
        if (await openScheme(item.scheme_id, true)) return true;
        problem = $("messages").textContent;
        return false;
      } catch (error) {
        if (openedId === item.scheme_id) { say(errorText(error), "notice"); return true; }
        problem = errorText(error);
        return false;
      }
    });
    // 搬回本身沒成：holdInput 已把原因寫進訊息列，抽屜開著時照抄在抽屜上方，抽屜不關。
    if (!restored) return;
    closeResults();
    // 開好了：鎖著時被擋下的「開一份新的」那一句（holdInput 的 finally 剛寫回去）已經過時，清掉（不變式四）。
    // 清單重讀失敗（5c 第三趟，複查第 3 條）：訊息列在空畫面被門藏著，原因寫在看得到的地方——開好了寫訊息列（門已開），
    // 沒開好寫卡片；只有結果的包不再指去「看算過的結果」（那顆照舊清單畫，可能藏著）。
    const reread = listProblem === null ? "" : `已經搬回，但清單更新不了，請重新整理這一頁：${listProblem}`;
    if (opened) {
      sayBeside("new-scheme-note", "", ""); leaveEmptyStart();
      if (reread) say(`「${item.scheme_id}」${reread}`, "notice");
    } else if (!withScheme) sayBeside("new-scheme-note", reread ? `「${item.scheme_id}」的 ${results} 筆結果${reread}` :
      `「${item.scheme_id}」的 ${results} 筆結果已經搬回；這一包裡沒有方案檔，沒有打開方案。結果在下面「看算過的結果」裡`,
      reread ? "notice" : "ok");
    else if (reread) sayBeside("new-scheme-note", `「${item.scheme_id}」${reread}${problem ? `；沒有打開：${problem}` : ""}`, "notice");
    else sayBeside("new-scheme-note", `「${item.scheme_id}」已經搬回，但沒有打開：${problem}`, "notice");
    return;
  }
  const data = await api(`/api/archive/packages/${item.package_id}/restore`, "POST", {});
  if ($("save-id").value === item.scheme_id) {
    openedId = item.scheme_id;
    $("scheme-archive-notice").hidden = true;
  }
  await refreshMovedLists(data.message);
}
async function deleteArchive(item, isPackage = true) {
  const label = isPackage
    ? `封存的「${item.scheme_id}」（${item.archived_text} 封存、${item.result_count} 筆結果、${item.size_text}）`
    : `舊的單筆封存「${item.scheme_id}」（${item.finished_text} 算完、${item.size_text}）`;
  if (!window.confirm(`會永久刪除${label}。這個資料夾沒有備份，刪了救不回來。確定刪除嗎？`)) return;
  const path = isPackage ? `packages/${item.package_id}` : item.run_id;
  try {
    const data = await api(`/api/archive/${path}/delete`, "POST", {});
    await refreshMovedLists(data.message);
  } catch (error) {
    await refreshMovedLists(errorText(error), "notice");
  }
}
async function clearArchive(preview) {
  let message = `會永久刪除封存區的 ${preview.package_count} 包方案（共 ${preview.result_count} 筆結果）和 ${preview.legacy_count} 筆舊的單筆封存，${preview.size_text}。清單上現在的方案和結果不受影響。這個資料夾沒有備份，刪了救不回來。確定全部刪除嗎？`;
  if (preview.skipped.length) message += `\n另有 ${preview.skipped.length} 包封存沒做完或清單讀不出，不會刪，請助理檢查。`;
  if (!window.confirm(message)) return;
  const data = await api("/api/archive/clear", "POST", {
    package_ids: preview.package_ids, run_ids: preview.run_ids});
  // 確認框已說不刪的包仍在；伺服器只處理這次送去的代號。
  const skipped = preview.skipped.length;
  const note = skipped ? `\n另有 ${skipped} 包封存已跳過，請助理檢查。` : "";
  await refreshMovedLists(data.message + note, skipped || data.skipped.length || data.failed.length ? "notice" : "ok");
}
async function loadArchiveList() {
  const data = await api("/api/archive");
  const archive = $("archived"), list = $("archived-list");
  const packages = data.packages || [];
  const preview = data.clear_preview;
  $("clear-archive").hidden = !preview || (!preview.package_ids.length && !preview.run_ids.length);
  $("clear-archive").onclick = () => action(() => clearArchive(preview));
  archive.hidden = !packages.length && !data.results.length;
  if (archive.hidden) archive.open = false;  // 清空就收合，之後再封存一筆時它是收著出現的（第三趟，審查建議 1）
  archive.querySelector("summary").textContent = `看已封存的結果（${packages.length} 包方案、${data.results.length} 筆舊的單筆封存）`;
  $("archive-say").hidden = archive.hidden;
  // 頂條「看已封存的」問過、而且真的一包也沒有：寫一句（home.js）；滑動提示只在真有表格時出現（第二趟 F5）。
  $("archive-none").hidden = !(archiveAsked && archive.hidden);
  archive.querySelector(".archive-scroll-note").hidden = archive.hidden;
  noteDrawerFocus();
  const packageList = $("archived-packages-list"); packageList.replaceChildren();
  $("archived-packages").hidden = !packages.length;
  // 空畫面卡片那一句（5c 第二趟，設計洞 3）：只有舊的單筆封存結果（archive/results/）也要打得開抽屜。
  const legacy = data.results.length;
  $("empty-archived").hidden = !packages.length && !legacy;
  $("empty-archived").textContent = !legacy ? `看已封存的方案（${packages.length} 份）` :
    !packages.length ? `看已封存的結果（${legacy} 筆）` : `看已封存的方案（${packages.length} 份）與結果（${legacy} 筆）`;
  for (const item of packages) {
    const row = document.createElement("tr"); row.dataset.packageId = item.package_id;
    row.dataset.schemeId = item.scheme_id;
    for (const value of [item.scheme_id, item.archived_text,
      item.result_count === null ? "讀不出" : `${item.result_count} 筆`]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    finishedCell(row.children[1], item.archived_text);
    if (item.status_text) {
      const note = document.createElement("span"); note.className = "run-status";
      note.textContent = item.status_text; row.children[0].append(note);
      row.classList.add("not-finished");
    }
    if (item.restore_available) rowButton(row, "搬回", () => movePackage(item));
    else row.append(document.createElement("td"));
    if (item.delete_available) rowButton(row, "永久刪除", () => deleteArchive(item), true);
    else row.append(document.createElement("td"));
    packageList.append(row);
  }
  $("legacy-archive").hidden = !data.results.length;
  list.replaceChildren();
  for (const item of data.results) {
    const row = resultRow(item, true);
    if (item.restore_available) rowButton(row, "搬回", () => moveResult(item, true));
    else row.append(document.createElement("td"));
    rowButton(row, "永久刪除", () => deleteArchive(item, false), true);
    list.append(row);
  }
  keepDrawerFocus();
}
async function loadResultList() {
  // 選單每份方案底下那一句白話跟著重讀（算完之後「電腦挑過材料」之類會變）；它讀不到只少那一句，不擋清單。
  if (typeof refreshHome === "function") refreshHome();
  const data = await api("/api/results");
  $("server-notice").hidden = !data.server_notice;
  $("server-notice").textContent = data.server_notice || "";
  finishedNames = new Set(data.results.filter(runFinished).map(row => row.scheme_id));
  resultRows = data.results;
  renderInputBars();
  drawResultRows();
  await loadArchiveList();
  return data.results;
}
// 抽屜裡的結果列：每次重讀清單、打開抽屜（「現在開著」跟著換）都重畫；勾選跟著計算代號留住（清單裡沒有了就放掉），
// 游標放回同一列的同一顆（那一列不在了放原本在它下面、還在的第一列，home.js 的 keepDrawerFocus）。
function drawResultRows() {
  const fresh = picked => resultRows.find(row => row.run_id === picked.run_id);
  dropCompare(picked => !fresh(picked));
  for (const side of ["a", "b"]) if (compareChoice[side]) compareChoice[side] = fresh(compareChoice[side]);
  noteDrawerFocus();
  $("results-list").replaceChildren(...resultRows.map(item => resultRow(item)));
  const empty = !resultRows.length;
  $("results-empty").hidden = !empty;
  // 空畫面（#1265 單元 5c）：頁底那顆「照現在牆上貼的算一次」不在畫面上，那一句改指卡片；有結果時卡片底下多一句打開抽屜。
  const start = document.body.dataset.inputStart === "empty";
  $("results-empty").textContent = start ? "還沒有算過的結果；先從範本開一份方案，算好就列在這裡。" :
    "還沒有算過的結果；在頁底按「照現在牆上貼的算一次」，算好就列在這裡。";
  $("empty-results").hidden = empty;
  $("empty-results").textContent = `看算過的結果（${resultRows.length} 筆）`;
  $("results-table").hidden = empty;
  $("compare-row").hidden = empty;
  drawCompareChoice();
  keepDrawerFocus();
}
async function resumeRuns() {
  // 只接回還在算的；已結束的那一筆不貼：開頁先開的那一份（openStartScheme）不一定是算的那一份，貼上去會讓人以為是這份的結果。
  // 同時在算好幾筆時主畫面跟最新開的那一筆（清單照開始時間新到舊），其餘的列在底下（#796）。
  // 接回跟「改看這一筆」走同一條路：同一套號碼，開頁時還在等回覆的開算也不會多排一條計時器。
  activeRuns = (await api("/api/runs")).running;
  const state = activeRuns[0];
  if (!state) return;
  await follow(state.run_id);
}
async function showOtherRuns() {
  // 主畫面只跟一筆；其餘還在算的每筆一行，按「改看這一筆」就換過去，停止鈕也跟著換（#796）。
  // 一次只讓一個請求在路上：上一個還沒回來就不再問，回來時照當下跟的那筆過濾，重問也只由它排，只留一條。
  if (otherAsking) return;
  otherAsking = true;
  let others;
  try {
    activeRuns = (await api("/api/runs")).running;
    others = activeRuns.filter((item) => item.run_id !== runId);
    renderInputBars();
  } finally {
    otherAsking = false;
  }
  const list = $("other-runs");
  list.replaceChildren(...others.map((item) => {
    const row = document.createElement("li");
    // 伺服器給的那句已經帶方案名與計時，前面只標「另一筆」。
    row.textContent = `另一筆：${item.display_text}`;
    const button = document.createElement("button");
    button.textContent = "改看這一筆";
    button.onclick = () => action(() => follow(item.run_id));
    row.append(button);
    return row;
  }));
  list.hidden = !others.length;
  // 列著別筆時每兩秒重問一次，算完的那筆就從這裡拿掉，不留過期的「計算中」。
  clearTimeout(otherTimer);
  if (others.length) otherTimer = setTimeout(() => action(showOtherRuns), 2000);
}
async function follow(id) {
  runId = id; $("stop").disabled = false;
  const mine = ++followSeq;
  clearInterval(timer);
  await poll();
  // 等回覆時又換過（連按改看，同一列按兩次也算）：計時器只歸最後那一次排，比號碼不比代號（#796 審查）。
  if (mine !== followSeq) return;
  if (!$("stop").disabled) timer = setInterval(() => action(poll), 1000);
  await showOtherRuns();
}
// sent：存檔時檢查的是按下那一刻抄下來的那一份（#1187）；其餘照表單現況。
// 一把序號 planSeq（單元 6 第二趟，審查發現 2）：每一次在 refreshPlan 開頭取號，頂條、畫圖、丟掉舊回覆都比這一把。
// 不帶 sent 的：等分區縮放（房間尺寸改了，#1266）醒來時已經不是最新那一次，就不送平面圖檢查，直接退出（最新那一次自己會問、會記）；
// 回來時已經不是最新、或表單已換成別份方案（例如開頁途中開機範本那一次晚到），整筆丟掉——不畫、不寫訊息列、不清任何提示。
// 帶 sent 的（存檔、另存、從範本開新的）一律不丟（5c 第三趟，複查第 1 條）：呼叫的人要拿 planProblems 與訊息列講沒存成的原因，
// 照 #1265 單元 5c 以前的做法畫圖、寫訊息列、更新 planProblems，再回伺服器收不收。
// 不變式：所有請求都結束之後，頂條那一句是最新那一次的結果（最新那一次一定走到 settle），不會停在「檢查中…」。
// 丟錯（連不上、表單收不起來）也記成「沒檢查成」。
// auto：改定一格自動發的那一次（#1265 單元 6，bootInputPage 的 AUTO_CHECK_AREAS）。限制照樣每一次都重問（第二趟發現 1：
// 牆面表單改厚度、流阻也動到限制判決，不另列哪一格要問）。它寫訊息列不清按鈕旁邊那幾句（say 的 besides）；
// 離開一格常常就是去按一顆鈕（儲存、另存、開新的、算）：那顆鈕的回話已經寫進訊息列，之後才回來的這一次只畫圖、記頂條，不寫訊息列。
let planSeq = 0;
async function refreshPlan(sent, auto = false) {
  const turn = ++planSeq, saidAt = saidCount;
  const latest = () => turn === planSeq;
  const settle = (state, count, words) => { if (latest()) planChecked(state, count, words); };
  const quiet = () => auto && saidCount !== saidAt;
  planChecking();
  try { return await checkPlan(sent, latest, settle, auto, quiet); } catch (error) {
    settle("error", 1, errorText(error));
    if (!auto) throw error;
    // 自動那一次連不上、表單收不起來：頂條已經記了；訊息列照上面同一條規矩寫（不清按鈕旁邊那幾句、別的回話先到就不寫）。
    if (!quiet()) say(errorText(error), "notice", false);
    return false;
  }
}
async function checkPlan(sent, latest, settle, auto, quiet) {
  const resize = window.pendingRoomResize();
  if (resize && !sent) {
    await resize;
    if (!latest()) return true;
  }
  // 每次確定的改動都走到這裡：順手重判超過限制的區（#1111；不等它、不鎖頁，晚到的回覆自己丟掉）。
  refreshZoneLimits();
  const asked = scheme;
  const {response, data: plan} = await reply(fetch("/api/plan", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(sent ?? collect())}));
  if (!sent && (!latest() || asked !== scheme)) return response.ok;
  if (!response.ok) {
    shownPlan = plan.plan ?? null;
    if (shownPlan) drawShownPlan();
    else for (const name of ["plan-xy", "plan-xz", "zoom-xy", "zoom-xz", "plan-legend", "plan-detail"])
      $(name).replaceChildren();
    planProblems = plan.problems ? problemLines(plan.problems) : (plan.error || "圖面檢查失敗");
    if (!quiet()) say(planProblems, "notice", !auto);
    settle(plan.problems?.length ? "problem" : "error", plan.problems?.length || 1, planProblems);
    return false;
  }
  shownPlan = plan;
  drawShownPlan();
  if (!quiet()) {
    say(plan.message, "ok", !auto);
    // 「檢查通過」頂條狀態旁那一句已經寫了（check_state.js）：訊息列這一句照寫（存檔、另存照抄它），但不再顯示第二次（home.css）。
    $("messages").dataset.echo = "check";
  }
  settle("ok");
  return true;
}
// 輸入頁上那幾張平面圖現在畫的是哪一份（沒畫是 null）。圖上的字靠量寬度避開彼此，輸入頁藏著的時候量不到：
// 「這間房貼什麼」那一頁關掉、輸入頁重新看得到時照同一份再畫一次（zone_overview.js），不必再問伺服器。
let shownPlan = null;
function drawShownPlan() {
  if (shownPlan) drawPlan(shownPlan, {planXY: "plan-xy", planXZ: "plan-xz", zoomXY: "zoom-xy",
    zoomXZ: "zoom-xz", detail: $("plan-detail"), legend: $("plan-legend")});
}
async function poll() {
  // 這一次問的是哪一筆：等回覆的時候可能已經開了另一筆計算（全域 runId 換了），只認問的那一筆。
  const polled = runId;
  const state = await api(`/api/runs/${polled}`);
  if (polled !== runId) return;
  // 主畫面只寫白話的進度；計算程式自己印的英文輸出收進可展開的技術細節，出錯時照樣找得到原文。
  followedState = state;
  activeRuns = activeRuns.filter(item => item.run_id !== polled);
  $("run-state").textContent = state.display_text;

  $("run-log-text").textContent = state.stderr_tail.join("\n") || "沒有留下出錯紀錄";
  renderInputBars();
  if (state.status !== "running") {
    clearInterval(timer);
    if (state.status !== "done") { $("stop").disabled = true; return; }
    // #755：名字放開後，算的可能是改之前的方案（重算舊結果）；先看清單上這一筆的那一列再決定掛不掛。
    let row;
    try {
      row = (await loadResultList()).find((item) => item.run_id === polled);
    } catch (error) {
      if (polled !== runId) return;
      $("stop").disabled = true;
      say(`「${state.scheme_id}」算完了，但結果清單讀不到：${errorText(error)}；重新整理頁面後在頂條「算過的結果與比較」裡看`, "notice");
      return;
    }
    if (polled !== runId) return;
    // 訊息寫好才把停止鈕灰掉：停止鈕灰了，畫面上就已經是算完的那一句。
    $("stop").disabled = true;
    // 表單還是算的那一份、開算後也沒改過、清單上那一列在而且沒標「改之前的方案算的」，連結才掛在表單旁邊；
    // 找不到那一列（例如另一個分頁已經封存）也不掛，不能說它跟表單是同一份。
    if (state.scheme_id === openedId && !runEdited && row !== undefined && !row.scheme_text) {
      // 講白話就好：結果檔路徑與 32 位計算代號不印，看結果交給頁底的「算好了 ▸ 看結果」連結。
      say(`「${state.scheme_id}」算完了，按頁底的「算好了 ▸ 看結果」看結果`, "ok");
      $("result-link").href = state.result_url;
      $("result-link").hidden = false;
      $("result-stale").hidden = true;
    } else {
      say(`「${state.scheme_id}」算完了；表單上現在不是算的那一份（開算後改過、換了方案，或算的是改之前的方案），結果在頂條「算過的結果與比較」裡`, "notice");
    }
    renderInputBars();
  }
}
async function action(work) {
  // 只印伺服器給的中文訊息，不帶 JS 的英文字首「Error: 」。
  try { await work(); } catch (error) { say(errorText(error), "notice"); }
}
const AUTO_CHECK_AREAS = ["room-fields", "speakers", "receivers", "scattering", "use-scattering", "source-model",
  "furniture-sofa-fields", "furniture-table-fields", "furniture-cloud-fields", "cabinet-fields", "walls"];
async function bootInputPage() {
  const example = await api("/api/example");
  inputDefaults = await api("/api/input-defaults");
  await loadZoneLibrary();
  fixedWallDefaults = structuredClone(example.scheme.scene.impedance_pa_s_per_m_by_wall);
  startupWallDefaults = structuredClone(fixedWallDefaults);
  // 名稱表載不到也照樣畫表單（列名退回代號）；說明寫在第 1 段右卡自己的一行（座標數字摺疊的上面），
  // 不寫訊息列（訊息列等一下就被「檢查通過」蓋掉，畫面上只剩代號配綠字）。
  // 那一行只講白話；為什麼載不到（伺服器的話、瀏覽器的英文原文）收在底下摺起來的技術細節。
  try { labels = await api("/api/labels"); } catch (error) {
    const cause = error instanceof Error && error.cause !== undefined ? `；瀏覽器原文：${errorText(error.cause)}` : "";
    $("labels-technical").textContent = `原因：${errorText(error)}${cause}`;
    $("labels-notice").hidden = false;
  }
  $("feature-note").textContent = example.feature_match_note;
  $("rho-c").textContent = example.rho_c_label;
  if (!await changeInputScheme(async () => example.scheme)) return;
  // 按鈕先綁：下面任一份清單載入失敗，也不能讓整頁按鈕都沒反應。
  // 表單任一格改了都重問阻抗倍數：別格空著時倍數會清掉，那一格補好就要回來。
  for (const id of ["room-fields", "walls", "scattering", "speakers", "receivers", "source-model", "use-scattering",
                    "furniture-sofa-fields", "furniture-table-fields", "furniture-cloud-fields", "cabinet-fields"]) {
    $(id).addEventListener("input", markStale);
    $(id).addEventListener("input", () => action(updateMultiples));
    $(id).addEventListener("input", () => action(updateInputPreview));
  }
  // 改了就自動檢查（#1265 單元 6）：這幾格改定（離開那一格、按 Enter，瀏覽器發 change）就重問平面圖與限制，不用再按「檢查」。
  // 規則只有一條：改到方案的才發（第二趟發現 3）。自己有交接的那幾格（onchange：換種類、套預設、家具勾選）交接完自己問，
  // 這裡不再多問一次；還沒套用的實測草稿不改方案，它的 change 在草稿那一塊就攔下（measured.js），走不到這裡。
  // 尺寸那一格改定時先把喇叭與聆聽點挪好（捕獲階段：排在自動檢查與分區縮放前面，兩邊問的都是挪好的座標）。
  $("room-fields").addEventListener("change", followRoomResize, true);
  for (const id of AUTO_CHECK_AREAS) $(id).addEventListener("change", event => {
    if (event.target instanceof HTMLElement && event.target.onchange) return;
    autoCheck();
  });
  $("walls").addEventListener("input", event => {
    if (!event.target.matches("[data-layer-field]")) return;
    collectWallMaterials();
    for (const name of formWalls()) if (["assembly", "measured_impedance"].includes(wallMaterial(name).kind) && !measuredDrafts[name]) showPresetState(name);
    // 原本那一面變了：右欄「選的跟原本一樣」那一句、牆那一行「原本的牆：…」照現在的寫（#1265 單元 3 之二收尾第二趟 B2）。
    window.refreshWallSection();
  });
  for (const id of ["walls", "elevation-panel"]) $(id).addEventListener("focusout", relabelWallsOnLeave);
  document.addEventListener("pointerdown", () => { pressing = true; }, true);
  for (const kind of ["pointerup", "pointercancel"]) document.addEventListener(kind, relabelWallsAfterPress, true);
  $("furniture-sofa").onchange = () => action(() => editInput($("furniture-sofa").checked ? "furniture" : "remove_furniture", "sofa"));
  $("furniture-cloud").onchange = () => action(() => editInput($("furniture-cloud").checked ? "furniture" : "remove_furniture", $("furniture-cloud").checked ? "ceiling_cloud" : "cloud"));
  $("furniture-table-kind").onchange = () => action(() => editInput($("furniture-table-kind").value === "none" ? "remove_furniture" : "furniture", $("furniture-table-kind").value === "none" ? "table" : $("furniture-table-kind").value));
  $("use-speaker-setup").onchange = () => action(() => editInput("setup", $("use-speaker-setup").checked));
  $("speaker-kind").onchange = () => action(() => editInput("speaker_kind", $("speaker-kind").value));
  $("speaker-mount").onchange = () => action(() => editInput("mount", $("speaker-mount").value));
  $("representative").onchange = () => action(() => editInput("representative", $("representative").checked));
  for (const name of Object.keys(inputDefaults.shortcuts)) {
    $(`shortcut-${name}`).textContent = `${inputDefaults.shortcuts[name].label}姿勢`;
    $(`shortcut-${name}`).onclick = () => action(() => editInput("shortcut", name));
  }
  $("save").onclick = () => action(save);
  bindBars();
  $("new-scheme").onsubmit = event => { event.preventDefault(); action(newFromTemplate); };
  // 「開一份新的」還沒做完（newSchemePending）：那一塊 inert，頁面程式按「取消」也不關（不變式一）。
  $("new-scheme-cancel").onclick = () => {
    if (newSchemePending) return;
    closeBarBoxes(); $("scheme-menu-button").focus();
  };
  // 新名字一邊打一邊看：打了收不下的字（中文、空格、/ 之類）就先在旁邊說，不用等按下去才知道。
  for (const [field, note, hint] of [["save-as-id", "save-as-note", newNameHint],
                                     ["new-scheme-id", "new-scheme-note", newNameHint.replace("「新名字」", "「新方案的名字」")]])
    $(field).addEventListener("input", () => {
      // 「開一份新的」還沒做完：那一塊 inert，程式寫進名字格也不換提示欄（它等一下要寫結果，不變式一）。
      if (field === "new-scheme-id" && newSchemePending) return;
      const typed = $(field).value.trim();
      const refused = typed !== "" && !newNameShape.test(typed);
      sayBeside(note, refused ? hint : "", refused ? "notice" : "");
    });
  $("calculate").onclick = () => action(async () => {
    const press = bottomPressed();
    if (inputBusy) { say(`${busyWords}，這一次沒有開算；等它做完再按一次`, "notice"); return; }
    // 鎖涵蓋「存＋開算」整段：開算的是存下去的那一份（按下那一刻表單上那一份），不是那時畫面上的。
    // 中間重讀清單也在鎖裡（跟改動前同一個順序：存、重讀清單、開算、馬上問第一次進度）。
    // 存之前那一次平面圖檢查沒過（含連不上）：沒存、沒送，storeThen 停在問題上。限制只擋搜尋，這裡不看。
    const state = await storeThen(press, async stored => {
      await refreshAfterSave(stored.message);
      return api("/api/runs", "POST", {scheme_id: stored.id});
    });
    if (!state) return;
    followedState = state;
    $("run-log").open = false;
    $("result-link").hidden = true;
    $("result-stale").hidden = true;
    renderInputBars();
    await follow(state.run_id);
  });
  $("stop").onclick = () => action(async () => { await api(`/api/runs/${runId}/stop`, "POST"); await poll(); });
  await refreshPlan();
  // 方案清單讀不到要記下來：不准跟「真的一份都沒有」混成同一個空清單（#1265 單元 5b）。
  let listProblem = null;
  try { await loadSchemeList(); } catch (error) { listProblem = errorText(error); }
  for (const load of [loadResultList, resumeRuns]) await action(load);
  // 已存的方案一份都沒有（網址帶的那一份當然也不在）或清單讀不到：整頁換成空畫面，不把藏著的開機範本攤出來。
  // 網址帶了名字（?scheme=、?zones=）而一份方案都沒有：那一句寫在空畫面卡片上，不寫頂條「打開」旁邊——「換一個方案」那顆
  // 在空畫面看不到（5c 第二趟，設計洞 5）。清單讀不到時不知道有沒有那一份，不講。
  if (listProblem !== null || !schemeNames.length) {
    enterEmptyStart(listProblem, [], listProblem === null ? missingFromLink() : []);
    return;
  }
  await openStartScheme();
}
// 網址帶的那一份（?scheme=、?zones=；?zones= 開好之後再進「這間房貼什麼」）。
function linkedScheme() {
  const params = new URLSearchParams(location.search);
  const zones = params.get("zones");
  return {zones, asked: zones || params.get("scheme")};
}
// 網址帶的名字不在清單裡那一句（沒帶、或在清單裡就沒有）。
function missingFromLink() {
  const {asked} = linkedScheme();
  return asked && !schemeNames.includes(asked) ? [`沒有打開「${asked}」：已存的方案裡沒有這一份（可能封存了或改了名字）`] : [];
}
// 有存過的方案時開頁先開哪一份（#1265 單元 5c）：網址帶的那一份（?scheme=、?zones=）在清單裡就先開它；沒帶、帶了不在、
// 或它打不開，照選單「最近的方案」同一個排序（home.js 的 schemesNewestFirst）從最近存的那一份往下試，開好一份為止。
// 門（body 的 data-input-start）一直關著、inputReady 不設，表單上那份開機範本一刻都不露出來；開好了才開門。
// 一份都打不開：進空畫面，卡片上寫每一份為什麼沒打開。沒打開的原因（網址那一份不在、哪一份打不開）等開好那一份之後
// 一次寫在「打開」旁邊：打開途中的圖面檢查會清掉旁邊的字（say）。
async function openStartScheme() {
  const {zones, asked} = linkedScheme();
  // missing：網址帶的名字不在清單裡；unopened：試過、打不開的那幾份（全都打不開時卡片上的份數只算這些）。
  const missing = missingFromLink();
  const unopened = [];
  const order = schemesNewestFirst();
  const tries = asked && schemeNames.includes(asked) ? [asked, ...order.filter(name => name !== asked)] : order;
  for (const name of tries) {
    // 開頁途中頁面程式動了表單、那一筆還在路上（使用者按不到：門關著）：等它做完再開，不把「上一個動作還沒做完」
    // 當成這一份打不開、也不往下一份跳。
    while (inputBusy) await new Promise(resolve => setTimeout(resolve, 50));
    let opened = false;
    try { opened = await openScheme(name, false, true); } catch (error) {
      // 已經換成這一份、只是之後接進度或圖面檢查丟錯：算打開了，原因留在訊息列（同 newSchemeSteps 的判法）。
      opened = openedId === name;
      if (opened) say(errorText(error), "notice");
      else { $("messages").textContent = errorText(error); delete $("messages").dataset.echo; }
    }
    // 原因是 changeInputScheme 寫進訊息列的那一句（門關著時不帶「保留原方案」），只留原因本身。
    if (!opened) {
      unopened.push(`沒有打開「${name}」：${$("messages").textContent.replace(/^方案與預覽換不過去：/u, "")}`);
      continue;
    }
    // openScheme 回來時這一份的平面圖已經回來（晚到的舊平面圖由 refreshPlan 的序號丟掉）。下面這一串中間沒有等待點：
    // 先寫「打開」旁邊那一句、開該開的那一頁，最後才開門、設 inputReady——inputReady 是「全部就緒」（5c 第二趟，設計洞 4）。
    const said = [...missing, ...unopened];
    if (said.length) sayBeside("open-note", `${said.join("；")}。先開了「${name}」；要別份從「換一個方案」另選`, "notice");
    const toZones = name === asked && zones;
    if (toZones) window.openZoneOverview();
    showInputPage();
    // 捲動要在開門之後：門關著時第 1 段藏著，捲不過去。
    if (asked && !toZones) $("scheme-section").scrollIntoView();
    return;
  }
  // 一份都沒打開：每一份的原因都寫在卡片上（quiet：「打開」旁邊本來就沒寫）。
  enterEmptyStart(null, unopened, missing);
}
window.addEventListener("DOMContentLoaded", () => action(async () => {
  try { await bootInputPage(); } catch (error) { say(errorText(error), "notice"); }
  // 開頁沒走到判定（範例、預設值讀不到，或表單換不上去）：門不開，訊息列藏在門後，原因改寫在「讀取中…」那一行。
  if (document.body.dataset.inputStart !== "loading") return;
  $("boot-note").textContent = `這一頁沒有讀完：${$("messages").textContent}；先確認本機的網頁伺服器開著，再重新整理這一頁。`;
  $("boot-note").className = "notice";
}));
