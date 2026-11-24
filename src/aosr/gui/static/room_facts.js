/* 輸入頁第 1 段「房間和擺位」（#1265）：右卡「這間房現在」一行一件事、平面圖上方切換看哪一張圖、「改…」打開那件事唯一的編輯處。
   - 房間、喇叭、主位、家具四句與姿勢由伺服器排（/api/input-preview 的 room_facts；每改一格都重問，字跟著資料走），這裡照印。
   - 結果那一行跟頁底條同一份狀態（home.js 的 renderInputBars 叫這裡），算好的時間與連結從結果清單那一份來。
   - 總厚度上限（房間預設）與改造前（原方案）兩格住在這張卡裡（以前在立面圖底下的做法表）；值仍記在設計設定（zoneDesign），
     存檔、重開、「這間房貼什麼」那一頁讀的都是同一份。
   - 四張圖疊在同一格、只露一張（不用 display:none：圖上的字靠量寬度躲開彼此，藏起來量不到）。 */
(() => {
const byId = (id) => document.getElementById(id);
const EDITORS = {"edit-depth": "depth-editor", "edit-speakers": "speaker-editor", "edit-furniture": "furniture-editor",
                 "edit-original": "original-editor"};
const NOT_FILLED = "還沒填完";
const ROOM_DEPTH_LABEL = "貼的東西總厚度最多（公尺；空白表示房間不設上限）";
const ROOM_DEPTH_NOTE = "從原牆表面往房間裡量；程式挑材料也不會超過。這是房間預設，某一面要不同在下面立面圖那一面另設。";

// 「改…」：打開（再按一次收起）那件事唯一的編輯處，焦點放進第一格；不另做一份彈出視窗複製同一批格子。
function toggleEditor(button) {
  const area = byId(EDITORS[button.id]);
  const opening = area.hidden;
  area.hidden = !opening;
  button.setAttribute("aria-expanded", String(opening));
  if (!opening) return;
  const first = area.querySelector("input:not([disabled]), select:not([disabled]), button:not([disabled])");
  first?.focus({preventScroll: true});
  area.scrollIntoView({block: "nearest"});
}
// 圖框記著現在露哪一張：長寬高三格只在平面圖貼圖邊，其他三張排到圖底下一列（同一組格子，換位置全交給 home.css）。
function showView(name) {
  for (const button of document.querySelectorAll("#plan-views [data-view]"))
    button.setAttribute("aria-pressed", String(button.dataset.view === name));
  for (const drawing of document.querySelectorAll("#plan-drawings .drawing"))
    drawing.classList.toggle("shown", drawing.dataset.view === name);
  byId("plan-frame").dataset.view = name;
}

// 房間、喇叭、主位、家具四行與姿勢：伺服器那一份（每次預覽回來都重畫）。
function renderRoomFacts() {
  const facts = inputPreview?.room_facts;
  if (!facts) return;
  for (const key of ["room", "speakers", "primary", "furniture"])
    byId(`fact-${key}-text`).textContent = facts[key] ?? NOT_FILLED;
  for (const name of Object.keys(inputDefaults?.shortcuts || {}))
    byId(`shortcut-${name}`).setAttribute("aria-pressed", String(facts.posture === name));
  // 喇叭那一行的小字只寫短句；長提示（怎麼改、圖怎麼畫）在「改喇叭與擺法」打開的那一塊裡。名字跟那句同一份名稱表。
  const setup = scheme?.speaker_setup, names = setup ? speakerSetupNames(setup) : null;
  byId("fact-speakers-kind").textContent = !setup ? "沒設喇叭類型與擺法"
    : names ? `${names[0]}、放${names[1]}（${names[2]}）` : "設了喇叭類型與擺法（中文名沒載到）";
  const around = (scheme?.receiver_set?.points || []).filter(point => point.role === "surrounding").length;
  byId("coordinates-summary").textContent = `座標數字（喇叭、主位，和主位周圍一起算聲音的 ${around} 個點；x、y、z）`;
}

// 結果那一行：正在算＞沒算完（兩句都跟頁底條同一份）＞算好的結果（清單裡這一份最新的那一列）＞還沒算過。
// 算好之後又改了（edited：頂條「有改動還沒存」那一份），字後面註明那份結果是修改前的設定算的。
function renderRoomResult(running, failed, reason, edited) {
  const row = byId("room-result"), title = byId("room-result-title"), text = byId("room-result-text"), link = byId("room-result-link");
  const finished = openedId ? (resultRows || []).find(item => item.scheme_id === openedId && ["done", "none"].includes(item.run_status)) : undefined;
  const state = running ? "running" : failed ? "failed" : finished ? "done" : "never";
  row.dataset.state = state;
  link.hidden = state !== "done";
  if (state === "running") {
    title.textContent = "正在算";
    text.textContent = `已跑 ${running.elapsed_s} 秒；算完這裡會出現「看結果 ▸」`;
  } else if (state === "failed") {
    title.textContent = "沒算完";
    text.textContent = `${reason}；看頁底的出錯紀錄`;
  } else if (state === "done") {
    title.textContent = "算好的結果";
    text.textContent = (finished.scheme_text ? `${finished.finished_text}，${finished.scheme_text}` : finished.finished_text) +
      (edited ? "（修改前的設定算的）" : "");
    link.href = finished.result_url;
  } else {
    title.textContent = "還沒算過";
    text.textContent = "按頁底「照現在牆上貼的算一次」；算好後，結果就出現在這一行。";
  }
}

function depthWords() {
  const value = zoneDesign?.room_max_depth_m;
  return value == null
    ? "貼的東西總厚度：房間預設沒設上限（照每一面、每一區和材料自己的限制）；某一面要不同在下面立面圖那一面另設"
    : `貼的東西總厚度最多 ${centimetres(value)} 公分（從原牆表面往房間裡量，程式挑材料也不會超過）；這是房間預設，某一面要不同在下面立面圖那一面另設`;
}
function originalWords() {
  const chosen = zoneDesign?.original_scheme_id;
  if (!chosen) return "每一面都只有原本的牆（沒指定原方案）";
  return schemeNames.includes(chosen) ? `照「${chosen}」那一份方案` : `照「${chosen}」那一份方案（不在方案清單裡）`;
}
// 正在填的那一格不換掉（立面圖每打一個字都會重畫做法表，連帶叫到這裡）；字照樣換。
function renderRoomEditors() {
  byId("fact-depth-text").textContent = depthWords();
  byId("fact-original-text").textContent = originalWords();
  if (!scheme) return;
  const depth = byId("depth-editor");
  if (!depth.contains(document.activeElement)) {
    const field = depthInput("depth-room", ROOM_DEPTH_LABEL, zoneDesign?.room_max_depth_m, value => {
      const design = designForEdit();
      if (value === null) delete design.room_max_depth_m; else design.room_max_depth_m = value;
    });
    const note = document.createElement("p"); note.className = "note"; note.textContent = ROOM_DEPTH_NOTE;
    const said = document.createElement("p"); said.id = "depth-room-note"; said.setAttribute("aria-live", "polite");
    depth.replaceChildren(field, note, said);
  }
  const original = byId("original-editor");
  if (!original.contains(document.activeElement)) {
    const clear = document.createElement("button"); clear.type = "button"; clear.id = "original-clear"; clear.className = "mini";
    clear.textContent = "清掉（每一面都只有原本的牆）";
    clear.disabled = !zoneDesign?.original_scheme_id;
    clear.onclick = () => {
      const select = byId("original-scheme");
      select.value = "";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    };
    original.replaceChildren(originalSchemeRow(), clear);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  for (const id of Object.keys(EDITORS)) byId(id).addEventListener("click", event => toggleEditor(event.currentTarget));
  for (const button of document.querySelectorAll("#plan-views [data-view]"))
    button.addEventListener("click", () => showView(button.dataset.view));
  // 總厚度上限填錯：做法表底下那一行照寫（depthInput），這張卡上也寫一份（那一行可能不在眼前）。
  // 捕獲階段先看：depthInput 自己的監聽會把格子畫回原值。
  byId("depth-editor").addEventListener("change", event => {
    if (event.target.id !== "depth-room") return;
    const text = event.target.value.trim(), number = Number(text);
    const refused = text !== "" && !(Number.isFinite(number) && number > 0);
    byId("depth-room-note").textContent = refused ? "總厚度上限要填大於 0 的數字（公尺）；空白表示房間不設上限" : "";
    byId("depth-room-note").className = refused ? "notice" : "";
  }, true);
  byId("depth-editor").addEventListener("change", () => { byId("fact-depth-text").textContent = depthWords(); });
  byId("original-editor").addEventListener("change", () => {
    byId("fact-original-text").textContent = originalWords();
    byId("original-clear").disabled = !zoneDesign?.original_scheme_id;
  });
});
window.renderRoomFacts = renderRoomFacts;
window.renderRoomResult = renderRoomResult;
window.renderRoomEditors = renderRoomEditors;
window.showPlanView = showView;
})();
