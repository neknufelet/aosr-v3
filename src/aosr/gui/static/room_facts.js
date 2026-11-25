/* 輸入頁第 1 段「房間和擺位」（#1265）：右卡「這間房現在」一行一件事、平面圖上方切換看哪一張圖、「改…」打開那件事唯一的編輯處。
   - 房間、喇叭、主位、家具四句與姿勢由伺服器排（/api/input-preview 的 room_facts；每改一格都重問，字跟著資料走），這裡照印。
   - 結果那一行跟頁底條同一份狀態（home.js 的 renderInputBars 叫這裡），算好的時間與連結從結果清單那一份來。
   - 總厚度上限（房間預設）與改造前（原方案）兩格住在這張卡裡（以前在立面圖底下的舊做法表，#1265 單元 3 之二第二趟拆了）；值仍記在設計設定（zoneDesign），
     存檔、重開、「這間房貼什麼」那一頁讀的都是同一份。
   - 四張圖疊在同一格、只露一張（不用 display:none：圖上的字靠量寬度躲開彼此，藏起來量不到）。 */
(() => {
const byId = (id) => document.getElementById(id);
// 第 2 段牆那一行的兩顆（#1265 單元 3）也走同一條：打開那一面唯一的編輯處（牆面表單只露選中那一面，elevation.js）。
const EDITORS = {"edit-depth": "depth-editor", "edit-speakers": "speaker-editor", "edit-furniture": "furniture-editor",
                 "edit-original": "original-editor", "edit-wall-original": "wall-original-editor", "edit-wall-depth": "wall-depth-editor"};
const NOT_FILLED = "還沒填完";
const ROOM_DEPTH_LABEL = "貼的東西總厚度最多（公分；空白表示房間不設上限）";
const ROOM_DEPTH_NOTE = "從原牆表面往房間裡量；程式挑材料也不會超過。這是房間預設，某一面要不同在下面立面圖那一面另設。";

// 「改…」：打開（再按一次收起）那件事唯一的編輯處，焦點放進第一格；不另做一份彈出視窗複製同一批格子。
function toggleEditor(button) {
  const area = byId(EDITORS[button.id]);
  const opening = area.hidden;
  area.hidden = !opening;
  button.setAttribute("aria-expanded", String(opening));
  // 第 2 段牆那一行的兩塊：房間尺寸空著時那一行只因為這一塊開著才留著，關掉就照 wall_section.js 再判一次（設計洞 4 走乙）。
  if (!opening && button.id.startsWith("edit-wall-")) window.refreshWallLine?.();
  if (!opening) return;
  // 第一個看得到的格子（牆面表單六面都在，只露選中那一面）。
  const first = [...area.querySelectorAll("input:not([disabled]), select:not([disabled]), button:not([disabled])")]
    .find(node => node.getClientRects().length > 0);
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
  // 只在 renderInputPreview 收尾叫：那時 inputPreview 一定是 /api/input-preview 的回覆（兩條路都是），伺服器每一份都帶
  // room_facts（input_setup.py :: input_preview）。開頁三種先後量過都不會沒有（#1265 單元 2 審查第 6 條），不另守。
  const facts = inputPreview.room_facts;
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
// 算好之後又改了（edited：頂條「有改動還沒存」那一份），字後面註明那份結果是修改前的設定算的；結果列自己已經寫
// 「改之前的方案算的」（存著的方案檔在算完之後改過，伺服器的 scheme_text）就不再加，同一件事不講兩次（審查設計洞 1）。
function renderRoomResult(running, failed, reason, edited) {
  const row = byId("room-result"), title = byId("room-result-title"), text = byId("room-result-text"), link = byId("room-result-link");
  const finished = openedId ? (resultRows || []).find(item => item.scheme_id === openedId && runFinished(item)) : undefined;
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
      (edited && !finished.scheme_text ? "（修改前的設定算的）" : "");
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
// 正在填的那一格不換掉（立面圖每打一個字都會走 renderZoneTable 重畫，連帶叫到這裡）；字照樣換。
function renderRoomEditors() {
  byId("fact-depth-text").textContent = depthWords();
  byId("fact-original-text").textContent = originalWords();
  // 還沒有方案（開頁讀不到範例）時之後的重畫照樣走到這裡（renderInputBars：鎖頁、輪詢都叫它）：沒有方案可改，不擺編輯格。
  if (!scheme) return;
  const depth = byId("depth-editor");
  if (!depth.contains(document.activeElement)) {
    // 填錯的那一句重畫照留（審查發現 2）；擁有者是「第幾次打開」（做法世代號：開方案、重開同一份、載入範本都加一），
    // 不用方案代號（範本都是空的、重開同一份一樣，換了也認不出來；#1265 單元 3 審查真錯 4）。
    const said = depthNote("depth-room-note", String(zoneDesignGeneration));
    const field = depthInput("depth-room", ROOM_DEPTH_LABEL, zoneDesign?.room_max_depth_m, "房間不設上限", said, value => {
      const design = designForEdit();
      if (value === null) delete design.room_max_depth_m; else design.room_max_depth_m = value;
    });
    const note = document.createElement("p"); note.className = "note"; note.textContent = ROOM_DEPTH_NOTE;
    depth.replaceChildren(field, note, said);
  }
  const original = byId("original-editor");
  original.dataset.stale = String(original.contains(document.activeElement));  // 焦點在裡面：這次不換，離開時補畫
  if (original.dataset.stale === "false") {
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
  byId("depth-editor").addEventListener("change", () => { byId("fact-depth-text").textContent = depthWords(); });
  // 方案清單晚到時焦點正在改造前那一塊裡（renderRoomEditors 那次沒換選單）：焦點離開那一塊就補畫一次（審查設計洞 3）。
  // 等焦點真的落到別處再畫（focusout 當下焦點還沒換過去）。
  byId("original-editor").addEventListener("focusout", event => {
    if (byId("original-editor").dataset.stale === "true" && !byId("original-editor").contains(event.relatedTarget))
      setTimeout(renderRoomEditors);
  });
  byId("original-editor").addEventListener("change", () => {
    byId("fact-original-text").textContent = originalWords();
    byId("original-clear").disabled = !zoneDesign?.original_scheme_id;
  });
});
window.renderRoomFacts = renderRoomFacts;
window.renderRoomResult = renderRoomResult;
window.renderRoomEditors = renderRoomEditors;
})();
