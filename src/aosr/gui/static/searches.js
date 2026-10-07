"use strict";
// 搜尋清單與單場共用輪詢間隔；畫面只畫伺服器給的文字。
const POLL_MS = 5000;
const searchId = window.location.pathname.split("/")[2];
let lastFetched = "";
let polling = false;
const $ = (id) => document.getElementById(id);
function node(tag, text) {
  const value = document.createElement(tag);
  value.textContent = text;
  return value;
}
function drawDetail(data) {
  $("title").textContent = `搜尋進度：${data.name}`;
  $("search-content").replaceChildren();
  for (const block of data.blocks) {
    const section = node("section", "");
    section.id = block.key;
    section.append(node("h2", block.title));
    for (const line of block.lines) {
      const p = node("p", line);
      // 伺服器標成警示的那一塊整塊標紅；其他塊裡帶這幾個字的那一行也標紅（複查）。
      // 交接摘要的未完成提示會提到「可能被中斷」；正常提醒只把實際讀取或計算失敗標紅。
      const warningWords = block.key === "crossover" ? ["讀不到", "失敗"] : ["讀不到", "中斷", "失敗", "判不出", "未確認", "已過期"];
      if ((block.warning && block.key !== "crossover") || warningWords.some((word) => line.includes(word))) {
        p.className = "notice";
      }
      section.append(p);
    }
    $("search-content").append(section);
  }
}
function drawList(data) {
  $("title").textContent = "搜尋清單";
  $("search-list").hidden = false;
  $("search-list").replaceChildren();
  if (data.error) $("search-list").append(node("p", data.error));
  else if (!data.searches.length) $("search-list").append(node("p", "還沒有搜尋資料夾"));
  for (const item of data.searches) {
    const section = node("section", "");
    const link = node("a", item.name);
    link.href = item.url;
    section.append(link, node("p", item.stage_text));
    $("search-list").append(section);
  }
}
async function refresh() {
  if (polling) return;
  polling = true;
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), POLL_MS);
  try {
    const response = await fetch(searchId ? `/api/searches/${searchId}` : "/api/searches", {cache: "no-store", signal: abort.signal});
    let data;
    try { data = await response.json(); } catch (_) { data = null; }
    if (!response.ok || data === null) {
      // 伺服器有回、但回的是錯誤：照它給的原因寫，不說成讀不到伺服器（複查）。
      const failure = new Error("server");
      failure.serverMessage = (data && data.error) || `回覆無法讀取（狀態 ${response.status}）`;
      throw failure;
    }
    if (searchId) { drawDetail(data); window.refreshBest(data); } else drawList(data);
    lastFetched = data.fetched_text;
    $("connection").textContent = `資料讀取於 ${lastFetched}；每 5 秒更新`;
    $("connection").className = "";
  } catch (error) {
    const reason = error.serverMessage ? `伺服器回覆：${error.serverMessage}` : "讀不到伺服器";
    $("connection").textContent = lastFetched ? `${reason}，上面是 ${lastFetched} 的資料` : `${reason}，尚未取得搜尋資料`;
    $("connection").className = "notice";
  } finally {
    clearTimeout(timeout);
    polling = false;
  }
}
window.addEventListener("DOMContentLoaded", () => { refresh(); setInterval(refresh, POLL_MS); });
