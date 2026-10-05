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
      if (line.includes("讀不到") || line.includes("中斷") || line.includes("失敗")) p.className = "notice";
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
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "回覆無法讀取");
    if (searchId) drawDetail(data); else drawList(data);
    lastFetched = data.fetched_text;
    $("connection").textContent = `資料讀取於 ${lastFetched}；每 5 秒更新`;
    $("connection").className = "";
  } catch (error) {
    $("connection").textContent = lastFetched ? `讀不到伺服器，上面是 ${lastFetched} 的資料` : "讀不到伺服器，尚未取得搜尋資料";
    $("connection").className = "notice";
  } finally {
    clearTimeout(timeout);
    polling = false;
  }
}
window.addEventListener("DOMContentLoaded", () => { refresh(); setInterval(refresh, POLL_MS); });
