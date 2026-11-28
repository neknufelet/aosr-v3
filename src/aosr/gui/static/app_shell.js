// 全站新外框（#1342）：在 body.app 的頁面上產生頂列與左功能列，每頁同一份清單，目前這一頁標出來。
// 只列伺服器真的有的頁；結果頁、比較頁從搜尋紀錄進去，不在這排。
(() => {
  const PAGES = [
    ["/", "方案輸入", "▭"],
    ["/room-model", "自訂房間", "◇"],
    ["/search-start", "喇叭座位", "◎"],
    ["/gradient", "挑材料", "⌕"],
    ["/searches", "搜尋紀錄", "▤"],
    ["/materials", "材料庫", "≋"],
  ];
  function current(path) {
    const hit = PAGES.filter(([href]) => href === "/" ? path === "/" : path === href || path.startsWith(href + "/"));
    return hit.length ? hit[0][0] : null;
  }
  function build() {
    const body = document.body;
    if (!body.classList.contains("app") || document.querySelector(".app-rail")) return;
    const here = current(location.pathname);
    const top = document.createElement("header");
    top.className = "app-top";
    const brand = Object.assign(document.createElement("a"), {className: "brand", href: "/", textContent: "AOSR"});
    const title = Object.assign(document.createElement("span"), {className: "title", textContent: document.title});
    const slot = Object.assign(document.createElement("div"), {className: "sp"});
    top.append(brand, title, slot);
    const rail = document.createElement("nav");
    rail.className = "app-rail";
    rail.setAttribute("aria-label", "網站導覽");
    for (const [href, label, icon] of PAGES) {
      const a = Object.assign(document.createElement("a"), {href});
      if (href === here) a.setAttribute("aria-current", "page");
      const i = document.createElement("i");
      i.setAttribute("aria-hidden", "true");
      i.textContent = icon;
      a.append(i, label);
      rail.append(a);
    }
    body.prepend(top, rail);
    const tools = document.querySelector("[data-top-tools]");
    if (tools) top.append(tools);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
