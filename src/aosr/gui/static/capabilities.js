"use strict";
// 能力表原文由伺服器彙整；不在瀏覽器另寫清單。
function drawCapabilities(view) {
  for (const field of ["not_modeled", "manual_checks"]) {
    const target = document.getElementById(field.replaceAll("_", "-"));
    target.replaceChildren();
    for (const item of view[field]) {
      const li = document.createElement("li");
      li.textContent = item;
      target.append(li);
    }
  }
}
