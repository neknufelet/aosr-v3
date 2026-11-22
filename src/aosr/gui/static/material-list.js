"use strict";

const entryKinds = ["fixed", "optimizable"];
const catalogKinds = [...entryKinds, "elements"];
const PREVIEW_TIMEOUT_MS = 60_000;
const previewCache = new Map();
let pendingPreviews = [];
let drawingPreview = false;

function sectionLayers(entry) {
  let layers, backing = "rigid";
  if (entry.kind === "fixed") {
    if (entry.material.kind !== "assembly") return {layers: [], backing: null};
    ({layers, backing} = entry.material);
  } else if (entry.kind === "optimizable") {
    layers = [entry.porous];
    if (entry.cavity?.initial_m) layers.push({kind: "air", thickness_m: entry.cavity.initial_m});
  } else {
    if (entry.kind === "construction") backing = entry.backing;
    layers = entry.layers;
  }
  return {backing, layers: layers.map((layer, index) => {
    const plate = entry.structure === "vpr" && index === layers.length - 2;
    const thickness = plate ? layer.surface_mass_kg_m2 / entry.plate_density_kg_m3 : layer.thickness_m;
    const adjustable = entry.kind === "optimizable" ? (index ? entry.cavity : entry.porous_thickness).adjustable :
      entry.variables?.some(variable => variable.layer === index);
    return {texture: plate ? "steel" : materialTexture(layer.kind, layer.material_class), name: form()[layer.kind].label,
      mm: (thickness ?? 0) * 1000, adjustable, hasThickness: thickness != null};
  })};
}

function sourceBadge(key, labels = listing.provenance_labels) {
  return el("span", {class: "source-badge " + key}, el("i", {"aria-hidden": "true"}), labels[key]);
}

function needsPreviewBadge(result) {
  return result.spans.some(overlapsSmallCurve) || result.warnings.some(line => line.level !== "note") || Boolean(result.reference_only);
}

function showPreview(box, {svg, warnings, showBadge}) {
  box.replaceChildren(svg.cloneNode(true));
  if (showBadge) {
    box.append(el("span", {class: "preview-warning", text: "⚠ 有提醒", title: warnings.join("\n")}));
  }
  if (!warnings.length) return;
  const details = box.closest("article").querySelector("dl");
  details.append(el("dt", {text: "曲線提醒"}), el("dd", {}, el("ul", {}, ...warnings.map(text => el("li", {text})))));
}

async function drawPreviews() {
  if (drawingPreview) return;
  drawingPreview = true;
  try {
    while (pendingPreviews.length) {
      const {entry, box} = pendingPreviews.shift();
      if (box.childElementCount || !box.isConnected || box.closest("[hidden],details:not([open])")) continue;
      const body = JSON.stringify(entry);
      const cached = previewCache.get(entry.id);
      if (cached?.body === body) {
        showPreview(box, cached);
        continue;
      }
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), PREVIEW_TIMEOUT_MS);
      try {
        // ?preview 只標示清單請求，讓瀏覽器考卷可單獨攔截編輯器的 /api/material-curve；伺服器的算法相同。
        const reply = await api("/api/material-curve?preview", {method: "POST", headers: {"Content-Type": "application/json"},
          body, signal: controller.signal});
        if (!reply.ok || reply.body.status !== "drawn") throw new Error(reply.body.message || reply.body.error || reply.status);
        const warnings = reply.body.warnings.map(line => line.text);
        const reference = reply.body.reference_only;
        if (reference) warnings.push(reference.sentence, reference.why_only);
        const preview = {body, svg: smallCurve(entry, reply.body), warnings, showBadge: needsPreviewBadge(reply.body)};
        previewCache.set(entry.id, preview);
        showPreview(box, preview);
      } catch (error) {
        const timedOut = controller.signal.aborted;
        const reason = timedOut ? `曲線計算逾時：已等候 ${PREVIEW_TIMEOUT_MS / 1000} 秒。` : error.message;
        box.replaceChildren(el("span", {text: timedOut ? "曲線沒算完（等太久）" : "曲線無法顯示", title: reason}));
        box.closest("article").querySelector("dl").append(el("dt", {text: "曲線"}), el("dd", {text: reason}));
      } finally {
        clearTimeout(timer);
      }
    }
  } finally {
    drawingPreview = false;
  }
}

function catalogRow(row, previews) {
  const {entry, builtin, placement} = row;
  const {layers, backing} = sectionLayers(entry);
  const total = layers.some(l => l.hasThickness) ? (Number(layers.reduce((sum, layer) => sum + layer.mm, 0).toFixed(2)) + " mm") : "未知";
  const stack = layers.map(layer => layer.name).join("、") || "無分層資料";
  const section = entry.kind === "construction" ? stack + "，" + backingWords(backing) : stack;
  const box = el("div", {class: "preview", text: "計算中…"});
  previews.push({entry, box});
  const actions = el("div", {class: "catalog-actions"});
  const inDesigner = entry.kind === "construction" || entry.kind === "optimizable";
  if (tunableLayers(entry)) actions.append(el("small", {text: "這一頁還不能看內容與編輯"}));
  else if (inDesigner) {
    actions.append(el("button", {class: "open", text: "打開", on: {click: () => location.assign(designerUrl(entry.id))}}),
      el("button", {class: "copy quiet", text: "複製到設計器", disabled: !!listing.user_file_problem,
        on: {click: () => location.assign(designerUrl(entry.id, true))}}));
  } else actions.append(el("button", {type: "button", class: "open", text: "打開", on: {click: () => openEntry(row, false)}}),
    el("button", {type: "button", class: "copy quiet", text: "複製", disabled: !!listing.user_file_problem, on: {click: () => copyEntry(entry)}}));
  if (!builtin) actions.append(el("button", {type: "button", class: "delete danger", text: "刪除", on: {click: () => deleteEntry(entry)}}));
  const details = el("dl");
  for (const [name, value] of [["代號", entry.id], ["種類", kindText(entry)], ["阻抗怎麼得到", row.basis],
    ["吸音曲線算法", layers.length ? "各層以傳遞矩陣法疊合，再算無規入射吸音率" :
      entry.material.kind === "measured_impedance" ? "按實測表面阻抗算無規入射吸音率，表外用底構造補值" :
        "按固定表面阻抗算無規入射吸音率"], ["房間側 → 牆側", section], ["出處與條件", entry.source], ["能貼在哪", placement.reason]]) {
    details.append(el("dt", {text: name}), el("dd", {text: value}));
  }
  const note = layers.some(l => !l.hasThickness) && total !== "未知" ? "（未計無厚度資料的層）" : "";
  return el("article", {class: "material-item", "data-id": entry.id},
    el("div", {class: "catalog-grid"},
      materialCutaway(layers, backing, section),
      el("div", {class: "material-name"}, el("strong", {}, inDesigner ?
        el("a", {href: designerUrl(entry.id), text: entry.label}) : entry.label),
        el("div", {class:"material-depth"}, "總厚 ", el("b", {text: total}), note),
        entry.kind === "construction" ? el("small", {text: backingWords(backing)}) : null, builtin ? el("small", {text: "內建・唯讀"}) : null),
      box, el("div", {class:"source-cell"}, sourceBadge(entry.provenance)),
      el("div", {class: "placement-cell"}, el("span", {class: "placement-tag", text: listing.surface_group_short[placement.group], title: placement.reason})),
      actions),
    el("details", {class: "material-details"}, el("summary", {text: "細節"}), details));
}

function catalogPanel(kind, rows) {
  const panel = byId("panel-" + kind);
  const list = panel.querySelector(".catalog-list");
  const headings = ["剖面", "名稱・總厚", "吸音曲線", "數據來源", "能貼在哪", ""];
  const hints = ["左前是房間側", "", "無規入射・0～1\n31.5 Hz～2 kHz"];
  const head = el("div", {class: "catalog-grid columns"}, ...headings.map((text, i) =>
    el("div", {}, text, hints[i] ? el("small", {text: hints[i]}) : null)));
  const openGroups = new Set([...list.querySelectorAll(".catalog-group[open]")].map(group => group.dataset.name));
  list.replaceChildren(head);
  const groups = new Map([["", []], ["示範房", []], ["草稿", []]]);
  for (const row of rows) {
    const name = row.placement.group === "demo109" ? "示範房" : row.entry.label.includes("草稿") ? "草稿" : "";
    groups.get(name).push(row);
  }
  const folded = rows.length - groups.get("").length;
  byId("tab-" + kind).querySelector("small").textContent = rows.length + " 筆" + (folded ? `（含分組 ${folded} 筆）` : "");
  const previews = [];
  for (const [name, members] of groups) {
    if (!members.length) continue;
    let parent = list;
    if (name) {
      parent = el("details", {class: "catalog-group", open: openGroups.has(name), "data-name": name},
        el("summary", {text: name + "　" + members.length + " 筆"}));
      parent.addEventListener("toggle", () => { if (parent.open) queuePreviews(); });
      list.append(parent);
    }
    for (const row of members) parent.append(catalogRow(row, previews));
  }
  if (!rows.length) list.append(el("p", {text: "目前沒有材料"}));
  return previews;
}

let catalogPreviews = [];
function queuePreviews() {
  pendingPreviews = catalogPreviews.filter(({box}) => !box.querySelector("svg,span") &&
    !box.closest("[hidden],details:not([open])"));
  void drawPreviews();
}

function renderCatalog(rows) {
  const legend = byId("source-legend");
  legend.replaceChildren(el("strong", {text:"數據來源怎麼看"}));
  for (const [key, text] of [["measured","量測取得"], ["literature","書籍或文獻提供"], ["model","模型推算"]]) {
    legend.append(el("span", {}, sourceBadge(key), text));
  }
  // 自己的材料排前面，接著列出唯讀的內建預設。
  const ordered = [...rows.filter(row => !row.builtin), ...rows.filter(row => row.builtin)];
  catalogPreviews = entryKinds.flatMap(kind => catalogPanel(kind, ordered.filter(row =>
    catalogKind(row.entry) === kind)));
  queuePreviews();
}

function selectCatalog(kind) {
  const workspace = byId("entry-workspace");
  const wasHidden = workspace.hidden;
  workspace.hidden = kind === "elements";
  for (const item of catalogKinds) {
    const active = kind === item, tab = byId("tab-" + item);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    byId("panel-" + item).hidden = !active;
  }
  if (wasHidden && !workspace.hidden) resizeEditorPlot();
  queuePreviews();
}

function revealSavedEntry(entry) {
  const kind = catalogKind(entry);
  const row = byId("panel-" + kind).querySelector(`[data-id="${CSS.escape(entry.id)}"]`);
  const group = row.closest(".catalog-group");
  if (group) group.open = true;
  selectCatalog(kind);
  const tab = kind === "fixed" ? "成品" : "參數構造";
  return `在「${tab}」頁「${group?.dataset.name ?? "我的材料"}」`;
}

function initCatalog() {
  for (const [i, kind] of catalogKinds.entries()) {
    const tab = byId("tab-" + kind);
    tab.addEventListener("click", () => selectCatalog(kind));
    tab.addEventListener("keydown", event => {
      const count = catalogKinds.length;
      const keys = {ArrowLeft: (i + count - 1) % count, ArrowRight: (i + 1) % count, Home: 0, End: count - 1};
      if (!Object.hasOwn(keys, event.key)) return;
      event.preventDefault();
      const next = catalogKinds[keys[event.key]];
      selectCatalog(next);
      byId("tab-" + next).focus();
    });
  }
}
