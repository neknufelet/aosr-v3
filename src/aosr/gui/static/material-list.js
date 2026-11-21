"use strict";

const catalogKinds = ["fixed", "optimizable"];
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
    layers = entry.layers;
  }
  return {backing, layers: layers.map((layer, index) => {
    const plate = entry.structure === "vpr" && index === layers.length - 2;
    const textures = {micro_perforated: "mpp", perforated: "perf", slotted: "slot",
      porous: layer.material_class === "foam" ? "foam" : "wool"};
    const thickness = plate ? layer.surface_mass_kg_m2 / entry.plate_density_kg_m3 : layer.thickness_m;
    const adjustable = entry.kind === "optimizable" ? (index ? entry.cavity : entry.porous_thickness).adjustable :
      entry.variables?.some(variable => variable.layer === index);
    return {texture: plate ? "steel" : textures[layer.kind] || layer.kind, name: form()[layer.kind].label,
      mm: (thickness ?? 0) * 1000, adjustable, hasThickness: thickness != null};
  })};
}

function sourceBadge(key) {
  return el("span", {class: "source-badge " + key}, el("i", {"aria-hidden": "true"}), listing.provenance_labels[key]);
}

// 取現有無規入射曲線的同一段對數頻軸，縱軸一律 0～1。實測表只在表內原有頻點加點。
function smallCurve(entry, result) {
  const w = 168, h = 56, low = 31.5, high = 2000;
  const x = hz => 2 + (w - 4) * Math.log(hz / low) / Math.log(high / low);
  const y = alpha => 2 + (h - 4) * (1 - alpha);
  const svg = materialSvg("svg", {class: "curve " + entry.provenance, viewBox: `0 0 ${w} ${h}`,
    role: "img", "aria-label": "無規入射吸音曲線，31.5 Hz 至 2 kHz，吸音率 0 至 1"});
  for (const hz of [63, 125, 250, 500, 1000]) svg.append(materialSvg("line", {x1:x(hz), x2:x(hz), y1:0, y2:h}));
  svg.append(materialSvg("line", {x1:0, x2:w, y1:y(.5), y2:y(.5)}));
  const measured = new Set(entry.material?.points?.map(p => p.frequency_hz));
  const points = [];
  for (const [index, hz] of result.frequencies_hz.entries()) {
    // 多留左右各一點，由 SVG 視窗裁切，避免軸端取最近頻點而出現空隙。
    if (result.frequencies_hz[index + 1] < low || result.frequencies_hz[index - 1] > high) continue;
    const cx = x(hz).toFixed(2), cy = y(result.absorption[index]).toFixed(2);
    points.push(cx + "," + cy);
    if (entry.provenance === "measured" && measured.has(hz)) svg.append(materialSvg("circle", {cx, cy, r: 2.4}));
  }
  svg.append(materialSvg("polyline", {points: points.join(" "), fill: "none", stroke: "currentColor", "stroke-width": 2}));
  return svg;
}

async function drawPreviews() {
  if (drawingPreview) return;
  drawingPreview = true;
  try {
    while (pendingPreviews.length) {
      const {entry, box} = pendingPreviews.shift();
      if (!box.isConnected || box.closest("[hidden],details:not([open])")) continue;
      const body = JSON.stringify(entry);
      const cached = previewCache.get(entry.id);
      if (cached?.body === body) {
        box.replaceChildren(cached.svg.cloneNode(true));
        continue;
      }
      try {
        const reply = await api("/api/material-curve?preview", {method: "POST", headers: {"Content-Type": "application/json"}, body});
        if (!reply.ok || reply.body.status !== "drawn") throw new Error(reply.body.message || reply.body.error || reply.status);
        const svg = smallCurve(entry, reply.body);
        previewCache.set(entry.id, {body, svg});
        box.replaceChildren(svg.cloneNode(true));
      } catch (error) {
        box.replaceChildren(el("span", {text: "曲線無法顯示", title: error.message}));
        box.closest("article").querySelector("dl").append(el("dt", {text: "曲線"}), el("dd", {text: error.message}));
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
  const box = el("div", {class: "preview", text: "計算中…"});
  previews.push({entry, box});
  const actions = el("div", {class: "catalog-actions"});
  if (tunableLayers(entry)) actions.append(el("small", {text: "這一頁還不能看內容與編輯"}));
  else actions.append(el("button", {type: "button", class: "open", text: "打開", on: {click: () => openEntry(row, false)}}),
    el("button", {type: "button", class: "copy quiet", text: "複製", disabled: !!listing.user_file_problem, on: {click: () => copyEntry(entry)}}));
  if (!builtin) actions.append(el("button", {type: "button", class: "delete danger", text: "刪除", on: {click: () => deleteEntry(entry)}}));
  const details = el("dl");
  for (const [name, value] of [["代號", entry.id], ["種類", kindText(entry)], ["阻抗怎麼得到", row.basis],
    ["吸音曲線算法", layers.length ? "各層以傳遞矩陣法疊合，再算無規入射吸音率" :
      entry.material.kind === "measured_impedance" ? "按實測表面阻抗算無規入射吸音率，表外用底構造補值" :
        "按固定表面阻抗算無規入射吸音率"], ["房間側 → 牆側", stack], ["出處與條件", entry.source], ["能貼在哪", placement.reason]]) {
    details.append(el("dt", {text: name}), el("dd", {text: value}));
  }
  const note = layers.some(l => !l.hasThickness) && total !== "未知" ? "（未計無厚度資料的層）" : "";
  return el("article", {class: "material-item", "data-id": entry.id},
    el("div", {class: "catalog-grid"},
      materialCutaway(layers, backing, stack),
      el("div", {class: "material-name"}, el("strong", {text: entry.label}),
        el("div", {class:"material-depth"}, "總厚 ", el("b", {text: total}), note), builtin ? el("small", {text: "內建・唯讀"}) : null),
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
  catalogPreviews = catalogKinds.flatMap(kind => catalogPanel(kind, ordered.filter(row =>
    (row.entry.kind === "fixed") === (kind === "fixed"))));
  queuePreviews();
}

function initCatalog() {
  function select(kind) {
    for (const item of catalogKinds) {
      const active = kind === item, tab = byId("tab-" + item);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      byId("panel-" + item).hidden = !active;
    }
    queuePreviews();
  }
  for (const [i, kind] of catalogKinds.entries()) {
    const tab = byId("tab-" + kind);
    tab.addEventListener("click", () => select(kind));
    tab.addEventListener("keydown", event => {
      const keys = {ArrowLeft: 1-i, ArrowRight: 1-i, Home: 0, End: 1};
      if (!Object.hasOwn(keys, event.key)) return;
      event.preventDefault();
      const next = catalogKinds[keys[event.key]];
      select(next);
      byId("tab-" + next).focus();
    });
  }
}
