"use strict";

const backingLabels = {rigid: "硬牆", air: "空氣", zone_wall: "這一區原本的牆"};
let elements, draft;
let selected = -1;
let readOnly = false;
let isNew = true;
let curveGeneration = 0;
let curveTimer;

function button(text, action, attributes = {}) {
  return el("button", {type: "button", text, ...attributes, on: {click: action}});
}

function textureChip(layer) {
  const texture = materialTexture(layer.kind, layer.material_class);
  const svg = materialSvg("svg", {class: "chip", viewBox: "0 0 24 24", "aria-hidden": "true"});
  svg.append(materialSvg("rect", {width: 24, height: 24, fill: texture === "air" ? "white" : `url(#mat-${texture})`}));
  return svg;
}

function kindName(kind) {
  return listing.material_form[kind].label;
}

function displayNumber(number) {
  return Number(number.toPrecision(15));
}

function thicknessText(layer) {
  return layer.thickness_m == null ? "" : displayNumber(layer.thickness_m * 1000) + " mm";
}

function selectLayer(index) {
  selected = index;
  renderStack();
  renderFigure();
}

function stackChanged() {
  renderStack();
  renderFigure();
  requestCurve();
}

function moveLayer(index, offset) {
  const layers = draft.layers;
  const [layer] = layers.splice(index, 1);
  selected = index + offset;
  layers.splice(selected, 0, layer);
  stackChanged();
}

function removeLayer(index) {
  draft.layers.splice(index, 1);
  selected = -1;
  stackChanged();
}

function parameterField(layer, key, spec) {
  const length = spec.unit === "m";
  const scale = length ? 1000 : key === "open_area" ? 100 : 1;
  const unit = length ? "mm" : key === "open_area" ? "%" : spec.unit;
  const required = listing.required_fields[layer.kind].includes(key);
  const value = layer[key] ?? spec.default;
  const options = Object.entries(spec.options);
  let input;
  if (options.length) {
    input = el("select", {name: key, disabled: readOnly});
    if (!required || value == null) input.append(el("option", {value: "", text: required ? "請選擇" : "（不填）"}));
    for (const [code, text] of options) input.append(el("option", {value: code, text}));
    input.value = value ?? "";
  } else {
    input = el("input", {type: "number", name: key, step: "any", disabled: readOnly,
      value: value == null ? "" : displayNumber(value * scale)});
  }
  input.addEventListener("input", () => {
    if (!input.value) delete layer[key];
    else layer[key] = options.length ? input.value : Number(input.value) / scale;
    byId("stack").querySelector(".selected .size").textContent = thicknessText(layer);
    renderFigure();
    requestCurve();
  });
  return el("label", {}, spec.label + (unit ? `（${unit}）` : "") + (required ? "＊" : ""), input);
}

function layerBody(layer, index) {
  const params = el("div", {class: "params"});
  for (const [key, spec] of Object.entries(listing.material_form[layer.kind].fields)) {
    if (key !== "kind") params.append(parameterField(layer, key, spec));
  }
  const tools = el("div", {class: "layer-tools"},
    button("往房間側移", () => moveLayer(index, -1), {class: "quiet", disabled: index === 0}),
    button("往牆側移", () => moveLayer(index, 1), {class: "quiet", disabled: index === draft.layers.length - 1}),
    button("拿掉", () => removeLayer(index), {class: "danger"}));
  return el("div", {class: "layer-body"}, params, readOnly ? null : tools);
}

function renderStack() {
  const rows = draft.layers.map((layer, i) => {
    const open = i === selected;
    const head = button("", () => selectLayer(open ? -1 : i), {class: "layer-select", "aria-expanded": String(open)});
    head.append(String(i + 1), textureChip(layer), kindName(layer.kind), el("span", {class: "size", text: thicknessText(layer)}));
    return el("article", {class: open ? "selected" : "", "data-kind": layer.kind},
      el("div", {class: "layer-head"}, head, el("div", {class: "tune"},
        el("span", {}, "給程式調 ", el("button", {type: "button", role: "switch", "aria-label": `第 ${i+1} 層給程式調`,
          "aria-checked": "false", disabled: true})), el("span", {text: "第二批才能給程式調"}))),
      open ? layerBody(layer, i) : null);
  });
  byId("stack").replaceChildren(...rows);
}

function renderFigure() {
  const layers = draft.layers.map(layer => ({name: kindName(layer.kind), texture: materialTexture(layer.kind, layer.material_class),
    mm: Math.max(0, (layer.thickness_m ?? 0) * 1000)}));
  byId("figure").replaceChildren(draft.layers.length ? materialCutaway(layers, draft.backing, layers.map(l => l.name).join("、"),
    {index: selected, pick: selectLayer}) : el("p", {class: "hint", text: "加一層後即可查看剖面"}));
  byId("figure-key").replaceChildren(...layers.map((layer, i) => button(`${i+1}　${layer.name} ${thicknessText(draft.layers[i])}`,
    () => selectLayer(i), {"aria-pressed": String(i === selected)})));
  byId("figure-key").append(el("small", {class: "back-key", text: "背後：" + backingLabels[draft.backing]}));
  const known = draft.layers.filter(layer => layer.thickness_m != null);
  const total = known.reduce((sum, layer) => sum + layer.thickness_m * 1000, 0);
  byId("total").textContent = !known.length ? "—" : `${displayNumber(total)} mm` +
    (known.length === layers.length ? "" : "（未計無厚度資料的層）");
}

function renderCurve(reply) {
  const chart = byId("curve");
  if (reply.status !== "drawn") {
    chart.textContent = reply.message || reply.error;
    return;
  }
  chart.replaceChildren(el("small", {text: "吸音率 0–1"}), smallCurve(draft, reply),
    el("div", {class: "axis"}, el("span", {text: SMALL_CURVE_HZ.low + " Hz"}), el("span", {text: SMALL_CURVE_HZ.high / 1000 + " kHz"})));
  const warnings = reply.warnings.map(note => note.text);
  if (reply.reference_only) warnings.unshift(reply.reference_only.sentence, reply.reference_only.why_only);
  byId("curve-notes").querySelector("ul").replaceChildren(...warnings.map(text => el("li", {text})));
  byId("curve-notes").hidden = !warnings.length;
}

function requestCurve() {
  const ticket = ++curveGeneration;
  clearTimeout(curveTimer);
  byId("curve-notes").hidden = true;
  byId("condition").textContent = "無規入射・背後：" + backingLabels[draft.backing] +
    (draft.backing === "zone_wall" ? "（先當硬牆算）" : "");
  byId("save-message").textContent = "";
  byId("curve").textContent = "計算中…";
  const body = JSON.stringify(draft);
  curveTimer = setTimeout(async () => {
    try {
      const reply = await api("/api/material-curve", {method: "POST", headers: {"Content-Type": "application/json"}, body});
      if (ticket === curveGeneration) renderCurve(reply.body);
    } catch (error) {
      if (ticket === curveGeneration) byId("curve").textContent = error.message;
    }
  }, 180);
}

function renderPalette() {
  const kinds = new Map();
  for (const row of elements) if (!kinds.has(row.kind)) kinds.set(row.kind, row);
  byId("palette").replaceChildren(...Array.from(kinds.values(), row => {
    const add = button("", () => {
      const layer = structuredClone(row.first_layer ?? {kind: row.kind});
      draft.layers.push(layer);
      selected = row.first_layer ? -1 : draft.layers.length - 1;
      stackChanged();
    }, {"data-kind": row.kind});
    add.append(textureChip(row), row.kind === "porous" ? "棉／泡棉（多孔）" : kindName(row.kind));
    return add;
  }));
}

function freshConstruction() {
  return {id: "construction_" + crypto.getRandomValues(new Uint32Array(2)).join("_"), label: freeLabel("未命名構造"), kind: "construction",
    layers: [], backing: "rigid", placement: "wall", product: false, provenance: "model", source: "材料設計器依各層物性推算", links: [], notes: []};
}

function renderDesigner() {
  byId("name").value = draft.label;
  byId("name").disabled = readOnly;
  for (const input of document.querySelectorAll("input[type=radio]")) {
    input.checked = input.value === draft[input.name];
    input.disabled = readOnly;
  }
  byId("product").checked = draft.product;
  byId("product").disabled = readOnly;
  byId("source").textContent = listing.provenance_labels[draft.provenance];
  byId("source").title = draft.source;
  byId("workspace").hidden = false;
  byId("entry-status").textContent = isNew ? "從房間側往牆側疊層，儲存後可在分區頁指定。" :
    (readOnly ? "成品唯讀。可複製到設計器。改這一筆，已經貼在分區上的不會跟著變" : "改這一筆，已經貼在分區上的不會跟著變");
  byId("palette").closest(".add").hidden = readOnly;
  byId("save").hidden = readOnly;
  byId("copy-designer").hidden = !readOnly;
  stackChanged();
}

async function initDesigner() {
  const [materials, parts] = await Promise.all([api("/api/materials"), api("/api/material-elements")]);
  if (!materials.ok || !parts.ok) throw new Error(materials.body.error || parts.body.error || "讀不到材料資料");
  listing = materials.body;
  if (listing.user_file_problem) throw new Error(listing.user_file_problem);
  elements = parts.body.elements;
  const query = new URLSearchParams(location.search);
  const copying = query.has("copy");
  const id = query.get(copying ? "copy" : "id");
  if (id === null) {
    draft = freshConstruction();
  } else {
    const row = listing.materials.find(item => item.entry.id === id);
    if (!row) throw new Error("材料庫找不到這一筆材料");
    const entry = row.entry;
    if (entry.kind === "construction") {
      draft = structuredClone(entry);
    } else if (copying && entry.kind === "fixed" && entry.material.kind === "assembly") {
      const {id, label, provenance, source, links, notes} = entry;
      const {layers, backing} = entry.material;
      const group = row.placement.group;
      draft = {id, label, provenance, source, links, notes, kind: "construction", layers, backing, product: false,
        placement: ["floor", "ceiling_build", "door_window"].includes(group) ? group : "wall"};
    } else {
      throw new Error("這種材料無法在設計器開啟");
    }
    if (copying) {
      draft.id = freeId(entry.id + "_copy");
      draft.label = freeLabel(entry.label + "（複製）");
      draft.product = false;
    } else {
      isNew = false;
      readOnly = entry.product || row.builtin;
    }
  }
  renderPalette();
  renderDesigner();
}

async function saveConstruction() {
  if (readOnly || byId("save").disabled) return;
  const message = byId("save-message");
  byId("save").disabled = true;
  byId("workspace").inert = true;
  byId("name").disabled = true;
  try {
    const reply = await api("/api/materials/" + encodeURIComponent(draft.id), {method: "PUT",
      headers: {"Content-Type": "application/json", ...(isNew ? {"If-None-Match": "*"} : {})}, body: JSON.stringify(draft)});
    if (!reply.ok) throw new Error(reply.body.error || String(reply.status));
    isNew = false;
    readOnly = draft.product;
    history.replaceState(null, "", designerUrl(draft.id));
    renderDesigner();
    message.className = "ok";
    message.textContent = `已存到「${draft.product ? "成品" : "參數構造"}」頁。`;
  } catch (error) {
    message.className = "notice";
    message.textContent = error.message;
  } finally {
    byId("workspace").inert = false;
    byId("name").disabled = readOnly;
    byId("save").disabled = false;
  }
}

byId("save").addEventListener("click", saveConstruction);
byId("copy-designer").addEventListener("click", () => location.assign(designerUrl(draft.id, true)));
byId("name").addEventListener("input", event => { draft.label = event.target.value; requestCurve(); });
for (const input of document.querySelectorAll("input[type=radio]")) {
  input.addEventListener("change", () => {
    draft[input.name] = input.value;
    renderFigure();
    requestCurve();
  });
}
byId("product").addEventListener("change", event => { draft.product = event.target.checked; requestCurve(); });
initDesigner().catch(error => { byId("entry-status").textContent = error.message; });
