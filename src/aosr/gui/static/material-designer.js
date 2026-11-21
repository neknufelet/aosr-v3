"use strict";

const backingLabels = {rigid: "硬牆", air: "空氣", zone_wall: "這一區原本的牆"};
let elements, dimensionFields, draft;
let savedSnapshot;
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

function elementName(layer) {
  const fields = listing.material_form[layer.kind].fields;
  const keys = Object.keys(fields).filter(key => key !== "flow_resistivity_source" && !dimensionFields[layer.kind].includes(key));
  const row = elements.find(row => row.kind === layer.kind && row.first_layer &&
    keys.every(key => (layer[key] ?? fields[key].default) === (row.first_layer[key] ?? fields[key].default)));
  return row?.name ?? kindName(layer.kind);
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

function updateLayer(layer, refreshFields = false) {
  if (refreshFields) {
    renderStack();
    byId("stack").querySelector(".advanced").open = true;
  } else {
    const head = byId("stack").querySelector(".selected .layer-select");
    head.querySelector(".element-name").textContent = elementName(layer);
    head.querySelector(".size").textContent = thicknessText(layer);
  }
  renderFigure();
  requestCurve();
}

function parameterField(layer, key, required = listing.required_fields[layer.kind].includes(key)) {
  const spec = listing.material_form[layer.kind].fields[key];
  const length = spec.unit === "m";
  const scale = length ? 1000 : key === "open_area" ? 100 : 1;
  const unit = length ? "mm" : key === "open_area" ? "%" : spec.unit;
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
    if (!input.value && !required) delete layer[key];
    else if (!input.value) layer[key] = null;
    else if (options.length) layer[key] = input.value;
    else layer[key] = scale === 1 ? Number(input.value) : displayNumber(Number(input.value) / scale);
    if (key === "model") {
      if (layer.model !== "wilson_2015") delete layer.tortuosity;
      if (!["wilson_2015", "mechel_1976"].includes(layer.model)) delete layer.porosity;
    }
    if (key === "hole_pattern" && layer.hole_pattern !== "rectangular") delete layer.hole_spacing_2_m;
    const refresh = key === "model" || key === "hole_pattern";
    updateLayer(layer, refresh);
    if (refresh) byId("stack").querySelector(`[name="${key}"]`).focus();
  });
  return el("label", {}, spec.label + (unit ? `（${unit}）` : "") + (required ? "＊" : ""), input);
}

function fieldGrid(layer, keys, required) {
  return el("div", {class: "params"}, ...keys.map(key => parameterField(layer, key, required)));
}

function porousOptions(layer) {
  const keys = ["material_class", "flow_resistivity_source", "model"];
  const model = layer.model ?? listing.material_form.porous.fields.model.default;
  const physics = model === "wilson_2015" ? ["porosity", "tortuosity"] : model === "mechel_1976" ? ["porosity"] : [];
  return [fieldGrid(layer, keys), fieldGrid(layer, physics, true),
    el("p", {class: "hint", text: "骨架壓縮模數與骨架損耗因數要一起填。填了，這一層就成了黏在薄板後面的背襯。"}),
    fieldGrid(layer, ["frame_compression_modulus_pa", "frame_loss_factor"])];
}

function plateOptions(layer) {
  const slotted = layer.kind === "slotted";
  const spacingKey = slotted ? "slot_spacing_m" : "hole_spacing_m";
  const useSpacing = Object.hasOwn(layer, spacingKey);
  const choices = el("fieldset", {class: "geometry"}, el("legend", {text: "開孔資料"}));
  for (const [spacing, label] of [[false, "用開孔率"], [true, slotted ? "用縫距" : "用孔距＋排法"]]) {
    const radio = el("input", {type: "radio", name: "geometry", checked: spacing === useSpacing, disabled: readOnly});
    radio.addEventListener("change", () => {
      for (const key of ["open_area", spacingKey, "hole_pattern", "hole_spacing_2_m"]) delete layer[key];
      layer[spacing ? spacingKey : "open_area"] = null;
      if (spacing && !slotted) layer.hole_pattern = "square";
      updateLayer(layer, true);
      byId("stack").querySelector('[name="geometry"]:checked').focus();
    });
    choices.append(el("label", {}, radio, label));
  }
  const geometry = useSpacing ? [spacingKey] : ["open_area"];
  if (useSpacing && !slotted) geometry.push("hole_pattern");
  if (useSpacing && layer.hole_pattern === "rectangular") geometry.push("hole_spacing_2_m");
  choices.append(fieldGrid(layer, geometry, true));
  const parts = [choices, fieldGrid(layer, ["surface_mass_kg_m2"])];
  if (layer.kind !== "micro_perforated") {
    const fabric = ["backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"];
    const enabled = Object.hasOwn(layer, fabric[0]);
    const check = el("input", {type: "checkbox", checked: enabled, disabled: readOnly});
    check.addEventListener("change", () => {
      for (const key of fabric) {
        if (check.checked) layer[key] = null;
        else delete layer[key];
      }
      updateLayer(layer, true);
      byId("stack").querySelector('input[type=checkbox]').focus();
    });
    parts.push(el("label", {}, check, "背面貼布"));
    if (enabled) parts.push(fieldGrid(layer, fabric, true));
  }
  return parts;
}

function layerBody(layer, index) {
  const keys = listing.required_fields[layer.kind].filter(key => key !== "kind" && key !== "material_class");
  const params = fieldGrid(layer, keys);
  const fields = listing.material_form[layer.kind].fields;
  const options = layer.kind === "porous" ? porousOptions(layer) : fields.open_area ? plateOptions(layer) : [];
  const advanced = options.length ? el("details", {class: "advanced"}, el("summary", {text: "進階，一般不用填"}), ...options) : null;
  const tools = el("div", {class: "layer-tools"},
    button("往房間側移", () => moveLayer(index, -1), {class: "quiet", disabled: index === 0}),
    button("往牆側移", () => moveLayer(index, 1), {class: "quiet", disabled: index === draft.layers.length - 1}),
    button("拿掉", () => removeLayer(index), {class: "danger"}));
  return el("div", {class: "layer-body"}, params, advanced, readOnly ? null : tools);
}

function renderStack() {
  const rows = draft.layers.map((layer, i) => {
    const open = i === selected;
    const head = button("", () => selectLayer(open ? -1 : i), {class: "layer-select", "aria-expanded": String(open)});
    head.append(String(i + 1), textureChip(layer), el("span", {class: "element-name", text: elementName(layer)}), el("span", {class: "size", text: thicknessText(layer)}));
    return el("article", {class: open ? "selected" : "", "data-kind": layer.kind},
      el("div", {class: "layer-head"}, head, el("div", {class: "tune"},
        el("span", {}, "給程式調 ", el("button", {type: "button", role: "switch", "aria-label": `第 ${i+1} 層給程式調`,
          "aria-checked": "false", disabled: true})), el("span", {text: "第二批才能給程式調"}))),
      open ? layerBody(layer, i) : null);
  });
  byId("stack").replaceChildren(...rows);
}

function renderFigure() {
  const layers = draft.layers.map(layer => ({name: elementName(layer), texture: materialTexture(layer.kind, layer.material_class),
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
  byId("curve-state").textContent = reply.status === "drawn" ? "" : reply.message || reply.error;
  byId("curve-basis").textContent = reply.basis || "";
  showPlotKey(reply);
  if (reply.status !== "drawn") return;
  drawPlot(reply);
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
  destroyPlot();
  byId("curve-basis").textContent = "";
  byId("curve-key").hidden = true;
  byId("curve-state").textContent = "計算中…";
  // 預覽只取疊層和背後，讓尚未填完的出處、名稱不影響物理曲線。
  const body = JSON.stringify({...draft, label: "構造預覽", provenance: "model", source: "依各層物性推算", notes: [], links: []});
  curveTimer = setTimeout(async () => {
    try {
      const reply = await api("/api/material-curve", {method: "POST", headers: {"Content-Type": "application/json"}, body});
      if (ticket === curveGeneration) renderCurve(reply.body);
    } catch (error) {
      if (ticket === curveGeneration) byId("curve-state").textContent = error.message;
    }
  }, 500);
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
  for (const input of document.querySelectorAll(".backs input, .where input")) {
    input.checked = input.value === draft[input.name];
    input.disabled = readOnly;
  }
  byId("product").checked = draft.product;
  byId("product").disabled = readOnly;
  byId("source").textContent = listing.provenance_labels.model;
  byId("citation-provenance").textContent = listing.provenance_labels.model;
  for (const key of ["source", "links", "notes"]) {
    const input = byId("citation-" + key);
    input.value = key === "source" ? draft.source : draft[key].join("\n");
    input.disabled = readOnly;
  }
  byId("source").title = draft.source;
  byId("workspace").hidden = false;
  byId("entry-status").textContent = isNew ? "從房間側往牆側疊層，儲存後可在分區頁指定。" :
    (readOnly ? "成品唯讀，可複製到設計器" : "改這一筆，已經貼在分區上的不會跟著變");
  byId("palette").closest(".add").hidden = readOnly;
  byId("save").hidden = readOnly;
  byId("copy-designer").hidden = isNew;
  byId("copy-designer").textContent = readOnly ? "複製到設計器" : "複製一份";
  renderStack();
  renderFigure();
}

function makeCopy(original) {
  const citation = `原來源：${listing.provenance_labels[original.provenance]}，出處：${original.source}，網址：${original.links.join("、") || "無"}`;
  draft = {...draft, id: freeId(original.id + "_copy"), label: freeLabel(original.label + "（複本）"), product: false,
    provenance: "model", source: `照「${original.label}」改的`, links: [], notes: [citation.replace(/[\r\n]+/g, " "), ...original.notes]};
  isNew = true;
  readOnly = false;
  savedSnapshot = null;
}

async function initDesigner() {
  const [materials, parts] = await Promise.all([api("/api/materials"), api("/api/material-elements")]);
  if (!materials.ok || !parts.ok) throw new Error(materials.body.error || parts.body.error || "讀不到材料資料");
  listing = materials.body;
  if (listing.user_file_problem) throw new Error(listing.user_file_problem);
  elements = parts.body.elements;
  dimensionFields = parts.body.dimension_fields;
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
      makeCopy(entry);
    } else {
      isNew = false;
      readOnly = entry.product || row.builtin;
    }
  }
  savedSnapshot = copying ? null : snapshot();
  renderPalette();
  renderDesigner();
  requestCurve();
}

async function saveConstruction() {
  if (readOnly || byId("save").disabled) return;
  const message = byId("save-message");
  byId("save").disabled = true;
  byId("workspace").inert = true;
  byId("name").disabled = true;
  try {
    const document = {...draft, provenance: "model"};
    const reply = await api("/api/materials/" + encodeURIComponent(draft.id), {method: "PUT",
      headers: {"Content-Type": "application/json", ...(isNew ? {"If-None-Match": "*"} : {})}, body: JSON.stringify(document)});
    if (!reply.ok) throw new Error(reply.body.error || String(reply.status));
    draft = document;
    const row = listing.materials.find(item => item.entry.id === draft.id);
    if (row) row.entry = structuredClone(draft);
    else listing.materials.push({entry: structuredClone(draft), builtin: false});
    isNew = false;
    savedSnapshot = snapshot();
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

function snapshot() {
  // 相同欄位刪掉再填回來，鍵的排列順序不算修改。
  return JSON.stringify(draft, (_, value) => value && typeof value === "object" && !Array.isArray(value)
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]])) : value);
}

window.addEventListener("beforeunload", event => {
  if (draft && !readOnly && snapshot() !== savedSnapshot) {
    event.preventDefault();
    event.returnValue = true;
  }
});

window.addEventListener("resize", () => {
  if (plot) plot.setSize({width: chartWidth(), height: 320});
});

for (const key of ["source", "links", "notes"]) {
  byId("citation-" + key).addEventListener("input", event => {
    const text = event.target.value;
    draft[key] = key === "source" ? text : text.split("\n").map(line => line.trim()).filter(Boolean);
    byId("source").title = draft.source;
    byId("save-message").textContent = "";
  });
}

byId("save").addEventListener("click", saveConstruction);
byId("copy-designer").addEventListener("click", () => {
  const url = designerUrl(draft.id, true);
  makeCopy(draft);
  history.replaceState(null, "", url);
  renderDesigner();
  byId("save-message").textContent = "這是複本，儲存後會新增一筆參數構造。";
});
byId("name").addEventListener("input", event => { draft.label = event.target.value; byId("save-message").textContent = ""; });
for (const input of document.querySelectorAll(".backs input, .where input")) {
  input.addEventListener("change", () => {
    draft[input.name] = input.value;
    byId("save-message").textContent = "";
    if (input.name === "backing") {
      renderFigure();
      requestCurve();
    }
  });
}
byId("product").addEventListener("change", event => { draft.product = event.target.checked; byId("save-message").textContent = ""; });
initDesigner().catch(error => { byId("entry-status").textContent = error.message; });
