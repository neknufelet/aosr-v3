"use strict";

const backingLabels = {rigid: "硬牆", air: "空氣", zone_wall: "這一區原本的牆"};
let elements, dimensionFields, draft;
let savedSnapshot;
let uncertainCreate;
let copiedOptimizable;
let selected = -1;
let readOnly = false;
let isNew = true;
let curveGeneration = 0;
let curveTimer;
const collidingIds = new Set();
const inactiveFields = new WeakMap();
const clothFields = ["backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"];

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
  const hasCloth = Object.hasOwn(layer, clothFields[0]);
  const row = elements.find(row => row.kind === layer.kind && row.first_layer &&
    Object.hasOwn(row.first_layer, clothFields[0]) === hasCloth &&
    keys.every(key => (layer[key] ?? fields[key].default) === (row.first_layer[key] ?? fields[key].default)));
  if (row) return row.name;
  return kindName(layer.kind) + (hasCloth ? "（背面貼布）" : "");
}

function displayNumber(number) {
  return Number(number.toPrecision(15));
}

function thicknessText(layer) {
  if (adjustableAir(layer) && !layer.thickness_m) return "初值無空腔";
  return layer.thickness_m == null ? "" : displayNumber(layer.thickness_m * 1000) + " mm";
}

function selectLayer(index) {
  selected = index;
  renderStack();
  renderFigure();
}

function stackChanged(advancedOpen = false) {
  if (tuningUnavailable()) {
    for (const layer of draft.layers) {
      const spec = thicknessSpecs.get(layer);
      if (spec) spec.adjustable = false;
    }
  }
  renderTuningOptions();
  renderStack(advancedOpen);
  renderFigure();
  requestCurve();
}

function moveLayer(index, offset) {
  const layers = draft.layers;
  const [layer] = layers.splice(index, 1);
  selected = index + offset;
  layers.splice(selected, 0, layer);
  stackChanged(byId("stack").querySelector(".advanced")?.open);
}

function removeLayer(index) {
  draft.layers.splice(index, 1);
  selected = -1;
  stackChanged();
}

function updateLayer(layer, refreshFields = false) {
  if (refreshFields) {
    renderStack(byId("stack").querySelector(".advanced")?.open);
  } else {
    const head = byId("stack").querySelector(".selected .layer-select");
    head.querySelector(".element-name").textContent = elementName(layer);
    head.querySelector(".size").textContent = thicknessText(layer);
  }
  renderFigure();
  requestCurve();
}

// 隱藏的值只留在這一層的畫面記憶中。草稿、曲線和存檔共用啟用中的欄位。
function setFieldActive(layer, key, active, initial = null) {
  if (!inactiveFields.has(layer)) inactiveFields.set(layer, {});
  const hidden = inactiveFields.get(layer);
  if (active && !Object.hasOwn(layer, key)) {
    layer[key] = hidden[key] ?? initial;
    delete hidden[key];
  } else if (!active && Object.hasOwn(layer, key)) {
    hidden[key] = layer[key];
    delete layer[key];
  }
}

function porousModelFields(layer) {
  const model = layer.model ?? listing.material_form.porous.fields.model.default;
  return model === "wilson_2015" ? ["porosity", "tortuosity"] : model === "mechel_1976" ? ["porosity"] : [];
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
    // 還原原本省略的選項（例如流阻來源「未註明」），保留開檔時的欄位。
    const original = layer.kind === "porous" && (copiedOptimizable ?? openedEntry)?.porous;
    const restoreOmitted = original && !Object.hasOwn(original, key) && input.value === spec.default;
    if ((!input.value && !required) || restoreOmitted) delete layer[key];
    else if (!input.value) layer[key] = null;
    else if (options.length) layer[key] = input.value;
    else layer[key] = scale === 1 ? Number(input.value) : displayNumber(Number(input.value) / scale);
    if (key === "model") {
      const active = porousModelFields(layer);
      for (const key of ["porosity", "tortuosity"]) setFieldActive(layer, key, active.includes(key));
    }
    if (key === "hole_pattern") setFieldActive(layer, "hole_spacing_2_m", layer.hole_pattern === "rectangular");
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
  return [fieldGrid(layer, ["flow_resistivity_source", "model"]), fieldGrid(layer, porousModelFields(layer), true),
    el("p", {class: "hint", text: "骨架壓縮模數與骨架損耗因數要一起填。填了，這一層就成了黏在薄板後面的背襯。"}),
    fieldGrid(layer, ["frame_compression_modulus_pa", "frame_loss_factor"])];
}

function plateGeometry(layer) {
  const pitch = layer.kind === "slotted" ? "slot_spacing_m" : "hole_spacing_m";
  if (!Object.hasOwn(layer, pitch)) return ["open_area"];
  if (layer.kind === "slotted") return [pitch];
  return layer.hole_pattern === "rectangular" ? [pitch, "hole_pattern", "hole_spacing_2_m"] : [pitch, "hole_pattern"];
}

function plateOptions(layer) {
  const slotted = layer.kind === "slotted";
  const spacingKey = slotted ? "slot_spacing_m" : "hole_spacing_m";
  const useSpacing = Object.hasOwn(layer, spacingKey);
  const choices = el("fieldset", {class: "geometry"}, el("legend", {text: "開孔資料"}));
  for (const [spacing, label] of [[false, "用開孔率"], [true, slotted ? "用縫距" : "用孔距＋排法"]]) {
    const radio = el("input", {type: "radio", name: "geometry", checked: spacing === useSpacing, disabled: readOnly});
    radio.addEventListener("change", () => {
      setFieldActive(layer, "open_area", !spacing);
      setFieldActive(layer, spacingKey, spacing);
      if (!slotted) {
        setFieldActive(layer, "hole_pattern", spacing, "square");
        setFieldActive(layer, "hole_spacing_2_m", spacing && layer.hole_pattern === "rectangular");
      }
      updateLayer(layer, true);
      byId("stack").querySelector('[name="geometry"]:checked').focus();
    });
    choices.append(el("label", {}, radio, label));
  }
  choices.append(fieldGrid(layer, plateGeometry(layer), true));
  const parts = [choices];
  if (layer.kind !== "micro_perforated") {
    const enabled = Object.hasOwn(layer, clothFields[0]);
    const check = el("input", {type: "checkbox", checked: enabled, disabled: readOnly});
    check.addEventListener("change", () => {
      for (const key of clothFields) setFieldActive(layer, key, check.checked);
      updateLayer(layer, true);
      byId("stack").querySelector('input[type=checkbox]').focus();
    });
    parts.push(el("label", {}, check, "背面貼布"));
    if (enabled) parts.push(fieldGrid(layer, clothFields, true));
  }
  return parts;
}

function layerBody(layer, index, advancedOpen) {
  const keys = listing.required_fields[layer.kind].filter(key => key !== "kind");
  const params = el("div", {class: "params"}, ...keys.map(key => parameterField(layer, key,
    !(layer.kind === "air" && key === "thickness_m" && tuningActive()))));
  const fields = listing.material_form[layer.kind].fields;
  const options = layer.kind === "porous" ? porousOptions(layer) : fields.open_area ? [fieldGrid(layer, ["surface_mass_kg_m2"])] : [];
  const advanced = options.length ? el("details", {class: "advanced", open: advancedOpen}, el("summary", {text: "進階，一般不用填"}), ...options) : null;
  const tools = el("div", {class: "layer-tools"},
    button("往房間側移", () => moveLayer(index, -1), {class: "quiet", disabled: index === 0}),
    button("往牆側移", () => moveLayer(index, 1), {class: "quiet", disabled: index === draft.layers.length - 1}),
    button("拿掉", () => removeLayer(index), {class: "danger"}));
  return el("div", {class: "layer-body"}, params, tuningFields(layer), ...(fields.open_area ? plateOptions(layer) : []), advanced, readOnly ? null : tools);
}

function renderStack(advancedOpen = false) {
  const rows = draft.layers.map((layer, i) => {
    const open = i === selected;
    const head = button("", () => selectLayer(open ? -1 : i), {class: "layer-select", "aria-expanded": String(open)});
    head.append(String(i + 1), textureChip(layer), el("span", {class: "element-name", text: elementName(layer)}), el("span", {class: "size", text: thicknessText(layer)}));
    return el("article", {class: open ? "selected" : "", "data-kind": layer.kind},
      el("div", {class: "layer-head"}, head, tuningControl(layer, i)),
      open ? layerBody(layer, i, advancedOpen) : null);
  });
  byId("stack").replaceChildren(...rows);
}

function renderFigure() {
  const backing = tuningActive() ? "rigid" : draft.backing;
  const layers = draft.layers.map(layer => ({name: elementName(layer), texture: materialTexture(layer.kind, layer.material_class),
    mm: Math.max(0, (layer.thickness_m ?? 0) * 1000)}));
  byId("figure").replaceChildren(draft.layers.length ? materialCutaway(layers, backing, layers.map(l => l.name).join("、"),
    {index: selected, pick: selectLayer}) : el("p", {class: "hint", text: "加一層後即可查看剖面"}));
  byId("figure-key").replaceChildren(...layers.map((layer, i) => button(`${i+1}　${layer.name} ${thicknessText(draft.layers[i])}`,
    () => selectLayer(i), {"aria-pressed": String(i === selected)})));
  byId("figure-key").append(el("small", {class: "back-key", text: "背後：" + backingLabels[backing]}));
  const known = draft.layers.filter(layer => layer.thickness_m != null || (layer.kind === "air" && tuningActive()));
  const total = known.reduce((sum, layer) => sum + (layer.thickness_m ?? 0) * 1000, 0);
  byId("total").textContent = !known.length ? "—" : `${displayNumber(total)} mm` +
    (known.length === layers.length ? "" : "（未計無厚度資料的層）");
}

function renderCurve(reply) {
  byId("curve-state").textContent = reply.status === "drawn" ? "" : reply.message || reply.error;
  if (reply.status !== "drawn") return;
  byId("curve-basis").textContent = reply.basis || "";
  showPlotKey(reply);
  drawPlot(reply);
  const warnings = reply.warnings.map(note => note.text);
  byId("curve-notes").querySelector("ul").replaceChildren(...warnings.map(text => el("li", {text})));
  byId("curve-notes").hidden = !warnings.length;
}

function missingLayerField() {
  for (const [i, layer] of draft.layers.entries()) {
    const fields = listing.material_form[layer.kind].fields;
    const required = [...listing.required_fields[layer.kind]];
    if (fields.open_area) required.push(...plateGeometry(layer));
    if (Object.hasOwn(layer, clothFields[0])) required.push(...clothFields);
    if (layer.kind === "porous") {
      required.push(...porousModelFields(layer));
      const frame = ["frame_compression_modulus_pa", "frame_loss_factor"];
      if (frame.some(key => layer[key] != null)) required.push(...frame);
    }
    const missing = required.find(key => layer[key] == null && fields[key].default == null &&
      !(key === "thickness_m" && adjustableAir(layer)));
    if (layer.kind === "air" && missing === "thickness_m") return `第 ${i + 1} 層 空氣層的厚度還沒填`;
    if (missing) return `第 ${i + 1} 層的${fields[missing].label}還沒填`;
  }
  return "";
}

function requestCurve() {
  const ticket = ++curveGeneration;
  clearTimeout(curveTimer);
  const backing = tuningActive() ? "rigid" : draft.backing;
  renderProvenance();
  byId("condition").textContent = (tuningActive() ? "初值排法・" : "") + "無規入射・背後：" + backingLabels[backing] +
    (backing === "zone_wall" ? "（先當硬牆算）" : "");
  byId("save-message").textContent = "";
  byId("curve-state").textContent = plot ? "改過了，停手後重算" : "計算中…";
  // 預覽只取疊層和背後，讓尚未填完的出處、名稱不影響物理曲線。
  const unchanged = unchangedOptimizable();
  const entry = unchanged ? openedEntry : draftDocument();
  const origin = unchangedPorousSource();
  const body = JSON.stringify(readOnly || unchanged ? entry : {...entry, label: "構造預覽",
    provenance: origin?.provenance ?? "model", source: origin?.source ?? "依各層物性推算", notes: [], links: origin?.links ?? []});
  curveTimer = setTimeout(async () => {
    const missing = !unchanged && (missingLayerField() || tuningProblem(entry));
    if (missing) {
      renderCurve({status: "incomplete", message: missing});
      return;
    }
    try {
      const reply = await api("/api/material-curve", {method: "POST", headers: {"Content-Type": "application/json"}, body});
      if (ticket === curveGeneration) renderCurve(reply.body);
    } catch (error) {
      if (ticket === curveGeneration) renderCurve({status: "error", message: error.message});
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

function newConstructionId() {
  return "construction_" + crypto.getRandomValues(new Uint32Array(2)).join("_");
}

function freshConstruction() {
  return {id: newConstructionId(), label: freeLabel("未命名構造"), kind: "construction",
    layers: [], backing: "rigid", placement: "wall", product: false, provenance: "model", source: "材料設計器依各層物性推算", links: [], notes: []};
}

function unchangedOptimizable() {
  return openedEntry?.kind === "optimizable" && snapshot() === savedSnapshot;
}

function unchangedPorousSource() {
  const original = copiedOptimizable ?? openedEntry;
  if (original?.kind !== "optimizable" || !tuningActive()) return null;
  const porous = draft.layers[0];
  const same = Object.keys({...original.porous, ...porous}).every(key =>
    key === "thickness_m" || (porous[key] ?? null) === (original.porous[key] ?? null));
  return same ? original : null;
}

function renderProvenance() {
  const provenance = listing.provenance_labels[readOnly ? draft.provenance : unchangedPorousSource()?.provenance ?? "model"];
  byId("source").textContent = provenance;
  byId("citation-provenance").textContent = provenance;
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
  renderProvenance();
  for (const key of ["source", "links", "notes"]) {
    const input = byId("citation-" + key);
    input.value = key === "source" ? draft.source : (draft[key] ?? []).join("\n");
    input.disabled = readOnly;
  }
  byId("source").title = draft.source;
  byId("workspace").hidden = false;
  byId("entry-status").textContent = isNew ? "從房間側往牆側疊層，儲存後可在分區頁指定。" :
    (readOnly ? (draft.product ? "成品唯讀，可複製到設計器" : "內建材料唯讀，可複製到設計器") : "改這一筆，已經貼在分區上的不會跟著變");
  byId("palette").closest(".add").hidden = readOnly;
  byId("save").hidden = readOnly;
  byId("copy-designer").hidden = isNew;
  byId("copy-designer").disabled = Boolean(listing.user_file_problem);
  byId("copy-designer").textContent = readOnly ? "複製到設計器" : "複製一份";
  renderTuningOptions();
  renderStack(byId("stack").querySelector(".advanced")?.open);
  renderFigure();
}

function modelNotes(original, notes = []) {
  if (original.provenance === "model") return notes;
  const citation = `原來源：${listing.provenance_labels[original.provenance]}，出處：${original.source}，網址：${(original.links ?? []).join("、") || "無"}`;
  const line = citation.replace(/[\r\n]+/g, " ");
  return [line, ...notes.filter(note => note !== line)];
}

function copyIdentity(original, recovered) {
  return {id: recovered?.id ?? freeId(original.id + "_copy", collidingIds), label: freeLabel(original.label + "（複本）", recovered?.id)};
}

function modelCopyFields(original, recovered) {
  const saved = openedEntry ?? original;
  return {...copyIdentity(original, recovered),
    provenance: "model", source: `照「${original.label}」改的`, links: [], notes: modelNotes(saved, original.notes)};
}

function makeCopy(original) {
  copiedOptimizable = openedEntry?.kind === "optimizable" ? openedEntry : null;
  const fields = copiedOptimizable ? copyIdentity(original) : modelCopyFields(original);
  draft = {...draft, ...fields, product: false};
  openedEntry = null;
  uncertainCreate = null;
  isNew = true;
  readOnly = false;
  savedSnapshot = null;
}

async function initDesigner() {
  const [materials, parts] = await Promise.all([api("/api/materials"), api("/api/material-elements")]);
  if (!materials.ok || !parts.ok) throw new Error(materials.body.error || parts.body.error || "讀不到材料資料");
  listing = materials.body;

  elements = parts.body.elements;
  dimensionFields = parts.body.dimension_fields;
  const query = new URLSearchParams(location.search);
  const copying = query.has("copy");
  const id = query.get(copying ? "copy" : "id");
  if (listing.user_file_problem && (id === null || copying)) throw new Error(listing.user_file_problem);
  if (id === null) {
    draft = freshConstruction();
  } else {
    const row = listing.materials.find(item => item.entry.id === id);
    if (!row) throw new Error(listing.user_file_problem || "材料庫找不到這一筆材料");
    const entry = row.entry;
    openedEntry = entry;
    if (entry.kind === "optimizable") {
      draft = optimizableDraft(entry);
    } else if (entry.kind === "construction") {
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
      readOnly = Boolean(entry.product || row.builtin);
    }
  }
  savedSnapshot = copying ? null : snapshot();
  renderPalette();
  renderDesigner();
  requestCurve();
}

async function putConstruction(document, create) {
  // 斷線時只記這一頁剛送出的那份；收到明確拒收就不是遺失回覆。
  if (create) uncertainCreate = structuredClone(document);
  const reply = await api("/api/materials/" + encodeURIComponent(document.id), {method: "PUT",
    headers: {"Content-Type": "application/json", ...(create ? {"If-None-Match": "*"} : {})}, body: JSON.stringify(document)});
  if (reply.ok && !reply.body.material_id) throw new Error(reply.body.error || "沒有收到存檔回覆");
  if (reply.ok || (create && reply.status < 500)) uncertainCreate = null;
  return reply;
}

function comparableEntry(entry) {
  // 清單會展開預設值，並修掉字串兩端空白；只在核對失去回覆的新筆時這樣比較。
  return JSON.stringify(entry, (_, value) => {
    if (typeof value === "string") return value.trim();
    if (!value || typeof value !== "object" || Array.isArray(value)) return value;
    const fields = {...value};
    if (fields.kind === "porous") {
      fields.model ??= listing.material_form.porous.fields.model.default;
      if (fields.flow_resistivity_source === "unspecified") delete fields.flow_resistivity_source;
    }
    return Object.fromEntries(Object.keys(fields).sort()
      .filter(key => fields[key] != null && !(Array.isArray(fields[key]) && !fields[key].length))
      .map(key => [key, fields[key]]));
  });
}

async function recoverCreate() {
  if (!uncertainCreate) return;
  const reply = await api("/api/materials");
  if (!reply.ok || reply.body.user_file_problem || !reply.body.materials) {
    throw new Error(reply.body.user_file_problem || reply.body.error || "無法核對上次存檔");
  }
  listing = reply.body;
  const found = listing.materials.find(row => row.entry.id === uncertainCreate.id);
  if (found && !found.builtin && comparableEntry(found.entry) === comparableEntry(uncertainCreate)) {
    if (!isNew) return found.entry;  // 換種類另存的複本已在檔內，畫面仍編輯原草稿。
    openedEntry = found.entry;
    isNew = false;
  }
  uncertainCreate = null;
}

async function saveConstruction() {
  if (readOnly || byId("save").disabled) return;
  const message = byId("save-message");
  byId("save").disabled = true;
  byId("workspace").inert = true;
  byId("name").disabled = true;
  try {
    const recovered = await recoverCreate();
    const unchanged = !isNew && snapshot() === savedSnapshot;
    const document = unchanged ? structuredClone(openedEntry) : draftDocument();
    if (!unchanged) {
      const problem = missingLayerField() || tuningProblem(document);
      if (problem) throw new Error(problem);
    }
    const switching = !isNew && document.kind !== openedEntry.kind;
    let copy;
    if (switching) {
      const names = listing.entry_kind_labels;
      if (!window.confirm(`這一筆已經存成${names[openedEntry.kind]}，改成${names[document.kind]}只能另存一份新的；原本那筆不動。`)) return;
      copy = modelCopyFields(draft, recovered?.kind === document.kind ? recovered : null);
      Object.assign(document, copy);
    } else if (!unchanged && !unchangedPorousSource()) {
      const original = copiedOptimizable ?? openedEntry ?? draft;
      document.provenance = "model";
      if (document.notes || original.provenance !== "model") document.notes = modelNotes(original, draft.notes);
    }
    const create = isNew || (switching && recovered?.kind !== document.kind);
    let reply = await putConstruction(document, create);
    if (create && reply.status === 409 && reply.body.conflict === "id") {
      collidingIds.add(document.id);
      document.id = newConstructionId();
      if (isNew) draft.id = document.id;
      reply = await putConstruction(document, true);
      if (reply.status === 409 && reply.body.conflict === "id") {
        collidingIds.add(document.id);
        if (isNew) draft.id = newConstructionId();
        throw new Error("自動編號又重複了，請再按一次儲存。");
      }
    }
    if (!reply.ok) throw new Error(reply.body.error || String(reply.status));
    if (copy) Object.assign(draft, copy);
    draft.id = document.id;
    draft.provenance = document.provenance;
    draft.notes = document.notes;
    openedEntry = structuredClone(document);
    copiedOptimizable = null;
    const row = listing.materials.find(item => item.entry.id === draft.id);
    if (row) row.entry = openedEntry;
    else listing.materials.push({entry: openedEntry, builtin: false});
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
  return JSON.stringify(draftDocument(), (_, value) => value && typeof value === "object" && !Array.isArray(value)
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
    renderProvenance();
    byId("save-message").textContent = "";
  });
}

byId("save").addEventListener("click", saveConstruction);
byId("copy-designer").addEventListener("click", () => {
  const url = designerUrl(draft.id, true);
  makeCopy(draft);
  history.replaceState(null, "", url);
  renderDesigner();
  requestCurve();
  byId("save-message").textContent = "這是複本，儲存後會新增一筆參數構造。";
});
byId("name").addEventListener("input", event => {
  draft.label = event.target.value;
  renderProvenance();
  byId("save-message").textContent = "";
});
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
byId("product").addEventListener("change", event => {
  const wasTuning = tuningActive();
  draft.product = event.target.checked;
  if (wasTuning) stackChanged();
  else {
    renderStack(byId("stack").querySelector(".advanced")?.open);
    byId("save-message").textContent = "";
  }
});
initDesigner().catch(error => { byId("entry-status").textContent = error.message; });
