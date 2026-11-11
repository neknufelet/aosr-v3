"use strict";
// 材料編輯頁（#1082 段 2-2a）：清單與表單。檢查全交給伺服器（材料庫那一套）；這裡只把表單排成材料文件、把回話照實印出來。
// 長度一律公尺，不在畫面換成公厘再換回來（來回換算會讓存回去的數字跟原本差末位）。

const LAYER_SKIP = new Set(["kind"]);
let listing = null;
let current = null; // {entry, builtin, isNew}

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === false || value === null || value === undefined) continue;
    if (name === "text") node.textContent = value;
    else if (name === "on") for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

const byId = (id) => document.getElementById(id);
const form = () => listing.material_form;
const clone = (value) => JSON.parse(JSON.stringify(value));
const shown = (value) => (value === null || value === undefined ? "" : String(value));

async function api(path, options = {}) {
  const reply = await fetch(path, options);
  let body;
  try { body = await reply.json(); } catch { body = {error: `伺服器回的不是 JSON（狀態 ${reply.status}）`}; }
  return {ok: reply.ok, status: reply.status, body};
}

function kindText(entry) {
  if (entry.kind === "optimizable") return listing.entry_kind_labels.optimizable + "：多孔層＋空腔";
  return `${listing.entry_kind_labels.fixed}：${form()[entry.material.kind].label}`;
}

function listTable(rows, builtin) {
  if (!rows.length) return el("p", {text: builtin ? "沒有內建材料" : "還沒有自己的材料；按上面的「新增」，或在下面內建材料按「複製」。"});
  const body = el("tbody");
  for (const row of rows) {
    const entry = row.entry;
    const actions = el("td", {},
      el("button", {type: "button", class: "open", text: builtin ? "看內容" : "打開", on: {click: () => openEntry(row, false)}}),
      el("button", {type: "button", class: "copy quiet", text: "複製", on: {click: () => copyEntry(entry)}}),
      builtin ? null : el("button", {type: "button", class: "delete danger", text: "刪除", on: {click: () => deleteEntry(entry)}}));
    body.append(el("tr", {"data-id": entry.id},
      el("td", {text: entry.label}), el("td", {text: entry.id, "data-head": "代號"}),
      el("td", {text: kindText(entry), "data-head": "種類"}), el("td", {text: row.basis, "data-head": "阻抗怎麼得到"}),
      el("td", {text: listing.provenance_labels[entry.provenance], "data-head": "數字從哪來"}), actions));
  }
  const head = el("tr", {}, ...["顯示名", "代號", "種類", "阻抗怎麼得到", "數字從哪來", ""].map((text) => el("th", {text})));
  return el("table", {}, el("thead", {}, head), body);
}

async function loadListing(reopenId = null) {
  const reply = await api("/api/materials");
  if (!reply.ok) {
    byId("list-state").textContent = `讀不到材料庫：${reply.body.error || reply.status}`;
    byId("list-state").className = "notice";
    return;
  }
  listing = reply.body;
  const rows = listing.materials;
  byId("builtin-list").replaceChildren(listTable(rows.filter((row) => row.builtin), true));
  byId("user-list").replaceChildren(listTable(rows.filter((row) => !row.builtin), false));
  const problem = listing.user_file_problem;
  byId("file-problem").hidden = !problem;
  byId("file-problem").textContent = problem ? `${problem}（修好之前不能存、不能刪）` : "";
  for (const id of ["new-optimizable", "new-fixed"]) byId(id).disabled = Boolean(problem);
  const mine = rows.filter((row) => !row.builtin).length;
  byId("list-state").textContent = `內建 ${rows.length - mine} 筆、我的材料 ${mine} 筆`;
  byId("list-state").className = "";
  if (reopenId) {
    const row = rows.find((item) => item.entry.id === reopenId);
    if (row) openEntry(row, false);
  }
}

function freeId(base) {
  const taken = new Set(listing.materials.map((row) => row.entry.id));
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) if (!taken.has(`${base}_${n}`)) return `${base}_${n}`;
}

function freeLabel(base) {
  const taken = new Set(listing.materials.map((row) => row.entry.label));
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) if (!taken.has(`${base} ${n}`)) return `${base} ${n}`;
}

function defaultsFor(kind) {
  const layer = {kind};
  for (const [name, spec] of Object.entries(form()[kind].fields)) {
    if (!LAYER_SKIP.has(name) && spec.default !== null && spec.default !== undefined) layer[name] = spec.default;
  }
  return layer;
}

function newEntry(kind) {
  const common = {id: freeId("new_material"), label: freeLabel("新材料"), provenance: null, source: "", links: [], notes: []};
  if (kind === "optimizable") {
    return {...common, kind, porous: {...defaultsFor("porous"), material_class: "fibrous", thickness_m: 0.05},
            porous_thickness: {adjustable: true, range_m: [0.05, 0.1], choices_m: [0.05, 0.1], initial_m: 0.05},
            cavity: {adjustable: true, range_m: [0, 0.2], choices_m: null, initial_m: null},
            max_total_depth_m: null, suggested_total_depth_m: null};
  }
  return {...common, kind, material: {kind: "assembly", backing: "rigid", layers: [{...defaultsFor("porous"), material_class: "fibrous"}]}};
}

function copyEntry(entry) {
  const copy = clone(entry);
  copy.id = freeId(`${entry.id}_copy`);
  copy.label = freeLabel(`${entry.label}（複本）`);
  // 複本是新材料：舊家族對應（構造預設代號、換樣板構造）只屬於內建那一筆，不帶過去。
  delete copy.preset_members;
  delete copy.template_preset;
  openEntry({entry: copy, builtin: false}, true);
  say("這是複本，還沒存；改好按「儲存」才會存進我的材料。", "");
}

async function deleteEntry(entry) {
  if (!window.confirm(`確定刪除「${entry.label}」？刪了就不見了。`)) return;
  const reply = await api(`/api/materials/${encodeURIComponent(entry.id)}/delete`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  if (!reply.ok) { window.alert(reply.body.error || `刪不掉（狀態 ${reply.status}）`); return; }
  if (current && current.entry.id === entry.id) closeEditor();
  await loadListing();
  byId("list-state").textContent += `；已刪除「${entry.label}」`;
}

function say(text, kind) {
  const node = byId("editor-message");
  node.textContent = text;
  node.className = kind;
}

// ---- 欄位 ----

function fieldLabel(spec, required) {
  return spec.label + (spec.unit ? `（${spec.unit}）` : "") + (required ? "＊" : "");
}

function inputFor(name, spec, value, required, readonly) {
  const options = Object.entries(spec.options || {});
  if (options.length) {
    const select = el("select", {name, disabled: readonly});
    if (!required) select.append(el("option", {value: "", text: "（不填）"}));
    for (const [code, text] of options) select.append(el("option", {value: code, text}));
    select.value = shown(value ?? spec.default);
    return select;
  }
  return el("input", {name, type: "number", step: "any", value: shown(value ?? spec.default), disabled: readonly});
}

function layerFields(kind, values, readonly, skip = new Set()) {
  const box = el("div", {class: "fields"});
  const required = new Set(listing.required_fields[kind] || []);
  for (const [name, spec] of Object.entries(form()[kind].fields)) {
    if (LAYER_SKIP.has(name) || skip.has(name)) continue;
    box.append(el("label", {}, fieldLabel(spec, required.has(name)), inputFor(name, spec, values[name], required.has(name), readonly)));
  }
  return box;
}

function readFields(box) {
  const values = {};
  for (const input of box.querySelectorAll("input[name], select[name]")) {
    if (input.value === "") continue;
    values[input.name] = input.type === "number" ? Number(input.value) : input.value;
  }
  return values;
}

function numberInput(id, value, readonly) {
  return el("input", {id, type: "number", step: "any", value: shown(value), disabled: readonly});
}

function labelled(text, input, hint = null) {
  return el("label", {}, text, input, hint ? el("span", {class: "hint", text: hint}) : null);
}

function readNumber(id) {
  const text = byId(id).value;
  return text === "" ? null : Number(text);
}

function lines(id) {
  return byId(id).value.split("\n").map((line) => line.trim()).filter(Boolean);
}

function keptLines(id, original) {
  // 沒動過那一格就原樣存回：一條註記裡本來就有換行時，不拆成好幾條。
  const before = original || [];
  return byId(id).value === before.join("\n") ? before : lines(id);
}

function numberList(text, what) {
  const parts = text.split(/[\s,，、]+/).filter(Boolean);
  const numbers = parts.map(Number);
  if (numbers.some((value) => !Number.isFinite(value))) throw new Error(`${what}只能填數字，用逗號或空白分開`);
  return numbers;
}

// ---- 多層構造：加層、刪層、換層、調順序（#850 那四件） ----

function layerEditor(id, layers, readonly) {
  const list = el("ol", {class: "layers", id});
  const kinds = listing.layer_kinds;
  const renumber = () => list.querySelectorAll(":scope > li").forEach((item, index) => {
    item.querySelector("strong").textContent = `第 ${index + 1} 層`;
  });
  const addLayer = (layer, after = null) => {
    const kindSelect = el("select", {name: "layer-kind", "aria-label": "這一層的種類", disabled: readonly});
    for (const kind of kinds) kindSelect.append(el("option", {value: kind, text: form()[kind].label}));
    kindSelect.value = layer.kind;
    let fields = layerFields(layer.kind, layer, readonly);
    const item = el("li", {class: "layer"});
    kindSelect.addEventListener("change", () => {
      // 換種類：同名的格子（例如厚度）留著，其餘換成新種類的預設。
      const kept = readFields(fields);
      const next = layerFields(kindSelect.value, {...defaultsFor(kindSelect.value), ...kept}, readonly);
      fields.replaceWith(next);
      fields = next;
    });
    const head = el("div", {class: "layer-head"}, el("strong"), kindSelect);
    if (!readonly) {
      head.append(
        el("button", {type: "button", class: "quiet up", text: "上移", on: {click: () => {
          if (item.previousElementSibling) item.previousElementSibling.before(item); renumber(); }}}),
        el("button", {type: "button", class: "quiet down", text: "下移", on: {click: () => {
          if (item.nextElementSibling) item.nextElementSibling.after(item); renumber(); }}}),
        el("button", {type: "button", class: "quiet insert", text: "下面加一層", on: {click: () => addLayer({kind: "air", thickness_m: null}, item)}}),
        el("button", {type: "button", class: "danger remove", text: "刪這層", on: {click: () => { item.remove(); renumber(); }}}));
    }
    item.append(head, fields);
    item.readLayer = () => ({kind: kindSelect.value, ...readFields(fields)});
    if (after) after.after(item); else list.append(item);
    renumber();
  };
  for (const layer of layers) addLayer(layer);
  const box = el("div", {}, el("p", {class: "hint", text: "由房間側往牆體側排；第一層是面向房間的那一層。"}), list);
  if (!readonly) box.append(el("button", {type: "button", class: "quiet add-layer", text: "最後加一層",
    on: {click: () => addLayer(defaultsFor("porous"))}}));
  box.readLayers = () => [...list.querySelectorAll(":scope > li")].map((item) => item.readLayer());
  return box;
}

function backingSelect(id, value, readonly) {
  const spec = form().assembly.fields.backing;
  const select = el("select", {id, disabled: readonly});
  for (const [code, text] of Object.entries(spec.options)) select.append(el("option", {value: code, text}));
  select.value = value || "rigid";
  return select;
}

// ---- 表單 ----

function commonBlock(entry, readonly, isNew) {
  const provenance = el("select", {id: "provenance", disabled: readonly}, el("option", {value: "", text: "（請選）"}));
  for (const [code, text] of Object.entries(listing.provenance_labels)) provenance.append(el("option", {value: code, text}));
  provenance.value = entry.provenance || "";
  return el("div", {class: "block"},
    el("div", {class: "fields"},
      labelled("顯示名＊", el("input", {id: "material-label", value: entry.label, disabled: readonly})),
      labelled("代號＊", el("input", {id: "material-id", value: entry.id, disabled: readonly || !isNew}),
               isNew ? "小寫英文字母開頭，只能用小寫英數與底線；存了就不能改" : "代號存了就不能改"),
      labelled("數字從哪來＊", provenance)),
    labelled("出處與條件＊", el("textarea", {id: "source", disabled: readonly}, entry.source || "")),
    labelled("出處網址（一行一個，https:// 開頭；選填）", el("textarea", {id: "links", disabled: readonly}, (entry.links || []).join("\n"))),
    labelled("註記（一行一條；選填）", el("textarea", {id: "notes", disabled: readonly}, (entry.notes || []).join("\n"))));
}

function toggle(id, checked, text, readonly, onChange) {
  const box = el("input", {id, type: "checkbox", disabled: readonly});
  box.checked = checked;
  box.addEventListener("change", onChange);
  return el("label", {class: "toggle"}, box, text);
}

function optimizableBlock(entry, readonly) {
  const porous = entry.porous_thickness;
  const cavity = entry.cavity;
  const porousBox = layerFields("porous", entry.porous, readonly, new Set(["thickness_m"]));
  porousBox.id = "porous-fields";
  const porousAdjustable = el("div", {id: "porous-adjustable-fields", class: "fields"},
    labelled("產品厚度（m，由薄到厚，逗號或空白分開）", el("input", {id: "porous-choices", value: (porous.choices_m || []).join(", "), disabled: readonly}),
             "搜尋最後取這幾個規格之一"),
    labelled("初值（m，要是產品厚度之一）", numberInput("porous-initial", porous.initial_m, readonly)));
  const porousFixed = el("div", {id: "porous-fixed-fields", class: "fields"},
    labelled("厚度（m）", numberInput("porous-thickness", porous.initial_m, readonly)));
  const showPorous = () => {
    const adjustable = byId("porous-adjustable").checked;
    porousAdjustable.hidden = !adjustable;
    porousFixed.hidden = adjustable;
  };
  const cavityFields = el("div", {id: "cavity-fields"},
    toggle("cavity-adjustable", cavity ? cavity.adjustable : true, "搜尋可以調空腔深度", readonly, () => showCavity()),
    el("div", {id: "cavity-range-fields", class: "fields"},
      labelled("下限（m）", numberInput("cavity-low", cavity ? cavity.range_m[0] : 0, readonly)),
      labelled("上限（m）", numberInput("cavity-high", cavity ? cavity.range_m[1] : 0.2, readonly)),
      labelled("初值（m）", numberInput("cavity-initial", cavity ? cavity.initial_m : null, readonly), "空白＝一開始沒有空氣層（下限要是 0）")),
    el("div", {id: "cavity-fixed-fields", class: "fields"},
      labelled("深度（m）", numberInput("cavity-depth", cavity ? cavity.range_m[0] : 0.1, readonly))));
  const showCavity = () => {
    const present = byId("cavity-present").checked;
    const adjustable = byId("cavity-adjustable").checked;
    cavityFields.hidden = !present;
    byId("cavity-range-fields").hidden = !adjustable;
    byId("cavity-fixed-fields").hidden = adjustable;
  };
  const block = el("div", {},
    el("div", {class: "block"}, el("h3", {text: "多孔層"}),
      el("p", {class: "hint", text: "流阻是產品本身的數字，搜尋不調；搜尋只調厚度與空腔。"}), porousBox),
    el("div", {class: "block"}, el("h3", {text: "多孔層厚度"}),
      toggle("porous-adjustable", porous.adjustable, "搜尋可以調厚度（取產品厚度）", readonly, showPorous), porousAdjustable, porousFixed),
    el("div", {class: "block"}, el("h3", {text: "空腔（多孔層後面、靠牆那一側的空氣層）"}),
      toggle("cavity-present", Boolean(cavity), "有空腔這一格", readonly, () => showCavity()), cavityFields),
    el("div", {class: "block"}, el("h3", {text: "總深度"}),
      el("div", {class: "fields"},
        labelled("材料自己的總深度上限（m，選填）", numberInput("max-total-depth", entry.max_total_depth_m, readonly), "硬上限：多孔層加空腔不准超過"),
        labelled("建議總深（m，選填）", numberInput("suggested-total-depth", entry.suggested_total_depth_m, readonly), "只當預設值"))));
  return {block, after: () => { showPorous(); showCavity(); }};
}

function pointsText(points) {
  return (points || []).map((row) => [row.frequency_hz, row.real_pa_s_m, row.imag_pa_s_m].filter((value) => value !== null && value !== undefined).join(" ")).join("\n");
}

function fixedBlock(entry, readonly) {
  const material = entry.material;
  const kindSelect = el("select", {id: "fixed-kind", disabled: readonly});
  for (const kind of listing.fixed_kinds) kindSelect.append(el("option", {value: kind, text: form()[kind].label}));
  kindSelect.value = material.kind;
  const assembly = material.kind === "assembly" ? material : {layers: [defaultsFor("porous")], backing: "rigid"};
  const measured = material.kind === "measured_impedance" ? material : {points: [], base_assembly: {layers: [{kind: "air", thickness_m: 0.1}], backing: "rigid"}, source_and_conditions: ""};
  const assemblyLayers = layerEditor("assembly-layers", assembly.layers, readonly);
  const baseLayers = layerEditor("base-layers", measured.base_assembly.layers, readonly);
  const pointSpec = form().measured_impedance.point_fields;
  const panels = {
    assembly: el("div", {id: "panel-assembly", class: "block"}, el("h3", {text: "多層構造"}), assemblyLayers,
      el("div", {class: "fields"}, labelled("背後負載", backingSelect("assembly-backing", assembly.backing, readonly)))),
    constant_impedance: el("div", {id: "panel-constant_impedance", class: "block"}, el("h3", {text: "固定阻抗"}),
      el("div", {class: "fields"}, labelled("阻抗（Pa·s/m）", numberInput("impedance", material.kind === "constant_impedance" ? material.impedance_pa_s_m : null, readonly))),
      el("p", {class: "hint", text: "每個頻率都同一個實數阻抗；畫面標「近似」。"})),
    measured_impedance: el("div", {id: "panel-measured_impedance", class: "block"}, el("h3", {text: "實測阻抗表"}),
      labelled(`一列一個頻率：${pointSpec.frequency_hz.label}（Hz） ${pointSpec.real_pa_s_m.label}（Pa·s/m） [${pointSpec.imag_pa_s_m.label}（Pa·s/m）]`,
        el("textarea", {id: "points", disabled: readonly, placeholder: "100 950 -320\n200 820 -210"}, pointsText(measured.points)),
        "沒有虛部就每列只寫兩個數（整張都不寫，存了標「近似」）；只缺幾列的表不收。頻率由低到高。"),
      labelled(fieldLabel(form().measured_impedance.fields.source_and_conditions, true),
        el("textarea", {id: "measured-source", disabled: readonly}, measured.source_and_conditions || "")),
      el("h3", {text: "底構造（表外的頻率用它補值）"}), baseLayers,
      el("div", {class: "fields"}, labelled("背後負載", backingSelect("base-backing", measured.base_assembly.backing, readonly)))),
  };
  const show = () => { for (const [kind, panel] of Object.entries(panels)) panel.hidden = kind !== kindSelect.value; };
  kindSelect.addEventListener("change", show);
  const block = el("div", {}, el("div", {class: "block"}, el("div", {class: "fields"}, labelled("固定材料的種類", kindSelect)),
    el("p", {class: "hint", text: "固定材料搜尋只能整塊選或不選，不會微調。換種類時另外兩種填的內容先留著，存的是現在選的這一種。"})),
    ...Object.values(panels));
  return {block, after: show, assemblyLayers, baseLayers};
}

let parts = null;

function openEntry(row, isNew) {
  const entry = row.entry;
  const readonly = row.builtin;
  current = {entry, builtin: row.builtin, isNew};
  byId("editor-title").textContent = isNew ? `新材料：${entry.label}` : `${row.builtin ? "內建材料" : "我的材料"}：${entry.label}`;
  byId("editor-readonly").hidden = !readonly;
  const kindBlock = entry.kind === "optimizable" ? optimizableBlock(entry, readonly) : fixedBlock(entry, readonly);
  parts = kindBlock;
  byId("editor-body").replaceChildren(
    el("p", {class: "hint", text: `種類：${entry.kind === "optimizable" ? listing.entry_kind_labels.optimizable + "（搜尋可以調厚度與空腔）" : listing.entry_kind_labels.fixed + "（整塊選或不選）"}`}),
    commonBlock(entry, readonly, isNew), kindBlock.block);
  kindBlock.after();
  byId("save").hidden = readonly;
  byId("save").disabled = Boolean(listing.user_file_problem);
  byId("copy").hidden = isNew;
  byId("copy").disabled = Boolean(listing.user_file_problem);
  byId("editor").hidden = false;
  say("", "");
  byId("editor").scrollIntoView({block: "start"});
}

function closeEditor() {
  current = null;
  parts = null;
  byId("editor").hidden = true;
  byId("editor-body").replaceChildren();
}

function collect() {
  const base = clone(current.entry);
  const document = {...base, id: byId("material-id").value.trim(), label: byId("material-label").value.trim(),
                    provenance: byId("provenance").value || null, source: byId("source").value.trim(),
                    links: keptLines("links", base.links), notes: keptLines("notes", base.notes)};
  if (document.kind === "optimizable") {
    const adjustable = byId("porous-adjustable").checked;
    const thickness = adjustable ? readNumber("porous-initial") : readNumber("porous-thickness");
    // 不能調的厚度可以帶一個產品規格（就是那個厚度）；原本有、厚度也沒改就留著，不悄悄刪掉。
    const kept = base.porous_thickness && base.porous_thickness.choices_m;
    const choices = adjustable ? numberList(byId("porous-choices").value, "產品厚度")
      : (kept && kept.length === 1 && kept[0] === thickness ? kept : null);
    if (adjustable && !choices.length) throw new Error("搜尋可以調厚度時，產品厚度至少要填一個");
    document.porous = {...readFields(byId("porous-fields")), kind: "porous", thickness_m: thickness};
    document.porous_thickness = {adjustable, choices_m: choices, initial_m: thickness,
      // 範圍取最薄到最厚；順序填錯時讓伺服器講「由薄到厚」那一條，不講畫面上沒有的「範圍」格。
      range_m: adjustable ? [Math.min(...choices), Math.max(...choices)] : [thickness, thickness]};
    if (!byId("cavity-present").checked) document.cavity = null;
    else if (byId("cavity-adjustable").checked) {
      document.cavity = {adjustable: true, choices_m: null, range_m: [readNumber("cavity-low"), readNumber("cavity-high")], initial_m: readNumber("cavity-initial")};
    } else {
      const depth = readNumber("cavity-depth");
      document.cavity = {adjustable: false, choices_m: null, range_m: [depth, depth], initial_m: depth};
    }
    document.max_total_depth_m = readNumber("max-total-depth");
    document.suggested_total_depth_m = readNumber("suggested-total-depth");
    return document;
  }
  const kind = byId("fixed-kind").value;
  const before = base.material.kind === kind ? base.material : {};
  if (kind === "assembly") {
    document.material = {kind, layers: parts.assemblyLayers.readLayers(), backing: byId("assembly-backing").value};
  } else if (kind === "constant_impedance") {
    document.material = {kind, impedance_pa_s_m: readNumber("impedance")};
  } else {
    const points = byId("points").value.split("\n").map((line) => line.trim()).filter(Boolean).map((line, index) => {
      const numbers = numberList(line, `實測表第 ${index + 1} 列`);
      if (numbers.length < 2 || numbers.length > 3) throw new Error(`實測表第 ${index + 1} 列要寫兩或三個數字：頻率 實部 [虛部]`);
      const row = {frequency_hz: numbers[0], real_pa_s_m: numbers[1]};
      if (numbers.length === 3) row.imag_pa_s_m = numbers[2];
      return row;
    });
    // 補值規則版本只有一種，原本有就原樣留著（之後物理換預設時，舊表不會被悄悄改讀）。
    document.material = {...(before.fill_rule ? {fill_rule: before.fill_rule} : {}), kind, points,
      base_assembly: {kind: "assembly", layers: parts.baseLayers.readLayers(), backing: byId("base-backing").value},
      source_and_conditions: byId("measured-source").value.trim()};
  }
  return document;
}

async function save() {
  let document;
  try { document = collect(); } catch (error) { say(error.message, "notice"); return; }
  const headers = {"Content-Type": "application/json"};
  if (current.isNew) headers["If-None-Match"] = "*";
  const target = current.isNew ? document.id : current.entry.id;
  byId("save").disabled = true;
  say("儲存中…", "");
  try {
    const reply = await api(`/api/materials/${encodeURIComponent(target)}`, {method: "PUT", headers, body: JSON.stringify(document)});
    if (!reply.ok) {
      say(`沒有存：\n${String(reply.body.error || reply.status).split("；").join("\n")}`, "notice");
      return;
    }
    await loadListing(reply.body.material_id);
    say(`已儲存「${document.label}」`, "ok");
  } finally {
    byId("save").disabled = Boolean(listing.user_file_problem);
  }
}

byId("new-optimizable").addEventListener("click", () => { openEntry({entry: newEntry("optimizable"), builtin: false}, true); say("新材料，還沒存。", ""); });
byId("new-fixed").addEventListener("click", () => { openEntry({entry: newEntry("fixed"), builtin: false}, true); say("新材料，還沒存。", ""); });
byId("save").addEventListener("click", save);
byId("copy").addEventListener("click", () => current && copyEntry(current.entry));
byId("close").addEventListener("click", closeEditor);
loadListing().catch((error) => { byId("list-state").textContent = `讀不到材料庫：${error.message}`; byId("list-state").className = "notice"; });
