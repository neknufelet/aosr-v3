/* 牆面表單只搬完整參數；欄名與單位由伺服器的模型描述提供，不寫聲學公式。 */
let fixedWallDefaults = {};
function wallMaterial(name) {
  return scheme.scene.surface_zones_by_wall?.[name][0].material ?? {
    kind: "constant_impedance", impedance_pa_s_m: scheme.scene.impedance_pa_s_per_m_by_wall[name]};
}
function sameMaterial(a, b) {
  if (a === b) return true;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key => Object.hasOwn(b, key) && sameMaterial(a[key], b[key]));
}
function matchingPreset(material) {
  return inputDefaults.construction_presets.find(preset => sameMaterial(preset.material, material));
}
function wallSelect(id, label, choices, value) {
  const wrap = document.createElement("label"); wrap.textContent = label;
  const select = document.createElement("select"); select.id = id;
  select.setAttribute("aria-label", label);
  chooseOptions(select, Object.keys(choices), choices, value);
  wrap.append(select); return wrap;
}
function assemblyMaterial(material) {
  return material.kind === "measured_impedance" ? material.base_assembly : material;
}
function showPresetState(name, material = assemblyMaterial(wallMaterial(name))) {
  const preset = matchingPreset(material);
  $(`preset-state-${name}`).textContent = preset ? `使用預設：${preset.label}` : "自訂構造（參數與預設不完全相同）";
  $(`preset-${name}`).value = preset?.id ?? "";
  $(`preset-notes-${name}`).textContent = preset ? preset.notes.join("；") : "預設出處只適用於與清單逐值相同的材料；目前以展開的參數計算。";
  const source = $(`preset-source-${name}`); source.replaceChildren();
  if (preset) {
    const text = document.createElement("p"); text.textContent = preset.source; source.append(text);
    for (const link of preset.links) {
      const anchor = document.createElement("a"); anchor.href = link; anchor.textContent = "查看文獻出處";
      source.append(anchor, document.createTextNode("　"));
    }
  }
}
function renderAssembly(name, material, row, onPreset = null) {
  const matched = matchingPreset(material);
  const choices = {"": "自訂構造", ...Object.fromEntries(inputDefaults.construction_presets.map(p => [p.id, p.label]))};
  const selector = wallSelect(`preset-${name}`, `${wallNames[name]}構造預設`, choices, matched?.id ?? "");
  selector.querySelector("option").disabled = true;
  selector.querySelector("select").onchange = event => action(() => onPreset
    ? onPreset(inputDefaults.construction_presets.find(p => p.id === event.target.value))
    : editInput("wall_preset", event.target.value, name));
  row.append(selector);
  const state = document.createElement("p"); state.id = `preset-state-${name}`; row.append(state);
  const details = document.createElement("details"); details.id = `construction-${name}`; details.open = true;
  const summary = document.createElement("summary"); summary.textContent = "展開構造參數（從房間側排到牆體側）"; details.append(summary);
  const spec = inputDefaults.material_form;
  material.layers.forEach((layer, index) => {
    const entry = document.createElement("div"); entry.className = "construction-layer";
    const title = document.createElement("h4"); title.textContent = `第 ${index + 1} 層：${spec[layer.kind].label}`; entry.append(title);
    const fields = document.createElement("div"); fields.className = "fields";
    for (const [key, field] of Object.entries(spec[layer.kind].fields)) {
      if (key === "kind" || !Object.hasOwn(layer, key)) continue;
      if (Object.keys(field.options).length) {
        const text = document.createElement("span"); text.textContent = `${field.label}：${field.options[layer[key]]}`; fields.append(text);
      } else {
        const input = numberField(`layer-${name}-${index}-${key}`, field.label + (field.unit ? `（${field.unit}）` : ""), layer[key]);
        input.querySelector("input").dataset.layerField = key; fields.append(input);
      }
    }
    entry.append(fields); details.append(entry);
  });
  const backing = document.createElement("p"); backing.textContent = `${spec.assembly.fields.backing.label}：${spec.assembly.fields.backing.options[material.backing]}`;
  details.append(backing); row.append(details);
  const notes = document.createElement("p"); notes.id = `preset-notes-${name}`; notes.className = "note"; row.append(notes);
  const source = document.createElement("details"); const heading = document.createElement("summary"); heading.textContent = "預設數字的出處";
  const body = document.createElement("div"); body.id = `preset-source-${name}`; source.append(heading, body); row.append(source);
  showPresetState(name, material);
}
function renderWalls() {
  $("walls").replaceChildren();
  const spec = inputDefaults.material_form;
  for (const name of Object.keys(wallNames)) {
    const material = wallMaterial(name);
    const row = document.createElement("div"); row.className = "wall-editor row"; row.id = `wall-editor-${name}`;
    const choice = wallSelect(`material-kind-${name}`, `${wallNames[name]}材料`,
      Object.fromEntries(["constant_impedance", "assembly", "measured_impedance"].map(kind => [kind, spec[kind].label])), material.kind);
    choice.querySelector("select").onchange = event => action(() => changeWallKind(name, event.target.value));
    row.append(choice); $("walls").append(row);
    if (measuredDrafts[name]) {
      choice.querySelector("select").value = "measured_impedance";
      renderMeasuredDraft(name, row);
    } else if (material.kind === "constant_impedance") {
      row.append(numberField(`wall-${name}`, `${wallNames[name]}阻抗（帕·秒／公尺）`, material.impedance_pa_s_m));
      const multiple = document.createElement("span"); multiple.id = `multiple-${name}`; row.append(multiple);
    } else if (material.kind === "assembly") renderAssembly(name, material, row);
    else if (material.kind === "measured_impedance") renderMeasured(name, material, row);
  }
}
function collectWallMaterials() {
  for (const name of Object.keys(wallNames)) {
    const material = wallMaterial(name);
    if (measuredDrafts[name]) continue;
    if (material.kind === "constant_impedance") {
      if (scheme.scene.surface_zones_by_wall) material.impedance_pa_s_m = numericInput(`wall-${name}`);
      else scheme.scene.impedance_pa_s_per_m_by_wall[name] = numericInput(`wall-${name}`);
    } else if (material.kind === "assembly") {
      collectAssembly(name, material);
    } else if (material.kind === "measured_impedance") collectMeasured(name, material);
  }
}
function wallDocument() {
  const document = structuredClone(collect());
  if (!document.scene.surface_zones_by_wall) {
    document.scene.surface_zones_by_wall = Object.fromEntries(Object.keys(wallNames).map(name =>
      [name, [{area: "whole_wall", material: structuredClone(wallMaterial(name))}]]));
    delete document.scene.impedance_pa_s_per_m_by_wall;
  }
  return document;
}
async function changeWallKind(name, kind) {
  if (kind === "measured_impedance") return startMeasuredDraft(name);
  if (measuredDrafts[name]) {
    delete measuredDrafts[name]; renderWalls();
    if (wallMaterial(name).kind === kind) return;
  }
  if (kind === "assembly" && wallMaterial(name).kind !== "measured_impedance") return editInput("wall_preset", inputDefaults.construction_presets[0].id, name);
  const document = wallDocument();
  document.scene.surface_zones_by_wall[name][0].material = kind === "assembly"
    ? structuredClone(wallMaterial(name).base_assembly) : {
    kind: "constant_impedance", impedance_pa_s_m: fixedWallDefaults[name]};
  if (await changeInputScheme(async () => document)) { markStale(); await refreshPlan(); }
}

function collectAssembly(name, material) {
  if (!material) return;
  material.layers.forEach((layer, index) => {
    for (const input of document.querySelectorAll(`#wall-editor-${name} input[data-layer-field]`)) {
      if (input.id === `layer-${name}-${index}-${input.dataset.layerField}`)
        layer[input.dataset.layerField] = numericInput(input.id);
    }
  });
}
