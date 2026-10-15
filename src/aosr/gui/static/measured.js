/* 實測草稿與列編輯；所有材料數字交既有方案預覽驗證，這裡不算吸音率。 */
const measuredDrafts = {};
function parsePastedTable(text) {
  const lines = text.replace(/\r/g, "").split("\n");
  while (lines.length && lines.at(-1).trim() === "") lines.pop();
  const points = [];
  lines.forEach((line, index) => {
    if (index === 0 && !/\d/.test(line)) return;
    const tokens = line.trim().split(/[\t,\s]+/);
    if (tokens.length !== 3 || tokens.some(token => !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(token) || !Number.isFinite(Number(token))))
      throw new Error(`貼上表第 ${index + 1} 行要填頻率、實部、虛部三個有限數字`);
    points.push(Object.fromEntries(Object.keys(inputDefaults.material_form.measured_impedance.point_fields).map((key, i) => [key, Number(tokens[i])])));
  });
  if (!points.length) throw new Error("請貼上實測表，每行三個數字");
  return points;
}
function measuredText(id, field, value) {
  const label = document.createElement("label"); label.textContent = field.label;
  const input = document.createElement("textarea"); input.id = id; input.value = value;
  input.placeholder = field.hint || ""; label.append(input); return label;
}
function measuredButton(id, label, work) {
  const button = document.createElement("button"); button.id = id; button.type = "button";
  button.textContent = label; button.onclick = () => action(work); return button;
}
function renderMeasured(name, material, row) {
  const spec = inputDefaults.material_form.measured_impedance;
  const rows = document.createElement("div"); rows.id = `measured-rows-${name}`;
  material.points.forEach((point, index) => {
    const entry = document.createElement("div"); entry.className = "measured-point"; entry.dataset.measuredRow = index;
    const heading = document.createElement("h4"); heading.textContent = `實測表第 ${index + 1} 列`; entry.append(heading);
    for (const [key, field] of Object.entries(spec.point_fields)) {
      const input = numberField(`point-${name}-${index}-${key}`, `${field.label}（${field.unit}）`, point[key]);
      const hint = document.createElement("p"); hint.className = "note"; hint.textContent = field.description;
      input.append(hint); entry.append(input);
    }
    entry.append(measuredButton(`measured-delete-${name}-${index}`, "刪除這列", () => {
      collect(); material.points.splice(index, 1); renderWalls(); markStale();
    }));
    rows.append(entry);
  });
  row.append(rows, measuredButton(`measured-add-${name}`, "增加一列", () => {
    collect(); material.points.push(Object.fromEntries(Object.keys(spec.point_fields).map(key => [key, null])));
    renderWalls(); markStale();
  }), measuredText(`measured-source-${name}`, spec.fields.source_and_conditions, material.source_and_conditions));
  renderAssembly(name, material.base_assembly, row);
}
function collectMeasured(name, material) {
  material.points = material.points.map((point, index) => Object.fromEntries(Object.keys(point).map(key =>
    [key, numericInput(`point-${name}-${index}-${key}`)])));
  material.source_and_conditions = $(`measured-source-${name}`).value;
  collectAssembly(name, material.base_assembly);
}
function startMeasuredDraft(name) {
  collect();
  const material = wallMaterial(name);
  measuredDrafts[name] = {paste: "", source_and_conditions: "", error: "",
    base_assembly: material.kind === "assembly" ? structuredClone(material) : null};
  renderWalls();
}
function renderMeasuredDraft(name, row) {
  const draft = measuredDrafts[name], spec = inputDefaults.material_form.measured_impedance;
  const panel = document.createElement("div"); panel.id = `measured-draft-${name}`; panel.className = "measured-draft";
  row.append(panel);
  const heading = document.createElement("h4"); heading.textContent = "實測阻抗表草稿（按套用才更換方案）";
  const paste = measuredText(`measured-paste-${name}`, {label: "貼上實測表", hint: "每行：頻率、阻抗實部、阻抗虛部；可用定位字元、逗號或空白分隔"}, draft.paste);
  paste.querySelector("textarea").oninput = event => { draft.paste = event.target.value; };
  const source = measuredText(`measured-source-${name}`, spec.fields.source_and_conditions, draft.source_and_conditions);
  source.querySelector("textarea").oninput = event => { draft.source_and_conditions = event.target.value; };
  const convention = document.createElement("p"); convention.textContent = spec.point_fields.imag_pa_s_m.description;
  panel.append(heading, paste, source, convention);
  if (draft.base_assembly) renderAssembly(name, draft.base_assembly, panel, preset => {
    draft.base_assembly = structuredClone(preset.material); renderWalls();
  });
  else {
    const choices = {"": "請選底構造", ...Object.fromEntries(inputDefaults.construction_presets.map(p => [p.id, p.label]))};
    const selector = wallSelect(`preset-${name}`, "同一面牆的底構造", choices, "");
    selector.querySelector("select").onchange = event => {
      draft.base_assembly = structuredClone(inputDefaults.construction_presets.find(p => p.id === event.target.value).material);
      renderWalls();
    };
    panel.append(selector);
  }
  panel.addEventListener("input", event => {
    if (event.target.matches("input[data-layer-field]")) {
      collectAssembly(name, draft.base_assembly); showPresetState(name, draft.base_assembly);
    }
    event.stopPropagation(); // 草稿未套用，不觸發原方案的預覽或存值。
  });
  const error = document.createElement("p"); error.id = `measured-error-${name}`; error.className = "notice";
  error.textContent = draft.error; error.setAttribute("aria-live", "polite"); panel.append(error);
  panel.append(measuredButton(`measured-apply-${name}`, "套用實測表", () => applyMeasuredDraft(name)),
    measuredButton(`measured-cancel-${name}`, "取消草稿", () => { delete measuredDrafts[name]; renderWalls(); }));
  row.append(panel);
}
async function applyMeasuredDraft(name) {
  const draft = measuredDrafts[name];
  try {
    if (!draft.base_assembly) throw new Error("請選底構造，描述同一面牆的構造");
    const points = parsePastedTable(draft.paste);
    const document = wallDocument();
    document.scene.surface_zones_by_wall[name][0].material = {kind: "measured_impedance", points,
      source_and_conditions: draft.source_and_conditions, base_assembly: structuredClone(draft.base_assembly),
      fill_rule: inputDefaults.material_form.measured_impedance.fields.fill_rule.default};
    // 白話拒收理由保留在草稿；正式交接走與其他種類相同的 changeInputScheme。
    if (await changeInputScheme(async () => document)) {
      delete measuredDrafts[name]; renderWalls(); markStale(); await refreshPlan();
    } else {
      draft.error = $("messages").textContent; $(`measured-error-${name}`).textContent = draft.error;
    }
  } catch (error) {
    draft.error = errorText(error); $(`measured-error-${name}`).textContent = draft.error;
  }
}
