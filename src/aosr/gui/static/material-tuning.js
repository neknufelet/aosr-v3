"use strict";

// 開關和範圍跟著各層移動，不把設計器的狀態混進物理層的欄位。
const thicknessSpecs = new WeakMap();
let depthLimits = {};
let openedEntry = null;

function optimizableDraft(entry) {
  const {porous, porous_thickness, cavity, max_total_depth_m, suggested_total_depth_m,
    preset_members, template_preset, ...common} = structuredClone(entry);
  const layers = [porous];
  thicknessSpecs.set(porous, porous_thickness);
  if (cavity) {
    const air = {kind: "air", thickness_m: cavity.initial_m};
    layers.push(air);
    thicknessSpecs.set(air, cavity);
  }
  depthLimits = {max_total_depth_m, suggested_total_depth_m};
  return {...common, kind: "construction", layers, backing: "rigid", placement: "wall", product: false};
}

function tuningUnavailable() {
  if (draft.product) return "成品不拆，不給程式調";
  const layers = draft.layers;
  if (layers[0]?.kind === "porous" && (layers.length === 1 || (layers.length === 2 && layers[1].kind === "air"))) return "";
  return "第二批才能給程式調";
}

function tuningActive() {
  return !tuningUnavailable() && draft.layers.some(layer => thicknessSpecs.get(layer)?.adjustable);
}

function tuningControl(layer, index) {
  const reason = tuningUnavailable();
  const spec = thicknessSpecs.get(layer);
  const checked = !reason && Boolean(spec?.adjustable);
  const toggle = button("", () => {
    const depth = layer.thickness_m ?? 0;
    if (!spec) thicknessSpecs.set(layer, {range_m: [layer.kind === "air" ? 0 : depth, depth], choices_m: null});
    thicknessSpecs.get(layer).adjustable = !checked;
    selected = index;
    stackChanged();
    byId("stack").querySelectorAll('[role="switch"]')[index].focus();
  }, {role: "switch", "aria-label": `第 ${index + 1} 層給程式調`, "aria-checked": String(checked), disabled: readOnly || Boolean(reason)});
  return el("div", {class: "tune"}, el("span", {}, "給程式調 ", toggle), reason ? el("span", {text: reason}) : null);
}

function meters(value) {
  return value === "" ? null : displayNumber(Number(value) / 1000);
}

function millimeters(value) {
  return value == null ? "" : displayNumber(value * 1000);
}

function tuningFields(layer) {
  const spec = thicknessSpecs.get(layer);
  if (!tuningActive() || !spec?.adjustable) return null;
  const fields = ["下限", "上限"].map((label, index) => {
    const input = el("input", {type: "number", step: "any", min: 0, disabled: readOnly, value: millimeters(spec.range_m[index]),
      "aria-label": `範圍${label}（mm）`});
    input.addEventListener("input", () => { spec.range_m[index] = meters(input.value); requestCurve(); });
    return el("label", {}, label, input);
  });
  if (layer.kind === "porous") {
    const input = el("input", {type: "text", disabled: readOnly, value: spec.choicesText ?? (spec.choices_m ?? []).map(millimeters).join(", "),
      placeholder: "多個厚度用逗號或空白分隔"});
    input.addEventListener("input", () => {
      spec.choicesText = input.value;
      spec.choices_m = input.value.trim().split(/[\s,，]+/).filter(Boolean).map(meters);
      requestCurve();
    });
    fields.push(el("label", {}, "買得到的厚度（mm）", input));
  }
  return el("fieldset", {class: "geometry"}, el("legend", {text: "範圍（mm）"}), el("div", {class: "params"}, ...fields));
}

function renderTuningOptions() {
  const active = tuningActive();
  document.querySelector(".backs").hidden = active;
  document.querySelector(".where").parentElement.hidden = active;
  byId("tuned-backing").hidden = byId("depth-limits").hidden = !active;
  for (const input of byId("depth-limits").querySelectorAll("input")) {
    input.value = millimeters(depthLimits[input.name]);
    input.disabled = readOnly;
  }
}

function thicknessDocument(layer) {
  const spec = thicknessSpecs.get(layer);
  const initial = layer.thickness_m ?? null;
  if (!spec?.adjustable) {
    const fixed = initial ?? 0;
    const result = {adjustable: false, range_m: [fixed, fixed], initial_m: initial};
    if (spec && Object.hasOwn(spec, "choices_m")) {
      result.choices_m = layer.kind === "porous" && spec.choices_m?.length ? [fixed] : null;
    }
    return result;
  }
  const {choicesText, ...active} = spec;
  return {...active, initial_m: initial};
}

function draftDocument() {
  if (!tuningActive()) {
    // 空腔初值為零或空白，在固定的初值排法裡沒有這一層。
    const layers = draft.layers.filter(layer => !(layer.kind === "air" && thicknessSpecs.has(layer) && !layer.thickness_m));
    return {...draft, layers};
  }
  const {layers, backing, placement, product, ...common} = draft;
  const [porous, air] = layers;
  const original = openedEntry?.kind === "optimizable" ? openedEntry : {};
  const entry = {...original, ...common, kind: "optimizable", porous, porous_thickness: thicknessDocument(porous)};
  if (air || Object.hasOwn(original, "cavity")) entry.cavity = air ? thicknessDocument(air) : null;
  for (const [key, value] of Object.entries(depthLimits)) {
    if (value != null || Object.hasOwn(original, key)) entry[key] = value ?? null;
  }
  return entry;
}

function tuningProblem(entry) {
  if (entry.kind !== "optimizable") return "";
  if (entry.porous.frame_compression_modulus_pa != null) return "棉厚給程式調時不能填骨架壓縮模數與骨架損耗因數";
  for (const [name, spec] of [["棉厚", entry.porous_thickness], ["空氣深度", entry.cavity]]) {
    if (!spec) continue;
    const [low, high] = spec.range_m;
    if (low == null || low < 0) return name + "範圍下限要填非負數";
    if (high == null || high < low) return name + "範圍上限要填數字，且不能小於下限";
    const initial = spec.initial_m ?? 0;
    if (initial < low || initial > high) return name + "初值要落在範圍內";
    if (spec === entry.porous_thickness && spec.adjustable) {
      const choices = spec.choices_m;
      if (!choices?.length || choices.some((v, i) => !Number.isFinite(v) || v <= 0 || (i && v <= choices[i - 1]))) {
        return "買得到的厚度必填，請從薄到厚填寫不重複的正數";
      }
      if (choices[0] !== low || choices.at(-1) !== high) return "棉厚範圍要等於買得到的厚度的最薄與最厚";
      if (!choices.includes(initial)) return "棉厚初值必須是買得到的厚度之一";
    }
  }
  const cap = entry.max_total_depth_m, suggested = entry.suggested_total_depth_m;
  const sum = (a, b) => a + (b ?? 0);
  const low = sum(entry.porous_thickness.range_m[0], entry.cavity?.range_m[0]);
  const high = sum(entry.porous_thickness.range_m[1], entry.cavity?.range_m[1]);
  if (cap != null && (cap <= 0 || low > cap || sum(entry.porous.thickness_m, entry.cavity?.initial_m) > cap)) {
    return "總深度上限要容得下最薄排法與初值總深";
  }
  if (suggested != null && (suggested <= 0 || suggested < low || suggested > Math.min(high, cap ?? Infinity))) {
    return "建議總深排不出來，請核對各層範圍與總深度上限";
  }
  return "";
}

for (const input of byId("depth-limits").querySelectorAll("input")) {
  input.addEventListener("input", () => { depthLimits[input.name] = meters(input.value); requestCurve(); });
}
