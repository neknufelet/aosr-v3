"use strict";

function elementSwatch(row) {
  const texture = materialTexture(row.kind, row.material_class);
  const svg = materialSvg("svg", {viewBox: "0 0 50 50", class: "element-swatch", role: "img",
    "aria-label": row.name + "花紋"});
  svg.append(materialSvg("rect", {width: 50, height: 50, fill: texture === "air" ? "white" : `url(#mat-${texture})`}));
  return svg;
}

function elementRow(row, labels) {
  const templates = Object.keys(row.templates);
  const sizes = Object.entries(row.templates).filter(([, lines]) => lines.length);
  const specs = Object.entries(row.families).map(([id, meters]) => el("div", {}, el("small", {text: id}),
    meters.map(m => Number((m * 1000).toPrecision(15))).join("、") + " mm"));
  const sizeList = el("dl", {class: "element-sizes"});
  for (const [label, lines] of sizes) {
    sizeList.append(el("dt", {text: label}), ...lines.map(text => el("dd", {text})));
  }
  return el("article", {class: "element-grid element-row"},
    elementSwatch(row),
    el("div", {class: "element-name"}, el("strong", {text: row.name}),
      el("small", {text: row.template_note + templates.join("、")})),
    el("div", {}, el("b", {class: "element-mobile-label", text: "自己的參數"}),
      ...(row.parameters.length ? row.parameters : ["沒有自己的參數"]).map(text => el("div", {text})),
      el("b", {class: "element-range", text: row.range_title}), ...row.ranges.map(text => el("div", {text}))),
    el("div", {}, el("b", {class: "element-mobile-label", text: "規格"}), row.spec_title, ...specs),
    el("div", {}, el("b", {class: "element-mobile-label", text: "內建樣板用過的尺寸"}),
      sizes.length ? sizeList : templates.length ? "這一種沒有尺寸欄" : "—"),
    el("div", {}, el("b", {class: "element-mobile-label", text: "數據來源"}),
      ...row.provenance.map(key => sourceBadge(key, labels)), ...row.flow_sources.map(text => el("small", {text})),
      row.provenance.length ? null : "—"));
}

async function loadElements() {
  const list = byId("element-list");
  try {
    const reply = await api("/api/material-elements");
    if (!reply.ok) throw new Error(reply.body.error || reply.status);
    const rows = reply.body.elements;
    const head = el("div", {class: "element-grid columns"}, ...["花紋", "元件", "自己的參數與範圍",
      "規格", "內建樣板用過的尺寸", "數據來源"].map(text => el("div", {text})));
    list.replaceChildren(head, ...rows.map(row => elementRow(row, reply.body.provenance_labels)));
    byId("tab-elements").querySelector("small").textContent = rows.length + " 種";
  } catch (error) {
    list.textContent = "讀不到元件：" + error.message;
  }
}
