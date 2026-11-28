"use strict";
// 「所有參數」頁（#1343）：照 /api/params 一頁一張表；數字全部從清單來，這裡不寫任何預設。

const byId = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value; else node.setAttribute(key, value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

function shown(param, value) {
  if (param.kind === "boolean") return value ? "開" : "關";
  if (param.kind === "choice") {
    const choice = param.choices.find((item) => item.value === value);
    return choice ? choice.label : String(value);
  }
  return String(value);
}

function range(param) {
  if (param.choices.length) return param.choices.map((item) => item.label).join("／");
  if (param.kind === "boolean") return "開／關";
  const parts = [];
  if (param.min !== null) parts.push(`${param.min_open ? "＞" : "≥"} ${param.min}`);
  if (param.max !== null) parts.push(`${param.max_open ? "＜" : "≤"} ${param.max}`);
  if (!parts.length && param.range_note) return param.range_note;
  const kind = param.kind === "integer" ? "整數" : "";
  const text = [parts.join("、"), kind].filter(Boolean).join(" 的");
  return [text, param.range_note].filter(Boolean).join("；") || "—";
}

function row(param) {
  const fixed = param.default !== null && param.default !== undefined;
  const value = fixed ? shown(param, param.default) : param.default_note;
  return el("tr", {},
    el("td", {text: param.name}),
    el("td", {class: fixed ? "value" : "value none", text: value}),
    el("td", {text: param.unit || "—"}),
    el("td", {text: range(param)}),
    el("td", {class: "blurb", text: param.blurb}),
    el("td", {class: "key", text: param.key}),
    el("td", {class: "from", text: param.default_from || "沒有固定值"}));
}

const COLUMNS = ["c-name", "c-value", "c-unit", "c-range", "c-blurb", "c-key", "c-from"];

function table(page, params) {
  const head = el("tr", {}, ...["參數", "預設", "單位", "範圍", "說明", "代號", "數字住在"].map((text) => el("th", {text})));
  return el("section", {},
    el("h2", {}, page.name, el("a", {href: page.path, text: "打開這一頁"})),
    el("div", {class: "wrap"}, el("table", {}, el("colgroup", {}, ...COLUMNS.map((name) => el("col", {class: name}))),
      el("thead", {}, head), el("tbody", {}, ...params.map(row)))));
}

// 吸音模型目錄：外殼 <details> 寫在 HTML、預設收起；這裡只填內容，不打開。
function cite(citation) {
  if (!citation) return "repo 未列引用";
  const parts = [citation.authors.join("、"), citation.year, citation.reference, citation.doi];
  if (citation.verified) parts.push(`書目核對：${citation.verified}`);
  if (citation.note) parts.push(citation.note);
  return parts.join("；");
}

// 核過的外部書目：書目核過不等於 aosr 照它實作，標語跟著伺服器給的 note 走。
function refList(block) {
  if (!block.items.length) return null;
  return el("div", {class: "mrefs"}, el("p", {class: "mlabel", text: block.note}),
    el("ul", {}, ...block.items.map((item) =>
      el("li", {text: `${item.text} ${item.doi}（書目核對：${item.verified}）`}))));
}

function unimplementedCard(item) {
  return el("article", {class: "model"},
    el("h3", {text: `${item.name}（沒有實作）`}),
    el("p", {class: "say", text: item.why}, el("span", {class: "mat", text: `（${item.at}）`})),
    refList({note: "出處文獻", items: item.references}));
}

function rangeText(span) {
  const rows = span.rows.map((item) => `${item.name} ${Array.isArray(item.value) ? item.value.join("、") : item.value}${item.unit ? " " + item.unit : ""}`);
  return [span.note, ...rows].join("；");
}

function modelCard(model) {
  const params = model.params.length
    ? el("ul", {class: "mparams"}, ...model.params.map((param) =>
      el("li", {text: [param.name, param.unit, param.bounds].filter(Boolean).join("｜")})))
    : el("p", {class: "faint", text: "沒有可填參數"});
  const line = (label, text, at) => el("p", {class: "mline"},
    el("span", {class: "mlabel", text: label}), text, at ? el("span", {class: "mat", text: `（${at}）`}) : null);
  return el("article", {class: "model"},
    el("h3", {text: model.name}),
    el("p", {class: "say", text: model.blurb}),
    params,
    line("有效範圍", rangeText(model.range), model.range.at),
    line("引用", cite(model.citation), model.citation ? model.citation.at : null),
    line("程式", [model.code, ...model.compute.filter((name) => name !== model.code)].join("、"), null),
    line("來源", model.origin.text, model.origin.at),
    refList(model.related),
    refList(model.background));
}

async function load() {
  const message = byId("message");
  let body;
  try {
    const reply = await fetch("/api/params", {cache: "no-store"});
    body = await reply.json();
    if (!reply.ok) throw new Error(body.error || `狀態 ${reply.status}`);
  } catch (error) {
    message.textContent = `參數清單讀不到：${error.message || error}`;
    message.classList.add("bad");
    return;
  }
  message.textContent = "";
  byId("source").textContent = `清單檔：${body.registry}`;
  byId("groups").replaceChildren(...body.pages
    .map((page) => [page, body.params.filter((param) => param.page === page.id)])
    .filter(([, params]) => params.length)
    .map(([page, params]) => table(page, params)));
  byId("catalogue").textContent = `目錄檔：${body.catalogue}`;
  byId("model-list").replaceChildren(...body.models.map(modelCard),
    ...(body.unimplemented.length ? [el("h3", {class: "msub", text: "沒有實作的模型"}), ...body.unimplemented.map(unimplementedCard)] : []));
}

window.addEventListener("DOMContentLoaded", load);
