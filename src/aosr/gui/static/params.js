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
}

window.addEventListener("DOMContentLoaded", load);
