"use strict";
// 只更新診斷區；原結果與排名保持可見。數字、原因与文案由伺服器格式化。
function loadModalDiagnosis(resultId) {
  const element = (id) => document.getElementById(id);
  const endpoint = `/api/results/${resultId}/modal`;
  let jobId = null, timer = null, busy = false;
  const text = (tag, value) => { const node = document.createElement(tag); node.textContent = value; return node; };
  const table = (headings, rows) => {
    const node = document.createElement("table");
    const head = document.createElement("tr");
    headings.forEach((value) => head.append(text("th", value))); node.append(head);
    rows.forEach((row) => { const line = document.createElement("tr");
      row.forEach((value) => line.append(text("td", value))); node.append(line); });
    const scroll = document.createElement("div"); scroll.className = "modal-table"; scroll.append(node); return scroll;
  };
  const folded = (caption, content) => { const node = document.createElement("details");
    node.append(text("summary", caption), content); return node; };
  const placementRows = (rows) => rows.map((row) => [row.index, row.frequency_text, row.magnitude_text,
    row.group_magnitude_text, row.relative_text]);
  function report(view) {
    const target = element("modal-report"); target.replaceChildren();
    if (view.state !== "diagnosed_not_scored") return;
    target.append(text("h3", "房間層"), text("p", view.group_note));
    target.append(table(["組號", "成員數", "頻率下限", "頻率上限", "T60 範圍", "超出成員數", "範圍外成員數"],
      view.groups.map((row) => [row.group_text, row.member_text, row.lower_text, row.upper_text,
        row.t60_range_text, row.exceeding_text, row.outside_text])));
    target.append(folded("完整模態表（每個解逐列）", table(["索引", "種類", "頻率", "T60", "Q", "目標 T60", "超出量", "來源", "組號"],
      view.modes.map((row) => [row.index, row.kind_text, row.frequency_text, row.t60_text, row.q_text,
        row.target_text, row.excess_text, row.origin_text, row.group_text]))));
    target.append(text("h3", "求解自檢"), text("p", view.guarantee_text), text("p", view.check_note));
    const counts = document.createElement("div");
    counts.append(text("p", `公式原文：${view.weyl_raw}`),
      table(["頻段", "共振個數", "Weyl 估計", "個數減估計", "剛性參考個數", "個數減剛性參考"], view.counts));
    target.append(folded("各段個數對 Weyl（幾何模態數估計）", counts));
    target.append(text("h3", "擺位層"), text("p", view.placement_note));
    view.placements.forEach((pair) => {
      const node = document.createElement("article"); node.append(text("h4", pair.heading_text));
      const headings = ["模態索引", "頻率", "自身絕對大小", "該頻率整群絕對大小", "相對同配對最強共振"];
      node.append(table(headings, placementRows(pair.rows.slice(0, 5))));
      node.append(folded("完整排序（所有共振）", table(headings, placementRows(pair.rows))));
      node.append(folded("整群大小沿成員頻率的剖面", table(["組號", "模態索引", "頻率", "整群絕對大小"], pair.profiles)));
      target.append(node);
    });
  }
  function draw(body) {
    const job = body.job;
    jobId = job?.run_id || null;
    const running = job?.status === "running";
    element("modal-progress").textContent = running ? job.display_text : "";
    element("modal-stop").hidden = !running;
    element("modal-calculate").hidden = true; element("modal-compute-note").hidden = true;
    if (body.view) {
      const view = body.view;
      element("modal-state").textContent = view.state_text;
      element("modal-reason").textContent = view.reason_text;
      element("modal-calculate").hidden = !view.can_calculate;
      element("modal-compute-note").hidden = !view.can_calculate;
      element("modal-compute-note").textContent = view.compute_note;
      report(view);
    } else { element("modal-state").textContent = body.state_text; element("modal-reason").textContent = ""; }
    if (running) timer = window.setTimeout(() => request("GET"), 1000);
  }
  async function request(method) {
    if (busy) return;
    busy = true; window.clearTimeout(timer);
    element("modal-calculate").disabled = true;
    try {
      const url = endpoint + (method === "GET" && jobId ? `?job_id=${jobId}` : "");
      const response = await fetch(url, method === "POST" ? {method, headers: {"Content-Type": "application/json"}, body: "{}"} : {});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || body.reason || `伺服器回應 ${response.status}`);
      draw(body);
    } catch (error) {
      element("modal-state").textContent = "失敗"; element("modal-reason").textContent = String(error);
      element("modal-progress").textContent = ""; element("modal-stop").hidden = true;
    } finally { busy = false; element("modal-calculate").disabled = false; }
  }
  element("modal-calculate").onclick = () => { jobId = null; request("POST"); };
  element("modal-stop").onclick = async () => {
    element("modal-stop").disabled = true;
    try {
      const response = await fetch(`/api/modal-jobs/${jobId}/stop`, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error);
      request("GET");
    } catch (error) { element("modal-reason").textContent = String(error); }
    finally { element("modal-stop").disabled = false; }
  };
  window.addEventListener("pagehide", () => window.clearTimeout(timer), {once: true});
  request("GET");
}
