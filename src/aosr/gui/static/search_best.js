"use strict";
// 數值、分區、超線標記和基準線都由伺服器給；這裡只轉成畫面座標。
(() => {
  const $ = (id) => document.getElementById(id);
  const colors = ["#1b6f8a", "#c0392b", "#8e44ad", "#b36b00"];
  let plots = [], selected = null, versions = null, shown = "", requestNumber = 0;
  function clearCharts() {
    plots.forEach((plot) => plot.destroy()); plots = [];
    $("best-frequency-chart").replaceChildren(); $("best-rfz-charts").replaceChildren();
    $("rfz-zones").replaceChildren(); $("rfz-window").textContent = "";
    $("best-plan-content").hidden = true;
    for (const key of ["frequency", "rfz", "plan"]) $("best-" + key).querySelector(".chart-error").textContent = "";
  }
  function axes(x, y) {
    return [{label: x, values: (u, values) => values.map((v) => v === null ? "" : v.toFixed(1))},
      {label: y, values: (u, values) => values.map((v) => v === null ? "" : v.toFixed(1))}];
  }
  function frequency(data) {
    const target = $("best-frequency-chart");
    plots.push(new uPlot({width: target.clientWidth, height: 340, legend: {live: false},
      scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}},
      axes: axes("頻率（Hz）", "聲級（dB）"),
      series: [{}, ...data.curves.map((curve, i) => ({label: curve.label, stroke: colors[i], width: 1.5}))]}, data.data, target));
  }
  function reflectionPlot(target, channel, data) {
    const points = channel.points.filter((p) => p.level_db !== null).sort((a, b) => a.delay_ms - b.delay_ms);
    const x = points.map((p) => p.delay_ms);
    const values = data.zones.map((zone) => points.map((p) => p.zone === zone.key ? p.level_db : null));
    const levels = [...points.map((p) => p.level_db), ...data.zones.map((z) => z.threshold_db)];
    const maxX = Math.max(data.window_ms, ...x, 1);
    plots.push(new uPlot({width: target.clientWidth, height: 310, legend: {live: false}, cursor: {show: false},
      scales: {x: {time: false, range: [0, maxX + 1]}, y: {range: [Math.min(...levels) - 5, Math.max(...levels) + 5]}},
      axes: axes("晚於直達音（毫秒）", "相對直達聲級（dB）"),
      series: [{}, ...data.zones.map((zone, i) => ({label: zone.label, stroke: colors[i], paths: () => null, points: {show: false}}))],
      hooks: {draw: [(u) => {
        const ctx = u.ctx, box = u.bbox;
        ctx.save(); ctx.beginPath(); ctx.rect(box.left, box.top, box.width, box.height); ctx.clip();
        data.zones.forEach((zone, i) => {
          ctx.strokeStyle = colors[i]; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
          const y = u.valToPos(zone.threshold_db, "y", true);
          ctx.beginPath(); ctx.moveTo(box.left, y); ctx.lineTo(box.left + box.width, y); ctx.stroke();
        });
        const windowX = u.valToPos(data.window_ms, "x", true);
        ctx.strokeStyle = "#344b58"; ctx.beginPath(); ctx.moveTo(windowX, box.top); ctx.lineTo(windowX, box.top + box.height); ctx.stroke();
        ctx.setLineDash([]);
        points.forEach((p) => {
          const color = colors[data.zones.findIndex((z) => z.key === p.zone)];
          ctx.strokeStyle = color; ctx.fillStyle = p.over_limit ? "white" : color;
          ctx.beginPath(); ctx.arc(u.valToPos(p.delay_ms, "x", true), u.valToPos(p.level_db, "y", true), p.over_limit ? 6 : 4, 0, Math.PI * 2);
          ctx.fill(); ctx.stroke();
        }); ctx.restore();
      }]}}, [x, ...values], target));
  }
  function rfz(data) {
    $("rfz-window").textContent = data.window_text;
    data.zones.forEach((zone, i) => {
      const span = document.createElement("span"); span.textContent = zone.text; span.style.borderColor = colors[i]; $("rfz-zones").append(span);
    });
    for (const channel of data.channels) {
      const group = document.createElement("div"), heading = document.createElement("h4"), target = document.createElement("div"), summary = document.createElement("p");
      heading.textContent = channel.label; summary.textContent = channel.summary_text;
      group.append(heading, target, summary); $("best-rfz-charts").append(group); reflectionPlot(target, channel, data);
    }
  }
  function draw(data) {
    $("best-title").textContent = data.title;
    for (const key of ["frequency", "rfz", "plan"]) {
      const chart = data[key], error = $("best-" + key).querySelector(".chart-error");
      if (chart.error) { error.textContent = chart.error; continue; }
      if (key === "frequency") frequency(chart);
      else if (key === "rfz") rfz(chart);
      else {
        $("best-plan-content").hidden = false;
        window.drawPlan(chart.data, {planXY: "best-plan-xy", planXZ: "best-plan-xz", detail: $("best-plan-detail"), legend: $("best-plan-legend")});
      }
    }
  }
  async function load(which, signal) {
    const version = JSON.stringify(versions[which]);
    if (version === shown) return;
    const request = ++requestNumber;
    clearCharts(); shown = ""; $("best-title").textContent = "目前最佳：正在讀取…";
    for (const choice of ["search", "refine"]) $("best-" + choice).setAttribute("aria-pressed", String(choice === which));
    try {
      const id = window.location.pathname.split("/")[2];
      const response = await fetch(`/api/searches/${id}/best?which=${which}`, {cache: "no-store", signal});
      const data = await response.json();
      if (request !== requestNumber) return;
      if (!response.ok) throw new Error(data.error || `回覆狀態 ${response.status}`);
      draw(data);
      // 狀態和結果之間換了第一名時，下次輪詢仍會重取，避免把新圖記成舊版本。
      if (data.version === versions[which].version) shown = version;
    } catch (error) {
      if (request !== requestNumber) return;
      clearCharts(); $("best-title").textContent = "目前最佳";
      for (const key of ["frequency", "rfz", "plan"]) $("best-" + key).querySelector(".chart-error").textContent = `讀不到：${error.message}`;
    }
  }
  window.refreshBest = async (data, signal) => {
    $("live-best").hidden = false; versions = data.best_versions;
    $("best-switch").hidden = data.best_default !== "refine";
    if (selected === "refine" && data.best_default !== "refine") selected = null;
    await load(selected || data.best_default, signal);
  };
  window.addEventListener("DOMContentLoaded", () => {
    for (const which of ["search", "refine"]) $("best-" + which).addEventListener("click", () => { selected = which; load(which); });
  });
})();
