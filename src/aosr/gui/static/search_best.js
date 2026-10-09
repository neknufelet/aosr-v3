"use strict";
// 數值、分區、超線標記和基準線都由伺服器給；這裡只轉成畫面座標。
(() => {
  const $ = (id) => document.getElementById(id);
  const colors = ["#1b6f8a", "#c0392b", "#8e44ad", "#b36b00"];
  const KEYS = ["frequency", "rfz", "plan"], TIMEOUT_MS = 60000;
  let plots = {frequency: [], rfz: []}, selected = null, versions = null, shown = "", loading = "", requestNumber = 0;
  function clearChart(key) {
    (plots[key] || []).forEach((plot) => plot.destroy()); plots[key] = [];
    if (key === "frequency") { $("best-frequency-chart").replaceChildren(); $("best-frequency-furniture-note").hidden = true; $("best-frequency-furniture-note").textContent = ""; }
    else if (key === "rfz") { $("best-rfz-charts").replaceChildren(); $("rfz-zones").replaceChildren(); $("rfz-window").textContent = ""; }
    else $("best-plan-content").hidden = true;
    $("best-" + key).querySelector(".chart-error").textContent = "";
  }
  function clearCharts() { KEYS.forEach(clearChart); }
  function axes(x, y) {
    return [{label: x, values: (u, values) => values.map((v) => v === null ? "" : v.toFixed(1))},
      {label: y, values: (u, values) => values.map((v) => v === null ? "" : v.toFixed(1))}];
  }
  function frequency(data) {
    $("best-frequency-furniture-note").textContent = data.furniture_note;
    $("best-frequency-furniture-note").hidden = !data.furniture_note;
    const target = $("best-frequency-chart");
    plots.frequency.push(new uPlot({width: target.clientWidth, height: 340, legend: {live: false},
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
    plots.rfz.push(new uPlot({width: target.clientWidth, height: 310, legend: {live: false}, cursor: {show: false},
      scales: {x: {time: false, range: [0, maxX + 1]}, y: {range: [Math.min(...levels) - 5, Math.max(...levels) + 5]}},
      axes: axes("晚於直達音（毫秒）", "相對直達聲級（dB）"),
      series: [{}, ...data.zones.map((zone, i) => ({label: zone.label, stroke: colors[i], paths: () => null, points: {show: false}}))],
      hooks: {draw: [(u) => {
        const ctx = u.ctx, box = u.bbox;
        ctx.save(); ctx.beginPath(); ctx.rect(box.left, box.top, box.width, box.height); ctx.clip();
        // 門檻相同的分區疊在同一條線上，只看得到最後畫的顏色；合成一條灰線，文字另寫明。
        const levels = new Map();
        data.zones.forEach((zone, i) => levels.set(zone.threshold_db, [...(levels.get(zone.threshold_db) || []), i]));
        levels.forEach((indexes, level) => {
          ctx.strokeStyle = indexes.length > 1 ? "#6b7780" : colors[indexes[0]]; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
          const y = u.valToPos(level, "y", true);
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
    for (const key of KEYS) {
      const chart = data[key], error = $("best-" + key).querySelector(".chart-error");
      if (chart.error) { error.textContent = chart.error; continue; }
      // 一張畫到一半出錯只清那一張，另兩張照常顯示。
      try {
        if (key === "frequency") frequency(chart);
        else if (key === "rfz") rfz(chart);
        else {
          $("best-plan-content").hidden = false;
          window.drawPlan(chart.data, {planXY: "best-plan-xy", planXZ: "best-plan-xz", detail: $("best-plan-detail"), legend: $("best-plan-legend")});
        }
      } catch (failure) {
        clearChart(key); error.textContent = `讀不到：這張圖畫不出來（${failure.message}）`;
      }
    }
  }
  async function load(which) {
    const version = JSON.stringify(versions[which]);
    // 同一版本已顯示或正在讀就不再問；伺服器第一次剖析大結果檔可能要幾秒。
    if (version === shown || version === loading) return;
    const request = ++requestNumber;
    loading = version;
    clearCharts(); shown = ""; $("best-title").textContent = "目前最佳：正在讀取…";
    for (const choice of ["search", "refine"]) $("best-" + choice).setAttribute("aria-pressed", String(choice === which));
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), TIMEOUT_MS);
    try {
      const id = window.location.pathname.split("/")[2];
      const response = await fetch(`/api/searches/${id}/best?which=${which}`, {cache: "no-store", signal: abort.signal});
      const data = await response.json();
      if (request !== requestNumber) return;
      if (!response.ok) throw new Error(data.error || `回覆狀態 ${response.status}`);
      draw(data);
      // 狀態和結果之間換了第一名時，下次輪詢仍會重取，避免把新圖記成舊版本。
      if (data.version === versions[which].version) shown = version;
    } catch (error) {
      if (request !== requestNumber) return;
      const reason = abort.signal.aborted ? `等伺服器整理圖表超過 ${TIMEOUT_MS / 1000} 秒，下次更新再試` : error.message;
      clearCharts(); $("best-title").textContent = "目前最佳";
      for (const key of KEYS) $("best-" + key).querySelector(".chart-error").textContent = `讀不到：${reason}`;
    } finally {
      clearTimeout(timer);
      if (request === requestNumber) loading = "";
    }
  }
  // 不跟單場狀態共用 5 秒的逾時：圖表自己的請求自己計時。
  window.refreshBest = (data) => {
    $("live-best").hidden = false; versions = data.best_versions;
    $("best-switch").hidden = data.best_default !== "refine";
    if (selected === "refine" && data.best_default !== "refine") selected = null;
    load(selected || data.best_default);
  };
  window.addEventListener("DOMContentLoaded", () => {
    for (const which of ["search", "refine"]) $("best-" + which).addEventListener("click", () => { selected = which; load(which); });
  });
})();
