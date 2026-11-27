// 自訂房間頁（#1315）：上傳 → 檢查與圖層 → 每層選吸音 → 背景算低頻，輪詢到結果畫曲線。
(() => {
  const $ = (id) => document.getElementById(id);
  const TICKS_HZ = [20, 30, 40, 50, 60, 80, 100, 150, 200, 300];
  const COLORS = ["#2a6fdb", "#e07b39", "#5fb36b", "#9b59b6"];
  let model = null, timer = null, plot = null, activeRun = null, upload = 0;

  async function call(url, body) {
    const response = await fetch(url, body === undefined ? {} :
      {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }

  function base64(buffer) {
    const bytes = new Uint8Array(buffer);
    let text = "";
    for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(text);
  }

  function showModel(data) {
    model = data;
    $("model").hidden = false;
    $("result").hidden = true;
    $("problems").replaceChildren(...data.problems.map((p) => {
      const item = document.createElement("li");
      item.textContent = p.text;
      if (p.blocking) item.className = "notice";
      return item;
    }));
    $("layers").replaceChildren(...data.layers.map((layer) => {
      const row = document.createElement("tr");
      const name = document.createElement("td"), area = document.createElement("td"), pick = document.createElement("td");
      const swatch = document.createElement("span");
      swatch.className = "swatch"; swatch.style.background = layer.color;
      name.append(swatch, layer.name);
      area.textContent = layer.area_m2.toFixed(2);
      const select = document.createElement("select");
      select.dataset.layer = layer.name;
      data.choices.forEach((choice) => select.add(new Option(choice.label, choice.key)));
      pick.append(select);
      row.append(name, area, pick);
      return row;
    }));
    $("preview").innerHTML = data.preview_svg;
    $("run").disabled = !data.ok;
    $("state").textContent = data.ok ? "" : "沒過檢查，不能算";
  }

  function draw(result) {
    $("result").hidden = false;
    if (plot) plot.destroy();
    const target = $("chart");
    plot = new uPlot({width: target.clientWidth || 900, height: 380, legend: {live: false},
      scales: {x: {time: false, distr: 3, range: (u, min, max) => [min, max]}},
      axes: [{label: "頻率（Hz）", splits: (u) => TICKS_HZ.filter((f) => f >= u.scales.x.min && f <= u.scales.x.max),
              filter: (u, splits) => splits},
             {label: "聲壓級（dB）"}],
      series: [{}, ...result.curves.map((curve, i) => ({label: curve.label, stroke: COLORS[i % COLORS.length], width: 1.5}))]},
      [result.frequencies_hz, ...result.curves.map((curve) => curve.spl_db)], target);
    $("modes").textContent = "模態（Hz）：" + result.modes_hz.map((f) => f.toFixed(1)).join("、");
    $("mesh").innerHTML = result.mesh_svg;
    $("facts").textContent = `網格 ${result.mesh.tetrahedra} 個四面體，空氣 ${result.mesh.volume_m3} m³，算了 ${result.seconds} 秒`;
  }

  function stopPolling() {
    clearInterval(timer); timer = null; activeRun = null;
  }

  async function poll(runId) {
    try {
      const data = await call(`/api/room-model-jobs/${runId}`);
      if (runId !== activeRun) return;  // 換了房或重按過：舊工作的回應不畫、不動計時器
      $("state").textContent = data.status === "done" ? "" : data.text + (data.error ? "\n" + data.error : "");
      if (data.status === "running") return;
      stopPolling();
      $("run").disabled = false;
      if (data.result) draw(data.result);
    } catch (error) {
      if (runId !== activeRun) return;
      stopPolling();
      $("run").disabled = false;
      $("state").textContent = error.message;
    }
  }

  $("file").addEventListener("change", async () => {
    const file = $("file").files[0];
    if (!file) return;
    // 一選新檔就放掉舊房：還在路上的起算回應、輪詢、較早的上傳回應都對不上，不會畫到新房上。
    const mine = ++upload;
    model = null;
    stopPolling();
    $("run").disabled = true;
    $("result").hidden = true;
    $("state").textContent = "讀取中…";
    try {
      const data = await call("/api/room-models", {file_name: file.name, data_base64: base64(await file.arrayBuffer())});
      if (mine !== upload) return;
      stopPolling();
      showModel(data);
    } catch (error) {
      if (mine !== upload) return;
      $("model").hidden = true;
      $("state").textContent = error.message;
    }
  });

  $("run").addEventListener("click", async () => {
    const absorption = {};
    document.querySelectorAll("#layers select").forEach((select) => { absorption[select.dataset.layer] = select.value; });
    $("run").disabled = true;
    try {
      const forModel = model.id;
      const job = await call(`/api/room-models/${forModel}/runs`, {absorption});
      if (!model || model.id !== forModel) return;  // 等回應時換了房
      $("state").textContent = job.display_text || "計算低頻中";
      stopPolling();
      activeRun = job.run_id;
      timer = setInterval(() => poll(job.run_id), 3000);
    } catch (error) {
      $("run").disabled = false;
      $("state").textContent = error.message;
    }
  });
})();
