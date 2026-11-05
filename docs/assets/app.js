/* Market Seismograph dashboard: reads data/latest.json written by the hourly pipeline. */
(() => {
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmtPrice = (p) => p >= 100 ? p.toLocaleString("en-US", { maximumFractionDigits: 0 })
    : p >= 1 ? p.toLocaleString("en-US", { maximumFractionDigits: 2 }) : p.toLocaleString("en-US", { maximumFractionDigits: 4 });
  const fmtTime = (iso) => new Date(iso + (iso.endsWith("Z") || iso.includes("+") ? "" : "Z"))
    .toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
  let data, selected, priceChart, sevChart;

  fetch("data/latest.json", { cache: "no-store" })
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => { data = d; render(); })
    .catch(() => { $("asof").textContent = "Live data isn't published yet. The hourly pipeline creates it."; });

  function render() {
    const mins = Math.max(0, Math.round((Date.now() - new Date(data.generated_at)) / 60000));
    $("asof").textContent = `Updated ${mins < 60 ? mins + " min" : Math.round(mins / 60) + " h"} ago, every hour`;
    board();
    feed();
    model();
    const hash = decodeURIComponent(location.hash.slice(1)).toUpperCase();
    select(data.coins.find((c) => c.symbol === hash) || data.coins[0]);
  }

  function board() {
    $("board").innerHTML = data.coins.map((c) => {
      const chg = c.change_24h ?? 0;
      const bars = c.heat.map((s) => {
        const h = Math.max(8, Math.min(100, Math.sqrt(s ?? 0) * 75));
        return `<i class="${s > 1 ? "hot" : ""}" style="height:${h}%"></i>`;
      }).join("");
      return `<li><button class="row" type="button" data-sym="${c.symbol}" aria-pressed="false">
        <span class="coin"><b>${c.symbol}</b><span>${c.name}</span></span>
        <span class="price">$${fmtPrice(c.price)}</span>
        <span class="chg ${chg >= 0 ? "up" : "down"}">${chg >= 0 ? "+" : ""}${(chg * 100).toFixed(1)}%</span>
        <span class="strip" aria-label="${c.anomalies_7d} anomalous hours in the last 7 days">${bars}</span>
        <span class="now ${c.severity_now > 1 ? "hot" : ""}">${c.severity_now.toFixed(2)}×</span>
      </button></li>`;
    }).join("");
    $("board").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) select(data.coins.find((c) => c.symbol === b.dataset.sym));
    });
  }

  function feed() {
    const f = data.feed;
    $("feed-sub").textContent = f.length
      ? `${data.totals.events_7d} anomalous coin-hours in the past 7 days. Select one to inspect it.`
      : "No hours crossed the threshold in the past 7 days.";
    $("feed").innerHTML = f.map((e) => `<li><button type="button" data-sym="${e.coin}">
        <span class="what">${e.coin}: ${e.reason}</span></button><span class="sev">${e.severity.toFixed(1)}×</span>
        <span class="when">${fmtTime(e.time)}${e.move != null ? `, price ${e.move >= 0 ? "+" : ""}${(e.move * 100).toFixed(2)}% that hour` : ""}</span></li>`).join("");
    $("feed").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) { select(data.coins.find((c) => c.symbol === b.dataset.sym)); $("coin-title").scrollIntoView({ behavior: "smooth", block: "start" }); }
    });
  }

  function select(c) {
    selected = c;
    history.replaceState(null, "", "#" + c.symbol);
    document.querySelectorAll(".row").forEach((r) => r.setAttribute("aria-pressed", r.dataset.sym === c.symbol));
    $("coin-title").textContent = `${c.name} (${c.symbol})`;
    $("coin-sub").textContent = `${c.anomalies_7d} anomalous ${c.anomalies_7d === 1 ? "hour" : "hours"} in the last 7 days`;
    charts(c);
  }

  function charts(c) {
    const s = c.series, labels = s.time;
    const muted = css("--muted"), grid = css("--rule"), calm = css("--calm"), tremor = css("--tremor");
    const hot = s.severity.map((v, i) => (v > 1 ? s.close[i] : null));
    const x = { ticks: { color: muted, maxTicksLimit: 7, maxRotation: 0, callback: (v) => fmtTime(labels[v]).split(" ")[0] }, grid: { display: false } };
    if (priceChart) priceChart.destroy();
    if (sevChart) sevChart.destroy();
    priceChart = new Chart($("price"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Price", data: s.close, borderColor: calm, borderWidth: 2, pointRadius: 0, tension: .2 },
        { label: "Anomaly", data: hot, showLine: false, pointRadius: 5, pointHoverRadius: 7, pointBackgroundColor: tremor, pointBorderColor: tremor },
      ] },
      options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: { callbacks: { title: (i) => fmtTime(labels[i[0].dataIndex]),
          label: (i) => i.raw == null ? null : `${i.dataset.label}: $${fmtPrice(i.raw)}` } } },
        scales: { x: { ...x, display: false }, y: { ticks: { color: muted, callback: (v) => "$" + fmtPrice(v) }, grid: { color: grid } } } },
    });
    sevChart = new Chart($("sev"), {
      type: "bar",
      data: { labels, datasets: [{ data: s.severity, backgroundColor: s.severity.map((v) => (v > 1 ? tremor : css("--calm-soft"))), barPercentage: 1, categoryPercentage: 1 }] },
      options: { maintainAspectRatio: false, animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: { callbacks: { title: (i) => fmtTime(labels[i[0].dataIndex]), label: (i) => `Severity ${i.raw.toFixed(2)}×` } } },
        scales: { x, y: { ticks: { color: muted, maxTicksLimit: 3 }, grid: { color: grid }, title: { display: true, text: "severity", color: muted } } } },
      plugins: [{ id: "threshold", afterDraw(ch) {
        const y = ch.scales.y.getPixelForValue(1), { left, right } = ch.chartArea, ctx = ch.ctx;
        ctx.save(); ctx.strokeStyle = tremor; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
        ctx.fillStyle = tremor; ctx.font = "12px Archivo, sans-serif"; ctx.fillText("threshold", right - 58, y - 4); ctx.restore();
      } }],
    });
  }

  function model() {
    const m = data.model, b = m.benchmark, muted = css("--muted"), grid = css("--rule");
    const names = { lstm_autoencoder: "LSTM autoencoder", isolation_forest: "Isolation Forest", zscore_rule: "Big-move rule" };
    const keys = Object.keys(b);
    const colors = [css("--calm"), css("--calm-soft"), muted];
    new Chart($("bench"), {
      type: "bar",
      data: { labels: ["ROC-AUC", "PR-AUC", "Precision at k"], datasets: keys.map((k, i) => ({
        label: names[k] || k, data: [b[k].roc_auc, b[k].pr_auc, b[k].precision_at_k], backgroundColor: colors[i], borderRadius: 3 })) },
      options: { plugins: { legend: { labels: { color: muted, boxWidth: 12 } } },
        scales: { x: { ticks: { color: muted }, grid: { display: false } }, y: { min: 0, max: 1, ticks: { color: muted }, grid: { color: grid } } } },
    });
    const ae = b.lstm_autoencoder;
    $("bench-note").textContent = `The autoencoder ranks planted anomalies with ROC-AUC ${ae.roc_auc}, against ${b.isolation_forest.roc_auc} for Isolation Forest. At the live threshold it catches ${Math.round(m.at_threshold.recall * 100)}% of them with ${Math.round(m.at_threshold.precision * 100)}% precision.`;
    new Chart($("loss"), {
      type: "line",
      data: { labels: m.history.map((h) => h.epoch), datasets: [
        { label: "Train", data: m.history.map((h) => h.train), borderColor: css("--calm"), pointRadius: 0, borderWidth: 2 },
        { label: "Validation", data: m.history.map((h) => h.val), borderColor: css("--tremor"), pointRadius: 0, borderWidth: 2 },
      ] },
      options: { plugins: { legend: { labels: { color: muted, boxWidth: 12 } } },
        scales: { x: { title: { display: true, text: "epoch", color: muted }, ticks: { color: muted, maxTicksLimit: 6 }, grid: { display: false } },
                  y: { ticks: { color: muted }, grid: { color: grid } } } },
    });
    $("train-note").textContent = `Trained ${new Date(m.trained_at).toLocaleDateString()} on ${m.n_train.toLocaleString()} windows of 24 hours. Retrained weekly.`;
  }
})();
