(function () {
  const D = window.VIOTECH_DATA;
  if (!D || typeof Chart === "undefined") return;

  const INK = "#26222e";
  const SOFT = "#5f5870";
  const LINE = "#e7e2ef";
  // Monochrome purple ramp (light → dark), matching the diagram's accent hue.
  const PURPLE = {
    l1: "#cbbde8", // lightest
    l2: "#a98fd6",
    l3: "#8a6bc9",
    l4: "#7c5fc0", // base accent
    l5: "#5d4494",
    l6: "#42305f"  // darkest
  };
  const PURPLE_RAMP = [PURPLE.l1, PURPLE.l2, PURPLE.l3, PURPLE.l4, PURPLE.l5, PURPLE.l6];

  Chart.defaults.font.family =
    "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
  Chart.defaults.color = SOFT;

  function hexToRgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }

  // ---- Per-class accuracy (grouped bar by task) ----
  let accChart;
  function renderAcc(ds) {
    const d = D.datasets[ds].perClass;
    const labels = [];
    const values = [];
    const colors = [];
    let idx = 0;
    ["Tonal", "Articulation", "Legato"].forEach((task) => {
      d[task].labels.forEach((cl, i) => {
        labels.push(`${task[0]}: ${cl}`);
        values.push(d[task].values[i]);
        colors.push(PURPLE_RAMP[idx % PURPLE_RAMP.length]);
        idx++;
      });
    });

    const cfg = {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Accuracy (%)",
          data: values,
          backgroundColor: colors.map((c) => hexToRgba(c, 0.85)),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => `${c.parsed.y.toFixed(2)}%` } }
        },
        scales: {
          y: { beginAtZero: true, max: 100, grid: { color: LINE }, title: { display: true, text: "Accuracy (%)" } },
          x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 30 } }
        }
      }
    };
    if (accChart) { accChart.data = cfg.data; accChart.update(); }
    else accChart = new Chart(document.getElementById("accChart"), cfg);
  }

  // ---- Task-level metrics (grouped bar) ----
  let taskChart;
  function renderTask(ds) {
    const t = D.datasets[ds].task;
    const cfg = {
      type: "bar",
      data: {
        labels: t.labels,
        datasets: [
          { label: "Accuracy", data: t.accuracy, backgroundColor: hexToRgba(PURPLE.l2, 0.9), borderRadius: 6 },
          { label: "Macro-F1", data: t.macroF1, backgroundColor: hexToRgba(PURPLE.l4, 0.9), borderRadius: 6 },
          { label: "Weighted-F1", data: t.weightedF1, backgroundColor: hexToRgba(PURPLE.l6, 0.9), borderRadius: 6 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(2)}%` } }
        },
        scales: {
          y: { beginAtZero: true, max: 100, grid: { color: LINE }, title: { display: true, text: "%" } },
          x: { grid: { display: false } }
        }
      }
    };
    if (taskChart) { taskChart.data = cfg.data; taskChart.update(); }
    else taskChart = new Chart(document.getElementById("taskChart"), cfg);
  }

  // ---- Gate routing (grouped bar over experts) ----
  let gateChart;
  function renderGate(ds) {
    const g = D.datasets[ds].gate;
    const cfg = {
      type: "bar",
      data: {
        labels: D.experts,
        datasets: ["Tonal", "Articulation", "Legato"].map((task, i) => {
          const shade = [PURPLE.l2, PURPLE.l4, PURPLE.l6][i];
          return {
            label: task,
            data: g[task],
            backgroundColor: hexToRgba(shade, 0.9),
            borderColor: shade,
            borderWidth: 1,
            borderRadius: 5
          };
        })
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%` } }
        },
        scales: {
          y: { beginAtZero: true, grid: { color: LINE }, title: { display: true, text: "Gate weight (%)" } },
          x: { grid: { display: false }, title: { display: true, text: "Expert receptive field" } }
        }
      }
    };
    if (gateChart) { gateChart.data = cfg.data; gateChart.update(); }
    else gateChart = new Chart(document.getElementById("gateChart"), cfg);
  }

  function bind(id, fn) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", (e) => fn(e.target.value));
  }

  renderAcc("rwc");
  renderTask("rwc");
  renderGate("rwc");
  bind("acc-dataset", renderAcc);
  bind("task-dataset", renderTask);
  bind("gate-dataset", renderGate);

  // ---- Copy BibTeX ----
  const copyBtn = document.getElementById("copyBib");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const bib = document.querySelector(".bibtex code").innerText;
      try {
        await navigator.clipboard.writeText(bib);
        copyBtn.textContent = "Copied!";
        copyBtn.classList.add("copied");
        setTimeout(() => {
          copyBtn.textContent = "Copy BibTeX";
          copyBtn.classList.remove("copied");
        }, 1800);
      } catch (e) {
        copyBtn.textContent = "Copy failed";
      }
    });
  }
})();
