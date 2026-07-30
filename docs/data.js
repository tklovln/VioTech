// VioTech evaluation results (extracted from results/ablation_expert_composition_v2/mert_moe.json)
// Proposed model: frozen MERT-v1-330M + 8-expert Multi-Scale MoE, checkpoint iter=2000.
window.VIOTECH_DATA = {
  experts: ["20ms", "40ms", "80ms", "160ms", "320ms", "640ms", "1280ms", "2560ms"],
  datasets: {
    rwc: {
      label: "RWC (cross-dataset)",
      perClass: {
        Tonal: { labels: ["none", "pizzicato", "harmonics"], values: [78.72, 59.51, 86.88] },
        Articulation: { labels: ["none", "spiccato"], values: [96.65, 84.22] },
        Legato: { labels: ["sustained", "bow-change"], values: [97.42, 56.54] }
      },
      task: {
        labels: ["Tonal", "Articulation", "Legato"],
        accuracy: [77.25, 95.26, 96.32],
        macroF1: [64.14, 88.62, 71.72],
        weightedF1: [81.66, 95.36, 96.67]
      },
      gate: {
        Tonal:        [10.26, 13.29, 15.62, 16.46, 14.48, 8.88, 10.78, 10.22],
        Articulation: [17.56, 9.14, 7.60, 18.42, 19.15, 10.97, 6.84, 10.32],
        Legato:       [6.30, 2.16, 3.51, 3.39, 35.88, 26.81, 4.61, 17.33]
      }
    },
    viotech: {
      label: "VioTech (in-domain)",
      perClass: {
        Tonal: { labels: ["none", "pizzicato", "harmonics"], values: [99.06, 0.01, 31.74] },
        Articulation: { labels: ["none", "spiccato"], values: [94.52, 75.18] },
        Legato: { labels: ["sustained", "bow-change"], values: [91.46, 41.08] }
      },
      task: {
        labels: ["Tonal", "Articulation", "Legato"],
        accuracy: [96.74, 93.33, 85.42],
        macroF1: [42.58, 77.24, 66.01],
        weightedF1: [96.03, 94.02, 85.53]
      },
      gate: {
        Tonal:        [10.27, 13.14, 17.50, 12.96, 11.03, 9.97, 13.27, 11.86],
        Articulation: [11.44, 8.21, 7.29, 16.31, 30.52, 11.85, 6.54, 7.84],
        Legato:       [7.75, 2.91, 3.76, 3.98, 31.00, 24.83, 5.55, 20.24]
      }
    }
  }
};
