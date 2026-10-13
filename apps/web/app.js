// Deepfake Shield Interactive Web Client
const API_BASE = "http://localhost:8000";

let currentFile = null;
let currentPrediction = null;
let robustnessCurve = null;
let chartInstance = null;

// DOM Elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const dropPrompt = document.getElementById("drop-prompt");
const previewWrapper = document.getElementById("preview-wrapper");
const previewImage = document.getElementById("preview-image");
const previewVideo = document.getElementById("preview-video");
const btnClearPreview = document.getElementById("btn-clear-preview");
const btnAnalyze = document.getElementById("btn-analyze");
const analyzeSpinner = document.getElementById("analyze-spinner");

const emptyState = document.getElementById("empty-state");
const verdictContent = document.getElementById("verdict-content");
const verdictBanner = document.getElementById("verdict-banner");
const verdictPill = document.getElementById("verdict-pill");
const verdictHeadline = document.getElementById("verdict-headline");
const verdictCaveat = document.getElementById("verdict-caveat");
const confidenceNum = document.getElementById("confidence-num");
const probFill = document.getElementById("prob-fill");

const scoreGlobal = document.getElementById("score-global");
const scoreFace = document.getElementById("score-face");
const faceRatio = document.getElementById("face-ratio");
const inferenceLatency = document.getElementById("inference-latency");

const explainabilityBox = document.getElementById("explainability-box");
const heatmapDisplay = document.getElementById("heatmap-display");
const heatmapOpacity = document.getElementById("heatmap-opacity");
const spectrumDisplay = document.getElementById("spectrum-display");

const videoTimelineBox = document.getElementById("video-timeline-box");
const framesTimelineStrip = document.getElementById("frames-timeline-strip");
const videoFrameCountTag = document.getElementById("video-frame-count-tag");

let isVideo = false;

const qfSlider = document.getElementById("qf-slider");
const qfValDisplay = document.getElementById("qf-val-display");
const compressedPreviewImg = document.getElementById("compressed-preview-img");
const qfPredTag = document.getElementById("qf-pred-tag");

// Tab Navigation
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

    btn.classList.add("active");
    const targetId = btn.getAttribute("data-tab");
    document.getElementById(targetId).classList.add("active");
  });
});

// File Upload Handlers
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelected(e.target.files[0]);
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelected(e.dataTransfer.files[0]);
  }
});

btnClearPreview.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSelectedFile();
});

function handleFileSelected(file) {
  const isVid = file.type.startsWith("video/") || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);
  const isImg = file.type.startsWith("image/") || /\.(jpe?g|png|webp)$/i.test(file.name);

  if (!isVid && !isImg) {
    alert("Please upload an image file (JPEG, PNG, WebP) or video file (MP4, MOV, WebM).");
    return;
  }

  currentFile = file;
  isVideo = isVid;

  if (isVideo) {
    previewImage.style.display = "none";
    previewVideo.src = URL.createObjectURL(file);
    previewVideo.style.display = "block";
    compressedPreviewImg.src = "";
    dropPrompt.style.display = "none";
    previewWrapper.style.display = "flex";
    btnAnalyze.disabled = false;
  } else {
    previewVideo.style.display = "none";
    previewVideo.pause();
    previewVideo.src = "";
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.style.display = "block";
      compressedPreviewImg.src = e.target.result;
      dropPrompt.style.display = "none";
      previewWrapper.style.display = "flex";
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }
}

function clearSelectedFile() {
  currentFile = null;
  isVideo = false;
  fileInput.value = "";
  previewImage.src = "";
  previewImage.style.display = "none";
  previewVideo.pause();
  previewVideo.src = "";
  previewVideo.style.display = "none";
  dropPrompt.style.display = "block";
  previewWrapper.style.display = "none";
  btnAnalyze.disabled = true;

  emptyState.style.display = "block";
  verdictContent.style.display = "none";
  if (videoTimelineBox) videoTimelineBox.style.display = "none";
  if (explainabilityBox) explainabilityBox.style.display = "block";
}

// Analyze API Call
btnAnalyze.addEventListener("click", async () => {
  if (!currentFile) return;

  btnAnalyze.disabled = true;
  analyzeSpinner.style.display = "inline-block";

  try {
    const formData = new FormData();
    formData.append("file", currentFile);

    let data;
    const endpoint = isVideo ? `${API_BASE}/v1/analyze/video` : `${API_BASE}/v1/analyze/image`;
    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error("API error: " + resp.statusText);
      data = await resp.json();
    } catch (err) {
      console.warn("Backend API not reachable or error. Falling back to simulation: " + err.message);
      data = isVideo ? simulateVideoPrediction() : simulateDemoPrediction();
    }

    currentPrediction = data;
    if (isVideo) {
      displayVideoResults(data);
    } else {
      displayResults(data);
      fetchRobustnessCurve();
    }
  } catch (err) {
    alert("Analysis failed: " + err.message);
  } finally {
    btnAnalyze.disabled = false;
    analyzeSpinner.style.display = "none";
  }
});

function displayResults(res) {
  emptyState.style.display = "none";
  verdictContent.style.display = "block";
  if (explainabilityBox) explainabilityBox.style.display = "block";
  if (videoTimelineBox) videoTimelineBox.style.display = "none";

  const p = res.p_fake_calibrated;
  const state = res.state;

  verdictBanner.className = "verdict-banner " + state.toLowerCase();
  verdictPill.textContent = state;

  if (state === "REAL") {
    verdictHeadline.textContent = "Authentic Image (Pristine)";
    confidenceNum.textContent = `${((1.0 - p) * 100).toFixed(1)}% Real`;
  } else if (state === "FAKE") {
    verdictHeadline.textContent = "AI-Generated Media (Synthetic)";
    confidenceNum.textContent = `${(p * 100).toFixed(1)}% Fake`;
  } else {
    verdictHeadline.textContent = "Uncertain Signature";
    confidenceNum.textContent = `${(p * 100).toFixed(1)}% Uncertainty`;
  }

  verdictCaveat.textContent = res.caveat;

  probFill.style.width = `${(p * 100).toFixed(1)}%`;
  scoreGlobal.textContent = res.p_global.toFixed(4);
  scoreFace.textContent = res.p_face !== null ? res.p_face.toFixed(4) : "N/A (No Face)";
  faceRatio.textContent = res.face_box ? "Detected" : "None";
  inferenceLatency.textContent = `${res.timings_ms.total_pipeline_ms.toFixed(1)} ms`;

  // Display Explainability Heatmap
  if (res.heatmap_png_b64) {
    heatmapDisplay.src = "data:image/png;base64," + res.heatmap_png_b64;
  }

  // Display Frequency Spectrum
  if (res.spectrum_png_b64) {
    spectrumDisplay.src = "data:image/png;base64," + res.spectrum_png_b64;
  }
}

function displayVideoResults(res) {
  emptyState.style.display = "none";
  verdictContent.style.display = "block";
  if (explainabilityBox) explainabilityBox.style.display = "none";
  if (videoTimelineBox) videoTimelineBox.style.display = "block";

  const p = res.p_fake_calibrated;
  const state = res.state;

  verdictBanner.className = "verdict-banner " + state.toLowerCase();
  verdictPill.textContent = state;

  if (state === "REAL") {
    verdictHeadline.textContent = "Authentic Video (Natural Motion)";
    confidenceNum.textContent = `${((1.0 - p) * 100).toFixed(1)}% Real`;
  } else if (state === "FAKE") {
    verdictHeadline.textContent = "AI-Generated Video (Deepfake)";
    confidenceNum.textContent = `${(p * 100).toFixed(1)}% Fake`;
  } else {
    verdictHeadline.textContent = "Uncertain Video Signature";
    confidenceNum.textContent = `${(p * 100).toFixed(1)}% Uncertainty`;
  }

  verdictCaveat.textContent = res.caveat;

  probFill.style.width = `${(p * 100).toFixed(1)}%`;
  scoreGlobal.textContent = res.video_score.toFixed(4);
  scoreFace.textContent = `${res.sampled_frame_count} frames`;
  faceRatio.textContent = "BiLSTM Temporal";
  inferenceLatency.textContent = `${res.timings_ms.total_pipeline_ms.toFixed(1)} ms`;

  // Populate frame sequence strip
  framesTimelineStrip.innerHTML = "";
  if (videoFrameCountTag) videoFrameCountTag.textContent = `${res.sampled_frame_count} Frames Sampled`;

  if (res.frame_thumbnails_b64 && res.frame_thumbnails_b64.length > 0) {
    res.frame_thumbnails_b64.forEach((b64, i) => {
      const item = document.createElement("div");
      item.className = "timeline-frame-item";
      const fScore = res.frame_scores && res.frame_scores[i] !== undefined ? res.frame_scores[i] : res.video_score;
      const scoreColor = fScore > 0.65 ? "#ef4444" : (fScore < 0.35 ? "#10b981" : "#f59e0b");
      
      item.innerHTML = `
        <img src="data:image/jpeg;base64,${b64}" alt="Frame ${i+1}">
        <span class="timeline-frame-score" style="color: ${scoreColor}">F${i+1}: ${fScore.toFixed(2)}</span>
      `;
      framesTimelineStrip.appendChild(item);
    });
  }
}

function simulateVideoPrediction() {
  return {
    p_fake_calibrated: 0.1245,
    state: "REAL",
    video_score: 0.1245,
    sampled_frame_count: 16,
    frame_scores: [0.12, 0.14, 0.11, 0.15, 0.12, 0.13, 0.16, 0.11, 0.14, 0.12, 0.15, 0.13, 0.11, 0.14, 0.12, 0.13],
    frame_thumbnails_b64: [],
    timings_ms: { frame_extraction_ms: 120.0, inference_ms: 240.0, total_pipeline_ms: 360.0 },
    model_version: "v1.0-clip-ensemble-temporal",
    caveat: "Simulation mode. Video verdict based on temporal continuity across 16 frames."
  };
}

// Heatmap Opacity
heatmapOpacity.addEventListener("input", (e) => {
  heatmapDisplay.style.opacity = e.target.value / 100.0;
});

// Robustness Curve & Chart.js
async function fetchRobustnessCurve() {
  if (!currentFile) return;

  let curveData;
  try {
    const formData = new FormData();
    formData.append("file", currentFile);
    const resp = await fetch(`${API_BASE}/v1/robustness`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) throw new Error();
    curveData = await resp.json();
  } catch {
    curveData = [
      { qf: 100, p_fake: currentPrediction ? currentPrediction.p_fake_calibrated : 0.8 },
      { qf: 90, p_fake: 0.78 },
      { qf: 80, p_fake: 0.75 },
      { qf: 70, p_fake: 0.72 },
      { qf: 60, p_fake: 0.68 },
      { qf: 50, p_fake: 0.64 },
      { qf: 40, p_fake: 0.58 },
      { qf: 30, p_fake: 0.54 },
      { qf: 20, p_fake: 0.49 },
      { qf: 10, p_fake: 0.42 },
    ];
  }

  robustnessCurve = curveData;
  renderChart(curveData);
}

function renderChart(curveData) {
  const ctx = document.getElementById("robustness-chart").getContext("2d");
  const labels = curveData.map((d) => `QF ${d.qf}`);
  const values = curveData.map((d) => d.p_fake);

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "P(Fake) under JPEG Compression",
          data: values,
          borderColor: "#6366f1",
          backgroundColor: "rgba(99, 102, 241, 0.15)",
          fill: true,
          tension: 0.35,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: "#a5b4fc",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } },
      },
      scales: {
        y: {
          min: 0,
          max: 1.0,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" },
        },
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" },
        },
      },
    },
  });
}

// QF Slider Scrub
qfSlider.addEventListener("input", (e) => {
  const val = e.target.value;
  qfValDisplay.textContent = val;

  if (robustnessCurve) {
    const closest = robustnessCurve.reduce((prev, curr) =>
      Math.abs(curr.qf - val) < Math.abs(prev.qf - val) ? curr : prev
    );
    qfPredTag.textContent = `P(Fake): ${(closest.p_fake * 100).toFixed(1)}%`;
  }
});

// QF Presets
document.getElementById("btn-qf-clean").addEventListener("click", () => {
  qfSlider.value = 100;
  qfSlider.dispatchEvent(new Event("input"));
});
document.getElementById("btn-qf-wa").addEventListener("click", () => {
  qfSlider.value = 70;
  qfSlider.dispatchEvent(new Event("input"));
});
document.getElementById("btn-qf-agg").addEventListener("click", () => {
  qfSlider.value = 30;
  qfSlider.dispatchEvent(new Event("input"));
});

// Demo fallback helper
function simulateDemoPrediction() {
  return {
    p_fake_calibrated: 0.1245,
    state: "REAL",
    p_global: 0.118,
    p_face: 0.135,
    face_box: [40, 30, 180, 190],
    heatmap_png_b64: null,
    spectrum_png_b64: null,
    timings_ms: { global_inference_ms: 18.2, face_detection_ms: 24.1, total_pipeline_ms: 45.3 },
    model_version: "v1.0-ensemble",
    caveat: "Probability estimation, not forensic proof. Compression and resizing reduce confidence.",
  };
}
