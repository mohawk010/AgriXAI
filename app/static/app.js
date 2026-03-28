/**
 * app.js — AgriXAI frontend logic
 * Handles: drag-drop upload, camera capture, /predict fetch, result rendering
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropzone        = document.getElementById('dropzone');
const dropzoneInner   = document.getElementById('dropzone-inner');
const fileInput       = document.getElementById('file-input');
const browseBtn       = document.getElementById('browse-btn');
const previewContainer= document.getElementById('preview-container');
const previewImg      = document.getElementById('preview-img');
const changeImgBtn    = document.getElementById('change-img-btn');

const cameraBtn       = document.getElementById('camera-btn');
const cameraContainer = document.getElementById('camera-container');
const cameraVideo     = document.getElementById('camera-video');
const captureBtn      = document.getElementById('capture-btn');
const closeCameraBtn  = document.getElementById('close-camera-btn');
const captureCanvas   = document.getElementById('capture-canvas');

const analyseBtn      = document.getElementById('analyse-btn');
const analyseBtnLabel = document.getElementById('analyse-btn-label');

const resultCard      = document.getElementById('result-card');
const resultClass     = document.getElementById('result-class');
const confidenceBar   = document.getElementById('confidence-bar');
const confidencePct   = document.getElementById('confidence-pct');
const top5List        = document.getElementById('top5-list');

const heatmapsPlaceholder = document.getElementById('heatmaps-placeholder');
const heatmapsContent     = document.getElementById('heatmaps-content');
const heatmapSpinner      = document.getElementById('heatmap-spinner');
const gradcamImg          = document.getElementById('gradcam-img');
const dctSpectrumImg      = document.getElementById('dct-spectrum-img');
const dctBandImg          = document.getElementById('dct-band-img');

const aiPlaceholder   = document.getElementById('ai-placeholder');
const aiSpinner       = document.getElementById('ai-spinner');
const aiContent       = document.getElementById('ai-content');

const toast           = document.getElementById('toast');

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile    = null;   // File | Blob currently queued for analysis
let cameraStream   = null;   // MediaStream when camera is active
let isAnalysing    = false;

// ── Toast helper ──────────────────────────────────────────────────────────────
function showToast(msg, type = 'info', duration = 3500) {
  toast.textContent = msg;
  toast.className   = `toast show ${type}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.className = 'toast'; }, duration);
}

// ── File / Preview helpers ─────────────────────────────────────────────────────
function setPreview(src) {
  previewImg.src = src;
  dropzoneInner.setAttribute('hidden', '');
  previewContainer.removeAttribute('hidden');
  analyseBtn.removeAttribute('disabled');
}

function clearPreview() {
  previewImg.src = '';
  currentFile = null;
  previewContainer.setAttribute('hidden', '');
  dropzoneInner.removeAttribute('hidden');
  analyseBtn.setAttribute('disabled', '');
  fileInput.value = '';
}

function loadFilePreview(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image file.', 'error');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => setPreview(e.target.result);
  reader.readAsDataURL(file);
}

// ── Browse button ─────────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('click', e => {
  if (!previewContainer.hasAttribute('hidden')) return; // click on preview → ignore
  if (e.target === browseBtn) return;
  fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFilePreview(fileInput.files[0]);
});
changeImgBtn.addEventListener('click', e => {
  e.stopPropagation();
  clearPreview();
});

// ── Drag & Drop ───────────────────────────────────────────────────────────────
['dragenter','dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  })
);
['dragleave','drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
  })
);
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) loadFilePreview(file);
});

// ── Camera ────────────────────────────────────────────────────────────────────
cameraBtn.addEventListener('click', async () => {
  if (cameraStream) return; // already open
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraContainer.removeAttribute('hidden');
    cameraBtn.setAttribute('disabled', '');
  } catch (err) {
    showToast('Camera access denied or unavailable.', 'error');
  }
});

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  cameraContainer.setAttribute('hidden', '');
  cameraBtn.removeAttribute('disabled');
}

closeCameraBtn.addEventListener('click', stopCamera);

captureBtn.addEventListener('click', () => {
  if (!cameraStream) return;
  const video = cameraVideo;
  captureCanvas.width  = video.videoWidth  || 640;
  captureCanvas.height = video.videoHeight || 480;
  captureCanvas.getContext('2d').drawImage(video, 0, 0);
  captureCanvas.toBlob(blob => {
    currentFile = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
    setPreview(captureCanvas.toDataURL('image/jpeg'));
    stopCamera();
  }, 'image/jpeg', 0.92);
});

// ── Analysis ──────────────────────────────────────────────────────────────────
analyseBtn.addEventListener('click', () => {
  if (!currentFile || isAnalysing) return;
  runAnalysis();
});

async function runAnalysis() {
  isAnalysing = true;

  // UI: loading state
  analyseBtn.classList.add('loading');
  analyseBtnLabel.textContent = 'Analysing…';
  analyseBtn.setAttribute('disabled', '');

  // Show spinners, hide old results
  show(heatmapSpinner);
  hide(heatmapsPlaceholder);
  hide(heatmapsContent);

  show(aiSpinner);
  hide(aiPlaceholder);
  hide(aiContent);

  resultCard.setAttribute('hidden', '');

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    showToast('Analysis complete ✓', 'success');

  } catch (err) {
    showToast(`Error: ${err.message}`, 'error', 6000);
    hide(heatmapSpinner);
    show(heatmapsPlaceholder);
    hide(aiSpinner);
    show(aiPlaceholder);
  } finally {
    isAnalysing = false;
    analyseBtn.classList.remove('loading');
    analyseBtnLabel.textContent = 'Analyse Plant';
    analyseBtn.removeAttribute('disabled');
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  /* ── Result card (left panel) ─────────────────────────────── */
  const label = formatClassName(data.predicted_class);
  resultClass.textContent = label;

  // Animated confidence bar
  confidenceBar.style.width = '0%';
  resultCard.removeAttribute('hidden');
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceBar.style.width = `${data.confidence}%`;
    });
  });
  confidencePct.textContent = `${data.confidence.toFixed(1)}%`;

  // Top-5 list
  top5List.innerHTML = '';
  data.top5.forEach((item, idx) => {
    const div = document.createElement('div');
    div.className = `top5-item${idx === 0 ? ' top1' : ''}`;
    div.innerHTML = `
      <span>${formatClassName(item.class)}</span>
      <span>${(item.probability * 100).toFixed(2)}%</span>
    `;
    top5List.appendChild(div);
  });

  /* ── Heatmaps (right panel) ───────────────────────────────── */
  hide(heatmapSpinner);
  gradcamImg.src       = `data:image/png;base64,${data.gradcam_b64}`;
  dctSpectrumImg.src   = `data:image/png;base64,${data.dct_spectrum_b64}`;
  dctBandImg.src       = `data:image/png;base64,${data.dct_band_b64}`;
  show(heatmapsContent);

  /* ── AI report (bottom panel) ─────────────────────────────── */
  hide(aiSpinner);
  // Render markdown using marked.js
  if (typeof marked !== 'undefined') {
    aiContent.innerHTML = marked.parse(data.ai_analysis || '');
  } else {
    // Fallback: plain text with basic newline rendering
    aiContent.innerHTML = `<pre style="white-space:pre-wrap;">${escapeHtml(data.ai_analysis)}</pre>`;
  }
  show(aiContent);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function show(el) { el.removeAttribute('hidden'); }
function hide(el) { el.setAttribute('hidden', ''); }

function formatClassName(name) {
  // e.g. "Tomato___Late_blight" → "Tomato — Late Blight"
  return name
    .replace(/___/g, ' — ')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(str) {
  return (str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
