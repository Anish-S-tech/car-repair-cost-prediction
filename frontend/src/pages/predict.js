/**
 * Predict page — drag-and-drop image upload + ML prediction (no auth).
 */

import { showToast } from '../toast.js';

export function renderPredict() {
  return `
    <div class="predict-container">
      <h2>Estimate Repair Cost</h2>
      <p class="predict-subtitle">Upload a photo of the damaged vehicle and get an instant estimate</p>

      <div class="glass-card-static">
        <!-- Drop zone -->
        <div class="drop-zone" id="drop-zone">
          <span class="drop-zone-icon">📁</span>
          <p class="drop-zone-text"><strong>Click or drag &amp; drop</strong> an image here</p>
          <p class="drop-zone-text" style="font-size: 0.8rem; margin-top: 0.5rem; color: var(--text-muted)">JPG, JPEG, or PNG — max 10 MB</p>
          <input type="file" id="file-input" accept=".jpg,.jpeg,.png" />
        </div>

        <!-- Preview -->
        <div id="preview-section" class="preview-container" style="display: none">
          <img id="preview-img" class="preview-image" alt="Uploaded preview" />
          <div style="display: flex; gap: 1rem; justify-content: center;">
            <button class="btn btn-primary btn-lg" id="predict-btn">🔍 Analyze Damage</button>
            <button class="btn btn-secondary" id="clear-btn">Clear</button>
          </div>
        </div>

        <!-- Loading -->
        <div id="loading-section" style="display: none">
          <div class="spinner-container">
            <div class="spinner"></div>
            <p class="spinner-text">Analyzing damage with AI…</p>
          </div>
        </div>
      </div>

      <!-- Result -->
      <div id="result-section" style="display: none">
        <div class="glass-card-static result-card">
          <p class="result-label">Estimated Repair Cost</p>
          <p class="result-cost" id="result-cost"></p>
          <div class="result-actions">
            <button class="btn btn-primary" id="new-prediction-btn">New Prediction</button>
            <a href="#/history" class="btn btn-secondary">View History</a>
          </div>
        </div>
      </div>
    </div>
  `;
}

export function mountPredict() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const previewSection = document.getElementById('preview-section');
  const previewImg = document.getElementById('preview-img');
  const predictBtn = document.getElementById('predict-btn');
  const clearBtn = document.getElementById('clear-btn');
  const loadingSection = document.getElementById('loading-section');
  const resultSection = document.getElementById('result-section');
  const resultCost = document.getElementById('result-cost');
  const newPredBtn = document.getElementById('new-prediction-btn');

  let selectedFile = null;

  // Drag & drop handlers
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['jpg', 'jpeg', 'png'].includes(ext)) {
      showToast('Please upload a JPG or PNG image', 'error');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast('File too large (max 10 MB)', 'error');
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropZone.style.display = 'none';
      previewSection.style.display = 'block';
      resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  // Clear
  clearBtn.addEventListener('click', resetForm);

  function resetForm() {
    selectedFile = null;
    fileInput.value = '';
    dropZone.style.display = '';
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
  }

  // Predict — simple fetch, no auth token needed
  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    previewSection.style.display = 'none';
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';

    try {
      const form = new FormData();
      form.append('file', selectedFile);

      const res = await fetch('/api/predict', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();

      loadingSection.style.display = 'none';
      resultCost.textContent = data.formatted_cost;
      resultSection.style.display = 'block';
      showToast('Prediction complete!', 'success');
    } catch (err) {
      loadingSection.style.display = 'none';
      dropZone.style.display = '';
      previewSection.style.display = 'block';
      showToast(err.message, 'error');
    }
  });

  // New prediction
  newPredBtn.addEventListener('click', resetForm);
}
