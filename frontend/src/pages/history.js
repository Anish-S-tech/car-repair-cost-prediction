/**
 * History page — grid of past predictions (no auth).
 */

import { showToast } from '../toast.js';

export function renderHistory() {
  return `
    <div class="history-container">
      <h2>Prediction History</h2>
      <p class="history-subtitle">Browse past damage assessments and cost estimates</p>
      <div id="history-content">
        <div class="spinner-container">
          <div class="spinner"></div>
          <p class="spinner-text">Loading history…</p>
        </div>
      </div>
    </div>
  `;
}

export function mountHistory() {
  loadHistory();
}

async function loadHistory() {
  const container = document.getElementById('history-content');

  try {
    const res = await fetch('/api/history');
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    const items = await res.json();

    if (!items.length) {
      container.innerHTML = `
        <div class="empty-state">
          <span class="empty-state-icon">📭</span>
          <p>No predictions yet</p>
          <a href="#/predict" class="btn btn-primary" style="margin-top: 1rem;">Make Your First Prediction</a>
        </div>
      `;
      return;
    }

    container.innerHTML = `
      <div class="history-grid">
        ${items.map(item => `
          <div class="glass-card history-card">
            <img
              class="history-card-image"
              src="${item.image_url}"
              alt="Damage photo"
              onerror="this.style.display='none'"
              loading="lazy"
            />
            <div class="history-card-body">
              <p class="history-card-cost">${item.formatted_cost}</p>
              <p class="history-card-date">${formatDate(item.created_at)}</p>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  } catch (err) {
    showToast(err.message, 'error');
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-state-icon">⚠️</span>
        <p>Failed to load history</p>
      </div>
    `;
  }
}

function formatDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}
