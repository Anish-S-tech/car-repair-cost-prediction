/**
 * SPA Router & App entry point (no auth).
 */

// Pages
import { renderHome } from './pages/home.js';
import { renderPredict, mountPredict } from './pages/predict.js';
import { renderHistory, mountHistory } from './pages/history.js';

const app = document.getElementById('app');

// ── Route table ─────────────────────────────────────
const routes = {
    '/': { render: renderHome },
    '/predict': { render: renderPredict, mount: mountPredict },
    '/history': { render: renderHistory, mount: mountHistory },
};

// ── Navigate helper (exported for pages) ───────────
export function navigate(path) {
    window.location.hash = path;
}

// ── Router ──────────────────────────────────────────
function router() {
    const hash = window.location.hash.slice(1) || '/';
    const route = routes[hash] || routes['/'];

    // Render page HTML
    app.innerHTML = route.render();

    // Mount event listeners
    if (route.mount) {
        route.mount();
    }

    // Update nav
    updateNav(hash);

    // Scroll to top
    window.scrollTo(0, 0);
}

// ── Nav bar ─────────────────────────────────────────
function updateNav(currentPath) {
    const navLinks = document.getElementById('nav-links');
    navLinks.innerHTML = `
    <a href="#/" class="nav-link ${currentPath === '/' ? 'active' : ''}">Home</a>
    <a href="#/predict" class="nav-link ${currentPath === '/predict' ? 'active' : ''}">Predict</a>
    <a href="#/history" class="nav-link ${currentPath === '/history' ? 'active' : ''}">History</a>
  `;
}

// ── Init ────────────────────────────────────────────
window.addEventListener('hashchange', router);
window.addEventListener('DOMContentLoaded', router);
