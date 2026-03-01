/**
 * API helper — fetch wrapper with JWT auto-attach.
 */

const BASE = '';  // Same origin via Vite proxy

/** Get stored JWT */
export function getToken() {
    return localStorage.getItem('token');
}

/** Save JWT + username */
export function setAuth(token, username) {
    localStorage.setItem('token', token);
    localStorage.setItem('username', username);
}

/** Clear auth */
export function clearAuth() {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
}

/** Check if logged in */
export function isLoggedIn() {
    return !!getToken();
}

/** Get stored username */
export function getUsername() {
    return localStorage.getItem('username') || '';
}

/**
 * JSON request helper.
 */
export async function apiFetch(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    const token = getToken();
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    // Don't set Content-Type for FormData (browser sets boundary automatically)
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = headers['Content-Type'] || 'application/json';
    }

    const res = await fetch(`${BASE}${path}`, { ...options, headers });

    if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
            const body = await res.json();
            detail = body.detail || detail;
        } catch { /* ignore */ }
        throw new Error(detail);
    }

    return res.json();
}
