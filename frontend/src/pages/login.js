/**
 * Login page.
 */

import { apiFetch, setAuth } from '../api.js';
import { showToast } from '../toast.js';
import { navigate } from '../main.js';

export function renderLogin() {
    return `
    <div class="auth-container">
      <div class="glass-card-static">
        <h2>Welcome Back</h2>
        <p class="auth-subtitle">Sign in to your account to continue</p>

        <form id="login-form">
          <div class="form-group">
            <label class="form-label" for="login-username">Username</label>
            <input class="form-input" type="text" id="login-username" placeholder="Enter your username" required />
          </div>
          <div class="form-group">
            <label class="form-label" for="login-password">Password</label>
            <input class="form-input" type="password" id="login-password" placeholder="Enter your password" required />
          </div>
          <button type="submit" class="btn btn-primary btn-lg" style="width: 100%" id="login-btn">Sign In</button>
        </form>

        <p class="auth-footer">
          Don't have an account? <a href="#/signup">Create one</a>
        </p>
      </div>
    </div>
  `;
}

export function mountLogin() {
    const form = document.getElementById('login-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('login-btn');
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;

        if (!username || !password) {
            showToast('Please fill in all fields', 'error');
            return;
        }

        btn.disabled = true;
        btn.textContent = 'Signing in…';

        try {
            const data = await apiFetch('/api/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password }),
            });
            setAuth(data.access_token, data.username);
            showToast(`Welcome back, ${data.username}!`, 'success');
            navigate('/predict');
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Sign In';
        }
    });
}
