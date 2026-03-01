/**
 * Signup page.
 */

import { apiFetch, setAuth } from '../api.js';
import { showToast } from '../toast.js';
import { navigate } from '../main.js';

export function renderSignup() {
    return `
    <div class="auth-container">
      <div class="glass-card-static">
        <h2>Create Account</h2>
        <p class="auth-subtitle">Sign up to start estimating repair costs</p>

        <form id="signup-form">
          <div class="form-group">
            <label class="form-label" for="signup-username">Username</label>
            <input class="form-input" type="text" id="signup-username" placeholder="Choose a username" required />
          </div>
          <div class="form-group">
            <label class="form-label" for="signup-password">Password</label>
            <input class="form-input" type="password" id="signup-password" placeholder="Choose a password (min 4 chars)" required minlength="4" />
          </div>
          <div class="form-group">
            <label class="form-label" for="signup-confirm">Confirm Password</label>
            <input class="form-input" type="password" id="signup-confirm" placeholder="Re-enter your password" required />
          </div>
          <button type="submit" class="btn btn-primary btn-lg" style="width: 100%" id="signup-btn">Create Account</button>
        </form>

        <p class="auth-footer">
          Already have an account? <a href="#/login">Sign in</a>
        </p>
      </div>
    </div>
  `;
}

export function mountSignup() {
    const form = document.getElementById('signup-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('signup-btn');
        const username = document.getElementById('signup-username').value.trim();
        const password = document.getElementById('signup-password').value;
        const confirm = document.getElementById('signup-confirm').value;

        if (!username || !password) {
            showToast('Please fill in all fields', 'error');
            return;
        }
        if (password !== confirm) {
            showToast('Passwords do not match', 'error');
            return;
        }

        btn.disabled = true;
        btn.textContent = 'Creating account…';

        try {
            const data = await apiFetch('/api/auth/signup', {
                method: 'POST',
                body: JSON.stringify({ username, password }),
            });
            setAuth(data.access_token, data.username);
            showToast('Account created successfully!', 'success');
            navigate('/predict');
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Create Account';
        }
    });
}
