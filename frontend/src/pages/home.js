/**
 * Home page — hero section + features.
 */

export function renderHome() {
  return `
    <section class="hero">
      <span class="hero-badge">✨ AI-Powered Estimation</span>
      <h1>Instant <span class="gradient-text">Car Repair</span><br/>Cost Estimates</h1>
      <p class="hero-subtitle">
        Upload a photo of vehicle damage and our deep-learning model delivers an 
        accurate repair cost estimate in seconds. No guesswork, no waiting.
      </p>
      <div class="hero-actions">
        <a href="#/predict" class="btn btn-primary btn-lg">Upload &amp; Predict</a>
        <a href="#/history" class="btn btn-secondary btn-lg">View History</a>
      </div>
    </section>

    <section class="features-grid">
      <div class="glass-card feature-card">
        <span class="feature-icon">📸</span>
        <h3>Upload a Photo</h3>
        <p>Simply snap or upload a picture of the damaged area — scratches, dents, or major collisions.</p>
      </div>
      <div class="glass-card feature-card">
        <span class="feature-icon">🤖</span>
        <h3>AI Analysis</h3>
        <p>Our deep-learning model analyses damage severity and computes an accurate repair cost in real-time.</p>
      </div>
      <div class="glass-card feature-card">
        <span class="feature-icon">💰</span>
        <h3>Instant Estimate</h3>
        <p>Get a detailed cost breakdown in US Dollars ($) so you can plan repairs with confidence.</p>
      </div>
    </section>
  `;
}
