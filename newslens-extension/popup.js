const API = 'http://localhost:5001';

const CAT_COLORS = {
  'Loaded Language':        '#ff6b6b',
  'Framing':                '#ffd166',
  'Epistemic Manipulation': '#06d6a0',
  'Anchoring':              '#38bdf8',
  'Sensationalism':         '#f97316',
  'False Balance':          '#a78bfa',
  'Whataboutism':           '#fb7185',
  'In-Group Framing':       '#34d399',
};

// ── UI helpers ────────────────────────────────────────────────────────────────

const STATE_DISPLAY = {
  stateIdle:    'flex',
  stateLoading: 'flex',
  stateError:   'flex',
  stateResult:  'block',
};
function show(id) {
  Object.keys(STATE_DISPLAY).forEach(s => {
    const el = document.getElementById(s);
    el.style.display = s === id ? STATE_DISPLAY[s] : 'none';
  });
}

function setLoading(msg) {
  show('stateLoading');
  document.getElementById('loadingMsg').textContent = msg;
}

function showError(msg) {
  show('stateError');
  document.getElementById('errorMsg').textContent = msg;
}

// ── Safe DOM helpers ──────────────────────────────────────────────────────────

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// ── Render result ─────────────────────────────────────────────────────────────

function render(data, url) {
  show('stateResult');

  // Score readout (now a number display, not a circle)
  const score  = Math.round(data.bias_score);
  const cls    = data.severity === 'High' ? 'high' : data.severity === 'Low' ? 'low' : 'medium';
  const circle = document.getElementById('scoreCircle');
  circle.textContent = score;
  circle.className = `score-num ${cls}`;

  const sev = document.getElementById('scoreSeverity');
  sev.textContent = `${data.severity} Bias`;
  sev.className = `score-severity ${cls}`;

  // URL
  try {
    const hostname = new URL(url).hostname.replace('www.', '');
    document.getElementById('scoreUrl').textContent = hostname;
  } catch { document.getElementById('scoreUrl').textContent = ''; }

  // RoBERTa badge
  const badge = document.getElementById('robertaBadge');
  if (data.roberta_used) {
    badge.textContent = '● RoBERTa active';
    badge.className = 'roberta-badge on';
  } else {
    badge.textContent = '● Rule-only';
    badge.className = 'roberta-badge off';
  }

  // Lean
  const leanLabel = data.political_lean.label;
  const leanConf  = data.political_lean.confidence;
  const fill  = document.getElementById('leanFill');
  const ltext = document.getElementById('leanText');

  fill.className  = `lean-fill ${leanLabel}`;
  ltext.className = `lean-readout ${leanLabel}`;

  if (leanLabel === 'left') {
    fill.style.cssText = `width:${Math.round(leanConf*100)}%;left:0;right:auto`;
    ltext.textContent  = `◀ Left ${Math.round(leanConf*100)}%`;
  } else if (leanLabel === 'right') {
    fill.style.cssText = `width:${Math.round(leanConf*100)}%;right:0;left:auto`;
    ltext.textContent  = `Right ${Math.round(leanConf*100)}% ▶`;
  } else if (leanLabel === 'center') {
    const pct = Math.round(leanConf*100);
    fill.style.cssText = `width:${pct}%;left:${(100-pct)/2}%;right:auto`;
    ltext.textContent  = `Center ${pct}%`;
  } else {
    fill.style.width  = '0';
    ltext.textContent = '— Unknown';
  }

  // Category bars — safe DOM construction (no innerHTML)
  const rows = document.getElementById('catRows');
  rows.textContent = '';   // clear safely

  const sorted = Object.entries(data.category_scores).sort((a,b) => b[1]-a[1]);
  let anyVisible = false;

  sorted.forEach(([name, score]) => {
    const pct = Math.round(score * 100);
    if (pct === 0) return;
    anyVisible = true;
    const color = CAT_COLORS[name] || '#7c3aed';

    const row   = el('div', 'cat-row');
    const label = el('span', 'cat-name', name);
    const track = el('div', 'cat-track');
    const bar   = el('div', 'cat-fill');
    bar.style.width      = `${pct}%`;
    bar.style.background = color;
    const pctEl = el('span', 'cat-pct', `${pct}%`);

    track.appendChild(bar);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pctEl);
    rows.appendChild(row);
  });

  if (!anyVisible) {
    rows.appendChild(el('div', 'no-bias-msg', '// No bias patterns detected.'));
  }

  // Footer
  const count = data.pattern_match_count;
  document.getElementById('footerMeta').textContent =
    `${count} pattern${count !== 1 ? 's' : ''} · ${Math.round(data.processing_ms)}ms`;
}

// ── Analyze current tab ───────────────────────────────────────────────────────

async function analyze() {
  setLoading('Getting page URL…');

  let tab;
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  } catch {
    showError('Could not access the current tab.');
    return;
  }

  const url = tab.url || '';
  if (!url || url.startsWith('chrome://') || url.startsWith('chrome-extension://') || url.startsWith('about:')) {
    showError('Navigate to a news article first, then click Analyze.');
    return;
  }

  setLoading('Extracting article…');

  try {
    // Try client-side DOM extraction first — works on paywalled/login-gated sites
    let data = null;
    let usedClientExtraction = false;

    try {
      const textResp = await chrome.tabs.sendMessage(tab.id, { type: 'GET_TEXT' });
      if (textResp && textResp.text && textResp.text.length > 200) {
        usedClientExtraction = true;
        setLoading('Analyzing article…');
        const resp = await fetch(`${API}/api/analyze`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ text: textResp.text }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || `Server error ${resp.status}`);
        }
        data = await resp.json();
      }
    } catch (clientErr) {
      if (usedClientExtraction) throw clientErr;
      // Content script unavailable — fall through to server scrape
    }

    // Fallback: server-side URL scraping
    if (!data) {
      setLoading('Analyzing article…');
      const resp = await fetch(`${API}/api/analyze/url`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ url }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      data = await resp.json();
    }

    // Cache for content script
    chrome.storage.local.set({ lastResult: data, lastUrl: url });
    // Ask content script to highlight phrases on the page
    chrome.tabs.sendMessage(tab.id, { type: 'HIGHLIGHT', data }).catch(() => {});

    render(data, url);

  } catch (err) {
    const msg = err.message || '';
    if (msg.includes('fetch') || msg.includes('Network')) {
      showError('Cannot reach NewsLens server.\nMake sure it is running at localhost:5001.');
    } else {
      showError(msg || 'Analysis failed. Please try again.');
    }
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  show('stateIdle');

  // Re-show cached result if same tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const { lastResult, lastUrl } = await chrome.storage.local.get(['lastResult','lastUrl']);
  if (lastUrl === tab?.url && lastResult) {
    render(lastResult, lastUrl);
    return;
  }

  document.getElementById('btnAnalyze').addEventListener('click', analyze);

  document.getElementById('btnRetry').addEventListener('click', () => {
    show('stateIdle');
    document.getElementById('btnAnalyze').addEventListener('click', analyze);
  });

  document.getElementById('btnReanalyze').addEventListener('click', () => {
    chrome.storage.local.remove(['lastResult','lastUrl']);
    show('stateIdle');
    document.getElementById('btnAnalyze').addEventListener('click', analyze);
  });

  document.getElementById('btnDashboard').addEventListener('click', () => {
    chrome.tabs.create({ url: 'http://localhost:5001' });
  });
});
