/**
 * NewsLens — Main JavaScript
 * DOM operations use textContent for user data and createContextualFragment
 * for server-generated annotated HTML (already html.escape()'d on the backend).
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let radarChart = null;
let lastResult = null;
let activeTab  = "text";

const CATEGORY_COLORS = {
  "Loaded Language":        "#ff6b6b",
  "Framing":                "#ffd166",
  "Epistemic Manipulation": "#06d6a0",
  "Anchoring":              "#38bdf8",
  "Sensationalism":         "#f97316",
  "False Balance":          "#a78bfa",
  "Whataboutism":           "#fb7185",
  "In-Group Framing":       "#34d399",
};
const LEAN_POSITIONS = { left: 10, center: 50, right: 90, unknown: 50 };
const LEAN_COLORS    = { left: "#3b82f6", center: "#8b5cf6", right: "#ef4444", unknown: "#9ba3c1" };
const SEVERITY_MESSAGES = {
  Low:    "This text appears relatively neutral. Few manipulation signals detected.",
  Medium: "Moderate bias signals detected. Some framing or loaded language present.",
  High:   "Strong bias patterns detected. Treat this content critically.",
};

// ── DOM ───────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

/**
 * Insert server-generated HTML (bias annotation) safely.
 * The backend escapes all user text via Python's html.escape() before adding
 * <mark> tags, so this fragment is structurally trusted.
 */
function setAnnotatedHtml(el, html) {
  const frag = document.createRange().createContextualFragment(html);
  el.replaceChildren(frag);
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initChips();
  initNav();
  initHistory();

  $("analyzeBtn").addEventListener("click", runAnalysis);
  $("clearInput").addEventListener("click", clearInput);
  $("textInput").addEventListener("input",  updateCharCount);
  $("textInput").addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runAnalysis();
  });
  $("urlInput").addEventListener("keydown", e => {
    if (e.key === "Enter") runAnalysis();
  });
  $("analyzeAnother").addEventListener("click", resetUI);
  $("copyResultBtn").addEventListener("click", copyResult);
  $("shareBtn").addEventListener("click", shareAnalysis);
  $("copyCode").addEventListener("click", copyCode);

  // API accordion
  const apiToggle = $("apiAccordionToggle");
  const apiBody   = $("apiAccordionBody");
  apiToggle.addEventListener("click", () => {
    const isOpen = apiBody.classList.toggle("open");
    apiToggle.setAttribute("aria-expanded", isOpen);
  });
});

// ── Theme ─────────────────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  document.body.className = saved;
  $("themeToggle").addEventListener("click", () => {
    const next = document.body.classList.contains("dark") ? "light" : "dark";
    document.body.className = next;
    localStorage.setItem("theme", next);
    if (radarChart && lastResult) {
      radarChart.destroy();
      radarChart = null;
      renderRadar(lastResult.category_scores);
    }
  });
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      activeTab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      $("panelText").style.display = activeTab === "text" ? "" : "none";
      $("panelUrl").style.display  = activeTab === "url"  ? "" : "none";
      $("charCount").textContent   = activeTab === "text"
        ? $("textInput").value.length.toLocaleString() : "";
    });
  });
}

// ── Chips ─────────────────────────────────────────────────────────────────────
function initChips() {
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      $("tabText").click();
      $("textInput").value = chip.dataset.text;
      updateCharCount();
    });
  });
}

function clearInput() {
  $("textInput").value = "";
  $("urlInput").value  = "";
  updateCharCount();
}
function updateCharCount() {
  $("charCount").textContent = $("textInput").value.length.toLocaleString();
}

// ── Drawers ───────────────────────────────────────────────────────────────────
function initNav() {
  $("historyToggle").addEventListener("click", () => openDrawer("history"));
  $("outletsToggle").addEventListener("click", () => openDrawer("outlets"));
  $("closeDrawer").addEventListener("click",  closeAllDrawers);
  $("closeOutlets").addEventListener("click", closeAllDrawers);
  $("drawerOverlay").addEventListener("click", closeAllDrawers);
}

function openDrawer(type) {
  closeAllDrawers();
  const id = type === "history" ? "historyDrawer" : "outletsDrawer";
  $(id).classList.add("open");
  $("drawerOverlay").classList.add("visible");
  document.body.style.overflow = "hidden";
  if (type === "history") refreshHistoryUI();
  else refreshOutletsUI();
}

function closeAllDrawers() {
  $("historyDrawer").classList.remove("open");
  $("outletsDrawer").classList.remove("open");
  $("drawerOverlay").classList.remove("visible");
  document.body.style.overflow = "";
}

// ── History UI ────────────────────────────────────────────────────────────────
function initHistory() {
  $("clearHistory").addEventListener("click", async () => {
    await fetch("/api/history", { method: "DELETE" });
    refreshHistoryUI();
  });
}

async function refreshHistoryUI() {
  const list = $("historyList");
  try {
    const res  = await fetch("/api/history");
    const data = await res.json();
    list.replaceChildren();
    if (!data.history?.length) {
      const msg = document.createElement("div");
      msg.className   = "empty-history";
      msg.textContent = "No analyses yet.";
      list.appendChild(msg);
      return;
    }
    data.history.forEach(h => list.appendChild(buildHistoryItem(h)));
  } catch {
    list.replaceChildren();
    const msg = document.createElement("div");
    msg.className   = "empty-history";
    msg.textContent = "History unavailable.";
    list.appendChild(msg);
  }
}

function buildHistoryItem(h) {
  const wrap  = document.createElement("div");
  wrap.className = "history-item";

  const textEl = document.createElement("div");
  textEl.className   = "history-item-text";
  textEl.textContent = h.text_preview;

  const meta  = document.createElement("div");
  meta.className = "history-item-meta";

  const score = document.createElement("span");
  score.className   = "history-score";
  score.textContent = `${h.bias_score} / 100`;
  score.style.color = `var(--${h.severity_color})`;

  const lean = document.createElement("span");
  lean.className   = `history-lean ${h.political_lean}`;
  lean.textContent = h.political_lean;

  const time = document.createElement("span");
  time.className   = "history-time";
  time.textContent = formatTime(h.timestamp);

  meta.append(score, lean, time);
  wrap.append(textEl, meta);
  return wrap;
}

// ── Outlets UI ────────────────────────────────────────────────────────────────
async function refreshOutletsUI() {
  const list = $("outletsList");
  try {
    const res  = await fetch("/api/outlets");
    const data = await res.json();
    list.replaceChildren();
    if (!data.outlets?.length) {
      const msg = document.createElement("div");
      msg.className   = "empty-history";
      msg.textContent = "Analyse articles via URL to populate the leaderboard.";
      list.appendChild(msg);
      return;
    }
    data.outlets.forEach(o => list.appendChild(buildOutletItem(o)));
  } catch {
    const msg = document.createElement("div");
    msg.className   = "empty-history";
    msg.textContent = "Leaderboard unavailable.";
    list.replaceChildren(msg);
  }
}

function buildOutletItem(o) {
  const wrap  = document.createElement("div");
  wrap.className = "outlet-item";

  const name  = document.createElement("div");
  name.className   = "outlet-name";
  name.textContent = o.name;

  const meta  = document.createElement("div");
  meta.className = "outlet-meta";

  const scoreEl = document.createElement("span");
  scoreEl.className   = "outlet-score";
  scoreEl.textContent = `${o.avg_bias_score.toFixed(1)} avg bias`;
  scoreEl.style.color = `var(--${scoreColor(o.avg_bias_score)})`;

  const lean = document.createElement("span");
  lean.className   = `outlet-lean ${o.dominant_lean}`;
  lean.textContent = o.dominant_lean;

  const count = document.createElement("span");
  count.className   = "outlet-count";
  count.textContent = `${o.article_count} article${o.article_count !== 1 ? "s" : ""}`;

  meta.append(scoreEl, lean, count);
  wrap.append(name, meta);
  return wrap;
}

function scoreColor(score) {
  return score < 33 ? "green" : score < 66 ? "orange" : "red";
}

// ── Analysis ──────────────────────────────────────────────────────────────────
function setInputHint(inputEl, msg) {
  const existing = inputEl.parentElement.querySelector(".input-hint");
  if (msg) {
    inputEl.classList.add("input-error");
    if (!existing) {
      const hint = document.createElement("p");
      hint.className = "input-hint";
      hint.textContent = msg;
      inputEl.parentElement.appendChild(hint);
    }
    const clearHint = () => {
      inputEl.classList.remove("input-error");
      inputEl.parentElement.querySelector(".input-hint")?.remove();
      inputEl.removeEventListener("input", clearHint);
    };
    inputEl.addEventListener("input", clearHint);
  } else {
    inputEl.classList.remove("input-error");
    existing?.remove();
  }
}

async function runAnalysis() {
  if (activeTab === "text") {
    const text = $("textInput").value.trim();
    if (!text) {
      setInputHint($("textInput"), "Please paste some text before analyzing.");
      shake($("textInput"));
      return;
    }
    setInputHint($("textInput"), null);
    await doAnalyze("/api/analyze", { text }, "Analyzing text for bias patterns…");
  } else {
    const url = $("urlInput").value.trim();
    if (!url) {
      setInputHint($("urlInput"), "Please enter a URL before analyzing.");
      shake($("urlInput"));
      return;
    }
    setInputHint($("urlInput"), null);
    await doAnalyze("/api/analyze/url", { url }, "Scraping and analyzing article…");
  }
}

async function doAnalyze(endpoint, body, loadingMsg) {
  $("loadingMsg").textContent = loadingMsg;
  showLoading();
  try {
    const res  = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      const raw = data.detail || data.error;
      const msg = Array.isArray(raw)
        ? raw.map(e => e.msg || e.message || JSON.stringify(e)).join("; ")
        : (typeof raw === "string" ? raw : `Server error (${res.status})`);
      showError(msg);
      return;
    }
    lastResult = data;
    showResults(data);
  } catch (err) {
    const msg = err?.message || err?.error || "An unexpected error occurred.";
    showError(msg);
  }
}

// ── UI States ─────────────────────────────────────────────────────────────────
function showLoading() {
  $("loadingCard").style.display = "block";
  $("resultsCard").style.display = "none";
  $("errorCard").style.display   = "none";
  $("loadingCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function showError(msg) {
  $("loadingCard").style.display = "none";
  $("errorCard").style.display   = "block";
  $("errorMsg").textContent      = msg;
}
function resetUI() {
  $("loadingCard").style.display = "none";
  $("resultsCard").style.display = "none";
  $("errorCard").style.display   = "none";
  (activeTab === "text" ? $("textInput") : $("urlInput")).focus();
}

// ── Results ───────────────────────────────────────────────────────────────────
function showResults(data) {
  $("loadingCard").style.display = "none";
  $("resultsCard").style.display = "block";

  // Score ring animation
  const circumference = 314;
  const offset = circumference - (data.bias_score / 100) * circumference;
  const ring = $("ringFill");
  ring.style.strokeDashoffset = circumference;
  setTimeout(() => { ring.style.strokeDashoffset = offset; }, 50);
  const colorMap = { green: "#4ade80", orange: "#fb923c", red: "#f87171" };
  ring.style.stroke = colorMap[data.severity_color] || "#6366f1";
  $("scoreVal").textContent = Math.round(data.bias_score);

  // Severity badge
  const badge = $("severityBadge");
  badge.textContent = data.severity;
  badge.className   = "severity-badge " + data.severity_color;
  $("severityText").textContent = SEVERITY_MESSAGES[data.severity] || "";

  // Meta row
  $("patternCount").textContent = `${data.pattern_match_count} pattern${data.pattern_match_count !== 1 ? "s" : ""}`;
  $("processTime").textContent  = `${data.processing_ms} ms`;
  $("modelStatus").textContent  = data.roberta_used ? "RoBERTa ✓" : "Pattern Engine";

  // Outlet badge
  if (data.outlet_name) {
    $("outletBadge").style.display = "inline-flex";
    $("outletName").textContent    = data.outlet_name;
  } else {
    $("outletBadge").style.display = "none";
  }

  // Political lean needle
  renderLean(data.political_lean);

  // Annotated text (server-escaped HTML with <mark> tags)
  setAnnotatedHtml($("highlightedText"), data.highlighted_html || "");

  // Category grid + radar
  renderCategoryGrid(data.category_scores, data.category_explanations);
  renderRadar(data.category_scores);

  $("resultsCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Political Lean ────────────────────────────────────────────────────────────
function renderLean(leanObj) {
  const { label, confidence } = leanObj;
  const pos   = LEAN_POSITIONS[label] ?? 50;
  const color = LEAN_COLORS[label]    ?? "#9ba3c1";

  const needle = $("leanNeedle");
  needle.style.left        = pos + "%";
  needle.style.borderColor = color;

  const verdict = $("leanVerdict");
  const labels  = { left: "Leans Left", center: "Center / Neutral", right: "Leans Right" };

  if (!label || label === "unknown" || confidence < 0.1) {
    verdict.textContent = "Undetermined";
    verdict.style.color = "var(--text3)";
    $("leanConf").textContent = "Train the model (ml/train.py) for political lean detection";
  } else {
    verdict.textContent = labels[label] || label;
    verdict.style.color = color;
    $("leanConf").textContent = `${Math.round(confidence * 100)}% confidence`;
  }
}

// ── Category Cards ────────────────────────────────────────────────────────────
function renderCategoryGrid(scores, explanations) {
  const grid = $("categoryGrid");
  grid.replaceChildren();

  Object.entries(scores).forEach(([cat, score]) => {
    const pct   = Math.round(score * 100);
    const color = CATEGORY_COLORS[cat] || "#6366f1";
    const phrases = (explanations?.[cat] || []).slice(0, 3);

    const card = document.createElement("div");
    card.className = "cat-card";
    card.style.setProperty("--cat-color", color);

    const nameEl = document.createElement("div");
    nameEl.className   = "cat-name";
    nameEl.textContent = cat;

    const barWrap = document.createElement("div");
    barWrap.className = "cat-bar-wrapper";
    const bar = document.createElement("div");
    bar.className        = "cat-bar";
    bar.style.width      = pct + "%";
    bar.style.background = color;
    barWrap.appendChild(bar);

    const scoreEl = document.createElement("div");
    scoreEl.className   = "cat-score";
    scoreEl.style.color = color;
    scoreEl.textContent = pct + "%";

    card.append(nameEl, barWrap, scoreEl);

    if (phrases.length) {
      const phrasesEl = document.createElement("div");
      phrasesEl.className = "cat-phrases";
      phrases.forEach(p => {
        const tag = document.createElement("span");
        tag.className   = "cat-phrase-tag";
        tag.textContent = p;
        phrasesEl.appendChild(tag);
      });
      card.appendChild(phrasesEl);
    }

    grid.appendChild(card);
  });
}

// ── Radar Chart ───────────────────────────────────────────────────────────────
function renderRadar(scores) {
  const labels  = Object.keys(scores);
  const values  = Object.values(scores).map(v => Math.round(v * 100));
  const isDark  = document.body.classList.contains("dark");
  const gridC   = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)";
  const textC   = isDark ? "#9ba3c1" : "#4a5280";

  if (radarChart) radarChart.destroy();
  radarChart = new Chart($("radarChart").getContext("2d"), {
    type: "radar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: "rgba(99,102,241,0.15)",
        borderColor: "#6366f1", borderWidth: 2,
        pointBackgroundColor: labels.map(l => CATEGORY_COLORS[l]),
        pointBorderColor: "#fff", pointBorderWidth: 1.5,
        pointRadius: 5, pointHoverRadius: 7,
      }],
    },
    options: {
      responsive: true, animation: { duration: 800 },
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, display: false },
          grid: { color: gridC }, angleLines: { color: gridC },
          pointLabels: { color: textC, font: { size: 10, family: "Inter" } },
        },
      },
    },
  });
}

// ── Copy / Share ──────────────────────────────────────────────────────────────
async function copyResult() {
  if (!lastResult) return;
  const clean = Object.fromEntries(
    Object.entries(lastResult).filter(([k]) => k !== "highlighted_html")
  );
  await navigator.clipboard.writeText(JSON.stringify(clean, null, 2));
  toast("JSON copied ✓");
}
async function shareAnalysis() {
  if (!lastResult) return;
  const lean = lastResult.political_lean?.label ?? "unknown";
  const txt  = `NewsLens: ${lastResult.bias_score}/100 (${lastResult.severity}) · Lean: ${lean}\n${lastResult.text.slice(0, 200)}`;
  try { await navigator.share({ title: "NewsLens Analysis", text: txt }); }
  catch { await navigator.clipboard.writeText(txt); toast("Copied ✓"); }
}
async function copyCode() {
  await navigator.clipboard.writeText($("codeExample").textContent);
  toast("Code copied ✓");
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function formatTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}
function shake(el) {
  el.style.animation = "none";
  el.offsetHeight;
  el.style.animation = "shake 0.4s ease";
  setTimeout(() => { el.style.animation = ""; }, 400);
}
function toast(msg) {
  const t = document.createElement("div");
  t.textContent = msg;
  Object.assign(t.style, {
    position: "fixed", bottom: "24px", right: "24px",
    background: "#6366f1", color: "#fff", padding: "10px 20px",
    borderRadius: "8px", fontSize: "0.85rem", fontWeight: "600",
    zIndex: "999", opacity: "1", transition: "opacity 0.4s ease",
  });
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 400); }, 2200);
}
const _kf = document.createElement("style");
_kf.textContent = `@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-4px)}80%{transform:translateX(4px)}}`;
document.head.appendChild(_kf);
