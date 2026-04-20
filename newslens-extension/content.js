/**
 * NewsLens Content Script
 * Receives HIGHLIGHT messages from popup.js and injects
 * color-coded underlines on matched bias phrases in the page body.
 */

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

const HIGHLIGHT_CLASS = 'newslens-highlight';
const TOOLTIP_CLASS   = 'newslens-tooltip';
const STYLE_ID        = 'newslens-style';

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = [
    '.'+HIGHLIGHT_CLASS+'{border-radius:2px;padding:0 1px;cursor:pointer;position:relative;transition:filter .15s;}',
    '.'+HIGHLIGHT_CLASS+':hover{filter:brightness(1.3);}',
    '.'+HIGHLIGHT_CLASS+':hover .'+TOOLTIP_CLASS+'{opacity:1;pointer-events:auto;}',
    '.'+TOOLTIP_CLASS+'{position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);',
    'background:#1a1a2e;color:#ede9ff;font-size:11px;font-family:system-ui,sans-serif;',
    'white-space:nowrap;padding:4px 9px;border-radius:5px;border:1px solid rgba(255,255,255,0.12);',
    'opacity:0;pointer-events:none;z-index:999999;transition:opacity .15s;}'
  ].join('');
  document.head.appendChild(style);
}

function clearHighlights() {
  document.querySelectorAll('.' + HIGHLIGHT_CLASS).forEach(mark => {
    const parent = mark.parentNode;
    if (!parent) return;
    parent.replaceChild(document.createTextNode(mark.textContent), mark);
    parent.normalize();
  });
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\]/g, '\$&');
}

function highlightPhrase(phrase, category) {
  const color = CAT_COLORS[category] || '#7c3aed';
  const regex = new RegExp('(?<![\w])' + escapeRegex(phrase) + '(?![\w])', 'gi');

  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        const tag = node.parentElement && node.parentElement.tagName
          ? node.parentElement.tagName.toUpperCase() : '';
        if (['SCRIPT','STYLE','NOSCRIPT','TEXTAREA','INPUT'].includes(tag)) {
          return NodeFilter.FILTER_REJECT;
        }
        if (node.parentElement && node.parentElement.classList &&
            node.parentElement.classList.contains(HIGHLIGHT_CLASS)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  const nodes = [];
  let node;
  while ((node = walker.nextNode())) {
    regex.lastIndex = 0;
    if (regex.test(node.textContent)) nodes.push(node);
  }

  nodes.forEach(textNode => {
    const text = textNode.textContent;
    const frag = document.createDocumentFragment();
    let last = 0;
    let match;
    regex.lastIndex = 0;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > last) {
        frag.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      const mark = document.createElement('mark');
      mark.className = HIGHLIGHT_CLASS;
      mark.textContent = match[0];
      mark.style.cssText = 'background:' + color + '22;border-bottom:2px solid ' + color + ';';

      const tip = document.createElement('span');
      tip.className = TOOLTIP_CLASS;
      tip.textContent = category;
      mark.appendChild(tip);
      frag.appendChild(mark);
      last = match.index + match[0].length;
    }
    if (last < text.length) {
      frag.appendChild(document.createTextNode(text.slice(last)));
    }
    if (textNode.parentNode) {
      textNode.parentNode.replaceChild(frag, textNode);
    }
  });
}

function applyHighlights(data) {
  injectStyles();
  clearHighlights();
  const explanations = data.category_explanations || {};
  Object.entries(explanations).forEach(function(entry) {
    var category = entry[0];
    var phrases  = entry[1];
    phrases.forEach(function(phrase) {
      if (phrase && phrase.length > 2) {
        try { highlightPhrase(phrase, category); } catch(e) {}
      }
    });
  });
}

function extractArticleText() {
  // Try semantic article containers first, then fall back to body
  const candidates = [
    document.querySelector('article'),
    document.querySelector('[role="main"]'),
    document.querySelector('main'),
    document.querySelector('.article-body'),
    document.querySelector('.post-content'),
    document.querySelector('.entry-content'),
    document.body,
  ];
  for (const el of candidates) {
    if (!el) continue;
    // Clone and strip scripts/styles/nav/ads before reading text
    const clone = el.cloneNode(true);
    clone.querySelectorAll('script,style,nav,header,footer,aside,[class*="ad"],[id*="ad"]').forEach(n => n.remove());
    const text = clone.innerText || clone.textContent || '';
    const cleaned = text.replace(/\s+/g, ' ').trim();
    if (cleaned.length > 200) return cleaned;
  }
  return null;
}

chrome.runtime.onMessage.addListener(function(msg, sender, sendResponse) {
  if (msg.type === 'HIGHLIGHT' && msg.data) applyHighlights(msg.data);
  if (msg.type === 'CLEAR') clearHighlights();
  if (msg.type === 'GET_TEXT') {
    sendResponse({ text: extractArticleText() });
  }
  return true; // keep channel open for async sendResponse
});
