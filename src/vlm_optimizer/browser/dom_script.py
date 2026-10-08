"""Browser-side DOM extraction script.

Runs inside the page via ``page.evaluate``. It walks the DOM for genuinely
*actionable* elements, filters out anything invisible or off-viewport, and
returns a flat, JSON-serializable list. All geometry is in viewport CSS pixels
(``getBoundingClientRect``), which maps 1:1 to screenshot pixels when
``device_scale_factor == 1``.

Keeping this logic in JS (not Python) means we only pay one round-trip per
capture instead of one per element, and visibility is computed with the
browser's own layout engine — the ground truth.
"""

from __future__ import annotations

EXTRACT_INTERACTABLES_JS = r"""
() => {
  const MAX_TEXT = 80;

  const INTERACTIVE_SELECTOR = [
    'a[href]',
    'button',
    'input',
    'select',
    'textarea',
    '[contenteditable=""]',
    '[contenteditable="true"]',
    '[role="button"]',
    '[role="link"]',
    '[role="checkbox"]',
    '[role="radio"]',
    '[role="menuitem"]',
    '[role="tab"]',
    '[role="switch"]',
    '[role="combobox"]',
    '[role="textbox"]',
    '[onclick]',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  const vw = window.innerWidth;
  const vh = window.innerHeight;

  const truncate = (s) => {
    if (!s) return '';
    s = ('' + s).replace(/\s+/g, ' ').trim();
    return s.length > MAX_TEXT ? s.slice(0, MAX_TEXT) + '…' : s;
  };

  const accessibleName = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria;
    const labelledby = el.getAttribute('aria-labelledby');
    if (labelledby) {
      const ref = document.getElementById(labelledby);
      if (ref) return ref.innerText || ref.textContent || '';
    }
    if (el.id) {
      const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lbl) return lbl.innerText || lbl.textContent || '';
    }
    const title = el.getAttribute('title') || el.getAttribute('placeholder');
    if (title) return title;
    if (el.tagName === 'INPUT' && (el.type === 'submit' || el.type === 'button')) {
      return el.value || '';
    }
    return el.innerText || el.textContent || '';
  };

  const normalizeRole = (el) => {
    const explicit = (el.getAttribute('role') || '').toLowerCase();
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'select';
    if (tag === 'textarea') return 'textarea';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t === 'checkbox') return 'checkbox';
      if (t === 'radio') return 'radio';
      if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
      return 'input';
    }
    if (el.isContentEditable) return 'editable';
    if (explicit === 'link') return 'link';
    if (explicit === 'checkbox') return 'checkbox';
    if (explicit === 'radio') return 'radio';
    if (explicit === 'textbox' || explicit === 'combobox') return 'input';
    if (explicit === 'button' || explicit === 'tab' || explicit === 'menuitem'
        || explicit === 'switch') return 'button';
    return 'other';
  };

  const isEnabled = (el) => {
    if (el.disabled) return false;
    if (el.getAttribute('aria-disabled') === 'true') return false;
    return true;
  };

  const seen = new Set();
  const results = [];

  for (const el of document.querySelectorAll(INTERACTIVE_SELECTOR)) {
    if (seen.has(el)) continue;
    seen.add(el);

    // --- visibility gates (browser layout engine is ground truth) --------
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    if (parseFloat(style.opacity || '1') < 0.05) continue;
    if (el.getAttribute('aria-hidden') === 'true') continue;

    const rect = el.getBoundingClientRect();
    if (rect.width < 4 || rect.height < 4) continue;

    // Intersect with the viewport; skip elements fully scrolled out of view.
    const ix = Math.max(rect.left, 0);
    const iy = Math.max(rect.top, 0);
    const ir = Math.min(rect.right, vw);
    const ib = Math.min(rect.bottom, vh);
    if (ir - ix < 2 || ib - iy < 2) continue;

    // Occlusion check: is the element's own center actually the top-most node?
    const cx = Math.min(Math.max((rect.left + rect.right) / 2, 1), vw - 1);
    const cy = Math.min(Math.max((rect.top + rect.bottom) / 2, 1), vh - 1);
    const topEl = document.elementFromPoint(cx, cy);
    if (topEl && topEl !== el && !el.contains(topEl) && !topEl.contains(el)) {
      continue;
    }

    results.push({
      tag: el.tagName.toLowerCase(),
      role: normalizeRole(el),
      name: truncate(accessibleName(el)),
      value: truncate(el.value || ''),
      enabled: isEnabled(el),
      // Report the *visible* (viewport-clipped) box so marks never land off-screen.
      x: ix,
      y: iy,
      width: ir - ix,
      height: ib - iy,
    });
  }

  return results;
}
"""
