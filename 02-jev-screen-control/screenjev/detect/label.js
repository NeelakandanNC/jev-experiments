// Returns the visible interactive elements of the page with their UI class, in CSS pixels.
// Used two ways: to auto-label screenshots for YOLO training, and as the "dom" oracle detector.
// Elements carrying data-ui="<class>" are labelled exactly (the synthetic generator sets it);
// everything else is classified from tag / role / type / aria-label heuristics.
() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const out = [];
  const taken = new Set();

  const ICON = [
    ["back", /(^|[\s_-])(back|arrow[-_ ]?left|chevron[-_ ]?left|caret[-_ ]?left|previous|prev|go back)([\s_-]|$)|^[←‹<«]$/i],
    ["close", /(^|[\s_-])(close|dismiss|x[-_ ]?lg|x[-_ ]?circle|cancel|times)([\s_-]|$)|^[×✕✖xX]$/],
    ["menu", /(^|[\s_-])(menu|hamburger|list|more|kebab|dots|ellipsis|options|three[-_ ]?dots)([\s_-]|$)|^[☰⋮⋯…]$/i],
    ["search", /(^|[\s_-])(search|magnif|magnify|find)([\s_-]|$)|^🔍$/i],
  ];
  const textOf = (el) => {
    let t = "";
    if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") t = el.value || el.placeholder || "";
    else if (el.tagName === "SELECT") t = el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : "";
    else t = el.innerText || el.textContent || "";
    return t.replace(/\s+/g, " ").trim().slice(0, 200);
  };
  const iconKind = (el, text) => {
    const hint = [el.getAttribute("aria-label"), el.getAttribute("title"), el.getAttribute("data-icon"),
                  typeof el.className === "string" ? el.className : "", text].filter(Boolean).join(" ");
    for (const [k, re] of ICON) if (re.test(hint) || re.test(text)) return k;
    return "icon";
  };
  const visibleRect = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 3 || r.height < 3) return null;
    if (r.right <= 0 || r.bottom <= 0 || r.left >= vw || r.top >= vh) return null;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || parseFloat(cs.opacity) < 0.15) return null;
    // occlusion: at least one sample point must hit the element (or something inside it / its label)
    const pts = [[0.5, 0.5], [0.2, 0.5], [0.8, 0.5], [0.5, 0.2], [0.5, 0.8]];
    let ok = false;
    for (const [fx, fy] of pts) {
      const x = Math.min(vw - 1, Math.max(0, r.left + r.width * fx));
      const y = Math.min(vh - 1, Math.max(0, r.top + r.height * fy));
      const hit = document.elementFromPoint(x, y);
      if (!hit) continue;
      if (hit === el || el.contains(hit) || (el.labels && [...el.labels].some((l) => l.contains(hit)))) { ok = true; break; }
    }
    if (!ok) return null;
    return [Math.max(0, r.left), Math.max(0, r.top), Math.min(vw, r.right), Math.min(vh, r.bottom)];
  };
  const classify = (el) => {
    const ui = el.getAttribute("data-ui");
    if (ui) return ui === "none" ? null : ui;
    const tag = el.tagName, role = (el.getAttribute("role") || "").toLowerCase();
    const text = textOf(el);
    if (tag === "INPUT") {
      const t = (el.type || "text").toLowerCase();
      if (t === "hidden") return null;
      if (t === "checkbox") return el.getAttribute("role") === "switch" ? "toggle" : "checkbox";
      if (t === "radio") return "radio";
      if (t === "range") return "slider";
      if (["button", "submit", "reset", "image"].includes(t)) return el.value ? "button" : iconKind(el, "");
      if (t === "search") return "text_input";
      return "text_input";
    }
    if (tag === "TEXTAREA" || el.isContentEditable && el.getAttribute("contenteditable") !== null) return "text_input";
    if (tag === "SELECT" || role === "combobox" || role === "listbox") return "dropdown";
    if (role === "switch") return "toggle";
    if (role === "tab") return "tab";
    if (role === "checkbox") return "checkbox";
    if (role === "radio") return "radio";
    if (role === "slider" || el.classList.contains("ui-slider")) return "slider";
    if (tag === "BUTTON" || role === "button" || role === "menuitem") {
      if (text.length <= 1 || !/[A-Za-z0-9]{2}/.test(text)) return iconKind(el, text);
      return "button";
    }
    if (tag === "A" && (el.hasAttribute("href") || el.onclick) || role === "link") {
      if (!text) return iconKind(el, "");
      return "link";
    }
    return null;
  };

  const SEL = "[data-ui], a, button, input, textarea, select, [role], [contenteditable], .ui-slider";
  for (const el of document.querySelectorAll(SEL)) {
    // a widget inside an already-labelled widget (e.g. <span> in a <button>) is part of it,
    // unless explicitly labelled itself
    let p = el.parentElement, inside = false;
    while (p) { if (taken.has(p)) { inside = true; break; } p = p.parentElement; }
    if (inside && !el.hasAttribute("data-ui")) continue;
    const cls = classify(el);
    if (!cls) continue;
    const box = visibleRect(el);
    if (!box) continue;
    taken.add(el);
    out.push({cls, box, text: cls === "scrollbar" ? "" : textOf(el), tag: el.tagName.toLowerCase()});
  }

  // Non-synthetic pages: clickable things that aren't semantic widgets (<span class="alink">, <div onclick>):
  // the outermost element with cursor:pointer, as a link if it's inline text or underlined, else a button.
  if (!document.querySelector("[data-ui]")) {
    for (const el of document.body.querySelectorAll("*")) {
      if (taken.has(el) || ["LABEL", "OPTION", "HTML", "BODY"].includes(el.tagName)) continue;
      const cs = getComputedStyle(el);
      if (cs.cursor !== "pointer") continue;
      const pcs = el.parentElement ? getComputedStyle(el.parentElement) : null;
      if (pcs && pcs.cursor === "pointer") continue;  // part of a bigger clickable
      let p = el.parentElement, inside = false;
      while (p) { if (taken.has(p)) { inside = true; break; } p = p.parentElement; }
      if (inside || el.querySelector("input, select, textarea, button, a[href]")) continue;
      const text = textOf(el);
      if (!text && !el.querySelector("svg, img, i")) continue;
      const box = visibleRect(el);
      if (!box) continue;
      const inline = cs.display.startsWith("inline") && cs.display !== "inline-block";
      const cls = !text ? iconKind(el, "") : (inline || /underline/.test(cs.textDecorationLine)) ? "link" : "button";
      taken.add(el);
      out.push({cls, box, text, tag: el.tagName.toLowerCase()});
    }
  }

  // scrollbars: classic (non-overlay) bars of scrollable boxes and of the page itself
  const bar = (x1, y1, x2, y2) => {
    x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(vw, x2); y2 = Math.min(vh, y2);
    if (x2 - x1 >= 3 && y2 - y1 >= 12 || x2 - x1 >= 12 && y2 - y1 >= 3) out.push({cls: "scrollbar", box: [x1, y1, x2, y2], text: "", tag: "scrollbar"});
  };
  const de = document.scrollingElement || document.documentElement;
  const pageBar = vw - document.documentElement.clientWidth;
  if (pageBar > 0 && de.scrollHeight > de.clientHeight + 1) bar(vw - pageBar, 0, vw, document.documentElement.clientHeight);
  for (const el of document.body.querySelectorAll("*")) {
    const cs = getComputedStyle(el);
    if (!/(auto|scroll)/.test(cs.overflowY + cs.overflowX)) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 20 || r.height < 20 || r.bottom <= 0 || r.top >= vh || r.right <= 0 || r.left >= vw) continue;
    if (cs.visibility === "hidden" || cs.display === "none") continue;
    const bl = parseFloat(cs.borderLeftWidth) || 0, br = parseFloat(cs.borderRightWidth) || 0;
    const bt = parseFloat(cs.borderTopWidth) || 0, bb = parseFloat(cs.borderBottomWidth) || 0;
    const vbar = el.offsetWidth - el.clientWidth - bl - br;
    const hbar = el.offsetHeight - el.clientHeight - bt - bb;
    if (vbar > 0 && el.scrollHeight > el.clientHeight + 1 && /(auto|scroll)/.test(cs.overflowY))
      bar(r.right - br - vbar, r.top + bt, r.right - br, r.top + bt + el.clientHeight);
    if (hbar > 0 && el.scrollWidth > el.clientWidth + 1 && /(auto|scroll)/.test(cs.overflowX))
      bar(r.left + bl, r.bottom - bb - hbar, r.left + bl + el.clientWidth, r.bottom - bb);
  }
  return out;
}
