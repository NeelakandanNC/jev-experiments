"""Random UI pages for detector training. Every interactive element carries data-ui="<class>", so
label.js labels them exactly.

Pages mix desktop web layouts (navbar, sidebar, forms, tables, cards, dialogs, scrolling lists),
phone app layouts (app bar with back/search/menu, list rows, bottom tabs, FAB) and desktop-app
windows (title bar, menu bar, toolbars), styled with Bootstrap, Bulma, Pico or random custom CSS,
in light and dark, with icons from Bootstrap Icons, Lucide and Material Design Icons.
"""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "data" / "assets"

WORDS = """account activity address admin advanced album alert analytics archive article audio author
backup balance banner battery billing block blog bookmark booking budget build calendar camera campaign
cart catalog category channel chart chat checkout city client cloud code collection color comment
community company config contact content contract country coupon course credit customer dashboard data
date deal delivery design detail device discount document domain download draft edit editor email
event expense export favorite feature feed file filter finance folder follow form forum gallery game
general goal group guide health help history home hotel image import inbox income invoice issue item
job journal language launch layout library license link list location login market media meeting
member menu message metric mobile model money month movie music network news note notification offer
office order owner page panel partner password payment people phone photo plan player playlist policy
portfolio post preview price privacy product profile project promotion property purchase quote rating
recipe record region report request resource result review reward room sale schedule school score
search security server service session settings share shipping shop size skill source space sport
staff status storage store story subscription summary supplier support survey system table tag task
team template theme ticket time title topic total tour track trade training transfer travel trend
trip update upload usage user value vendor version video view visit wallet weather website week widget
work workspace year""".split()
NAMES = "Alice Bruno Chen Dana Elif Farah Gabe Hana Ivan Jade Kofi Lena Mateo Nina Omar Priya Quinn Rosa Sami Tara".split()
BUTTONS = ["Submit", "Cancel", "OK", "Save", "Delete", "Sign in", "Log in", "Sign up", "Log out", "Next",
           "Continue", "Back to list", "Add to cart", "Buy now", "Learn more", "Apply", "Reset", "Confirm",
           "Send", "Upload", "Download", "Edit", "Share", "Follow", "Subscribe", "Create", "Update", "Done",
           "Retry", "Accept", "Decline", "Search", "Open", "Close window", "Install", "Get started", "Yes", "No"]
LINKS = ["Home", "About", "Contact", "Pricing", "Docs", "Blog", "Careers", "Help", "Privacy", "Terms",
         "Forgot password?", "View all", "Read more", "See details", "Support", "Status", "Settings"]

SEMANTIC = {
    "back": {"bi": ["arrow-left", "chevron-left", "arrow-left-short", "arrow-left-circle"],
             "lucide": ["arrow-left", "chevron-left", "move-left", "circle-arrow-left"],
             "mdi": ["arrow-left", "chevron-left", "keyboard-backspace", "arrow-left-circle"]},
    "close": {"bi": ["x", "x-lg", "x-circle", "x-square"], "lucide": ["x", "circle-x", "square-x"],
              "mdi": ["close", "close-circle", "close-thick", "window-close"]},
    "menu": {"bi": ["list", "three-dots", "three-dots-vertical", "grid-3x3-gap"],
             "lucide": ["menu", "ellipsis", "ellipsis-vertical", "align-justify"],
             "mdi": ["menu", "dots-vertical", "dots-horizontal", "hamburger"]},
    "search": {"bi": ["search"], "lucide": ["search"], "mdi": ["magnify", "search-web"]},
}
_LOOKALIKE = re.compile(r"arrow|chevron|caret|x-|^x$|close|menu|search|magnif|list|dots|ellipsis|hamburger|grid-3x3")
SETS = {"bi": ("bootstrap-icons/icons", "{}.svg"), "lucide": ("lucide/icons", "{}.svg"), "mdi": ("mdi/svg", "{}.svg")}


@lru_cache(maxsize=None)
def _generic_names(s: str) -> list[str]:
    d = ASSETS / SETS[s][0]
    return sorted(p.stem for p in d.glob("*.svg") if not _LOOKALIKE.search(p.stem))


@lru_cache(maxsize=4096)
def _svg_src(s: str, name: str) -> str:
    p = ASSETS / SETS[s][0] / SETS[s][1].format(name)
    src = re.sub(r"<!--.*?-->", "", p.read_text(), flags=re.S).strip()
    src = re.sub(r"\s+", " ", src)
    src = re.sub(r'\s(width|height|class|id)="[^"]*"', "", src)
    if s == "mdi":
        src = src.replace("<svg ", '<svg fill="currentColor" ', 1)
    return src


class R:
    """All randomness for one page, so a seed reproduces it."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.iconset = self.r.choice(list(SETS))

    def p(self, prob: float) -> bool:
        return self.r.random() < prob

    def words(self, a: int, b: int, cap: bool = True) -> str:
        w = " ".join(self.r.choice(WORDS) for _ in range(self.r.randint(a, b)))
        return w[:1].upper() + w[1:] if cap else w

    def sentence(self) -> str:
        return self.words(6, 16) + "."

    def icon(self, kind: str | None = None, size: int | None = None) -> str:
        s = self.iconset if self.p(0.8) else self.r.choice(list(SETS))
        if kind:
            names = [n for n in SEMANTIC[kind][s] if (ASSETS / SETS[s][0] / SETS[s][1].format(n)).exists()]
            name = self.r.choice(names)
        else:
            name = self.r.choice(_generic_names(s))
        size = size or self.r.choice([16, 18, 20, 22, 24])
        return _svg_src(s, name).replace("<svg ", f'<svg width="{size}" height="{size}" ', 1)


# --- themes -------------------------------------------------------------------------------

@dataclass
class Theme:
    name: str
    head: str
    btn: list[str]
    input: str = ""
    select: str = ""
    check: str = ""
    dark: bool = False


PALETTE = ["#2563eb", "#7c3aed", "#db2777", "#dc2626", "#ea580c", "#16a34a", "#0d9488", "#0891b2", "#4f46e5",
           "#334155", "#9333ea", "#c2410c", "#15803d", "#1d4ed8", "#be123c"]
FONTS = ["Inter", "DejaVu Sans", "Liberation Sans", "Liberation Serif", "FreeSans", "Carlito", "Caladea",
         "Roboto", "Open Sans", "Lato", "Noto Sans", "WenQuanYi Zen Hei"]


def _fontfaces() -> str:
    out = []
    for fam, d in (("Roboto", "font-roboto"), ("Open Sans", "font-open-sans"), ("Lato", "font-lato")):
        for w in (400, 700):
            f = ASSETS / d / "files" / f"{d.split('-', 1)[1]}-latin-{w}-normal.woff2"
            if f.exists():
                out.append(f"@font-face{{font-family:'{fam}';font-weight:{w};src:url('{f.as_uri()}')}}")
    return "\n".join(out)


def theme(r: R) -> Theme:
    dark = r.p(0.25)
    font = r.r.choice(FONTS)
    base = r.r.choice([12, 13, 14, 15, 16, 17, 18])
    fw = _fontfaces() + f"\nhtml{{font-size:{base}px}} body{{font-family:'{font}',sans-serif}}"
    kind = r.r.choices(["bootstrap", "bulma", "pico", "custom"], weights=[3, 2, 2, 4])[0]
    if kind == "bootstrap":
        css = (ASSETS / "bootstrap/dist/css/bootstrap.min.css").as_uri()
        head = f'<link rel="stylesheet" href="{css}"><style>{fw}</style>'
        return Theme("bootstrap", head, ["btn btn-primary", "btn btn-outline-secondary", "btn btn-success",
                                         "btn btn-light", "btn btn-danger", "btn btn-dark", "btn btn-outline-primary",
                                         "btn btn-sm btn-secondary", "btn btn-lg btn-primary"],
                     "form-control", "form-select", "form-check-input", dark)
    if kind == "bulma":
        css = (ASSETS / "bulma/css/bulma.min.css").as_uri()
        head = f'<link rel="stylesheet" href="{css}"><style>{fw}</style>'
        return Theme("bulma", head, ["button is-primary", "button", "button is-link", "button is-danger is-outlined",
                                     "button is-small is-info", "button is-success is-light", "button is-dark"],
                     "input", "", "", dark)
    if kind == "pico":
        color = r.r.choice(["amber", "blue", "cyan", "green", "indigo", "pink", "purple", "red", "slate", "violet"])
        css = (ASSETS / f"pico/css/pico.{color}.min.css").as_uri()
        head = f'<link rel="stylesheet" href="{css}"><style>{fw} body{{padding:0}}</style>'
        return Theme("pico", head, ["", "secondary", "contrast", "outline", "outline secondary"], "", "", "", dark)
    c = r.r.choice(PALETTE)
    rad = r.r.choice([0, 2, 4, 6, 8, 12, 20])
    bg, fg, card, line = ("#121418", "#e5e7eb", "#1d2026", "#3a3f48") if dark else ("#ffffff", "#1f2937", "#f6f7f9", "#d1d5db")
    if not dark and r.p(0.3):
        bg = r.r.choice(["#fafafa", "#f3f4f6", "#fffdf7", "#f5f8ff"])
    pad = r.r.choice(["4px 10px", "6px 14px", "8px 18px", "10px 22px"])
    sb = ""
    if r.p(0.5):  # custom scrollbars of varying width; 0 = hidden
        w = r.r.choice([0, 6, 8, 10, 12, 14])
        sb = (f"::-webkit-scrollbar{{width:{w}px;height:{w}px}} ::-webkit-scrollbar-thumb{{background:{r.r.choice(['#999', c, '#555', '#bbb'])};"
              f"border-radius:{r.r.choice([0, 4, 8])}px}} ::-webkit-scrollbar-track{{background:{r.r.choice(['transparent', line, card])}}}")
    bar = r.r.choice([f".bar{{background:{c};color:#fff}}", f".bar{{background:{card};color:{fg}}}", f".bar{{background:{bg};color:{fg}}}"])
    css = f"""{fw}
    body{{margin:0;background:{bg};color:{fg}}} a{{color:{c}}}
    .b{{border:1px solid transparent;border-radius:{rad}px;padding:{pad};font:inherit;cursor:pointer}}
    .b0{{background:{c};color:#fff}} .b1{{background:transparent;color:{c};border-color:{c}}}
    .b2{{background:{card};color:{fg};border-color:{line}}} .b3{{background:transparent;color:{c}}}
    .b4{{background:{fg};color:{bg}}}
    .in{{border:1px solid {line};border-radius:{rad}px;padding:6px 8px;font:inherit;background:{bg};color:{fg}}}
    .ul .in{{border:none;border-bottom:2px solid {line};border-radius:0}}
    .card{{background:{card};border-radius:{rad + 2}px;padding:14px;margin:10px 0}}
    .sw{{display:inline-block;width:38px;height:22px;border-radius:11px;background:{line};position:relative;vertical-align:middle}}
    .sw.on{{background:{c}}} .sw::after{{content:'';position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;background:#fff}}
    .sw.on::after{{left:19px}} .ib{{background:transparent;border:none;color:inherit;padding:6px;cursor:pointer;line-height:0}}
    .tab{{padding:8px 14px;border-bottom:2px solid transparent;cursor:pointer;color:inherit;text-decoration:none}}
    .tab.on{{border-color:{c};color:{c}}} {bar}
    input[type=checkbox],input[type=radio]{{accent-color:{c};width:{r.r.choice([13, 16, 18])}px;height:{r.r.choice([13, 16, 18])}px}}
    {sb}"""
    return Theme("custom", f"<style>{css}</style>", ["b b0", "b b1", "b b2", "b b3", "b b4"], "in", "in", "", dark)


# --- components ---------------------------------------------------------------------------

def esc(s: str) -> str:
    return html.escape(s, quote=True)


def button(r: R, t: Theme, text: str | None = None) -> str:
    text = text or r.r.choice(BUTTONS)
    icon = r.icon(size=16) + " " if r.p(0.15) else ""
    return f'<button data-ui="button" class="{r.r.choice(t.btn)}" style="margin:4px">{icon}{esc(text)}</button>'


def icon_button(r: R, t: Theme, kind: str | None = None) -> str:
    cls = {"bootstrap": "btn btn-link p-1", "bulma": "button is-white is-small", "pico": "outline"}.get(t.name, "ib")
    if t.name == "pico" and r.p(0.6):
        cls = "secondary outline"
    label = f' aria-label="{kind}"' if kind and r.p(0.3) else ""
    return (f'<button data-ui="{kind or "icon"}" class="{cls}"{label} '
            f'style="margin:2px;line-height:0;padding:6px">{r.icon(kind)}</button>')


def link(r: R, text: str | None = None) -> str:
    return f'<a data-ui="link" href="#">{esc(text or r.r.choice(LINKS + [r.words(1, 3)]))}</a>'


def text_input(r: R, t: Theme, placeholder: bool = True) -> str:
    typ = r.r.choice(["text", "text", "email", "password", "search", "number", "tel", "date"])
    ph = f' placeholder="{esc(r.words(1, 3))}"' if placeholder and r.p(0.6) else ""
    val = f' value="{esc(r.r.choice([r.words(1, 2), r.r.choice(NAMES), str(r.r.randint(1, 9999))]))}"' if r.p(0.3) and typ != "date" else ""
    w = r.r.choice(["", "", "width:180px", "width:260px", "width:100%", "width:120px"])
    if r.p(0.1):
        return f'<textarea data-ui="text_input" class="{t.input}" rows="{r.r.randint(2, 4)}" style="{w}"{ph}></textarea>'
    return f'<input data-ui="text_input" type="{typ}" class="{t.input}" style="{w}"{ph}{val}>'


def dropdown(r: R, t: Theme) -> str:
    opts = "".join(f"<option>{esc(r.words(1, 2))}</option>" for _ in range(r.r.randint(2, 6)))
    sel = f'<select data-ui="dropdown" class="{t.select}" style="max-width:260px">{opts}</select>'
    return f'<div class="select" style="display:inline-block">{sel}</div>' if t.name == "bulma" else sel


def check(r: R, t: Theme, kind: str = "checkbox") -> str:
    name = f"g{r.r.randint(0, 3)}"
    on = " checked" if r.p(0.35) else ""
    cls = f' class="{t.check}"' if t.check else ""
    return (f'<label style="display:inline-flex;align-items:center;gap:6px;margin:4px 12px 4px 0">'
            f'<input data-ui="{kind}" type="{kind}" name="{name}"{cls}{on}> {esc(r.words(1, 3))}</label>')


def toggle(r: R, t: Theme) -> str:
    text = esc(r.words(1, 3))
    on = r.p(0.5)
    if t.name == "bootstrap":
        return (f'<div class="form-check form-switch"><input data-ui="toggle" class="form-check-input" type="checkbox" '
                f'role="switch"{" checked" if on else ""}> <label class="form-check-label">{text}</label></div>')
    if t.name == "pico":
        return f'<label><input data-ui="toggle" type="checkbox" role="switch"{" checked" if on else ""}> {text}</label>'
    return f'<div style="margin:6px 0"><span data-ui="toggle" class="sw{" on" if on else ""}"></span> {text}</div>'


def slider(r: R, t: Theme) -> str:
    cls = ' class="form-range"' if t.name == "bootstrap" else ""
    return (f'<label style="display:block;margin:6px 0">{esc(r.words(1, 2))}<br><input data-ui="slider" type="range"{cls} '
            f'value="{r.r.randint(0, 100)}" style="width:{r.r.choice([140, 200, 260])}px"></label>')


def tabs(r: R, t: Theme) -> str:
    n = r.r.randint(2, 6)
    on = r.r.randrange(n)
    items = [r.words(1, 2) for _ in range(n)]
    if t.name == "bootstrap":
        lis = "".join(f'<li class="nav-item"><a data-ui="tab" class="nav-link{" active" if i == on else ""}" href="#">{esc(x)}</a></li>'
                      for i, x in enumerate(items))
        return f'<ul class="nav {r.r.choice(["nav-tabs", "nav-pills", "nav-underline"])}" style="margin:8px 0">{lis}</ul>'
    if t.name == "bulma":
        lis = "".join(f'<li class="{"is-active" if i == on else ""}"><a data-ui="tab">{esc(x)}</a></li>' for i, x in enumerate(items))
        return f'<div class="tabs {r.r.choice(["", "is-boxed", "is-toggle"])}"><ul>{lis}</ul></div>'
    a = "".join(f'<a data-ui="tab" role="tab" class="tab{" on" if i == on else ""}" href="#">{esc(x)}</a>'
                for i, x in enumerate(items))
    return f'<nav style="display:flex;gap:2px;margin:8px 0;border-bottom:1px solid #8884">{a}</nav>'


def field(r: R, t: Theme) -> str:
    lab = esc(r.r.choice(["Name", "Email", "Password", "Username", "Phone", "City", "Company", "Title", "Amount",
                          "Date", "Address", "Zip code", "Comment"] + [r.words(1, 2)]))
    ctrl = r.r.choices([lambda: text_input(r, t), lambda: dropdown(r, t)], weights=[4, 1])[0]()
    if r.p(0.5):
        return f'<div style="margin:8px 0"><label style="display:block">{lab}</label>{ctrl}</div>'
    return f'<div style="margin:8px 0"><label style="margin-right:8px">{lab}:</label>{ctrl}</div>'


def form(r: R, t: Theme) -> str:
    parts = [f"<h{r.r.choice([3, 4, 5])}>{esc(r.words(1, 3))}</h3>"]
    for _ in range(r.r.randint(1, 5)):
        parts.append(r.r.choices([lambda: field(r, t), lambda: "".join(check(r, t) for _ in range(r.r.randint(1, 4))),
                                  lambda: "".join(check(r, t, "radio") for _ in range(r.r.randint(2, 4))),
                                  lambda: toggle(r, t), lambda: slider(r, t)], weights=[6, 2, 2, 2, 1])[0]())
    parts.append("<div>" + "".join(button(r, t) for _ in range(r.r.randint(1, 3))) + "</div>")
    if r.p(0.4):
        parts.append(f"<p>{link(r)}</p>")
    cls = {"bootstrap": "card card-body", "bulma": "box", "pico": ""}.get(t.name, "card")
    tag = "article" if t.name == "pico" else "div"
    return f'<{tag} class="{cls}" style="max-width:{r.r.choice([320, 420, 560, 720])}px">{"".join(parts)}</{tag}>'


def paragraph(r: R) -> str:
    out = []
    for _ in range(r.r.randint(1, 4)):
        out.append(link(r, r.words(1, 3, cap=False)) if r.p(0.15) else esc(r.sentence()))
    return f'<p style="max-width:{r.r.choice([400, 600, 800])}px">{" ".join(out)}</p>'


def card_grid(r: R, t: Theme) -> str:
    cards = []
    for _ in range(r.r.randint(2, 6)):
        img = f'<div style="height:{r.r.randint(50, 120)}px;background:{r.r.choice(PALETTE)}66;border-radius:6px"></div>' if r.p(0.6) else ""
        foot = button(r, t) if r.p(0.6) else link(r, r.r.choice(["Read more", "View", "Details", "Open"]))
        extra = icon_button(r, t, r.r.choice([None, None, "menu", "close"])) if r.p(0.3) else ""
        cards.append(f'<div class="card" style="width:{r.r.choice([160, 200, 240])}px;padding:10px;margin:6px;'
                     f'border:1px solid #8884;border-radius:8px">{img}<b>{esc(r.words(1, 3))}</b>{extra}'
                     f'<p style="font-size:.85em">{esc(r.sentence())}</p>{foot}</div>')
    return f'<div style="display:flex;flex-wrap:wrap">{"".join(cards)}</div>'


def table(r: R, t: Theme) -> str:
    cols = r.r.randint(2, 5)
    head = "".join(f"<th>{esc(r.words(1, 1))}</th>" for _ in range(cols))
    rows = []
    for _ in range(r.r.randint(2, 8)):
        cells = []
        for c in range(cols):
            u = r.r.random()
            cells.append(check(r, t) if c == 0 and u < 0.3 else
                         link(r, r.words(1, 2)) if u < 0.4 else
                         icon_button(r, t, r.r.choice([None, "close", "menu"])) if u < 0.5 and c == cols - 1 else
                         esc(r.r.choice([r.words(1, 2), str(r.r.randint(1, 999)), r.r.choice(NAMES)])))
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    cls = {"bootstrap": "table table-sm", "bulma": "table is-narrow"}.get(t.name, "")
    return f'<table class="{cls}" style="border-collapse:collapse;margin:8px 0"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def scroll_list(r: R, t: Theme) -> str:
    items = []
    for _ in range(r.r.randint(8, 25)):
        u = r.r.random()
        body = (link(r, r.words(1, 3)) if u < 0.3 else check(r, t) if u < 0.4 else esc(r.words(2, 5)))
        tail = icon_button(r, t, r.r.choice([None, "close", "menu"])) if r.p(0.2) else ""
        items.append(f'<div style="padding:6px 8px;border-bottom:1px solid #8883;display:flex;justify-content:space-between;align-items:center">{body}{tail}</div>')
    return (f'<div style="height:{r.r.choice([120, 160, 200, 260])}px;width:{r.r.choice([200, 260, 320, 420])}px;'
            f'overflow-y:{r.r.choice(["auto", "scroll"])};border:1px solid #8886;margin:8px 0">{"".join(items)}</div>')


def toolbar(r: R, t: Theme) -> str:
    n = r.r.randint(3, 9)
    kinds = [r.r.choice([None, None, None, None, "search", "menu", "close"]) for _ in range(n)]
    return f'<div style="display:flex;gap:2px;align-items:center;margin:6px 0">{"".join(icon_button(r, t, k) for k in kinds)}</div>'


def navbar(r: R, t: Theme) -> str:
    left = (icon_button(r, t, "menu") if r.p(0.4) else "") + f'<b style="margin:0 12px">{esc(r.words(1, 2))}</b>'
    links = "".join(f'<span style="margin:0 8px">{link(r)}</span>' for _ in range(r.r.randint(0, 5)))
    right = ""
    if r.p(0.5):
        right += text_input(r, t) if r.p(0.5) else icon_button(r, t, "search")
    if r.p(0.5):
        right += button(r, t, r.r.choice(["Sign in", "Log in", "Sign up", "Get started", "Log out"]))
    if r.p(0.3):
        right += icon_button(r, t)
    cls = {"bootstrap": "navbar bg-body-tertiary px-2", "bulma": "navbar px-2"}.get(t.name, "bar")
    return f'<nav class="{cls}" style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;flex-wrap:wrap"><div style="display:flex;align-items:center">{left}{links}</div><div style="display:flex;align-items:center;gap:6px">{right}</div></nav>'


def sidebar(r: R, t: Theme) -> str:
    items = "".join(f'<div style="padding:5px 0">{link(r, r.words(1, 2))}</div>' for _ in range(r.r.randint(4, 14)))
    ov = f'overflow-y:auto;height:{r.r.choice([300, 400, 500])}px;' if r.p(0.4) else ""
    return f'<aside style="width:{r.r.choice([160, 200, 240])}px;padding:10px;{ov}border-right:1px solid #8884;flex:none">{items}</aside>'


def dialog(r: R, t: Theme) -> str:
    body = r.r.choice([lambda: esc(r.sentence()), lambda: field(r, t), lambda: check(r, t) + check(r, t)])()
    btns = "".join(button(r, t, b) for b in r.r.sample(["OK", "Cancel", "Yes", "No", "Delete", "Save", "Continue", "Close"], r.r.randint(1, 3)))
    bg = "#1d2026" if t.dark else "#fff"
    return (f'<div style="position:fixed;inset:0;background:#0007;display:flex;align-items:center;justify-content:center;z-index:50">'
            f'<div style="background:{bg};padding:16px;border-radius:10px;min-width:260px;max-width:420px;box-shadow:0 10px 30px #0006">'
            f'<div style="display:flex;justify-content:space-between;align-items:center"><b>{esc(r.words(1, 4))}</b>'
            f'{icon_button(r, t, "close") if r.p(0.8) else ""}</div><div style="margin:12px 0">{body}</div>'
            f'<div style="text-align:right">{btns}</div></div></div>')


def pagination(r: R, t: Theme) -> str:
    n = r.r.randint(3, 7)
    btns = [button(r, t, "Prev")] if r.p(0.5) else []
    btns += [button(r, t, str(i + 1)) for i in range(n)]
    btns += [button(r, t, "Next")] if r.p(0.5) else []
    return f'<div style="margin:8px 0">{"".join(btns)}</div>'


# --- page layouts -------------------------------------------------------------------------

def web_page(r: R, t: Theme) -> str:
    blocks = [navbar(r, t)] if r.p(0.8) else []
    main = []
    if r.p(0.3):
        main.append(f'<h{r.r.choice([1, 2])}>{esc(r.words(2, 5))}</h1>')
    for _ in range(r.r.randint(2, 6)):
        main.append(r.r.choices([lambda: form(r, t), lambda: paragraph(r), lambda: card_grid(r, t), lambda: table(r, t),
                                 lambda: scroll_list(r, t), lambda: tabs(r, t), lambda: toolbar(r, t),
                                 lambda: pagination(r, t), lambda: "<div>" + "".join(button(r, t) for _ in range(r.r.randint(1, 4))) + "</div>"],
                                weights=[5, 4, 2, 2, 2, 2, 1, 1, 2])[0]())
    content = f'<main style="padding:12px 18px;flex:1;min-width:0">{"".join(main)}</main>'
    blocks.append(f'<div style="display:flex">{sidebar(r, t) if r.p(0.35) else ""}{content}</div>')
    if r.p(0.3):
        blocks.append(f'<footer style="padding:16px;border-top:1px solid #8884">{" · ".join(link(r) for _ in range(r.r.randint(2, 6)))}</footer>')
    if r.p(0.15):
        blocks.append(dialog(r, t))
    wrap = '<main class="container">' if t.name == "pico" and r.p(0.5) else "<div>"
    return wrap + "".join(blocks) + ("</main>" if wrap.startswith("<main") else "</div>")


def app_page(r: R, t: Theme) -> str:
    """Phone app screen."""
    bar = ""
    if r.p(0.9):
        left = icon_button(r, t, r.r.choice(["back", "back", "menu", "close"]))
        right = "".join(icon_button(r, t, k) for k in r.r.sample(["search", "menu", None, None], r.r.randint(0, 3)))
        bar = (f'<header class="bar" style="display:flex;align-items:center;padding:6px 6px;gap:6px;position:sticky;top:0;z-index:5">'
               f'{left}<b style="flex:1;font-size:1.15em">{esc(r.words(1, 3))}</b>{right}</header>')
    body = []
    if r.p(0.3):
        body.append(f'<div style="padding:8px">{text_input(r, t)}</div>')
    if r.p(0.3):
        body.append(tabs(r, t))
    for _ in range(r.r.randint(4, 14)):
        u = r.r.random()
        lead = r.icon(size=24) if r.p(0.5) else ""
        tail = (toggle(r, t) if u < 0.15 else check(r, t).replace("</label>", "</label>") if u < 0.25 else
                icon_button(r, t, r.r.choice([None, "menu"])) if u < 0.45 else "")
        title = link(r, r.words(1, 3)) if r.p(0.15) else f"<div>{esc(r.words(1, 4))}</div>"
        sub = f'<div style="font-size:.8em;opacity:.7">{esc(r.words(3, 8))}</div>' if r.p(0.6) else ""
        body.append(f'<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;border-bottom:1px solid #8883">'
                    f'{lead}<div style="flex:1">{title}{sub}</div>{tail}</div>')
    if r.p(0.4):
        body.append(f'<div style="padding:10px">{form(r, t)}</div>')
    if r.p(0.4):
        body.append(f'<div style="padding:10px">{button(r, t)}</div>')
    nav = ""
    if r.p(0.5):
        n = r.r.randint(3, 5)
        items = "".join(f'<a data-ui="tab" href="#" style="flex:1;text-align:center;color:inherit;text-decoration:none;padding:6px 0">'
                        f'{r.icon(size=22)}<div style="font-size:.7em">{esc(r.words(1, 1))}</div></a>' for _ in range(n))
        nav = f'<nav class="bar" style="position:fixed;bottom:0;left:0;right:0;display:flex;border-top:1px solid #8884;z-index:5">{items}</nav>'
    fab = ""
    if r.p(0.25):
        fab = (f'<button data-ui="icon" style="position:fixed;right:18px;bottom:{80 if nav else 20}px;width:56px;height:56px;border-radius:50%;'
               f'border:none;background:{r.r.choice(PALETTE)};color:#fff;z-index:6;line-height:0">{r.icon(size=24)}</button>')
    dlg = dialog(r, t) if r.p(0.1) else ""
    return f'{bar}<div style="padding-bottom:70px">{"".join(body)}</div>{nav}{fab}{dlg}'


def window_page(r: R, t: Theme) -> str:
    """Desktop application window."""
    ctrl = "".join(f'<button data-ui="{k}" class="ib" style="padding:4px 8px;line-height:0">{r.icon(k if k != "icon" else None, 14)}</button>'
                   for k in ("icon", "icon", "close"))
    title = (f'<div class="bar" style="display:flex;align-items:center;padding:2px 6px;font-size:.85em">'
             f'<span style="flex:1">{esc(r.words(1, 3))} — {esc(r.r.choice(["Editor", "Explorer", "Settings", "Mail", "Viewer", "Terminal"]))}</span>{ctrl}</div>')
    menu = "".join(f'<span data-ui="button" role="menuitem" style="padding:3px 8px;cursor:default">{m}</span>'
                   for m in r.r.sample(["File", "Edit", "View", "Insert", "Format", "Tools", "Window", "Help", "Go", "Run"], r.r.randint(3, 7)))
    body = f'<div style="display:flex">{sidebar(r, t)}<div style="padding:10px;flex:1">{toolbar(r, t)}'
    for _ in range(r.r.randint(1, 3)):
        body += r.r.choice([lambda: form(r, t), lambda: table(r, t), lambda: scroll_list(r, t), lambda: paragraph(r), lambda: tabs(r, t)])()
    body += "</div></div>"
    status = f'<div style="position:fixed;bottom:0;left:0;right:0;font-size:.75em;padding:2px 8px;border-top:1px solid #8884">{esc(r.words(2, 5))}</div>'
    return f'{title}<div style="display:flex;font-size:.9em;border-bottom:1px solid #8884">{menu}</div>{body}{status}{dialog(r, t) if r.p(0.1) else ""}'


VIEWPORTS = {
    "desktop": [(1280, 800), (1366, 768), (1440, 900), (1024, 768), (1920, 1080), (1280, 720)],
    "tablet": [(768, 1024), (820, 1180), (1024, 1366)],
    "mobile": [(390, 844), (360, 780), (412, 915), (375, 667), (393, 873)],
}


def make_page(seed: int) -> tuple[str, dict]:
    """(html, render options: width, height, scale, mobile, scroll)."""
    r = R(seed)
    t = theme(r)
    kind = r.r.choices(["web", "app", "window"], weights=[5, 3, 2])[0]
    if kind == "app":
        w, h = r.r.choice(VIEWPORTS["mobile"])
        scale, mobile = r.r.choice([1, 1, 2]), True
        body = app_page(r, t)
    elif kind == "window":
        w, h = r.r.choice(VIEWPORTS["desktop"])
        scale, mobile = 1, False
        body = window_page(r, t)
    else:
        w, h = r.r.choice(VIEWPORTS["desktop"] + VIEWPORTS["tablet"])
        scale, mobile = r.r.choice([1, 1, 1, 2]) if w < 1400 else 1, False
        body = web_page(r, t)
    dark = ' data-theme="dark" data-bs-theme="dark" class="theme-dark"' if t.dark else ' data-theme="light"'
    page = (f'<!doctype html><html{dark}><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">{t.head}</head><body>{body}</body></html>')
    return page, {"width": w, "height": h, "scale": scale, "mobile": mobile, "kind": kind, "theme": t.name,
                  "scroll": r.r.choice([0, 0, 0, 200, 500]) if kind == "web" else 0, "seed": seed}
