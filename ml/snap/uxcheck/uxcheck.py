"""UX check suite for the REAL Calorie Lens site (built dist served by `vite preview`, headless Chromium, ONE browser).

    # serve the existing build yourself:
    #   cd web && MSYS_NO_PATHCONV=1 npx vite preview --base=/calorie-lens/ --port 4175 --strictPort
    python ml/snap/uxcheck/uxcheck.py --sample 15 --tag baseline        # quick run
    python ml/snap/uxcheck/uxcheck.py --full                              # every photo of every set
    python ml/snap/uxcheck/uxcheck.py --serve --sample 15                 # start + stop the preview server itself
    options: --url, --sets meals,multi,..., --sample N, --full, --tag NAME (default: timestamp), --out DIR

Output: SP/uxcheck/<tag|timestamp>/{report.json, report.html}  (SP = session scratchpad, see ml/snap/common.py).

Metrics (see metric dict keys in report.json):
 1 not_food_logged   2 drinks_logged   3 multi (recall of GT food boxes @IoU>=0.3, items/img vs GT)
 4 layout (pin/label overlap, outside/clipped, min font)   5 fits_on_screen (390x844, 375x667)
 6 cjk per screen   7 contrast (WCAG, light+dark)   8 tap targets <44px   9 timing   10 errors + behaviour checks
 11 names (top-1 vs Open Images class, lenient regex mapping, see NAME_MAP)
UI hooks: data-testid snap, choose, progress, result, result-photo, pin, result-item, meal-total, no-food, error, log, meal,
day-total, week-bars, settings-button, settings, undo-toast. Missing ones are reported as n/a (with legacy-UI fallbacks).
Notes: videos (.y4m) are not photo inputs and are skipped. Contrast skips text over images/gradients (unknowable from the DOM).
"""
import argparse, base64, io, json, random, re, statistics as st, subprocess, sys, time, datetime, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import SP, ROOT  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UX = SP / "uxset"
TESTIDS = ["snap", "choose", "progress", "result", "result-photo", "pin", "result-item", "meal-total", "no-food", "error", "log",
           "meal", "day-total", "week-bars", "settings-button", "settings", "undo-toast"]
FILE_SEL = 'input[type=file][accept="image/*"]:not([capture])'
CJK_RE = r"[　-鿿＀-￯]"
VIEWPORTS = [(390, 844), (375, 667)]
MIN_FONT_OK = 11  # px; labels smaller than this are flagged

# ---- metric 11: Open Images class -> lenient regex on the English item name (top-1) (or any candidate name for "in candidates").
NAME_MAP = {
    "Apple": r"apple", "Artichoke": r"artichoke", "Bagel": r"bagel", "Banana": r"banana", "Bell pepper": r"pepper|capsicum",
    "Bread": r"bread|toast|bun|roll|baguette|loaf|bagel|croissant", "Broccoli": r"broccoli", "Burrito": r"burrito|wrap|taco",
    "Cabbage": r"cabbage|coleslaw|kale|lettuce", "Cake": r"cake|cupcake|tiramisu|torte|cheesecake|brownie",
    "Candy": r"candy|chocolate|sweet|gummy|lollipop|toffee|caramel|marshmallow", "Cantaloupe": r"melon|cantaloupe",
    "Carrot": r"carrot", "Cheese": r"cheese", "Coconut": r"coconut", "Common fig": r"\bfig", "Cookie": r"cookie|biscuit|cracker",
    "Crab": r"crab", "Cream": r"cream|custard|pudding|mousse|yogurt|whipped", "Croissant": r"croissant|pastry|danish",
    "Cucumber": r"cucumber|pickle", "Doughnut": r"doughnut|donut", "Egg": r"\begg|omelet|omelette",
    "French fries": r"fries|french fr|chips|potato", "Garden Asparagus": r"asparagus", "Grape": r"grape|raisin",
    "Grapefruit": r"grapefruit|pomelo|orange|citrus", "Hamburger": r"burger|hamburger|sandwich", "Honeycomb": r"honey",
    "Hot dog": r"hot dog|sausage|frankfurter", "Ice cream": r"ice cream|gelato|sundae|frozen|sorbet",
    "Lemon (plant)": r"lemon|lime|citrus", "Lobster": r"lobster|crab|shrimp|crawfish|seafood", "Mango": r"mango",
    "Muffin": r"muffin|cupcake|cake", "Mushroom": r"mushroom", "Orange (fruit)": r"orange|tangerine|mandarin|citrus",
    "Oyster": r"oyster|clam|mussel|seafood", "Pancake": r"pancake|crepe|waffle|hotcake", "Pasta": r"pasta|spaghetti|noodle|macaroni|lasagna|ravioli|penne",
    "Pastry": r"pastry|croissant|danish|pie|tart|cake|donut|doughnut|bun|bread", "Peach": r"peach|nectarine|apricot|plum",
    "Pear": r"\bpear", "Pineapple": r"pineapple", "Pizza": r"pizza", "Pomegranate": r"pomegranate", "Popcorn": r"popcorn",
    "Potato": r"potato|fries|hash", "Pretzel": r"pretzel", "Pumpkin": r"pumpkin|squash", "Radish": r"radish|turnip|beet",
    "Salad": r"salad|lettuce|greens|slaw", "Sandwich": r"sandwich|burger|sub\b|panini|wrap|toast",
    "Seafood": r"fish|shrimp|crab|lobster|squid|oyster|clam|mussel|salmon|tuna|seafood|prawn|scallop|octopus",
    "Shrimp": r"shrimp|prawn", "Squash (Plant)": r"squash|pumpkin|zucchini", "Squid": r"squid|calamari|octopus|cuttlefish",
    "Strawberry": r"strawberr", "Submarine sandwich": r"sandwich|sub\b|hoagie|burger|wrap", "Sushi": r"sushi|sashimi|maki|nigiri|roll",
    "Taco": r"taco|burrito|wrap|quesadilla|tortilla", "Tart": r"tart|pie|pastry|cake", "Tomato": r"tomato",
    "Waffle": r"waffle|pancake", "Watermelon": r"watermelon|melon", "Zucchini": r"zucchini|squash|cucumber|courgette",
    # generic OI classes: accept any food-ish word of the family
    "Baked goods": r"bread|cake|cookie|pastry|muffin|pie|bun|roll|biscuit|bagel|croissant|tart|donut|doughnut|brownie|scone|pretzel",
    "Dairy Product": r"cheese|milk|yogurt|cream|butter|pudding|custard|ice cream|cake",
    "Dessert": r"cake|pie|ice cream|pudding|cookie|tart|pastry|custard|mousse|brownie|donut|doughnut|sweet|candy|tiramisu|cheesecake|waffle|sundae",
    "Fast food": r"burger|fries|pizza|nugget|hot dog|sandwich|taco|fried|chicken|wrap|burrito|sub\b",
    "Food": r".", "Fruit": r"apple|banana|orange|grape|berry|melon|peach|pear|mango|pineapple|cherry|plum|fig|kiwi|lemon|lime|fruit|citrus|papaya|guava|pomelo|persimmon|apricot",
    "Snack": r"chip|cracker|popcorn|pretzel|nut|bar|cookie|candy|fries|snack|biscuit",
    "Vegetable": r"vegetable|carrot|broccoli|cabbage|lettuce|spinach|greens|pepper|tomato|cucumber|onion|bean|pea|corn|squash|zucchini|eggplant|salad|kale|cauliflower|celery|radish|asparagus",
}
FOOD_BOX_CLASSES = set(NAME_MAP) | {"Pizza", "Salad", "Juice_excluded"}  # GT "food" boxes for multi recall (drinks/people/tableware excluded)


# =============================== page-side JS ===============================
JS_COMMON = r"""
const q=(s,r=document)=>r.querySelector(s), qa=(s,r=document)=>[...r.querySelectorAll(s)];
const R=e=>{const r=e.getBoundingClientRect();return {l:r.left,t:r.top,r:r.right,b:r.bottom,w:r.width,h:r.height}};
const shown=e=>{const s=getComputedStyle(e);const r=e.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0};
const sel=e=>e.tagName.toLowerCase()+(e.getAttribute('data-testid')?'[testid='+e.getAttribute('data-testid')+']':'')+(e.className&&typeof e.className==='string'?'.'+e.className.trim().split(/\s+/).join('.'):'');
"""

JS_MEASURE = JS_COMMON + r"""
(() => {
  const cjkRe=new RegExp(%CJK%,'g');
  const body=document.body.innerText||'';
  const out={cjk:(body.match(cjkRe)||[]).length, vw:innerWidth, vh:innerHeight, scrollY:scrollY, docH:document.documentElement.scrollHeight};
  out.testids={}; for (const id of %IDS%) out.testids[id]=qa('[data-testid="'+id+'"]').length;
  let card=q('[data-testid=result]'), legacy=false;
  if(!card){const ph=q('.photo'); if(ph){card=ph.closest('.card'); legacy=true}}
  out.card=!!card; out.legacy=legacy;
  if(!card) return out;
  const cr=R(card); out.cardRect=cr;
  const photoEl=q('[data-testid=result-photo]',card)||q('.photo',card)||q('img',card);
  const imgEl=photoEl&&(photoEl.tagName==='IMG'?photoEl:(q('img',photoEl)||photoEl));
  const bounds=imgEl?R(imgEl):null; out.photoRect=bounds;
  out.photoOk = !!(imgEl && (imgEl.tagName!=='IMG'||imgEl.complete));
  let pins=qa('[data-testid=pin]',card); if(!pins.length&&legacy&&photoEl) pins=qa('.box',photoEl);
  out.nPins=pins.length;
  out.nItems=qa('[data-testid=result-item]',card).length||(legacy?qa('.row',card).length:0);
  const labels=[];
  for(const p of pins){let l=q('[data-testid=pin-label]',p)||(legacy?q('span',p):null); if(l) labels.push(l);}
  const inter=(a,b)=>Math.max(0,Math.min(a.r,b.r)-Math.max(a.l,b.l))*Math.max(0,Math.min(a.b,b.b)-Math.max(a.t,b.t));
  const pairs=(els)=>{let n=0;const rs=els.filter(shown).map(R);for(let i=0;i<rs.length;i++)for(let j=i+1;j<rs.length;j++)if(inter(rs[i],rs[j])>1)n++;return n};
  out.labelOverlaps = labels.length?pairs(labels):null;
  out.pinOverlaps = (!legacy&&pins.length)?pairs(pins):null;   // legacy .box elements are whole-object boxes: overlap is natural
  let outside=0, clipped=0, minFont=null, small=0, trunc=0;
  const marks=[...(legacy?[]:pins),...labels];
  for(const m of marks){
    if(!shown(m)) continue; const r=R(m);
    if(bounds && (r.l<bounds.l-1.5||r.t<bounds.t-1.5||r.r>bounds.r+1.5||r.b>bounds.b+1.5)) outside++;
    if(r.l<-1||r.r>innerWidth+1) clipped++;
    if(m.scrollWidth>m.clientWidth+1 && getComputedStyle(m).overflow!=='visible') trunc++;
    if((m.innerText||'').trim()){const f=parseFloat(getComputedStyle(m).fontSize); if(minFont==null||f<minFont)minFont=f; if(f<%MINFONT%) small++;}
  }
  out.outside=outside; out.clipped=clipped; out.truncated=trunc; out.minFont=minFont; out.smallLabels=small;
  const tot=q('[data-testid=meal-total]',card)||q('.total',card);
  const btn=qa('button,[role=button],a[href]',card).find(b=>/done|dismiss|close|\bok\b|完成|關閉/i.test((b.innerText||'')+' '+(b.getAttribute('aria-label')||'')));
  out.totalBottom=tot?R(tot).b:null; out.doneBottom=btn?R(btn).b:null; out.doneTop=btn?R(btn).t:null;
  return out;
})()
"""

JS_AUDIT = JS_COMMON + r"""
(() => {
  const cv=document.createElement('canvas'); cv.width=cv.height=1; const cx=cv.getContext('2d',{willReadFrequently:true});
  const cache={};
  const rgba=c=>{ if(cache[c]) return cache[c]; cx.clearRect(0,0,1,1); cx.fillStyle='#000'; cx.fillStyle=c; cx.fillRect(0,0,1,1);
    const d=cx.getImageData(0,0,1,1).data; const a=d[3]/255; const v=a>0?[d[0]*255/d[3],d[1]*255/d[3],d[2]*255/d[3],a]:[0,0,0,0]; return cache[c]=v; };
  const over=(f,b)=>{const a=f[3];return [f[0]*a+b[0]*(1-a),f[1]*a+b[1]*(1-a),f[2]*a+b[2]*(1-a),1]};
  const lum=c=>{const f=v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)};return 0.2126*f(c[0])+0.7152*f(c[1])+0.0722*f(c[2])};
  const ratio=(a,b)=>{const x=lum(a),y=lum(b);return (Math.max(x,y)+0.05)/(Math.min(x,y)+0.05)};
  const dark=(getComputedStyle(document.documentElement).colorScheme||'').includes('dark');
  const canvasBg=dark?[0,0,0,1]:[255,255,255,1];
  const res={checked:0,skipped:0,fails:[]}; const seen=new Set();
  const tw=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  for(let n;n=tw.nextNode();){
    const t=n.textContent.trim(); if(!t) continue; const el=n.parentElement; if(!el) continue;
    if(/^(SCRIPT|STYLE|NOSCRIPT)$/.test(el.tagName)||el.closest(':disabled')) continue;
    const r=R(el); const s=getComputedStyle(el);
    if(s.display==='none'||s.visibility==='hidden'||r.w<3||r.h<3) continue;
    let op=1, bad=false; const layers=[]; let unknown=null, overlay=false;
    for(let a=el;a;a=a.parentElement){
      const cs=getComputedStyle(a); op*=parseFloat(cs.opacity);
      if(cs.display==='none'){bad=true;break}
      if(cs.position==='absolute'||cs.position==='fixed') overlay=true;
      if(cs.backgroundImage!=='none'){unknown='bg-image';break}
      const bc=rgba(cs.backgroundColor);
      if(bc[3]>0) layers.push(bc);
      if(bc[3]>=0.999) break;
      if(overlay && q(':scope > img, :scope > video, :scope > canvas',a)){unknown='over-image';break}
    }
    if(bad||op<0.05) continue;
    if(unknown){res.skipped++;continue}
    let bg=canvasBg; for(let i=layers.length-1;i>=0;i--) bg=over(layers[i],bg);
    const fg0=rgba(s.color); const fg=over([fg0[0],fg0[1],fg0[2],fg0[3]*op],bg);
    const size=parseFloat(s.fontSize), bold=parseInt(s.fontWeight)>=700||s.fontWeight==='bold';
    const large=size>=24||(size>=18.66&&bold); const need=large?3:4.5; const cr=ratio(fg,bg);
    res.checked++;
    if(cr<need){const key=sel(el)+'|'+t.slice(0,30); if(seen.has(key)) continue; seen.add(key);
      res.fails.push({sel:sel(el),text:t.slice(0,40),ratio:+cr.toFixed(2),need,size,fg:fg.slice(0,3).map(Math.round).join(','),bg:bg.slice(0,3).map(Math.round).join(',')});}
  }
  return res;
})()
"""

JS_TAP = JS_COMMON + r"""
(() => {
  const res={checked:0,small:[]};
  for(const e of qa('button,a[href],input,select,textarea,[role=button],[role=link],summary')){
    if(e.tagName==='INPUT'&&e.type==='hidden') continue;
    if(e.disabled) continue;
    const r=R(e); const s=getComputedStyle(e);
    if(s.display==='none'||s.visibility==='hidden'||r.w<=0||r.h<=0) continue;
    if(r.r<0||r.b<0&&false) continue;
    if(e.tagName==='INPUT'&&e.type==='file'&&(r.l<-1000||r.w<=2||s.opacity==='0')) continue;
    if(r.l<-1000||r.t<-1000) continue;   // off-screen "sr" inputs
    res.checked++;
    if(r.w<43.5||r.h<43.5) res.small.push({sel:sel(e),text:((e.innerText||e.getAttribute('aria-label')||e.value||'')+'').trim().slice(0,30),w:+r.w.toFixed(1),h:+r.h.toFixed(1)});
  }
  return res;
})()
"""

JS_MSG = JS_COMMON + r"""
(() => {
  const el=q('[data-testid=error]')||q('[data-testid=no-food]')||q('.card .err')||q('.card .warn')||q('#status .card');
  const t=el?(el.closest('.card')||el).innerText.trim():'';
  return {testid:!!(q('[data-testid=error]')||q('[data-testid=no-food]')), text:t.slice(0,300), hasResult:!!(q('[data-testid=result]')||q('.photo'))};
})()
"""


# =============================== helpers ===============================
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def thumb(path, gt=None, logged=None, w=190):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((w, w * 2))
        d = ImageDraw.Draw(im); W, H = im.size
        for b, c in ((gt, (40, 200, 60)), (logged, (230, 40, 40))):
            for bb in b or []:
                d.rectangle([bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H], outline=c, width=2)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def pct(a, b):
    return None if not b else round(100 * a / b, 1)


def quant(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


# =============================== suite ===============================
class Suite:
    def __init__(self, a):
        self.a = a
        self.vocab = {v["id"]: v for v in json.load(open(ROOT / "web/public/models/vocab.json", encoding="utf-8"))}
        self.errors = []      # (job, kind, text)
        self.audits = {}      # screen -> {light:{..}, dark:{..}}
        self.testid_max = {i: 0 for i in TESTIDS}
        self.job = "-"

    # ---- page plumbing
    def settle(self, page, items):
        try:
            if items:
                page.wait_for_function("document.querySelector('[data-testid=result]')||document.querySelector('.photo img')", timeout=5000)
                page.wait_for_function("(()=>{const i=document.querySelector('[data-testid=result-photo] img')||document.querySelector('.photo img');return !i||i.complete})()", timeout=5000)
            else:
                page.wait_for_function("document.querySelector('[data-testid=no-food],[data-testid=error]')||document.querySelector('.card .warn,.card .err')", timeout=3000)
        except Exception:
            pass
        page.wait_for_timeout(250)

    def upload(self, page, src, name=None, timeout=240_000):
        """src: path or dict(name,mimeType,buffer). Returns (wall_s, __clLast or None)."""
        n0 = page.evaluate("(window.__clLast&&window.__clLast.n)||0")
        page.evaluate("window.scrollTo(0,0)")
        t0 = time.time()
        page.locator(FILE_SEL).first.set_input_files(src)
        try:
            page.wait_for_function(f"window.__clLast && window.__clLast.n > {n0}", timeout=timeout)
        except Exception:
            return time.time() - t0, None
        return time.time() - t0, page.evaluate("window.__clLast")

    def measure(self, page):
        js = JS_MEASURE.replace("%CJK%", json.dumps(CJK_RE)).replace("%IDS%", json.dumps(TESTIDS)).replace("%MINFONT%", str(MIN_FONT_OK))
        m = page.evaluate(js)
        for k, v in m["testids"].items():
            self.testid_max[k] = max(self.testid_max[k], v)
        return m

    def audit(self, page, screen):
        r = {}
        for scheme in ("light", "dark"):
            page.emulate_media(color_scheme=scheme)
            page.wait_for_timeout(150)
            r[scheme] = page.evaluate(JS_AUDIT)
            if scheme == "light":
                r["tap"] = page.evaluate(JS_TAP)
                r["cjk"] = self.measure(page)["cjk"]
        page.emulate_media(color_scheme="light")
        self.audits[screen] = r
        return r

    # ---- one photo
    def run_photo(self, page, grp, item, path, first=False):
        self.job = f"{grp}/{Path(path).name}"
        e0 = len(self.errors)
        wall, last = self.upload(page, str(path))
        rec = {"group": grp, "id": item.get("id"), "path": str(path), "wall_s": round(wall, 2), "first_visit": first}
        if last is None:
            rec["error"] = "timeout (no __clLast)"; return rec
        if "error" in last:
            rec["error"] = last["error"]; self.settle(page, False); rec["ui"] = page.evaluate(JS_MSG); return rec
        res = last["result"]
        rec.update(ms=last["ms"], backend=last["backend"], total=res.get("total"), nBoxes=res.get("nBoxes"),
                   items=[{k: it.get(k) for k in ("name", "nameEn", "unsure", "box", "p50", "cands")} for it in res["items"]])
        self.settle(page, bool(res["items"]))
        m = self.measure(page)
        rec["layout"] = {k: m.get(k) for k in ("card", "legacy", "nPins", "nItems", "labelOverlaps", "pinOverlaps", "outside", "clipped",
                                               "truncated", "minFont", "smallLabels", "cjk", "photoOk")}
        fits = {}
        if res["items"] and m["card"]:
            fits["390x844"] = self.fits(m)
            for (w, h) in VIEWPORTS[1:]:
                page.set_viewport_size({"width": w, "height": h}); page.wait_for_timeout(250)
                fits[f"{w}x{h}"] = self.fits(self.measure(page))
            page.set_viewport_size({"width": VIEWPORTS[0][0], "height": VIEWPORTS[0][1]}); page.wait_for_timeout(150)
            if not self.audits.get("result") and grp in ("meals", "multi", "food101"):
                self.audit(page, "result")
        elif not res["items"] and not self.audits.get("no-food"):
            self.audit(page, "no-food")
        rec["fits"] = fits
        rec["page_errors"] = len(self.errors) - e0
        return rec

    @staticmethod
    def fits(m):
        vh = m["vh"]
        parts = {"card": m["cardRect"]["b"], "total": m["totalBottom"], "done": m["doneBottom"]}
        res = {"scrollY": m["scrollY"], "bottoms": {k: (round(v) if v is not None else None) for k, v in parts.items()}, "vh": vh}
        res["missing"] = [k for k, v in parts.items() if v is None and k != "card"]
        res["ok"] = all(v <= vh + 0.5 for v in parts.values() if v is not None) and m["cardRect"]["t"] >= -0.5
        return res

    # ---- behaviour checks
    def behaviours(self, page, tmp, any_meal):
        out = {}
        tmp.mkdir(parents=True, exist_ok=True)
        cases = {}
        cases["non_image_txt"] = {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"hello, this is not an image\n" * 20}
        cases["corrupt_jpg"] = {"name": "broken.jpg", "mimeType": "image/jpeg", "buffer": b"\xff\xd8\xff\xe0" + os.urandom(2000)}
        p100 = tmp / "tiny_100x100.jpg"
        Image.effect_noise((100, 100), 40).convert("RGB").save(p100, quality=85)
        cases["tiny_100x100"] = str(p100)
        if any_meal:
            p6k = tmp / "huge_6000x4000.jpg"
            Image.open(any_meal).convert("RGB").resize((6000, 4000), Image.BICUBIC).save(p6k, quality=80)
            cases["huge_6000x4000"] = str(p6k)
        for name, src in cases.items():
            self.job = "behaviour/" + name
            e0 = len(self.errors)
            wall, last = self.upload(page, src, timeout=120_000)
            r = {"wall_s": round(wall, 2)}
            if last is None:
                r.update(verdict="FAIL: no response (hang)")
            else:
                self.settle(page, bool(last.get("result", {}).get("items")))
                ui = page.evaluate(JS_MSG); r["ui"] = ui
                r["items"] = len(last.get("result", {}).get("items", [])) if "result" in last else 0
                r["error"] = last.get("error")
                raw = bool(re.search(r"TypeError|ReferenceError|undefined|\[object|Error:|NaN|\bnull\b|Failed to|InvalidState|\bat \w+.*:\d+", (ui["text"] or "") + " " + (r["error"] or "")))
                shown = bool(ui["text"]) and not ui["hasResult"] or bool(ui["text"])
                if name == "huge_6000x4000":
                    r["verdict"] = "ok" if (r["error"] is None and r["items"] > 0) else ("FAIL: error " + str(r["error"])[:80] if r["error"] else "no items logged")
                else:
                    r["verdict"] = "ok" if (r["items"] == 0 and ui["text"] and not raw) else (
                        "FAIL: items logged" if r["items"] else "FAIL: message not friendly/absent (raw: %s) %r" % (raw, ui["text"][:80]))
                if name in ("non_image_txt", "corrupt_jpg") and not self.audits.get("error") and ui["text"]:
                    self.audit(page, "error")
            r["page_errors"] = len(self.errors) - e0
            out[name] = r
        return out

    def settings_screen(self, page):
        try:
            btn = page.locator('[data-testid=settings-button]')
            if not btn.count():
                btn = page.locator('[aria-label*="Settings"], [aria-label*="設定"]').first
            if not btn.count():
                return False
            btn.click(); page.wait_for_timeout(400)
            self.audit(page, "settings")
            return True
        except Exception as e:
            self.errors.append(("settings", "harness", str(e)[:150])); return False

    def log_screen(self, page):
        # dismiss the result card to expose the day log
        try:
            page.reload(); page.wait_for_function("document.querySelectorAll('input[type=file]').length>=1", timeout=30000)
            page.wait_for_timeout(800)
            self.audit(page, "log")
        except Exception as e:
            self.errors.append(("log", "harness", str(e)[:150]))


def pick_sets(a):
    man = json.load(open(UX / "manifest.json", encoding="utf-8"))["items"]
    by = {}
    for it in man:
        by.setdefault(it["set"], []).append(it)
    random.seed(0)
    S = {}
    want = a.sets.split(",") if a.sets else ["meals", "multi", "negatives", "drinks", "degraded", "n5k", "food101"]
    for s in want:
        items = [i for i in by.get(s, []) if (UX / i["file"]).exists()]
        if not a.full and len(items) > a.sample:
            items = random.sample(items, a.sample)
        S[s] = items
    user = sorted((SP / "negatives").glob("*.jpg"))
    return S, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4175/calorie-lens/")
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sets", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--serve", action="store_true", help="start/stop vite preview on port 4175 (web/dist must exist)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else SP / "uxcheck" / (a.tag or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    srv = None
    if a.serve:
        srv = subprocess.Popen('npx vite preview --base=/calorie-lens/ --port 4175 --strictPort', cwd=ROOT / "web", shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
        time.sleep(6)
    S, user = pick_sets(a)
    su = Suite(a)
    recs = []
    t_start = time.time()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)   # fresh context = first visit
            page = ctx.new_page()
            page.on("pageerror", lambda e: su.errors.append((su.job, "pageerror", str(e)[:200])))
            page.on("console", lambda m: su.errors.append((su.job, "console." + m.type, m.text[:200])) if m.type == "error" else None)
            page.goto(a.url)
            page.wait_for_function("document.querySelectorAll('input[type=file][accept=\"image/*\"]').length >= 1", timeout=60000)
            page.wait_for_timeout(500)
            su.audit(page, "idle")
            # job order: meals first (first-visit timing + result-screen audit)
            jobs = [("meals", i, UX / i["file"]) for i in S.get("meals", [])]
            jobs += [(g, i, UX / i["file"]) for g in S if g != "meals" for i in S[g]]
            jobs += [("user_neg", {"id": p_.name}, p_) for p_ in user]
            first = True
            for k, (g, it, path) in enumerate(jobs, 1):
                r = su.run_photo(page, g, it, path, first=first and g == "meals")
                first = False
                r["gt"] = {x: it.get(x) for x in ("main_class", "label", "kcal", "boxes", "parent_id") if it.get(x) is not None}
                recs.append(r)
                print(f"[{k}/{len(jobs)}] {g:9s} {Path(path).name[:30]:30s} items {len(r.get('items', []))} {r.get('wall_s')}s {r.get('error', '')}", flush=True)
            any_meal = next((UX / i["file"] for i in S.get("meals", [])), None)
            beh = su.behaviours(page, out / "tmp", any_meal)
            su.log_screen(page)
            su.settings_screen(page)
            b.close()
    finally:
        if srv:
            subprocess.run(f"taskkill /F /T /PID {srv.pid}", shell=True, capture_output=True)
    metrics = compute(a, su, recs, beh, S, time.time() - t_start)
    fo = metrics.pop("_failobjs")
    json.dump({"metrics": metrics, "records": recs}, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (out / "report.html").write_text(html(metrics, recs, fo), encoding="utf-8")
    (out / "tmp" / "huge_6000x4000.jpg").unlink(missing_ok=True)
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=1))
    print("report:", out / "report.html")


# =============================== metrics ===============================
def top_names(su, it, k=1):
    names = [it["nameEn"] or ""]
    for c in (it.get("cands") or [])[:k if k > 1 else 0]:
        v = su.vocab.get(c["id"]); names.append(v["en"] if v else "")
    return [n.lower() for n in names]


def compute(a, su, recs, beh, S, elapsed):
    g = lambda n: [r for r in recs if r["group"] == n and "items" in r]
    logged = lambda r: bool(r["items"])
    M, F = {}, {}  # metrics, failures
    fails = {}

    for key, grp in (("not_food_logged_user", "user_neg"), ("not_food_logged_uxset", "negatives"), ("drinks_logged", "drinks")):
        rs = g(grp); bad = [r for r in rs if logged(r)]
        M[key] = {"n": len(rs), "logged": len(bad), "pct": pct(len(bad), len(rs)),
                  "examples": [[i["nameEn"] for i in r["items"]] for r in bad[:5]]}
        fails[key] = bad

    # multi
    rs = g("multi"); rec_hits = tot = 0; ipi = []; gti = []; mfails = []
    for r in rs:
        gt = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in r["gt"].get("boxes", []) if b["class"] in FOOD_BOX_CLASSES]
        lg = [i["box"] for i in r["items"]]
        hit = sum(any(iou(x, y) >= 0.3 for y in lg) for x in gt)
        rec_hits += hit; tot += len(gt); ipi.append(len(lg)); gti.append(len(gt))
        r["multi"] = {"gt": len(gt), "hit": hit, "logged": len(lg)}
        if gt and hit / len(gt) < 0.5:
            mfails.append((r, gt, lg))
    M["multi"] = {"n_images": len(rs), "gt_food_boxes": tot, "matched": rec_hits, "recall_pct": pct(rec_hits, tot),
                  "items_per_image_logged": round(st.mean(ipi), 2) if ipi else None, "gt_per_image": round(st.mean(gti), 2) if gti else None}
    fails["multi"] = mfails

    # layout over every result with items
    ls = [r for r in recs if r.get("items") and r.get("layout", {}).get("card")]
    nl = len(ls)
    cnt = lambda f: sum(1 for r in ls if f(r["layout"]))
    mf = [r["layout"]["minFont"] for r in ls if r["layout"].get("minFont") is not None]
    M["layout"] = {"results_measured": nl,
                   "label_overlap_results": cnt(lambda l: (l["labelOverlaps"] or 0) > 0), "pin_overlap_results": cnt(lambda l: (l["pinOverlaps"] or 0) > 0),
                   "outside_photo_results": cnt(lambda l: (l["outside"] or 0) > 0), "clipped_results": cnt(lambda l: (l["clipped"] or 0) > 0),
                   "truncated_label_results": cnt(lambda l: (l["truncated"] or 0) > 0),
                   "pin_count_mismatch_results": sum(1 for r in ls if r["layout"]["nPins"] != len(r["items"])),
                   "min_label_font_px": min(mf) if mf else "n/a", "labels_below_%dpx_results" % MIN_FONT_OK: cnt(lambda l: (l["smallLabels"] or 0) > 0),
                   "legacy_ui_fallback": any(r["layout"]["legacy"] for r in ls),
                   "note": "n/a for pins/labels if none found in DOM (legacy UI: .box span labels used; pin overlap not scored)"}
    fails["layout"] = [r for r in ls if (r["layout"]["labelOverlaps"] or 0) or (r["layout"]["pinOverlaps"] or 0) or r["layout"]["outside"] or r["layout"]["clipped"] or r["layout"]["smallLabels"]]

    # fits
    fit = {}
    for vp in ("390x844", "375x667"):
        vs = [r["fits"][vp] for r in ls if vp in r.get("fits", {})]
        bad = [r for r in ls if vp in r.get("fits", {}) and not r["fits"][vp]["ok"]]
        miss = sorted({m for v in vs for m in v["missing"]})
        fit[vp] = {"n": len(vs), "fit": len(vs) - len(bad), "pct_fit": pct(len(vs) - len(bad), len(vs)),
                   "median_card_bottom_px": round(st.median([v["bottoms"]["card"] for v in vs])) if vs else None,
                   "n/a_elements": miss}
        fails["fits_" + vp] = bad
    M["fits_on_screen"] = fit

    # cjk / contrast / tap per screen
    scr = {}
    for name, r in su.audits.items():
        scr[name] = {"cjk_chars": r["cjk"],
                     "contrast_light": {"checked": r["light"]["checked"], "skipped_over_image": r["light"]["skipped"], "fails": len(r["light"]["fails"]), "worst": sorted(r["light"]["fails"], key=lambda x: x["ratio"])[:8]},
                     "contrast_dark": {"checked": r["dark"]["checked"], "skipped_over_image": r["dark"]["skipped"], "fails": len(r["dark"]["fails"]), "worst": sorted(r["dark"]["fails"], key=lambda x: x["ratio"])[:8]},
                     "tap_small": {"checked": r["tap"]["checked"], "small": len(r["tap"]["small"]), "items": r["tap"]["small"][:15]}}
    for s in ("idle", "result", "no-food", "error", "log", "settings"):
        scr.setdefault(s, "n/a (screen not reached)")
    M["screens"] = scr
    resc = [r["layout"]["cjk"] for r in ls]
    M["cjk_in_results"] = {"results_with_cjk": sum(1 for c in resc if c), "of": len(resc), "median_chars": st.median(resc) if resc else None}

    # timing
    first = [r for r in recs if r.get("first_visit") and "ms" in r]
    rest = [r for r in recs if "ms" in r and not r.get("first_visit")]
    W = [r["wall_s"] for r in rest]; A = [r["ms"] / 1000 for r in rest]
    M["timing"] = {"first_visit_to_first_result_s": first[0]["wall_s"] if first else None, "first_visit_app_ms": first[0]["ms"] if first else None,
                   "per_photo_wall_median_s": round(st.median(W), 2) if W else None, "per_photo_wall_p90_s": round(quant(W, .9), 2) if W else None,
                   "per_photo_app_median_s": round(st.median(A), 2) if A else None, "per_photo_app_p90_s": round(quant(A, .9), 2) if A else None,
                   "n": len(W), "note": "shared CPU, headless desktop Chromium (wasm); wall includes harness waits"}

    # errors
    kinds = {}
    for j, k, t in su.errors:
        kinds[k] = kinds.get(k, 0) + 1
    uniq = []
    for j, k, t in su.errors:
        if (k, t) not in [(x[0], x[1]) for x in uniq]:
            uniq.append((k, t, j))
    M["errors"] = {"total": len(su.errors), "by_kind": kinds, "unique": uniq[:12], "photos_with_error_result": sum(1 for r in recs if "error" in r),
                   "behaviour": beh}

    # names
    def plaus(grp):
        rs = [r for r in g(grp) if logged(r) and r["gt"].get("main_class") in NAME_MAP and r["gt"]["main_class"] != "Food"]
        # top-1 of any logged item / any candidate of any logged item
        a1 = [any(re.search(NAME_MAP[r["gt"]["main_class"]], top_names(su, it)[0]) for it in r["items"]) for r in rs]
        ak = [any(any(re.search(NAME_MAP[r["gt"]["main_class"]], n) for n in top_names(su, it, 5)) for it in r["items"]) for r in rs]
        return rs, a1, ak
    nm = {}
    for grp in ("meals", "degraded"):
        rs, a1, ak = plaus(grp)
        nm[grp] = {"n_logged_mappable": len(rs), "of_total": len(g(grp)), "top1_plausible_pct": pct(sum(a1), len(a1)), "in_top5_cands_pct": pct(sum(ak), len(ak)),
                   "wrong_examples": [(r["gt"]["main_class"], r["items"][0]["nameEn"]) for r, ok in zip(rs, a1) if not ok][:6]}
        fails["names_" + grp] = [r for r, ok in zip(rs, a1) if not ok]
    from evalsets import FOOD101_MAP
    rs = [r for r in g("food101") if r["gt"].get("label") in FOOD101_MAP and logged(r)]
    ok = [any(it["cands"] and it["cands"][0]["id"] in FOOD101_MAP[r["gt"]["label"]] for it in r["items"]) for r in rs]
    nm["food101"] = {"n_logged_mappable": len(rs), "top1_right_pct": pct(sum(ok), len(ok)), "note": "vocab-id map from evalsets.FOOD101_MAP"}
    M["names"] = nm
    for grp in ("degraded", "n5k", "food101", "meals"):
        rs = g(grp)
        M.setdefault("logged_rate", {})[grp] = {"n": len(rs), "logged_pct": pct(sum(map(logged, rs)), len(rs))}

    M["testids_found"] = {k: (v if v else "n/a (missing)") for k, v in su.testid_max.items()}
    summary = {
        "1 not_food_logged (user 10 / uxset)": f"{M['not_food_logged_user']['logged']}/{M['not_food_logged_user']['n']} | {M['not_food_logged_uxset']['logged']}/{M['not_food_logged_uxset']['n']}",
        "2 drinks_logged": f"{M['drinks_logged']['logged']}/{M['drinks_logged']['n']}",
        "3 multi recall@IoU.3": f"{M['multi']['recall_pct']}% ({M['multi']['matched']}/{M['multi']['gt_food_boxes']}); items/img {M['multi']['items_per_image_logged']} vs GT {M['multi']['gt_per_image']}",
        "4 layout": {k: v for k, v in M["layout"].items() if k != "note"},
        "5 fits_on_screen": {k: f"{v['fit']}/{v['n']} {v['n/a_elements'] or ''}" for k, v in fit.items()},
        "6 cjk chars per screen": {k: (v["cjk_chars"] if isinstance(v, dict) else v) for k, v in scr.items()},
        "7 contrast fails light/dark": {k: (f"{v['contrast_light']['fails']}/{v['contrast_dark']['fails']}" if isinstance(v, dict) else v) for k, v in scr.items()},
        "8 tap targets <44px": {k: (f"{v['tap_small']['small']}/{v['tap_small']['checked']}" if isinstance(v, dict) else v) for k, v in scr.items()},
        "9 timing": M["timing"],
        "10 errors": {"total": M["errors"]["total"], "by_kind": kinds, "behaviour": {k: v["verdict"] for k, v in beh.items()}},
        "11 names top1 plausible %": {k: v.get("top1_plausible_pct", v.get("top1_right_pct")) for k, v in nm.items()},
        "testids missing": [k for k, v in M["testids_found"].items() if isinstance(v, str)],
        "elapsed_s": round(elapsed), "args": {"sample": a.sample, "full": a.full},
    }
    return {"summary": summary, "metrics": M, "_fails": {k: [(x[0]["path"] if isinstance(x, tuple) else x["path"]) for x in v] for k, v in fails.items()},
            "_failobjs": fails}


# =============================== HTML ===============================
def html(MM, recs, fails):
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    h = ["<!doctype html><meta charset=utf-8><title>Calorie Lens UX check</title><style>body{font:14px system-ui;margin:20px;max-width:1100px}"
         "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px;text-align:left;vertical-align:top}"
         ".g{display:flex;flex-wrap:wrap;gap:10px}.c{width:200px;font-size:11px}.c img{display:block;border:1px solid #888}"
         "pre{background:#f4f4f4;padding:8px;overflow:auto;font-size:12px}h2{margin-top:28px}</style>"]
    h.append("<h1>Calorie Lens UX check</h1><p>%s</p>" % esc(datetime.datetime.now().isoformat(timespec="seconds")))
    h.append("<h2>Summary</h2><pre>%s</pre>" % esc(json.dumps(MM["summary"], ensure_ascii=False, indent=1)))
    h.append("<h2>Test ids found (max count on any screen)</h2><table>%s</table>" % "".join(f"<tr><td>{k}</td><td>{esc(v)}</td></tr>" for k, v in MM["metrics"]["testids_found"].items()))
    h.append("<h2>Screens: CJK / contrast / tap targets</h2>")
    for name, r in MM["metrics"]["screens"].items():
        if not isinstance(r, dict):
            h.append(f"<h3>{name}</h3><p>{esc(r)}</p>"); continue
        h.append(f"<h3>{name}</h3><p>CJK chars: {r['cjk_chars']} | contrast checked light/dark {r['contrast_light']['checked']}/{r['contrast_dark']['checked']}</p><table><tr><th>scheme<th>text<th>ratio<th>need<th>size<th>fg<th>bg<th>element</tr>")
        for sch in ("light", "dark"):
            for f in r["contrast_" + sch]["worst"]:
                h.append(f"<tr><td>{sch}<td>{esc(f['text'])}<td>{f['ratio']}<td>{f['need']}<td>{f['size']}<td>{f['fg']}<td>{f['bg']}<td>{esc(f['sel'])}</tr>")
        h.append("</table>")
        if r["tap_small"]["items"]:
            h.append("<p>Tap targets &lt;44px (%d of %d):</p><table>%s</table>" % (r["tap_small"]["small"], r["tap_small"]["checked"],
                     "".join(f"<tr><td>{esc(t['sel'])}<td>{esc(t['text'])}<td>{t['w']}x{t['h']}</tr>" for t in r["tap_small"]["items"])))
    h.append("<h2>Behaviour checks</h2><pre>%s</pre>" % esc(json.dumps(MM["metrics"]["errors"], ensure_ascii=False, indent=1)))
    h.append("<h2>Other metrics</h2><pre>%s</pre>" % esc(json.dumps({k: v for k, v in MM["metrics"].items() if k in ("multi", "layout", "fits_on_screen", "names", "timing", "logged_rate", "cjk_in_results")}, ensure_ascii=False, indent=1)))
    h.append("<h2>Failures</h2><p>Green = ground-truth food boxes, red = boxes the app logged.</p>")
    for key, lst in fails.items():
        if not lst:
            continue
        h.append(f"<h3>{key} ({len(lst)})</h3><div class=g>")
        for x in lst[:40]:
            gt = lg = None
            if isinstance(x, tuple):
                r, gt, lg = x
            else:
                r = x
                lg = [i["box"] for i in r.get("items", [])]
                gt = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in r["gt"].get("boxes", []) if b["class"] in FOOD_BOX_CLASSES] if r["group"] == "multi" else None
            cap = ", ".join(i["nameEn"] or "?" for i in r.get("items", []))[:90]
            extra = ""
            if key.startswith("fits") or key == "layout":
                l = r.get("layout", {}); extra = f" ov={l.get('labelOverlaps')}/{l.get('pinOverlaps')} out={l.get('outside')} clip={l.get('clipped')} font={l.get('minFont')} " + json.dumps(r.get("fits", {}).get(key[5:], "")) if key.startswith("fits") else f" ov={l.get('labelOverlaps')}/{l.get('pinOverlaps')} out={l.get('outside')} clip={l.get('clipped')} font={l.get('minFont')}"
            h.append(f"<div class=c><img src='{thumb(r['path'], gt, lg)}'>{esc(Path(r['path']).name)}<br>{esc(r['gt'].get('main_class', ''))}<br>{esc(cap)}{esc(extra)}</div>")
        h.append("</div>")
    return "".join(h)


if __name__ == "__main__":
    main()
