# Capture only: screenshots of the live screen for the blind design check; not part of the product.
# usage: uv run --with playwright --with pillow python tools/capture_shots.py [scratch dir]   (shots land in ./shots)
import sys, time, subprocess, urllib.request, json
from playwright.sync_api import sync_playwright
import os, tempfile; REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(); OUT = f"{REPO}/shots"
import os; os.makedirs(OUT, exist_ok=True)
def state():
    return json.load(urllib.request.urlopen("http://localhost:8777/state", timeout=5))
p = subprocess.Popen(["uv", "run", "python", "live.py", "--rhythm"], cwd=REPO,
                     stdout=open(f"{S}/live7.log", "w"), stderr=subprocess.STDOUT)
TAPS = {1: [0, 520, 1050, 1600, 2150], 2: [0, 300, 450, 1000, 1200], 3: [0, 300, 700, 800, 1500]}
WIDTH_JS = """() => [document.documentElement.scrollWidth, [...document.querySelectorAll('body *')]
  .filter(e => e.getBoundingClientRect().right > innerWidth + 0.5 && e.getClientRects().length)
  .slice(0, 6).map(e => e.tagName + '#' + e.id + '.' + e.className + ' ' + Math.round(e.getBoundingClientRect().right))]"""
SMALL_JS = """() => { let MIN = 99; const out = [], w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (w.nextNode()) { const n = w.currentNode, e = n.parentElement; if (!n.textContent.trim() || !e) continue;
    const r = e.getBoundingClientRect(); if (!r.width || !e.getClientRects().length || getComputedStyle(e).visibility === 'hidden') continue;
    let px = parseFloat(getComputedStyle(e).fontSize); const svg = e.closest('svg');
    if (svg && e.getScreenCTM) px *= Math.hypot(e.getScreenCTM().a, e.getScreenCTM().b);
    MIN = Math.min(MIN, px); if (px < 11.95) out.push(e.tagName + ' ' + px.toFixed(1) + 'px ' + n.textContent.trim().slice(0, 30)); }
  return [MIN.toFixed(2)].concat(out.slice(0, 8)); }"""
def width(pg, tag):
    small = pg.evaluate(SMALL_JS)
    if len(small) > 1 or tag.endswith(("after", "revealed")): print("SMALL" if len(small) > 1 else "minpx", tag, pg.viewport_size["width"], small)
    if pg.viewport_size["width"] == 390: print("width", tag, pg.evaluate(WIDTH_JS))
PARK = """() => { const w = document.querySelector('#scene-slot iframe')?.contentWindow; if (!w || w.__raf) return;
  w.__raf = w.requestAnimationFrame; w.__q = []; w.requestAnimationFrame = cb => (w.__q.push(cb), 0); }"""
RESUME = """() => { const w = document.querySelector('#scene-slot iframe')?.contentWindow; if (!w || !w.__raf) return;
  w.requestAnimationFrame = w.__raf; delete w.__raf; w.__q.splice(0).forEach(cb => w.requestAnimationFrame(cb)); }"""
def rnd(pg, tag, targets):
    pg.click("#blind"); time.sleep(1)
    pg.evaluate("document.getElementById('piles').scrollIntoView({block: 'start'})"); time.sleep(0.3)
    pg.screenshot(path=f"{OUT}/piles-{tag}-1-before.png"); width(pg, tag + " before")
    for n in (1, 2, 3):
        # headless software WebGL starves the page (~5 s per input at 1280): park the scene's frame loop while taps land
        pg.evaluate(PARK); pg.wait_for_function("() => { const w = document.querySelector('#scene-slot iframe')?.contentWindow; return !w || w.__q.length > 0; }", timeout=60000)
        pg.click("#bgo"); time.sleep(0.3)
        t0 = time.time()
        # taps scheduled inside the page: headless software WebGL delays each protocol input by ~5 s (the capture artefact)
        pg.evaluate("ts => ts.forEach(t => setTimeout(() => document.getElementById('bpad').dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, cancelable: true})), t))", TAPS[n])
        time.sleep(1.2)
        if n == 2: pg.screenshot(path=f"{OUT}/piles-{tag}-2-during.png")
        time.sleep(max(0, 7 - (time.time() - t0)))
        pg.evaluate(RESUME)
    for _ in range(60):
        if state()["piles_phase"] == "heard": break
        time.sleep(1)
    time.sleep(0.6)
    for t in targets: pg.click(f'#bwho [data-t="{t}"]')
    pg.evaluate("document.getElementById('bwho').scrollIntoView({block: 'center'})"); time.sleep(0.3)
    pg.screenshot(path=f"{OUT}/piles-{tag}-3-heard.png")
    print("picks", tag, pg.evaluate("""() => [...document.querySelectorAll('#bwho [data-t]')].map(b =>
      [b.dataset.t, b.getAttribute('aria-pressed'), getComputedStyle(b).borderColor, getComputedStyle(b).color])"""))
    pg.click("#breveal"); time.sleep(1.5)
    pg.evaluate("document.getElementById('piles').scrollIntoView({block: 'start'})"); time.sleep(0.4)
    width(pg, tag + " revealed")
    pg.screenshot(path=f"{OUT}/piles-{tag}-4-revealed.png")
    pg.screenshot(path=f"{OUT}/piles-{tag}-4-revealed-full.png", full_page=True)
    s = state(); print(tag, json.dumps({k: s[k] for k in ("piles_phase", "targets", "flagged", "software_flagged", "tally", "software_tally", "outcome")}), [p["dish_verdict"] for p in s["piles"]])
try:
    for _ in range(60):
        try: state(); break
        except Exception: time.sleep(1)
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        pg = b.new_page(viewport={"width": 1280, "height": 633})
        pg.on("console", lambda m: m.type in ("error", "warning") and print("CONSOLE", m.type, m.text[:200]))
        pg.on("pageerror", lambda e: print("PAGEERROR", str(e)[:200]))
        pg.goto("http://localhost:8777/"); time.sleep(1)
        def both(tag, full=False):
            for w, h in ((1280, 633), (390, 757)):
                pg.set_viewport_size({"width": w, "height": h}); time.sleep(0.8)
                pg.evaluate("document.getElementById('pcard').scrollIntoView({block: 'start'})"); time.sleep(0.3)
                pg.screenshot(path=f"{OUT}/{tag}-{w}.png"); width(pg, tag)
                if full:
                    pg.evaluate("document.getElementById('scene-slot').scrollIntoView({block: 'center'})"); time.sleep(1)
                    pg.screenshot(path=f"{S}/scene-{tag}-{w}-view.png", clip=pg.locator('#scene-slot').bounding_box())
                    pg.screenshot(path=f"{OUT}/{tag}-{w}-full.png", full_page=True)
                    box = pg.locator('#scene-slot').bounding_box(); sy = pg.evaluate("scrollY")
                    from PIL import Image, ImageStat
                    full_img = Image.open(f"{OUT}/{tag}-{w}-full.png").convert("L")
                    crop = full_img.crop((int(box["x"]), int(box["y"] + sy), int(box["x"] + box["width"]), int(box["y"] + sy + box["height"])))
                    print("scene", tag, w, "in view mean", round(ImageStat.Stat(Image.open(f"{S}/scene-{tag}-{w}-view.png").convert("L")).mean[0], 1),
                          "in full-page mean", round(ImageStat.Stat(crop).mean[0], 1))
            pg.set_viewport_size({"width": 1280, "height": 633}); time.sleep(0.5)
        both("rasters-1-before", full=True)
        pg.click("#prun")
        for _ in range(300):
            g = state().get("probe_progress")
            if g and g["stage"] == "trials" and g["done"] >= 30: break
            time.sleep(1)
        both("rasters-2-running")
        for _ in range(300):
            if state().get("probe") and not state().get("probe_progress"): break
            time.sleep(1)
        time.sleep(1)
        both("rasters-3-after", full=True)
        print("probe", json.dumps({k: state()["probe"].get(k) for k in ("primary", "means", "effect", "error")}))
        pg.click("#run")
        for _ in range(200):
            if state()["phase"] == "ready": break
            time.sleep(2)
        time.sleep(1)
        pg.screenshot(path=f"{OUT}/piles-0-ready-1280-full.png", full_page=True)
        rnd(pg, "w1280", [1])
        pg.click("#bclose"); time.sleep(0.5)
        pg.set_viewport_size({"width": 390, "height": 757}); time.sleep(0.8)
        pg.screenshot(path=f"{OUT}/piles-0-ready-390-full.png", full_page=True)
        rnd(pg, "w390", [2])
        b.close()
finally:
    p.terminate()
