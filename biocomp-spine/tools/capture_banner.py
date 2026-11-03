# Capture only: screenshots of the live screen for the blind design check; not part of the product.
# usage: uv run --with playwright --with pillow python tools/capture_banner.py [scratch dir]   (shots land in ./shots)
import sys, time, subprocess, json, urllib.request
from playwright.sync_api import sync_playwright
import os, tempfile; REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
p = subprocess.Popen(["uv", "run", "python", "live.py", "--rhythm"], cwd=REPO, stdout=open(f"{S}/live9.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        try: urllib.request.urlopen("http://localhost:8777/state", timeout=3); break
        except Exception: time.sleep(1)
    def fake(route):
        s = json.load(urllib.request.urlopen("http://localhost:8777/state", timeout=5))
        s.update(probe=None, running=True, phase="probe", probe_progress=dict(stage="trials", done=4, total=120), probe_no_echo=True, probe_live=[dict(i=i, cond=c, raster=[], flag="5 pulse(s) with no echo", win=[20, 250]) for i, c in enumerate(["A+", "A-", "B+", "B-"])])
        route.fulfill(status=200, content_type="application/json", body=json.dumps(s))
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        for w, h in ((1280, 633), (390, 757)):
            pg = b.new_page(viewport={"width": w, "height": h}); pg.route("**/state*", fake)
            pg.goto("http://localhost:8777/"); time.sleep(2.5)
            pg.evaluate("document.getElementById('pcard').scrollIntoView({block: 'start'})"); time.sleep(0.3)
            pg.screenshot(path=f"{REPO}/shots/echo-banner-{w}.png")
            print(w, pg.evaluate("[document.documentElement.scrollWidth, document.getElementById('pecho').hidden]"))
        b.close()
finally:
    p.terminate()
