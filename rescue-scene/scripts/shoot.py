"""Headless screenshots at 1280x633 (and a few at 390x757), the drone's on-screen height, and an fps reading.

Run (with the page served on :8791): uv run --no-project --with playwright python -I scripts/shoot.py <out_dir>
Software GL (SwiftShader) draws about 1 frame/s, so stills use ?capture (big time steps) and
fps is measured separately on the normal page as requestAnimationFrame callbacks over 5 s.
"""
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8791/"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
DESK, PHONE = {"width": 1280, "height": 633}, {"width": 390, "height": 757}
ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def shot(pg, name):
    pg.screenshot(path=out / name, timeout=600_000)
    st = pg.evaluate("rescue.state")
    print(name, "| drone px:", st["dronePx"], "| mode:", pg.evaluate("document.body.dataset.mode"), "| node:", st["node"],
          "|", pg.text_content("#v-label"), "|", pg.inner_text("#prov"), flush=True)


def layout(pg, name):
    """No HUD box may overlap another or a tag, no two tags may touch or leave the screen, no text under 12 px."""
    bad = pg.evaluate("""() => {
      const r = s => { const e = document.querySelector(s); return e && !e.hidden && getComputedStyle(e).display !== 'none' ? e.getBoundingClientRect() : null; };
      const hit = (a, b) => a && b && a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
      const out = [], boxes = ['.brand', '#mode', '#telemetry', '#verdict', '#foot'];
      for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++)
        if (hit(r(boxes[i]), r(boxes[j]))) out.push(boxes[i] + ' overlaps ' + boxes[j]);
      const tags = [...document.querySelectorAll('.tag:not([hidden])')];
      tags.forEach((t, i) => { const b = t.getBoundingClientRect();
        if (b.right > innerWidth || b.left < 0 || b.top < 0 || b.bottom > innerHeight) out.push('tag off screen: ' + t.textContent);
        for (const s of boxes) if (hit(b, r(s))) out.push('tag ' + t.textContent + ' overlaps ' + s);
        const g = u => { const q = u.getBoundingClientRect(); return { left: q.left - 7, right: q.right + 7, top: q.top - 7, bottom: q.bottom + 7 }; };
        for (const u of tags.slice(i + 1)) if (hit(b, g(u))) out.push('tag ' + t.textContent + ' under 8 px from tag ' + u.textContent); });
      for (const e of document.querySelectorAll('body *')) { const fs = parseFloat(getComputedStyle(e).fontSize);
        if (e.childNodes.length && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()) && e.offsetParent !== null && fs < 12)
          out.push('text under 12 px: ' + e.textContent.slice(0, 30)); }
      return out; }""")
    if bad:
        errors.append(f"{name}: {bad}")


def phone(pg, name, node=False, piles=False, seek=None):
    """Returns the pill's phase word as shot. `seek` re-times the phone frame to the desktop one's story moment."""
    pg.set_viewport_size(PHONE); pg.wait_for_timeout(8000)
    if seek is not None:
        pg.evaluate(f"rescue.seek({seek})"); pg.wait_for_function("rescue.state.arrived", timeout=120_000); pg.wait_for_timeout(2500)
    pill = pg.inner_text("#mode-label")
    vis = "[...document.querySelectorAll('.node-tag')].filter(t => !t.hidden).length"
    before = pg.evaluate(vis) if piles else 3
    shot(pg, name); layout(pg, name)
    if piles and 3 != before or piles and 3 != pg.evaluate(vis):   # an off-screen or unplaceable pile hides its tag
        errors.append(f"{name}: a pile out of frame")
    if pg.evaluate("rescue.state.tagCross"):   # a leader crosses a tag or another leader, or a tag covers an anchor
        errors.append(f"{name}: a leader crosses a callout")
    if node and pg.evaluate("document.querySelector('.node-tag').hidden"):   # the hero's node is in frame, tagged
        errors.append(f"{name}: node not in frame")
    bad = pg.evaluate("""() => { const d = rescue.state.droneBox, out = [];
      if (rescue.state.dronePx < 10) d.left = d.right = -1e9;   // a landed drone a few px tall in the wide shot is ground, not a subject
      for (const s of ['.brand', '#mode', '#verdict', '#foot']) { const b = document.querySelector(s).getBoundingClientRect();
        if (d.left < b.right && b.left < d.right && d.top < b.bottom && b.top < d.bottom) out.push('drone touches ' + s); }
      const c = document.querySelector('#credit');
      if (c.offsetHeight > 52 || c.getBoundingClientRect().right > document.querySelector('#foot').getBoundingClientRect().right) out.push('credit over three lines');
      return out; }""")
    if bad:
        errors.append(f"{name}: {bad}")
    pg.set_viewport_size(DESK); pg.wait_for_timeout(5000)
    return pill


def mic(pg, name, says=False):
    """The node on the rubble is a microphone: no callout may put a dish, cells or neurons out there."""
    txt = pg.evaluate("[...document.querySelectorAll('.node-tag:not([hidden]), #tag:not([hidden])')].map(t => t.textContent).join(' | ')")
    if re.search(r"dish|neuron|cell|living|culture", txt, re.I):
        errors.append(f"{name}: callout implies a dish on the rubble: {txt}")
    if says and "Microphone" not in txt:
        errors.append(f"{name}: no callout says Microphone: {txt}")


def fresh(b, query):
    for c in b.contexts:  # one SwiftShader page at a time, or they starve
        c.close()
    pg = b.new_page(viewport=DESK)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL + query)
    pg.wait_for_function("window.rescue && window.rescue.state.fps > 0", timeout=120_000)
    return pg


def rounds(pg):
    """a second drop replacing a pending one drops the old round's held calls; a non-confirming verdict clears a pile hit"""
    hit = "rescue.setPileVerdicts([{p_survivor: .2}, {p_survivor: .79}, {p_survivor: .1}], [2], 'practice', {target: 2})"
    pg.evaluate(f"rescue.reset(); rescue.dropNodes(undefined, 'practice'); {hit}; rescue.markSpot(); rescue.listenPile(1); rescue.dropNodes(undefined, 'practice')")
    pg.wait_for_function("rescue.state.nodes === 3", timeout=120_000); pg.wait_for_timeout(500)
    s = pg.evaluate("rescue.state")
    assert s["pileVerdicts"] is None and not s["marked"] and s["pile"] is None, ("old round replayed", s["pileVerdicts"], s["marked"], s["pile"])
    pg.evaluate(hit)
    assert pg.evaluate("document.querySelectorAll('[data-state=survivor]').length") > 0, "the hit did not show"
    pg.evaluate("rescue.setVerdict('unsure', .5)")
    assert pg.evaluate("rescue.state.pileVerdicts") is None and pg.evaluate("document.querySelectorAll('[data-state=survivor]').length") == 0, "unsure kept the pile hit"
    assert pg.text_content("#v-label").endswith("Unclear"), pg.text_content("#v-label")
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .2}, {p_survivor: .79}, {p_survivor: .1}], [2], 'practice', {target: null}); rescue.setVerdict('survivor', .9, 'practice')")
    assert pg.evaluate("rescue.state.pileVerdicts") is None and pg.text_content("#v-label") == "Practice · Target rhythm flagged", ("old pile result masks a survivor", pg.text_content("#v-label"))


errors = []
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    # the story in order on one page, so the clock and telemetry run on without a break
    pg = b.new_page(viewport=DESK)
    pg.on("console", lambda m: m.type in ("error", "warning") and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("response", lambda r: r.status >= 400 and errors.append(f"{r.status} {r.url}"))   # e.g. a favicon 404
    pg.goto(URL + "?capture=1&t=20")
    pg.wait_for_function("window.rescue && window.rescue.state.fps > 0", timeout=120_000)
    pg.evaluate("rescue.seek(24)")
    pg.wait_for_function("rescue.state.arrived", timeout=120_000)   # a fixed phase on both screens: wall time per frame varies under load
    pg.wait_for_timeout(2500)
    assert float(pg.inner_text("#t-spd").split()[0]) < 40, pg.inner_text("#t-spd")   # no first-frame spike
    d1 = pg.inner_text("#mode-label")
    shot(pg, "1-flyover.png"); layout(pg, "1-flyover.png")
    assert pg.evaluate("!document.getElementById('drone-tag').hidden"), "the drone has no callout"   # a few px drone still reads as one
    assert pg.evaluate("document.body.innerText").count("Simulated flight") == 1, "Simulated flight said twice"
    assert "Haiti" not in pg.evaluate("document.body.innerText") and pg.text_content("#credit").startswith("Aerial: World Bank / RIT"), pg.text_content("#credit")   # no event or place name; one text at every size
    m1 = phone(pg, "m1-flyover.png", seek=24)
    assert d1 == m1, (d1, m1)   # one phase word for one story moment on both screens
    assert pg.evaluate("document.getElementById('landmark')") is None   # no real-collapse name near the piles
    pg.evaluate("rescue.seek(43); rescue.listen()")
    pg.wait_for_timeout(6000)
    shot(pg, "2-listen.png"); layout(pg, "2-listen.png")

    # the opening node: listening only, no score, no verdict, no green
    pg.evaluate("rescue.dropNode('practice')")   # the shoot runs on the practice dish, and the panel must say so
    pg.wait_for_function("rescue.state.node === 'landed'", timeout=600_000)
    pg.wait_for_timeout(2500)
    shot(pg, "3-node-listening.png"); layout(pg, "3-node-listening.png"); mic(pg, "3", says=True)
    assert pg.inner_text("#t-spd") == "0.0 m/s" and pg.inner_text("#t-alt") == "0.0 m AGL", (pg.inner_text("#t-spd"), pg.inner_text("#t-alt"))
    assert pg.evaluate("rescue.state.verdict") is None and pg.inner_text("#v-chip") == ""
    tele = pg.evaluate("document.getElementById('telemetry').textContent")
    assert "°" not in tele, tele   # no real lat/lon
    assert "backdrop only" in pg.inner_text("#prov"), pg.inner_text("#prov")
    assert "%" not in pg.evaluate("document.body.innerText"), "a score before the reveal"
    assert pg.inner_text("#v-sub") == "Knocks are sent to the practice dish (simulated) at base.", pg.inner_text("#v-sub")
    phone(pg, "m2-node.png", node=True)
    assert pg.inner_text("#t-rot") == "stopped", pg.inner_text("#t-rot")

    # three piles: the drone takes off from the opening node and places one on each
    pg.evaluate("rescue.dropNodes()")
    assert "Awaiting verdict" in pg.inner_text("#prov"), pg.inner_text("#prov")
    pg.wait_for_function("rescue.state.node === 'dropping' && [...document.querySelectorAll('.node-tag')].filter(t => !t.hidden).length >= 1", timeout=600_000)
    shot(pg, "4-placing.png"); layout(pg, "4-placing.png")
    pg.wait_for_function("rescue.state.node === 'landed'", timeout=600_000)
    pg.wait_for_timeout(2500)
    shot(pg, "5-three-nodes.png"); layout(pg, "5-three-nodes.png"); mic(pg, "5", says=True)
    assert "%" not in pg.evaluate("document.body.innerText"), "a score before the reveal"
    # no leak: listening to each pile in turn, nothing on screen differs but the pile number
    looks = []
    for i in (1, 2, 3):
        pg.evaluate(f"rescue.listenPile({i})")
        pg.wait_for_timeout(8000)
        lk = pg.evaluate("rescue.look()")
        nd = lk.pop("nodes"); lk["nodes"] = [nd[i - 1]] + nd[:i - 1] + nd[i:]   # the listened pile first
        looks.append(re.sub(r"\d", "#", json.dumps(lk, sort_keys=True)))
    assert looks[0] == looks[1] == looks[2], looks
    pg.evaluate("rescue.listenPile(1)")
    pg.wait_for_timeout(3000)
    shot(pg, "6-listen-pile1.png"); layout(pg, "6-listen-pile1.png"); mic(pg, "6", says=True)
    tags = pg.evaluate("[...document.querySelectorAll('.node-tag')].map(t => t.dataset.state)")
    assert tags == ["idle"] * 3, tags
    NOTE = "Reader 55% on held-out sequences · shuffled labels 49%"
    pg.evaluate(f"rescue.setPileVerdicts([{{p_survivor: .18}}, {{p_survivor: .91}}, {{p_survivor: .22}}], [2], 'practice', {{target: 2, note: '{NOTE}'}})")   # scripted: the truth is known
    pg.wait_for_timeout(4000)
    shot(pg, "7-reveal.png"); layout(pg, "7-reveal.png"); mic(pg, "7")
    assert pg.inner_text("#v-chip") == "Practice dish (simulated)", pg.inner_text("#v-chip")
    assert pg.inner_text("#mode-label") == "Practice · Target rhythm flagged", pg.inner_text("#mode-label")   # the pill names the verdict, not "Landed"
    # a practice verdict says so on the pill and on the coloured callout, so a cropped frame never reads as a real hit
    hit = pg.evaluate("document.querySelector('.node-tag[data-state=survivor]').textContent")
    assert pg.inner_text("#v-label").split("\n")[0] == "Practice", pg.inner_text("#v-label")   # desktop: its own line, so the verdict never wraps
    assert pg.evaluate("(r => (r.selectNodeContents(document.getElementById('v-label').lastChild), r.getClientRects().length))(document.createRange())") == 1, "the desktop verdict wraps"
    assert "Practice" in pg.inner_text("#mode-label") and "Practice" in hit, (pg.inner_text("#mode-label"), hit)
    assert pg.evaluate("getComputedStyle(document.querySelector('.brand .dot')).backgroundColor") != "rgb(0, 212, 106)", "brand dot green under practice"
    assert pg.inner_text("#v-note") == NOTE, pg.inner_text("#v-note")   # the caller's caveat is the only number on the card
    assert "%" not in pg.evaluate("document.body.innerText.replace(document.getElementById('v-note').innerText, '')"), "a score outside the note"
    phone(pg, "m3-reveal.png", piles=True)
    tags = pg.evaluate("[...document.querySelectorAll('.node-tag')].map(t => t.dataset.state)")
    assert tags == ["debris", "survivor", "debris"], tags
    assert pg.evaluate("[...document.querySelectorAll('.node-tag')].map(t => t.textContent).join()").count("%") == 0   # no score on a pile
    for bad in ["[2, 2]", "[4]", "2"]:
        try:
            pg.evaluate(f"rescue.setPileVerdicts([{{p_survivor: .1}}, {{p_survivor: .2}}, {{p_survivor: .15}}], {bad})"); errors.append("no throw: " + bad)
        except Exception:
            pass
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .8}, {p_survivor: .2}, {p_survivor: .9}], [3, 1], 'practice')")
    tags = pg.evaluate("[...document.querySelectorAll('.node-tag')].map(t => t.dataset.state)")
    assert tags == ["neutral", "debris", "neutral"], tags   # no truth given: a flag is the dish's claim, ink, never green
    assert pg.text_content("#v-label") == "Practice · Piles 1 and 3 flagged", pg.text_content("#v-label")
    mic(pg, "flagged")
    pg.wait_for_function("document.body.dataset.mode === 'busy'", timeout=60_000)   # the pill's LED is ink, not green, after the next frame
    # the spine sends target null when the reader is at chance: pile 2 flagged reads neutral, nothing green
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .18}, {p_survivor: .91}, {p_survivor: .22}], [2], 'practice', {target: null, note: 'x'})")
    assert pg.text_content("#v-label") == "Practice · Pile 2 flagged" and pg.evaluate("rescue.look().card") == "neutral", pg.text_content("#v-label")
    assert pg.evaluate("document.querySelectorAll('[data-state=survivor]').length") == 0 and pg.evaluate("rescue.state.pileVerdicts.target") is None
    # the truth given: the false flag is amber, the missed target outlined, nothing green
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .2}, {p_survivor: .79}, {p_survivor: .1}], [2], 'practice', {target: 1})")
    tags = pg.evaluate("[...document.querySelectorAll('.node-tag')].map(t => [t.dataset.state, t.querySelector('b').textContent])")
    assert tags == [["missed", "Missed"], ["flagged", "Flagged, no survivor"], ["debris", "No target rhythm"]], tags
    assert pg.text_content("#v-label") == "Practice · Flagged, no survivor" and pg.evaluate("rescue.look().card") == "flagged", pg.text_content("#v-label")
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .8}, {p_survivor: .2}, {p_survivor: .9}], [3, 1], 'practice')")
    assert pg.inner_text("#v-sub") == "Not confirmed", pg.inner_text("#v-sub")
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .1}, {p_survivor: .2}, {p_survivor: .15}], [], 'practice')")
    assert pg.text_content("#v-label") == "Practice · No target rhythm heard", pg.text_content("#v-label")
    assert "practice dish (simulated)" in pg.inner_text("#prov"), pg.inner_text("#prov")
    # a pile the dish did not score: a neutral ink "No score" tag, no percentage, never flagged even if asked
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .91}, {p_survivor: null}, {p_survivor: .2}], [1, 2], 'live')")
    tags = pg.evaluate("[...document.querySelectorAll('.node-tag')].map(t => [t.dataset.state, t.textContent])")
    assert [t[0] for t in tags] == ["neutral", "idle", "debris"] and tags[1][1].startswith("No score") and "%" not in tags[1][1], tags
    assert pg.evaluate("rescue.state.pileVerdicts.flagged") == [1] and pg.text_content("#v-label") == "Pile 1 flagged", pg.text_content("#v-label")
    assert pg.inner_text("#v-chip") == "CL1 dish", pg.inner_text("#v-chip")
    pg.evaluate("rescue.setPileVerdicts([{p_survivor: .1}, {p_survivor: null}, {p_survivor: .2}], [], 'practice')")
    assert pg.inner_text("#v-sub") == "No pile flagged" and pg.evaluate("document.getElementById('v-note').hidden"), pg.inner_text("#v-sub")   # no note given, none shown
    assert pg.inner_text("#t-rot") == "stopped", pg.inner_text("#t-rot")
    # the status follows the source: a live run says the dish, never the practice one
    pg.evaluate("rescue.dropNodes(undefined, 'live')")
    pg.wait_for_function("rescue.state.nodes === 3", timeout=120_000)
    assert pg.inner_text("#v-sub") == "Knocks are sent to the living dish at base.", pg.inner_text("#v-sub")
    assert pg.evaluate("getComputedStyle(document.querySelector('.brand .dot')).backgroundColor") == "rgb(0, 212, 106)", "brand dot not green on the live dish"   # the check can go red
    # API guards, then the operator key as the default source
    for bad in ["rescue.setVerdict('maybe', .5)", "rescue.setVerdict('debris', 2)", "rescue.setVerdict('debris', .5, 'dish')",
                "rescue.setPileVerdicts([{p_survivor: 'x'}, {p_survivor: .1}, {p_survivor: .1}], [])", "rescue.listen(true, 'dish')",
                "rescue.listen(true, 'constructor')", "rescue.dropNode('toString')", "rescue.setVerdict('constructor', .5)",
                "rescue.setPileVerdicts([{p_survivor: .1}, {p_survivor: .1}, {p_survivor: .1}], [], 'practice', {target: 4})",
                "rescue.setPileVerdicts([{p_survivor: .1}, {p_survivor: .1}, {p_survivor: .1}], [], 'practice', {note: 55})"]:
        try:
            pg.evaluate(bad); errors.append("no throw: " + bad)
        except Exception:
            pass
    pg.evaluate("rescue.reset(); rescue.setVerdict('survivor', .9); rescue.markSpot()")
    assert pg.evaluate("rescue.state.marked"), "markSpot did not mark"
    pg.evaluate("rescue.setVerdict('unsure', .5)")
    pg.wait_for_timeout(500)
    assert not pg.evaluate("rescue.state.marked") and not pg.evaluate("rescue.state.pinVisible"), "unsure kept the old mark"
    pg.evaluate("rescue.reset(); rescue.setVerdict('unsure', .5)")
    assert pg.text_content("#v-label") == "Unclear", pg.text_content("#v-label")
    assert "operator key" in pg.inner_text("#prov") and pg.inner_text("#v-chip") == "Operator key", pg.inner_text("#prov")
    # calls sent before the drop (the spine's "Listen to pile N" while the drone is en route) are held, then applied
    pg.evaluate("rescue.reset()")
    try:
        pg.evaluate("rescue.setPileVerdicts([{p_survivor: .1}, {p_survivor: .1}], [])"); errors.append("no throw: two piles held")
    except Exception:
        pass
    pg.evaluate("rescue.listenPile(2); rescue.setPileVerdicts([{p_survivor: .2}, {p_survivor: .79}, {p_survivor: .1}], [2], 'practice', {target: null, note: 'held'})")
    assert pg.evaluate("rescue.state.pileVerdicts") is None, "applied before dropNodes"
    pg.evaluate("rescue.dropNodes(undefined, 'practice')")
    pg.wait_for_function("rescue.state.pileVerdicts !== null", timeout=120_000)
    assert pg.text_content("#v-label") == "Practice · Pile 2 flagged" and pg.inner_text("#v-note") == "held", pg.text_content("#v-label")
    assert pg.evaluate("document.querySelectorAll('[data-state=survivor]').length") == 0, "green at chance"
    rounds(pg)
    pg.evaluate("rescue.reset()")
    print("errors after story:", errors[:10])

    for c in b.contexts:  # one SwiftShader page at a time, or the fps page starves
        c.close()
    pg2 = b.new_page(viewport=DESK)
    pg2.goto(URL)
    pg2.wait_for_function("window.rescue && window.rescue.state.fps > 0", timeout=120_000)
    n = pg2.evaluate("""() => new Promise(res => { let n = 0; const t0 = performance.now();
        (function f(){ n++; performance.now() - t0 < 5000 ? requestAnimationFrame(f) : res(n); })(); })""")
    print(f"fps (headless SwiftShader, rAF over 5 s): {n / 5:.1f}")
    pg2.close()
    pg3 = b.new_page(viewport=DESK)
    pg3.on("pageerror", lambda e: errors.append(str(e)))
    pg3.goto(URL + "?lite=1&capture=1&t=24")
    pg3.wait_for_function("window.rescue && window.rescue.state.fps > 0", timeout=120_000)
    pg3.wait_for_timeout(3000)
    assert pg3.evaluate("rescue.state.lite") is True
    print("errors at end:", errors[:10])
    b.close()
sys.exit(1 if errors else 0)
