#!/usr/bin/env bash
# Offline smoke: spine tests on the simulated dish, then a headless load of the demo screen (live.py --rhythm)
# that fails on any request leaving the local server, any page error, or any failed request.
set -euo pipefail
cd "$(dirname "$0")"; KIT=$(pwd)
# SIM=<organisers' simulator checkout> runs the demo screen on their LIF simulator, the way the run card does.
echo "== spine pytest (cl_mock simulated dish)"
(cd biocomp-spine && uv run pytest -q)

echo "== kit wiring (empty HOME): ear finds ../rescue-data, night flags parse"
FAKEHOME=$(mktemp -d)
(cd biocomp-spine && HOME="$FAKEHOME" USERPROFILE="$FAKEHOME" .venv/bin/python -c '
import bci, ear
n = len(ear.index()); assert n > 0, "ear.index() is empty"
a = bci.parse_args(["--rhythm", "--short", "--cal", "20"]); assert a.short and a.cal == 20
print(f"ear.index(): {n} clips from {ear.DATA}; --short --cal 20 accepted")')
rm -rf "$FAKEHOME"

# A free port of its own, so a demo already running on 8777 is left alone.
PORT=$(biocomp-spine/.venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
echo "== demo screen headless load (live.py --rhythm --port $PORT, scene expected: $([ -d rescue-scene ] && echo yes || echo no))"
# live.py runs with an empty HOME so a developer's own copy of the scene in their home folder can't stand in for the kit's own scene.
# With rescue-scene/ in the kit the scene slot must show and the scene must load; without it the slot must stay hidden.
EXPECT_SCENE=$([ -d rescue-scene ] && echo 1 || echo 0)
FAKEHOME=$(mktemp -d)
SRVLOG=$(mktemp)
if [ -n "${SIM:-}" ]; then
  export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}   # keep uv's cache when HOME is swapped
  (cd "$SIM" && HOME="$FAKEHOME" USERPROFILE="$FAKEHOME" PYTHONUNBUFFERED=1 CL_SDK_VISUALISATION=0 PYTHONPATH="$KIT/biocomp-spine" \
    exec uv run --with scikit-learn python "$KIT/biocomp-spine/live.py" --rhythm --port "$PORT") >"$SRVLOG" 2>&1 &
else
  (cd biocomp-spine && HOME="$FAKEHOME" USERPROFILE="$FAKEHOME" PYTHONUNBUFFERED=1 exec .venv/bin/python live.py --rhythm --port "$PORT") >"$SRVLOG" 2>&1 &
fi
SRV=$!; trap 'kill $SRV 2>/dev/null || true; rm -rf "$FAKEHOME" "$SRVLOG"' EXIT
for _ in $(seq 180); do curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 1; done
kill -0 $SRV 2>/dev/null || { echo "live.py failed to start" >&2; tail -20 "$SRVLOG" >&2; exit 1; }
grep -m1 "^http://localhost" "$SRVLOG" || true
if [ -n "${SIM:-}" ]; then grep -q "cl-sdk simulator" "$SRVLOG" || { echo "not on the cl-sdk simulator:" >&2; tail -20 "$SRVLOG" >&2; exit 1; }; fi
uv run --no-project --with playwright python - "$PORT" "$EXPECT_SCENE" <<'PYW'
import sys
from playwright.sync_api import sync_playwright
expect_scene = sys.argv[2] == "1"
origins = (f"http://127.0.0.1:{sys.argv[1]}/", f"http://localhost:{sys.argv[1]}/")
bad, errors, seen = [], [], []
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page()
    def route(r):
        u = r.request.url
        if u.startswith(origins) or u.startswith(("data:", "blob:")):
            seen.append(u); r.continue_()
        else:
            bad.append(u); r.abort()
    pg.route("**/*", route)
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("requestfailed", lambda q: errors.append(f"failed: {q.url}"))
    pg.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status}: {r.url}"))
    pg.goto(origins[0], wait_until="load", timeout=60000)
    pg.wait_for_timeout(3000)   # the page polls /state; give it a few rounds
    title = pg.title()
    slot_hidden = pg.evaluate("(() => { const s = document.getElementById('scene-slot'); return !s || s.hidden; })()")
    scene_ok = False
    if expect_scene and not slot_hidden:
        # the scene is an iframe at /scene/index.html; it is ready when it exposes window.rescue
        fr = next((f for f in pg.frames if "/scene/" in f.url), None)
        for _ in range(30):
            if fr and fr.evaluate("typeof window.rescue === 'object'"):
                scene_ok = True; break
            pg.wait_for_timeout(1000)
    b.close()
print(f"local requests: {len(seen)}  title: {title!r}  scene slot hidden: {slot_hidden}  scene loaded: {scene_ok}")
for u in bad: print("OUTSIDE REQUEST:", u)
for e in errors: print(e)
scene_wrong = (slot_hidden or not scene_ok) if expect_scene else not slot_hidden
if bad or errors or not seen or scene_wrong:
    sys.exit(f"demo screen smoke FAILED (outside={len(bad)} errors={len(errors)} expect_scene={expect_scene} slot_hidden={slot_hidden} scene_loaded={scene_ok})")
print("demo screen smoke OK")
PYW
echo "== smoke OK"
