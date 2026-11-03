"""Live view: `uv run python live.py [--reps 12] [--replay [NAME]] [--rhythm] [--port 8777]`, open http://localhost:8777, press Start.
Stdlib server; the page polls /state. --rhythm serves Rescue Listener (rhythm.html): calibrate, then POST /knock
{"knocks": [ms, ...]} plays the judge's knocks into the dish and the verdict lands in /state "verdicts". On the mock (and replay) it runs at MOCK_SPEED x real time (default 5).
MOCK_SPEED is ignored by the real CL1."""
import collections
import hashlib
import json
import mimetypes
import os
import pickle
import re
import secrets
import threading
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("MOCK_SPEED", "5")
# the organisers' LIF simulator, when cl-sdk and organoid_simulator are installed (README: the sim run command); no seed
os.environ.setdefault("CL_SDK_DATA_SOURCE", "organoid_simulator.lif_data_source:make_lif_source")
os.environ.setdefault("CL_SDK_DATA_SOURCE_CONFIG", '{"coupling_gain_mV":4.0}')
import bci  # noqa: E402
import ear  # noqa: E402
import noisefilter  # noqa: E402
import piles as piles_mod  # noqa: E402
import probe  # noqa: E402
import rhythm  # noqa: E402

ARGS = bci.parse_args([]) if __name__ != "__main__" else bci.parse_args()
SPEED = float(os.environ["MOCK_SPEED"])
CORNERS = (0, 7, 56, 63)
REPLAY_CELLS = [c for c in range(64) if c not in CORNERS]  # shortcut: their channel c drawn at the c-th non-corner cell, not the true MEA layout


def _source():
    if ARGS.replay is not None:
        return dict(source="replay", pill="Replay · rat cortex, Wagenaar 2006", trial_s=bci.REPLAY_TRIAL_S,
                    off=list(CORNERS), array="MEA · 60 electrodes")
    src = bci.source()
    if src == "mock":
        what = "network off (control)" if os.environ.get("NO_CELLS") == "1" else "not the LIF network"
        pill = f"Practice simulator · {what} · {SPEED:g}× speed"
    else:
        pill = "CL1 · live neurons" if src == "CL1" else "Simulator running live · LIF spiking network"
    return dict(source=src, pill=pill, cells_off=src == "mock" and os.environ.get("NO_CELLS") == "1", trial_s=bci.TRIAL_S, off=list(bci.EXCLUDED), array="CL1 · 64 electrodes")


def no_echo(live):
    """True once the whole first block is in and every trial in it is flagged for missing stim echoes: the device
    likely reports no echoes at all, so the screen explains the flags instead of showing 120 mystery ones."""
    first = [t for t in live if t["i"] < 4]
    return len(first) == 4 and all("no echo" in (t["flag"] or "") for t in first)


def _fresh():
    if ARGS.rhythm:
        return dict(_source(), mode="rhythm", listen=_listen[0], running=False, finished=False, phase="idle", cal_done=0,
                    cal_total=2 * (ear.N_CAL if _listen[0] == "ear" else ARGS.cal), calib=None, verdicts=[],
                    cal_s=None if _listen[0] == "ear" else rhythm.cal_duration_s(rhythm.calibration(ARGS.seed, ARGS.cal)[0]),
                    pending=0, dropped=0, drops=[], skip_ms=bci.SKIP_MS, snr=ear_snr(), sound=_sound_ok(),
                    input=rhythm.IN_CH, read=rhythm.READ, picked=_rh.get("picked"), scan_done=0, scan_total=0,
                    piles=[], flagged=None, software_flagged=None, piles_phase=None, targets=None, outcome=None, tally=None, software_tally=None,
                    scene=os.path.isfile(os.path.join(SCENE_DIR, "index.html")),
                    history=dict(bins=GAP_BINS, series=_hist, current=None), probe=_rh.get("probe_result"), probe_progress=None, probe_design=probe.D, probe_live=_rh.get("probe_live", []), probe_no_echo=no_echo(_rh.get("probe_live", [])),
                    probe_trials=4 * probe.D["blocks_short" if ARGS.short else "blocks"],
                    nf=_rh.get("nf_result"), nf_live=_rh.get("nf_live", []), nf_running=False,
                    nf_plan=dict(n=noisefilter.N_STIM, isi_ms=noisefilter.ISI_MS, rest_s=noisefilter.REST_S,
                                 duration_s=noisefilter.duration_s()))
    return dict(mode="scaling", **_source(), running=False, finished=False, trials=0, target=sum(bci.STAGES) * ARGS.reps, stage=None,
                stage_trial=0, stage_total=0, alphabet=[], stim=None, stim_seq=0, grid=[0] * 64,
                res={}, reps=ARGS.reps, skip_ms=bci.SKIP_MS)


SCENE_DIR = os.path.realpath(os.path.expanduser(os.environ.get(  # the kit puts the scene beside the spine
    "RESCUE_SCENE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rescue-scene"))))
mimetypes.add_type("text/javascript", ".js")  # Windows' registry can say text/plain, and Chrome refuses module scripts
GAP_BINS = [60, 100, 150, 250, 400, 600, 1000, 4000]  # ms before the pulse
_rh = dict(F=[], T=[], TR=[], model=None, timing_model=None, next_id=0, picked=None)
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json"), encoding="utf-8") as _f:
        _hist = json.load(_f)  # earlier runs' series (e.g. the simulator check), shown beside this one
except (OSError, ValueError):
    _hist = {}
_clips = {}  # opaque clip id -> wav path: rescue-data ids (sNNN/dNNN) carry the class, so they never leave the server
_listen, _cal, _stop, _thread = ["knock"], {}, [threading.Event()], [None]  # mode; calibration kept per mode


def ear_snr():
    return os.environ.get("EAR_SNR", "10")


def _sound_ok():
    """Sound needs the rendered WAVs, which are gitignored in rescue-data: a kit without them can't offer it."""
    try:
        return all(os.path.isfile(t[2]) for t in ear.calibration(ear_snr()))
    except (OSError, ValueError, KeyError):
        return False
_ear_rng = bci.np.random.default_rng()
STATE = _fresh()
_recent, _lock, _busy, _pool = [], threading.Lock(), [False], []


def _on_bin(chbin):
    _recent.append(chbin)
    del _recent[:-3]
    with _lock:
        STATE["grid"] = [float(v) for v in sum(_recent)]


def _set(N, r, done):
    with _lock:
        if STATE["res"].get(str(N), {}).get("done") and not done:
            return  # a late quick estimate never overwrites the stage's final number
        if r:
            STATE["res"][str(N)] = dict(live=r["live"], ctrl=r["ctrl"], pre=r["pre"], trap=r["trap"], acc=r["acc"], n=r["n"], done=done)


def _eval_done(fut, N, done):
    try:
        _set(N, fut.result(), done)
    finally:
        if not done:
            _busy[0] = False


def _eval(d, N, done):
    # a separate process, not a thread: a forest training in-process holds the GIL and pushed tick p99 to 3.5 ms
    if not _pool:
        _pool.append(ProcessPoolExecutor(1))
    fut = _pool[0].submit(bci.evaluate_stage, d, N, bci.TREES if done else 40)
    fut.add_done_callback(lambda f: _eval_done(f, N, done))


def _on_trial(n, out, b):
    N = b["N"]
    cell = REPLAY_CELLS[b["electrode"]] if ARGS.replay is not None else b["electrode"]
    alphabet = [REPLAY_CELLS[int(e)] for e in b["alphabet"]] if "alphabet" in b else bci.ALPHABET
    with _lock:
        STATE.update(trials=n, stage=N, stage_trial=b["stage_trial"], stage_total=b["stage_total"], stim=cell,
                     reps=b["stage_total"] // N,
                     stim_seq=STATE["stim_seq"] + 1, alphabet=alphabet[:N])
        if "counts" in b:
            STATE["grid"] = [0.0] * 64
            for c, v in enumerate(b["counts"]):
                STATE["grid"][REPLAY_CELLS[c]] = float(v)
    if b["last_in_stage"] or not _busy[0]:
        d = {k: bci.np.array(out[k]) for k in ("X", "pre", "trap", "y", "stage")}  # snapshot; decode off the loop process
        if not b["last_in_stage"]:
            _busy[0] = True
        _eval(d, N, b["last_in_stage"])


def _run():
    try:
        if ARGS.replay is not None:
            bci.replay_session(ARGS.replay or None, ARGS.reps, SPEED, on_trial=_on_trial)
        else:
            with bci.cl.open() as n:
                bci.run_session(n, ARGS.reps, ARGS.seed, on_bin=_on_bin, on_trial=_on_trial)
    finally:
        with _lock:
            STATE.update(running=False, finished=True)


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rhythm.html" if ARGS.rhythm else "live.html"), encoding="utf-8").read()


# ---- Rescue Listener (--rhythm) -----------------------------------------------------------------
_knocks = collections.deque()


def _pool_submit(fn, *a):
    if not _pool:
        _pool.append(ProcessPoolExecutor(1))
    return _pool[0].submit(fn, *a)


def _calibrated(fut):
    r = fut.result()
    _rh.update(model=r.pop("model"), timing_model=r.pop("timing_model"))
    if _listen[0] == "knock":  # the blind-pile thresholds, frozen before any pile is heard
        e = probe.freeze("knock", dict(experiment="knock (whole-sequence, demo)", threshold=r["threshold"],
                                       software_threshold=r["software_threshold"], n_cal=r["n"], cv_acc=r["acc"],
                                       model_sha256=hashlib.sha256(pickle.dumps(_rh["model"])).hexdigest(),
                                       input=rhythm.IN_CH, read=list(rhythm.READ), skip_ms=bci.SKIP_MS, seed=ARGS.seed))
        r["frozen"] = dict(sha256=e["sha256"], frozen_at=e["frozen_at"])
    _cal[_listen[0]] = dict(model=_rh["model"], timing_model=_rh["timing_model"], calib=r)
    with _lock:
        STATE.update(calib=r, phase="ready")


def _p_survivor(v):
    return round(v["confidence"] if v["verdict"] == "survivor" else 1 - v["confidence"], 3)


def _pile_heard(n, entry):
    """Blind piles: hold each pile's score privately; nothing is published until the judge names the targets."""
    with _lock:
        _rh["piles"][n] = entry
        if len(_rh["piles"]) == 3:
            STATE["piles_phase"] = "heard"


def _reveal(body):
    """{"targets": [pile, ...]} (0-3 piles, the judge's secret): publish every pile's score, the flags and the tally."""
    try:
        targets = json.loads(body)["targets"]
        assert isinstance(targets, list) and set(targets) <= {1, 2, 3} and len(set(targets)) == len(targets)
    except (ValueError, TypeError, KeyError, AssertionError):
        return 400, dict(error='send {"targets": [pile, ...]}, 0-3 of 1, 2, 3')
    fz = probe.load().get("knock", {}).get("config")
    if not fz:
        return 409, dict(error="no frozen knock threshold: calibrate again")
    with _lock:
        if STATE.get("piles_phase") != "heard":
            return 409, dict(error="not all three piles heard yet")
        piles = [_rh["piles"][k] for k in (1, 2, 3)]
        fl = piles_mod.flags([p["p_survivor"] for p in piles], fz["threshold"])
        sf = piles_mod.flags([p["software_p_survivor"] for p in piles], fz["software_threshold"])
        STATE.update(piles=piles, piles_phase="done", targets=sorted(targets), flagged=fl, software_flagged=sf,
                     threshold=fz["threshold"], software_threshold=fz["software_threshold"],
                     tally=piles_mod.tally(fl, targets), software_tally=piles_mod.tally(sf, targets),
                     outcome=piles_mod.outcome([p["p_survivor"] for p in piles], fl))
    return 200, dict(piles_phase="done")


def _judge_verdict(f, tf, meta):
    dish = _pool_submit(rhythm.verdict, _rh["model"], f).result()  # forest predicts off the tick process
    soft = rhythm.verdict(_rh["timing_model"], tf)
    if meta.get("pile"):
        _pile_heard(meta["pile"], dict(id=meta["pile"], taps=meta["knocks"], p_survivor=_p_survivor(dish),
                                       dish_verdict=dish["verdict"], software_verdict=soft["verdict"],
                                       software_p_survivor=_p_survivor(soft)))
        with _lock:
            STATE["pending"] -= 1
        return
    v = dict(id=meta["id"], knocks=meta["knocks"], verdict=dish["verdict"], confidence=dish["confidence"],
             timing_only=soft, truth=meta.get("truth"), clip=meta.get("clip"), bed=meta.get("bed"))
    with _lock:
        STATE["verdicts"] = (STATE["verdicts"] + [v])[-20:]
        STATE.update(pending=STATE["pending"] - 1, latest=v)


HISTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")


def _series_name():
    src = _source()
    return "Live: CL1" if src["source"] == "CL1" else (
        "Practice simulator · network off (control)" if os.environ.get("NO_CELLS") == "1" else "Practice simulator (not the LIF network)") \
        if src["source"] == "mock" else src["pill"]


def _history(meta):
    """Response size (spikes 10-50 ms after the pulse, read set) against the gap before it, binned: does the
    response depend on recent history? Every run adds to its own series in history.json; the screen shows them all."""
    g, c = meta.get("gaps_ms"), meta.get("pulse_counts")
    if not g:
        return
    k = bci.np.clip(bci.np.searchsorted(GAP_BINS, g, side="right") - 1, 0, len(GAP_BINS) - 2)
    with _lock:
        ser = _hist.setdefault(_series_name(), dict(sum=[0.0] * (len(GAP_BINS) - 1), n=[0] * (len(GAP_BINS) - 1)))
        for b, v in zip(k, c[1:]):
            ser["sum"][b] += float(v)
            ser["n"][b] += 1
        STATE["history"] = dict(bins=GAP_BINS, series=_hist, current=_series_name())
        try:
            with open(HISTORY, "w", encoding="utf-8") as f:
                json.dump(_hist, f)
        except OSError:
            pass


def _on_seq(f, tf, trap, meta):
    _history(meta)
    if meta["kind"] == "judge":
        threading.Thread(target=_judge_verdict, args=(f, tf, meta), daemon=True).start()
        return
    _rh["F"].append(f)
    _rh["T"].append(tf)
    _rh["TR"].append(trap)
    _rh["kept"].append(meta["i"])
    _cal_check()


def _cal_check():
    with _lock:
        STATE["cal_done"] = len(_rh["F"]) + STATE["dropped"]
        if STATE["cal_done"] < STATE["cal_total"]:
            return
        STATE["phase"] = "fitting"
    y = _rh["y"][_rh["kept"]]  # dropped sequences leave the tally
    _pool_submit(rhythm.evaluate, bci.np.array(_rh["F"]), bci.np.array(_rh["T"]), y, ARGS.seed, 20,
                 bci.np.array(_rh["TR"])).add_done_callback(_calibrated)


def _on_drop(meta, why):
    """A sequence the loop could not play whole (skipped ticks, a late pulse): never scored, always counted."""
    with _lock:
        STATE["dropped"] += 1
        STATE["drops"] = (STATE["drops"] + [why])[-5:]
        if meta["kind"] == "judge":
            STATE["pending"] -= 1
    if meta["kind"] == "cal":
        _cal_check()
    elif meta.get("pile"):
        _pile_heard(meta["pile"], dict(id=meta["pile"], taps=meta["knocks"], p_survivor=None, dish_verdict="dropped: " + why,
                                       software_verdict="dropped: " + why, software_p_survivor=None))


def _run_rhythm():
    is_ear, cached = _listen[0] == "ear", _cal.get(_listen[0])
    _rh.update(F=[], T=[], TR=[], kept=[], model=None, timing_model=None)
    try:  # the WAV load too: a failure reports itself and clears running
        if cached:  # this mode was calibrated before a switch: listen straight away
            seqs = []
            _rh.update(model=cached["model"], timing_model=cached["timing_model"])
            with _lock:
                STATE.update(calib=cached["calib"], phase="ready", cal_done=STATE["cal_total"])
        else:
            if is_ear:
                cal = ear.calibration(ear_snr())
                seqs, _rh["y"] = [ear.events(*ear.load_wav(t[2])) for t in cal], bci.np.array([t[1] for t in cal])
            else:
                seqs, _rh["y"] = rhythm.calibration(ARGS.seed, ARGS.cal)
            with _lock:
                STATE.update(phase="calibrating", cal_total=len(seqs))
        with bci.cl.open() as n:  # runs until a mode switch sets _stop: the dish keeps listening
            rhythm.run_rhythm(n, seqs, on_seq=_on_seq, more=lambda: _knocks.popleft() if _knocks else None,
                              stop=_stop[0], read_set=ear.READ_SET if is_ear else None, on_drop=_on_drop)
    except Exception as e:
        with _lock:
            STATE.update(phase="idle", error=f"Calibration failed: {type(e).__name__}: {e}"[:300])
        raise
    finally:
        with _lock:
            STATE.update(running=False, finished=True)


def _scan():
    """30 s response scan: picks the input electrode and its readout before calibration."""
    def progress(i, n):
        with _lock:
            STATE.update(scan_done=i, scan_total=n)
    try:
        with bci.cl.open() as n:
            r = rhythm.scan(n, on_progress=progress)
        rhythm.set_input(r["input"], r["read"])
        _rh["picked"] = r
        with _lock:
            STATE.update(input=rhythm.IN_CH, read=rhythm.READ, picked=r)
    finally:
        with _lock:
            STATE.update(running=False, phase="idle")


def _probe():
    """The pre-registered probe-omission experiment: label-blind scan, freeze, the trials. Before knock calibration."""
    live = _rh["probe_live"] = []

    def progress(stage, done, total):
        with _lock:
            STATE["probe_progress"] = dict(stage=stage, done=done, total=total)

    def trial(rec):
        with _lock:
            live.append(dict(i=rec["i"], cond=rec["cond"], raster=rec["raster"], flag=rec["flag"], win=rec["win"]))
            STATE["probe_live"] = list(live)
            STATE["probe_no_echo"] = no_echo(live)
    try:
        with bci.cl.open() as n:
            r = probe.run(n, ARGS.seed, probe.D["blocks_short" if ARGS.short else "blocks"], on_progress=progress, on_trial=trial)
        _rh["picked"] = dict(input=r["input"], read=r["read"])
    except Exception as e:  # shown, never swallowed: a failed run is a result too
        r = dict(error=f"{type(e).__name__}: {e}"[:300])
    _rh["probe_result"] = r  # kept across Calibrate and mode switches, which reset STATE
    with _lock:
        STATE.update(probe=r, running=False, phase="idle", probe_progress=None, input=rhythm.IN_CH, read=rhythm.READ,
                     picked=_rh.get("picked"))


def _noisefilter():
    """The exploratory noise-filter run (noisefilter.py): a flip-flop oddball on the scan's two strongest drivers."""
    live = _rh["nf_live"] = []
    a, b = noisefilter.electrodes()
    print(f"noise filter: planned chip time {noisefilter.duration_s():.0f} s on electrodes {a} and {b}", flush=True)

    def stim(rec):
        with _lock:
            live.append(dict(i=rec["i"], cond=rec["cond"], ch=rec["ch"], rare=rec["rare"], y=rec["y"], flag=rec["flag"]))
            STATE["nf_live"] = list(live)
    try:
        with bci.cl.open() as n:
            r = noisefilter.run(n, ARGS.seed, on_stim=stim, a=a, b=b)
    except Exception as e:  # shown, never swallowed
        r = dict(error=f"{type(e).__name__}: {e}"[:300])
    _rh["nf_result"] = r
    with _lock:
        STATE.update(nf=r, running=False, nf_running=False, phase="idle")


def _queue(t, meta):
    with _lock:
        _rh["next_id"] += 1
        meta["id"] = _rh["next_id"]
        STATE["pending"] += 1
    _knocks.append((t, meta))
    return 200, dict(id=meta["id"], knocks=meta["knocks"], clip=meta.get("clip"))


def _hear(body):
    """One random rescue-data clip, optionally {"pattern": "five_tap_reply"} (any sequences.json pattern).
    Ear mode plays the sound into the dish (never a calibration clip); knock mode plays its knock times."""
    if STATE.get("phase") != "ready":
        return 409, dict(error="calibrate first")
    if _blind():
        return 409, dict(error="a blind round is under way")
    try:
        pattern = json.loads(body).get("pattern") if body else None
    except (ValueError, AttributeError):
        return 400, dict(error='send {} or {"pattern": "five_tap_reply"}')
    idx = ear.index()
    seen = {t[0] for t in ear.calibration(ear_snr())} if _listen[0] == "ear" else set()
    pool = [t for t in ear.trials(ear_snr()) if t[0] not in seen and pattern in (None, idx[t[0]]["pattern"])]
    if not pool:
        return 400, dict(error=f"no clips with pattern {str(pattern)[:40]!r}")
    tid, y, path, bed = pool[_ear_rng.integers(len(pool))]  # not rng.choice: it turns the tuples into strings
    cid = secrets.token_hex(8)
    _clips[cid] = path
    meta = dict(kind="judge", clip=cid, bed=bed, truth=rhythm.LABELS[y])
    if _listen[0] == "ear":
        ev = ear.events(*ear.load_wav(path))
        return _queue(ev, dict(meta, knocks=[round(t * 1000) for t, _ in ev]))
    t = rhythm.clean(idx[tid]["knock_times_ms"])
    return _queue(t, dict(meta, knocks=[round(v * 1000) for v in t]))


def _switch(body):
    """{"listen": "knock" | "ear"}: stop the dish loop, keep each mode's calibration, resume the other mode."""
    try:
        to = json.loads(body)["listen"]
    except (ValueError, TypeError, KeyError):
        to = None
    if to not in ("knock", "ear"):
        return 400, dict(error="listen: knock or ear")
    if _blind():
        return 409, dict(error="a blind round is under way")
    if STATE["phase"] in ("scanning", "probe", "noisefilter", "calibrating", "fitting") or STATE["pending"]:  # they own the dish
        return 409, dict(error="busy: wait for the dish")
    if to == "ear" and not STATE.get("sound"):
        return 409, dict(error="Sound clips not in this kit")
    if to != _listen[0]:
        _stop[0].set()
        if _thread[0]:
            _thread[0].join(10)
        _stop[0], _listen[0] = threading.Event(), to
        _knocks.clear()
        with _lock:
            STATE.clear()
            STATE.update(_fresh())
        if to in _cal:
            _start()
    return 200, dict(listen=to)


def _start():
    with _lock:
        STATE.clear()
        STATE.update(_fresh(), running=True)
    _busy[0] = False
    _thread[0] = threading.Thread(target=_run_rhythm if ARGS.rhythm else _run, daemon=True)
    _thread[0].start()


def _knock(body):
    """Judge input (untrusted): {"knocks": [ms, ...]} -> queued for the dish, or an error string.
    With "pile": N during a blind round, it is that pile's window (fewer than 3 taps: P(survivor) 0)."""
    if STATE.get("phase") != "ready" or _listen[0] != "knock":
        return 409, dict(error="calibrate first")
    try:
        d = json.loads(body)
        knocks, pile = d["knocks"], d.get("pile")
        t = rhythm.clean(knocks[:200]) if isinstance(knocks, list) else []
    except (ValueError, TypeError, KeyError, AttributeError):
        return 400, dict(error="send {\"knocks\": [ms, ...]}")
    if pile is not None:
        if pile not in (1, 2, 3) or STATE.get("piles_phase") != f"listening:{pile}":
            return 409, dict(error=f"not listening to pile {str(pile)[:8]}")
        with _lock:
            STATE["piles_phase"] = f"listening:{pile + 1}" if pile < 3 else "judging"
        ms = [round(v * 1000) for v in t]
        if len(t) < 3:
            _pile_heard(pile, dict(id=pile, taps=ms, p_survivor=None, dish_verdict="no response: too few taps",
                                   software_verdict="no response: too few taps", software_p_survivor=None))
            return 200, dict(pile=pile, knocks=ms)
        return _queue(t, dict(kind="judge", knocks=ms, pile=pile))
    if _blind():
        return 409, dict(error="a blind round is under way: taps go to a pile")
    if len(t) < 3:
        return 400, dict(error="at least 3 knocks")
    return _queue(t, dict(kind="judge", knocks=[round(v * 1000) for v in t]))


def _blind():
    return STATE.get("piles_phase") not in (None, "done")


def _piles():
    """Start a blind round: three windows, the judge picks the survivor's pile in secret."""
    if STATE.get("phase") != "ready" or _listen[0] != "knock":
        return 409, dict(error="calibrate knock mode first")
    with _lock:
        if STATE["piles_phase"] not in (None, "done"):
            return 409, dict(error="a blind round is under way")
        _rh["piles"] = {}
        STATE.update(piles=[], flagged=None, software_flagged=None, targets=None, outcome=None, tally=None, software_tally=None,
                     piles_phase="listening:1")
    return 200, dict(piles_phase="listening:1")


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path == "/state":
            with _lock:
                self._send(json.dumps(STATE), "application/json")
        elif self.path.startswith("/scene/"):  # the 3D rescue scene, served from its own folder, same origin
            p = os.path.realpath(os.path.join(SCENE_DIR, self.path[7:].split("?")[0]))
            if not p.startswith(SCENE_DIR + os.sep) or not os.path.isfile(p):
                return self._send("not found", "text/plain", 404)
            with open(p, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(p)[0] or "application/octet-stream")
            self.end_headers()
            self.wfile.write(body)
        elif re.fullmatch(r"/wav/[0-9a-f]{16}", self.path):  # the clip the dish hears, for the room too: opaque id
            p = _clips.get(self.path[5:])
            if not p or not os.path.isfile(p):
                return self._send("not found", "text/plain", 404)
            with open(p, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/static/") and ".." not in self.path:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.path[1:]), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "font/woff2" if self.path.endswith(".woff2") else "text/plain")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path in ("/knock", "/mode", "/hear", "/piles", "/reveal") and ARGS.rhythm:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if 0 < n <= 20_000 else b""
            code, out = (_knock(body) if self.path == "/knock" else _switch(body) if self.path == "/mode"
                         else _piles() if self.path == "/piles" else _reveal(body) if self.path == "/reveal" else _hear(body))
            return self._send(json.dumps(out), "application/json", code)
        if self.path == "/scan" and ARGS.rhythm and _listen[0] == "knock" and not STATE["running"] and STATE["phase"] == "idle":
            with _lock:
                STATE.update(running=True, phase="scanning")
            threading.Thread(target=_scan, daemon=True).start()
        elif self.path == "/probe" and ARGS.rhythm and _listen[0] == "knock" and not STATE["running"] and STATE["phase"] == "idle":
            with _lock:
                STATE.update(running=True, phase="probe", probe=None, probe_live=[], probe_no_echo=False)
            threading.Thread(target=_probe, daemon=True).start()
        elif self.path == "/noisefilter" and ARGS.rhythm and _listen[0] == "knock" and not STATE["running"] and STATE["phase"] == "idle":
            with _lock:
                a, b = noisefilter.electrodes()
                STATE.update(running=True, nf_running=True, phase="noisefilter", nf=None, nf_live=[],
                             nf_plan=dict(STATE["nf_plan"], a=a, b=b))
            threading.Thread(target=_noisefilter, daemon=True).start()
        elif self.path == "/start" and not STATE["running"]:
            _start()
        self._send("{}", "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"http://localhost:{ARGS.port}  ({STATE['pill']})")
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), H).serve_forever()
