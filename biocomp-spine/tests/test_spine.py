import os
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bci


def _run(monkeypatch, no_cells, reps):
    if no_cells:
        monkeypatch.setenv("NO_CELLS", "1")
    else:
        monkeypatch.delenv("NO_CELLS", raising=False)
    with bci.cl.open() as n:
        d = bci.run_session(n, reps, seed=0)
    r = bci.evaluate_stage(d, 16)
    r["tick_s"], r["tick_cpu_s"] = d["tick_s"], d["tick_cpu_s"]
    return r


def _tick_fast(r):
    """The tick body only reads, buffers and stims: well under the 1 ms tick period. Its p99 is checked on the body's
    own CPU time: wall p99 is the scheduler's when other jobs share the VM (1.09 ms at load ~7). Wall median too."""
    cpu, wall = np.asarray(r["tick_cpu_s"]) * 1000, np.asarray(r["tick_s"]) * 1000
    assert np.percentile(cpu, 99) < 0.5, np.percentile(cpu, 99)
    assert np.median(wall) < 0.5, np.median(wall)


def test_scaling_on_mock_rises_above_the_control(monkeypatch):
    r = _run(monkeypatch, False, 12)   # 12 not the default 20: keeps the suite near a minute
    assert r["n"] == 16 * 12, r
    assert r["live"] > 1.0 and r["live"] > r["ctrl"] + 0.8, r   # stage 16: well clear of the shuffled control
    assert abs(r["pre"]) < 0.3, r                               # the window before the pulse carries nothing
    _tick_fast(r)


def test_no_cells_sits_on_the_floor_but_springs_the_artefact_trap(monkeypatch):
    r = _run(monkeypatch, True, 6)   # fewer reps: the floor needs no precision, and it keeps the suite short
    assert abs(r["live"]) < 0.3, r   # honest window (from SKIP_MS after the pulse): nothing without cells
    assert r["trap"] > 1.5, r        # a window that includes the stimulation decodes the artefact alone


def test_mutual_information_bounds():
    y = np.repeat(np.arange(4), 10)
    assert abs(bci.mi_bits(y, y, 4) - 2.0) < 1e-9          # a perfect reader gets log2 N
    assert bci.mi_bits(y, np.zeros_like(y), 4) == 0.0       # a constant guess carries nothing


import rhythm  # noqa: E402


def _rhythm(monkeypatch, no_cells, n):
    if no_cells:
        monkeypatch.setenv("NO_CELLS", "1")
    else:
        monkeypatch.delenv("NO_CELLS", raising=False)
    with bci.cl.open() as neurons:
        return rhythm.calibrate(neurons, seed=0, n=n)


def test_rhythm_beats_chance_on_the_memory_mock(monkeypatch):
    # SOS/motif draws only (seed 0: 68%). One seed of 80 trials swings 50-72% on the mock (seeds 0-3 with five-tap:
    # 55/68/50/72%), so this guards the pipeline, not the pattern mix.
    monkeypatch.setattr(rhythm, "FIVE_TAP", False)
    r = _rhythm(monkeypatch, False, rhythm.N_CAL)
    assert r["acc"] >= 0.6 and r["p"] < 0.05, rhythm.fmt(r)
    assert abs(r["shuffled"] - 0.5) < 0.1 and 0 < r["shuffled_sd"] < 0.2, rhythm.fmt(r)   # the spread the screen's verdict uses
    assert r["timing_only"] > 0.9, rhythm.fmt(r)                 # software reads timing: shown, never hidden
    _tick_fast(r)


def test_rhythm_no_cells_sits_near_chance_but_springs_the_trap(monkeypatch):
    r = _rhythm(monkeypatch, True, 20)   # fewer: chance needs no precision
    assert abs(r["acc"] - 0.5) < 0.2, rhythm.fmt(r)
    assert r["trap"] > 0.85, rhythm.fmt(r)   # a trace that includes the stimulation reads the knocks themselves


def test_scan_picks_an_input_that_drives_its_readout(monkeypatch):
    monkeypatch.delenv("NO_CELLS", raising=False)
    with bci.cl.open() as neurons:
        r = rhythm.scan(neurons, seconds=10)
    assert r["input"] in bci.USABLE and r["input"] not in r["read"] and len(r["read"]) == rhythm.N_READ, r
    assert r["drive"] > 0, r
    old = (rhythm.IN_CH, rhythm.READ)
    rhythm.set_input(r["input"], r["read"])
    assert tuple(rhythm.INPUT) == (r["input"],) and rhythm.READ_SET == set(r["read"])
    rhythm.set_input(*old)


def test_judge_knocks_are_cleaned():
    t = rhythm.clean([300, 0, 20, float("nan"), 9e9, 500])
    assert list(np.round(t * 1000)) == [0, 300, 500]   # sorted, the 20 ms bounce merged, nan and the far knock dropped
    assert len(rhythm.clean(range(0, 100_000, 100))) == rhythm.MAX_KNOCKS


# ---- real-SDK surface: the mock enforces the CL1's limits, the adapter checks them again --------------------
import pytest  # noqa: E402

import adapter  # noqa: E402
import artefact  # noqa: E402
import cl_mock  # noqa: E402


@pytest.mark.parametrize("phases", [(160, -3.5, 160, 3.5),     # over 3 uA
                                    (1200, -2.9, 1200, 2.9),   # over 3 nC per phase
                                    (160, -2.0),               # monophasic
                                    (160, -2.0, 160, -2.0),    # not alternating
                                    (150, -2.0, 150, 2.0),     # not a multiple of 20 us
                                    (160, float("nan"), 160, 2.0),   # NaN current passes every comparison
                                    (float("nan"), -2.0, 160, 2.0)])  # NaN duration
def test_unsafe_designs_raise_in_the_mock_and_the_adapter(phases):
    with pytest.raises(ValueError):
        cl_mock.StimDesign(*phases)
    with pytest.raises(ValueError):
        adapter.StimDesign(*phases)


def test_mock_refuses_bad_channels_rates_and_lead_and_positional_read():
    n = cl_mock.Neurons(no_cells=True)
    d = cl_mock.StimDesign(160, -2.0, 160, 2.0)
    for ch in (0, 4, 7, 56, 63):
        with pytest.raises(ValueError):
            n.stim(cl_mock.ChannelSet(ch), d)
    with pytest.raises(ValueError):
        n.stim(cl_mock.ChannelSet(27), d, None, 40)                 # lead time under 80 us
    with pytest.raises(ValueError):
        cl_mock.BurstDesign(2, 250)                                 # burst over 200 Hz
    n.stim(cl_mock.ChannelSet(27), d)
    with pytest.raises(cl_mock.ChannelQueueFull):
        n.stim(cl_mock.ChannelSet(27), d)                           # same frame: over 200 Hz on channel 27
    with pytest.raises(TypeError):
        n.read(25, from_timestamp=0)                                # the bci.py:90 bug: positional-only
    assert n.read(25, 0).dtype == np.int16


def test_adapter_refuses_reference_and_rate_before_the_device_sees_it():
    with adapter.open() as nr:
        p = nr.create_stim_plan()
        with pytest.raises(ValueError):
            p.stim(adapter.ChannelSet(4), bci.STIM)
        for tick in nr.loop(1000):
            p = nr.create_stim_plan()
            p.stim(adapter.ChannelSet(27), bci.STIM)
            t = p.run()
            p = nr.create_stim_plan()
            p.stim(adapter.ChannelSet(27), bci.STIM)
            with pytest.raises(ValueError):
                p.run()                                             # second pulse in the same tick: over 200 Hz
            break
        for tick in nr.loop(1000):                                  # the echo lands where run() said
            if tick.analysis.stims:
                assert adapter.stim_time(tick, (27,)) == t
                assert abs(tick.frames).max() > 100                 # uV: the artefact, scaled from int16
                break


def test_adapter_recovers_from_jitter():
    class Loop:
        def __init__(self):
            self.recovered, self.i = 0, 0

        def recover_from_jitter(self, callback=None, timeout_seconds=5.0):
            self.recovered += 1

        def __iter__(self):
            while self.i < 5:
                self.i += 1
                if self.i == 3 and not self.recovered:
                    raise TimeoutError("tick overrun")
                yield cl_mock._Tick(self, self.i, self.i * 25, self.i * 25 + 25, [], [], np.zeros((25, 64), np.int16))

    lp = Loop()
    nr = adapter.Neurons(type("N", (), {"loop": lambda self, *a, **k: lp})())
    ticks = [t.timestamp for t in nr.loop(1000)]
    assert lp.recovered == 1 and nr.jitter_recoveries == 1 and ticks == [25, 50, 100, 125], ticks


def test_artefact_probe_suggests_skip_and_can_go_red(monkeypatch):
    monkeypatch.delenv("NO_CELLS", raising=False)
    with bci.cl.open() as n:
        r = artefact.measure(n, 27, 10)
    assert 1 <= r["end_ms"] < 8 and r["skip_ms"] == 10, r          # mock artefact: 40 sigma, 2 ms decay
    monkeypatch.setattr(cl_mock, "ARTEFACT_TAU", 0.01)
    with bci.cl.open() as n:
        r = artefact.measure(n, 27, 10)
    assert r["skip_ms"] >= 20, r                                    # a long artefact moves the suggestion


def test_ear_events_are_tonotopic_and_capped():
    import ear
    tid, y, path, bed = ear.trials("10")[0]
    if not os.path.exists(path):   # the rendered WAVs are gitignored; the night's kit ships without them
        pytest.skip(f"rendered WAV missing ({path}); render rescue-data's sequences to run ear tests")
    ev = ear.events(*ear.load_wav(path))
    assert len(ev) >= 2
    for t, cs in ev:
        chs = list(cs)
        assert 1 <= len(chs) <= ear.MAX_PULSES and set(chs) <= set(ear.BAND_CH)
        assert abs(t / ear.FRAME_S - round(t / ear.FRAME_S)) < 1e-9
    assert not set(ear.BAND_CH) & set(ear.READ)


def test_five_tap_reply_is_a_survivor_pattern():
    rng = np.random.default_rng(0)
    fives = [s for s in (rhythm.survivor(rng) for _ in range(300)) if len(s) == 5]
    assert 60 < len(fives) < 140                                   # about a third
    g = np.diff(fives, axis=1)
    assert g.min() > 0.4 * 0.7 and g.max() < 0.7 * 1.3 and (g.std(1) / g.mean(1)).max() < 0.25


def test_blind_piles_taps_and_flag():
    import piles
    rng = np.random.default_rng(0)
    h = [piles.human_taps(rng) for _ in range(500)]
    g = np.concatenate([np.diff(t) for t in h])
    assert {len(t) for t in h} == {4, 5, 6, 7} and g.min() >= 0.15 and g.max() <= 0.9
    assert np.median(g) < 0.45                                     # skewed short: rubble taps clump


def test_blind_piles_flag_only_above_the_frozen_threshold():
    """Prometheus: argmax always flagged a pile, even with three rubble piles, and a tie went to pile 1."""
    import piles
    assert piles.flags([0.20, 0.31, 0.12], 0.6) == []                       # all rubble: "No pile flagged"
    assert piles.flags([0.7, 0.7, 0.1], 0.6) == [1, 2]                      # a tie: both, never just the first
    assert piles.flags([0.7, None, 0.9], 0.6) == [1, 3]                     # no response is never flagged
    sc, perm = [0.2, 0.8, 0.65], [2, 0, 1]                                   # same taps, piles in another order
    moved = piles.flags([sc[i] for i in perm], 0.6)
    assert sorted(perm[n - 1] + 1 for n in moved) == piles.flags(sc, 0.6) == [2, 3]
    assert piles.tally([2, 3], [3]) == dict(hits=1, false_flags=1, misses=0)
    assert piles.tally([], []) == dict(hits=0, false_flags=0, misses=0)
    assert piles.outcome([0.2, 0.3, 0.1], []) == "Dish flagged no pile"
    assert piles.outcome([0.7, None, 0.9], [1, 3]) == "Dish unsure: no score for Pile 2"
    assert piles.outcome([None, None, None], []) == "Dish unsure: no score for Piles 1, 2 and 3"
    assert piles.outcome([0.7, 0.7, 0.1], [1, 2]) == "Dish flagged Piles 1 and 2"


def _two(neurons, seqs):
    dropped = []
    F, _, _, recs = rhythm.run_rhythm(neurons, seqs, on_drop=lambda m, why: dropped.append((m["i"], why)))
    return dropped, [r["meta"]["i"] for r in recs]


def test_jitter_gap_drops_the_sequence_instead_of_bunching_pulses(monkeypatch):
    """Atlas T-1022 #1: after recover_from_jitter the overdue knocks fired 1 ms apart and broke the 200 Hz guard."""
    import cl_mock
    monkeypatch.delenv("NO_CELLS", raising=False)
    orig, hit = cl_mock.Neurons._ticks, []

    def ticks(self, loop):
        for k, tick in enumerate(orig(self, loop)):
            yield tick
            if self._last_pulse and not hit:
                hit.append(self._t)
                raise TimeoutError("tick overran")  # the device raises; the adapter recovers and iterates again
    monkeypatch.setattr(cl_mock.Neurons, "_ticks", ticks)
    monkeypatch.setattr(cl_mock._Loop, "recover_from_jitter", lambda self, **k: setattr(self.n, "_t", self.n._t + 7000))
    seq = np.array([0, 0.1, 0.2])
    with bci.cl.open() as n:
        dropped, kept = _two(n, [seq, seq.copy()])
        assert n.jitter_recoveries == 1
    assert dropped == [(0, "ticks skipped")] and kept == [1]


def test_every_pulse_window_is_whole_at_a_long_artefact_skip(monkeypatch):
    """Atlas T-1022 #2: at --skip-ms 50, pulses 60 ms apart cut the earlier 40 ms window short ([8, 40] spikes)."""
    monkeypatch.delenv("NO_CELLS", raising=False)
    monkeypatch.setattr(bci, "SKIP_MS", 50)
    assert rhythm.min_gap_s() >= 0.092
    assert list(rhythm.clean([0, 60, 120, 240])) == [0, 0.12, 0.24]          # judge input: too-close knocks merge
    assert np.diff(rhythm.debris(np.random.default_rng(0), 9, 1.0)).min() >= 0.092
    with bci.cl.open() as n:
        dropped, kept = _two(n, [np.array([0, 0.06, 0.2]), np.array([0, 0.1, 0.2])])
    assert dropped == [(0, "pulse inside the previous window")] and kept == [1]


def test_clips_and_blind_rounds_leak_no_answer(monkeypatch):
    """Atlas T-1022 #3: /hear returned class-coded ids (s217), and clip play and mode switches ran mid blind round."""
    import re
    had = "MOCK_SPEED" in os.environ
    import live
    if not had:
        os.environ.pop("MOCK_SPEED", None)  # live paces the mock at 5x real time; later mock tests must not crawl
    monkeypatch.setattr(live, "STATE", dict(live.STATE, phase="ready", piles_phase=None, pending=0))
    code, out = live._hear(b"{}")
    assert code == 200 and re.fullmatch(r"[0-9a-f]{16}", out["clip"]) and live._clips[out["clip"]].endswith(".wav")
    live._knocks.clear()
    live.STATE["piles_phase"] = "listening:2"
    for code, _ in (live._hear(b"{}"), live._switch(b'{"listen": "ear"}'), live._knock(b'{"knocks": [0, 500, 1000]}')):
        assert code == 409
    assert live.STATE["pending"] == 1 and not live._knocks          # nothing new was queued



@pytest.mark.parametrize("phase", ["noisefilter", "probe"])
def test_mode_switch_is_refused_while_a_run_owns_the_dish(monkeypatch, phase):
    """Atlas T-1027 #1: a switch mid noise filter returned 200, then the old worker overwrote the new mode's state."""
    had = "MOCK_SPEED" in os.environ
    import live
    if not had:
        os.environ.pop("MOCK_SPEED", None)
    monkeypatch.setattr(live, "STATE", dict(live.STATE, phase=phase, running=True, piles_phase=None, pending=0))
    monkeypatch.setattr(live, "_listen", ["knock"])
    code, out = live._switch(b'{"listen": "ear"}')
    assert code == 409 and out["error"].startswith("busy") and live._listen == ["knock"] and live.STATE["phase"] == phase

def test_history_bins_each_pulse_by_the_gap_before_it(monkeypatch, tmp_path):
    """The history chart: pulse k's response goes in the bin of the gap before it; the first pulse has no gap."""
    had = "MOCK_SPEED" in os.environ
    import live
    if not had:
        os.environ.pop("MOCK_SPEED", None)
    monkeypatch.setattr(live, "_hist", {})
    monkeypatch.setattr(live, "HISTORY", str(tmp_path / "history.json"))
    monkeypatch.setattr(live, "STATE", dict(live.STATE))
    live._history(dict(gaps_ms=[70.0, 500.0, 5000.0], pulse_counts=[99, 3, 5, 7]))
    ser = live._hist[live._series_name()]
    assert ser["n"] == [1, 0, 0, 0, 1, 0, 1] and ser["sum"] == [3.0, 0, 0, 0, 5.0, 0, 7.0]  # 5 s clips into the last bin
    assert json.loads((tmp_path / "history.json").read_text()) == live._hist


def test_omission_design_blocks_and_pulses(tmp_path):
    import json
    import probe
    d = probe.load_design()
    assert d["histories_ms"] == {"A": [0, 200, 400, 600], "B": [0, 100, 500, 600]} and d["probe_ms"] == 800
    plan = probe.schedule(np.random.default_rng(0), 30)
    assert len(plan) == 120 and all(sorted(c for b, c in plan if b == k) == sorted(probe.CONDS) for k in range(30))
    assert [c for _, c in plan[:8]] != [c for _, c in plan[4:12]]                     # orders differ between blocks
    assert probe.pulses_ms("A+") == [0, 200, 400, 600, 800] and probe.pulses_ms("B-") == [0, 100, 500, 600]
    for bad in (dict(d, histories_ms=dict(A=[0, 200, 400, 600], B=[0, 50, 500, 600])),   # under 100 ms apart
                dict(d, histories_ms=dict(A=[0, 200, 400, 600], B=[0, 100, 500, 650])),  # span differs
                dict(d, onset_s=1.5)):                                                     # trials would overlap
        (tmp_path / "d.json").write_text(json.dumps(bad))
        with pytest.raises(AssertionError):
            probe.load_design(str(tmp_path / "d.json"))


def _omission_recs(probe, interaction, blocks=20, seed=0):
    """Rasters built from an additive model: a per-block drift, a history response h (A and B differ), a probe
    response p; `interaction` scales p after history A only (a true history x probe effect)."""
    rng = np.random.default_rng(seed)
    d, read = probe.D, list(range(8, 20))
    n = d["burst_lookback_ms"] + d["probe_ms"] + d["y_window_ms"][1]
    t0 = d["burst_lookback_ms"]
    h = {k: rng.poisson(0.3, (n, 64)) * (np.arange(n) >= t0)[:, None] for k in "AB"}
    p = rng.poisson(0.5, (n, 64)) * (np.arange(n) >= t0 + d["probe_ms"])[:, None]
    recs = []
    for i, (b, c) in enumerate(probe.schedule(rng, blocks)):
        drift = np.full((n, 64), b % 3)                                                   # slow, shared by the block
        r = drift + h[c[0]] + (p * (1 + interaction * (c[0] == "A")) if c[1] == "+" else 0)
        recs.append(dict(i=i, block=b, cond=c, flag=None, **probe.trial_record(r, read, d["y_window_ms"], 99)))
    return recs


def test_additive_response_gives_d_of_exactly_zero():
    """prep.md: a purely additive model (y = h + p) gives D = 0 on this pipeline; an interaction does not."""
    import probe
    r = probe.analyse(_omission_recs(probe, 0.0))
    assert set(r["d_by_block"].values()) == {0} and r["primary"]["mean"] == 0 and r["primary"]["p"] == 1.0
    assert r["sensitivity"]["baseline_subtracted"]["mean"] == 0 and all(x["mean"] == 0 for x in r["descriptive"]["sub_windows"])
    assert not r["effect"] and r["null_line"].startswith("No history effect larger than")
    r = probe.analyse(_omission_recs(probe, 1.0))
    assert r["primary"]["mean"] > 0 and r["primary"]["ci"][0] > 0 and r["primary"]["p"] < 0.05 and r["effect"]
    assert r["primary"]["ci"][0] < r["primary"]["mean"] < r["primary"]["ci"][1] and r["null_line"] is None


def test_sign_flip_p_is_exact_for_small_n():
    import probe
    assert probe.sign_flip_p([1, 1, 1, 1]) == pytest.approx(2 / 16)                  # only all-plus and all-minus
    assert probe.sign_flip_p([0, 0, 0]) == 1.0
    assert 0.0 < probe.sign_flip_p(np.ones(20)) < 0.001                                # Monte Carlo, +1 corrected
    d = np.random.default_rng(0).normal(2.0, 3.0, 30)
    lo, hi = probe.flip_ci(d)                                                           # inverts the same test
    assert lo < d.mean() < hi and probe.sign_flip_p(d, lo - 0.05) < 0.05 <= probe.sign_flip_p(d, lo + 0.05)
    assert probe.flip_ci([1.0, 2.0]) == [None, None]                                    # 2 blocks bound nothing


def test_omission_runs_end_to_end_with_the_readout_frozen_first(monkeypatch, tmp_path):
    """Cells off: D near 0. The readout is in frozen.json before the first trial, and an edit to it during the
    trials stops the run instead of being used."""
    import json
    import probe
    monkeypatch.setenv("NO_CELLS", "1")
    path = tmp_path / "frozen.json"
    seen, orig = [], probe.run_trials

    def spy(*a, **k):
        seen.append(probe.load(str(path))["omission"]["config"]["read"])
        return orig(*a, **k)
    monkeypatch.setattr(probe, "run_trials", spy)
    with bci.cl.open() as n:
        r = probe.run(n, seed=0, blocks=2, path=str(path))
    assert seen == [r["read"]] and r["n_trials"] == 8 and r["flagged"]["n"] == 0 and r["y_window_ms"][1] == 250
    assert r["primary"]["n"] == 2 and not r["effect"] and r["null_line"] == "Too few blocks to bound the history effect."

    def tamper(rec):
        d = json.loads(path.read_text())
        d["omission"]["config"]["read"] = [1]
        path.write_text(json.dumps(d))
    with bci.cl.open() as n, pytest.raises(ValueError, match="hash"):
        probe.run(n, seed=0, blocks=1, path=str(path), on_trial=tamper)


def test_omission_flags_a_late_pulse_and_keeps_the_trial(monkeypatch):
    """prep.md amendment: a trial after a failed or late pulse is flagged, never dropped; the late pulse never fires."""
    import cl_mock
    import probe
    orig, hit = cl_mock.Neurons._ticks, []

    def ticks(self, loop):
        for tick in orig(self, loop):
            yield tick
            if self._last_pulse and not hit:
                hit.append(self._t)
                raise TimeoutError("tick overran")
    monkeypatch.setattr(cl_mock.Neurons, "_ticks", ticks)
    monkeypatch.setattr(cl_mock._Loop, "recover_from_jitter", lambda self, **k: setattr(self.n, "_t", self.n._t + 2500))
    plan = [(0, c) for c in probe.CONDS]
    with bci.cl.open() as n:
        recs = probe.run_trials(n, plan, list(rhythm.READ), probe.D["y_window_ms"], 99)
    assert len(recs) == 4 and recs[0]["flag"] and not any(r["flag"] for r in recs[1:])  # kept, flagged
    assert len(probe.block_d(recs)) == 1                                                 # and it still counts


def test_live_rhythm_screen_builds_its_state():
    """live.py --rhythm crashed at import (GAP_BINS used before it was defined); tests only imported the default mode."""
    import subprocess, sys
    code = ("import bci; a = bci.parse_args(['--rhythm', '--short']); bci.parse_args = lambda *x: a\n"
            "import live; assert live.STATE['mode'] == 'rhythm' and live.STATE['history']['bins']")
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]



def test_scene_source_is_live_only_on_a_cl1():
    """The scene says "the live dish" only for a real CL1; the cl-sdk simulator is "simulator", the mock practice."""
    import re, shutil, subprocess
    if not shutil.which("node"):
        pytest.skip("node not installed")
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rhythm.html")).read()
    fn = re.search(r"const sceneSource = [^\n]+;", html).group(0)
    js = fn + 'console.log(JSON.stringify(["CL1", "mock", "cl-sdk simulator", undefined].map(sceneSource)));'
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30).stdout
    assert json.loads(out) == ["live", "practice", "simulator", "practice"]


def test_header_and_result_line_say_practice_under_the_mock():
    """A blind judge saw no "simulated" on the first screen: the probe card header, the page pill and the
    result line all carry the source label, so a mock run never reads as a real dish."""
    import re, shutil, subprocess
    if not shutil.which("node"):
        pytest.skip("node not installed")
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rhythm.html")).read()
    fns = "\n".join(re.search(rf"const {n} = [^\n]+;", html).group(0) for n in ("PROV", "provOf", "resultTail"))
    null = "No history effect larger than 20.6 spikes either way in this dish tonight."
    js = fns + f"console.log(JSON.stringify([provOf('cl-sdk simulator'), resultTail({json.dumps(null)}, 'cl-sdk simulator'), resultTail(null, 'cl-sdk simulator'), provOf('CL1')]));"
    head, tail, bare, live = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30).stdout)
    assert head == "Simulator running live · LIF spiking network" and live == "Live: CL1"
    assert tail.startswith("No history effect larger") and tail.endswith("tonight (Simulator running live · LIF spiking network). ")
    assert bare == "Simulator running live · LIF spiking network. "
    assert '$("ptag").textContent = provOf(s.source);' in html and '$("pill").textContent = provOf(s.source)' in html
    assert re.search(r"<small>\$\{resultTail\(r\.null_line, s\.source\)\}", html)


def _html():
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rhythm.html")).read()


def test_blind_pile_picks_stay_neutral_until_the_reveal():
    """The final blind check saw the operator's target pile light up before Reveal: the pick rule must undo the
    pressed colour, and come after it so it wins the cascade."""
    html = _html()
    pressed = html.index('button.b.ghost[aria-pressed="true"] {')
    neutral = html.index('#bwho button.b.ghost[aria-pressed] { border-color: var(--line); color: var(--ink); }')
    assert neutral > pressed


def test_piles_and_calibration_cards_carry_the_source_label():
    """Every card with a number on it says where the number came from (Practice dish, simulated, under the mock)."""
    html = _html()
    assert '$("ctag").textContent = provOf(s.source);' in html
    assert '$("btag").textContent = `${provOf(s.source)} · Manual taps`;' in html


def test_no_scene_no_slot():
    """The demo ships without the 3D scene: with /state scene false the slot must not render (no placeholder text,
    no dashed box). It stays hidden until mountScene() runs, which only happens when the server has a scene."""
    import re
    html = _html()
    slot = re.search(r'<section[^>]*id="scene-slot"[^>]*>(.*?)</section>', html, re.S)
    assert " hidden" in slot.group(0).split(">")[0] and slot.group(1).strip() == ""
    mount = html[html.index("function mountScene()"):html.index("const rescue = ")]
    assert '$("scene-slot").hidden = false' in mount
    assert "if (s.scene && !scene" in html

# ---- T-1026 fault injection (from Atlas's repro harness): a scripted device behind the real adapter ----------
class _FakePlan:
    def __init__(self, dev):
        self.dev, self.channels = dev, []

    def stim(self, channels, *args):
        self.channels = list(channels)

    def run(self):
        import adapter
        if self.dev.mode == "reject":
            raise adapter.TransactionRejected("injected device rejection")
        if self.dev.mode != "drop":
            import cl_mock
            self.dev.echoes += [cl_mock.Stim(c, self.dev.now + 2 + self.dev.echo_shift) for c in self.channels]


class _FakeDevice:
    """1 kHz ticks, `phase` frames off the ms grid; `skip` ms produce no tick; `spikes` are (frame, channel)."""
    def __init__(self, mode="good", skip=(), spikes=(), phase=0, echo_shift=0):
        self.mode, self.skip, self.spikes, self.phase, self.echo_shift = mode, set(skip), list(spikes), phase, echo_shift
        self.now, self.echoes = 0, []

    def timestamp(self):
        return self.now

    def create_stim_plan(self):
        return _FakePlan(self)

    def loop(self, *args, **kwargs):
        import cl_mock
        for ms in range(13000):
            lo = ms * 25 + self.phase
            hi = lo + 25
            if ms in self.skip:
                continue
            self.now = hi
            sp = [cl_mock.Spike(c, f) for f, c in self.spikes if lo <= f < hi]
            yield cl_mock._Tick(None, ms, lo, hi, sp, [s for s in self.echoes if lo <= s.timestamp < hi],
                                np.zeros((25, 64), np.int16))


def _fake_trials(dev):
    import adapter
    import probe
    return probe.run_trials(adapter.Neurons(dev), [(0, c) for c in probe.CONDS], list(rhythm.READ), [20, 250], 99)


def _expected_y(spikes, n=4):
    """Independent count: spikes on read electrodes 20-250 ms after each trial's probe slot (onsets 2 s + 3 s k)."""
    import probe
    onsets = [2000 + k * int(probe.D["onset_s"] * 1000) for k in range(n)]
    return [sum(c in rhythm.READ_SET and 20 <= f // 25 - o - probe.D["probe_ms"] < 250 for f, c in spikes) for o in onsets]


def test_t1026_valid_echoes_keep_four_clean_trials():
    recs = _fake_trials(_FakeDevice("good"))
    assert [r["y"] for r in recs] == [0, 0, 0, 0] and not any(r["flag"] for r in recs)


def test_t1026_rejected_pulse_is_flagged_and_the_run_goes_on():
    recs = _fake_trials(_FakeDevice("reject"))
    assert len(recs) == 4 and all("rejected" in r["flag"] for r in recs)


def test_t1026_pulse_without_an_echo_is_flagged():
    recs = _fake_trials(_FakeDevice("drop"))
    assert len(recs) == 4 and all("no echo" in r["flag"] for r in recs)


def test_t1026_stale_ring_rows_never_reach_y_across_a_tick_gap():
    spikes = [(1729 * 25 + 1, rhythm.READ[0])]  # 5825 % 4096 == 1729: the skipped 5820-5830 ms would read it
    recs = _fake_trials(_FakeDevice("good", skip=range(5820, 5831), spikes=spikes))
    assert [r["y"] for r in recs] == _expected_y(spikes) == [0, 0, 0, 0]
    assert [bool(r["flag"]) for r in recs] == [False, True, False, False] and "ticks skipped" in recs[1]["flag"]


def test_t1026_off_grid_ticks_keep_every_real_spike():
    spikes = [(70630, rhythm.READ[0]), (2950 * 25 + 20, rhythm.READ[1]), (5900 * 25 + 24, rhythm.READ[0])]  # 2825.2 ms ...
    recs = _fake_trials(_FakeDevice("good", spikes=spikes, phase=13))
    assert [r["y"] for r in recs] == _expected_y(spikes) == [2, 1, 0, 0] and not any(r["flag"] for r in recs)


def test_t1026_late_echo_is_paired_but_flagged():
    """Atlas round 2: an echo 4 ms late sat inside the 5 ms pairing window and passed as clean."""
    assert not any(r["flag"] for r in _fake_trials(_FakeDevice("good", echo_shift=25)))      # 1 ms: on time
    recs = _fake_trials(_FakeDevice("good", echo_shift=100))                                 # 4 ms late
    assert len(recs) == 4 and all(r["flag"] == "pulse delivered over 2 ms off schedule" for r in recs)


def test_zero_echo_banner_shows_after_a_first_block_with_no_echoes():
    """Lead 3 Nov: a device that reports no stim echoes gets one loud explanation, not 120 mystery flags."""
    import live
    rows = lambda recs: [dict(i=r["i"], flag=r["flag"]) for r in recs]
    dropped = rows(_fake_trials(_FakeDevice("drop")))
    assert live.no_echo(dropped) and not live.no_echo(dropped[:3])          # only once the whole first block is in
    assert not live.no_echo(rows(_fake_trials(_FakeDevice("good"))))
    assert not live.no_echo(rows(_fake_trials(_FakeDevice("reject"))))     # rejections are a different failure


# ---- T-1027 noise filter (exploratory) and the artefact chart -------------------------------------------------
def test_noise_filter_schedule_rare_share_no_back_to_back_counts_limits_seed():
    import cl_mock
    import noisefilter as nf
    m = nf.schedule()
    assert len(m) == 150 and m.sum() == 15                       # exactly 10% rare
    assert not (m[1:] & m[:-1]).any()                            # never two rare in a row
    assert (nf.schedule(0) == m).all() and not (nf.schedule(1) == m).all()   # fixed seed, and the seed matters
    for seed in range(50):                                       # the rule holds for every seed, not one lucky draw
        s = nf.schedule(seed)
        assert s.sum() == 15 and not (s[1:] & s[:-1]).any()
    pl = nf.plan(27, 45)
    assert len(pl) == 300 and {c for _, c, _ in pl} == {27, 45} and not {27, 45} & set(bci.EXCLUDED)
    assert 1000 / nf.ISI_MS <= cl_mock.MAX_HZ                    # 2 Hz per electrode, limit 200
    ph = bci.STIM.phases                                         # probe's pulse: inside 3 uA and 3 nC per phase
    assert all(abs(ua) <= 3 and us * abs(ua) / 1000 <= 3 for us, ua in zip(ph[::2], ph[1::2]))
    assert "p.stim(bci.ChannelSet(ch), bci.STIM, rhythm.BURST)" in open(nf.__file__).read()
    assert 170 <= nf.duration_s() <= 190                         # about 3 minutes of chip time


def test_noise_filter_flip_flop_pairs():
    """Condition 2 is condition 1 with A and B swapped, pulse for pulse, so each electrode is rare exactly once."""
    import noisefilter as nf
    pl = nf.plan(27, 45, seed=3)
    c1, c2 = [p for p in pl if p[0] == 1], [p for p in pl if p[0] == 2]
    assert len(c1) == len(c2) == 150
    for (_, e1, r1), (_, e2, r2) in zip(c1, c2):
        assert r1 == r2 and {e1, e2} == {27, 45} and e1 == (45 if r1 else 27)
    assert all(e == 45 for _, e, r in c1 if r) and all(e == 27 for _, e, r in c2 if r)


def test_noise_filter_ssa_index_maths_including_zero_response():
    import noisefilter as nf
    assert nf.ssa(3, 1) == 0.5 and nf.ssa(1, 3) == -0.5 and nf.ssa(2, 2) == 0 and nf.ssa(5, 0) == 1
    assert nf.ssa(0, 0) == 0.0                                   # a dead dish: nothing to filter, no NaN
    rec = lambda ch, rare, y: dict(ch=ch, rare=rare, y=y, flag=None)
    recs = [rec(1, True, 30)] * 3 + [rec(1, False, 10)] * 27 + [rec(2, True, 20)] * 3 + [rec(2, False, 20)] * 27
    r = nf.analyse(recs, 1, 2)
    assert r["ssa_a"] == 0.5 and r["ssa_b"] == 0 and r["mean"] == 0.25 and r["ci"] == [0.25, 0.25]  # no spread
    assert r["verdict"] == "positive" and r["label"] == "exploratory" and "p" not in r   # a contrast, never "filtering"
    dead = nf.analyse([dict(d, y=0) for d in recs], 1, 2)
    assert dead["mean"] == 0 and dead["ci"] == [0, 0] and dead["verdict"] == "none"


def test_noise_filter_practice_run_end_to_end_reads_no_filtering(monkeypatch):
    """The mock has no stimulus-specific adaptation (its short-term depression is shared by every electrode), so
    the honest practice result is no filtering; nothing in cl_mock was added to fake one."""
    import noisefilter as nf
    monkeypatch.delenv("NO_CELLS", raising=False)
    seen = []
    with bci.cl.open() as n:
        r = nf.run(n, n=60, rest_s=2, on_stim=seen.append, a=27, b=45)
    assert r["n"] == len(seen) == 120 and r["flagged"]["n"] == 0
    assert r["means"]["27_frequent"] > 0 and r["ci"][0] <= 0 <= r["ci"][1] and r["verdict"] == "none"
    assert [s["cond"] for s in seen] == [1] * 60 + [2] * 60 and sum(s["rare"] for s in seen) == 12


@pytest.mark.parametrize("mode,why", [("reject", "rejected"), ("drop", "no echo")])
def test_noise_filter_flags_failed_pulses_and_keeps_them(mode, why):
    import adapter
    import noisefilter as nf
    recs = nf.run_stimuli(adapter.Neurons(_FakeDevice(mode)), nf.plan(27, 45, n=10), 27, 45, rest_s=1)
    assert len(recs) == 20 and all(why in r["flag"] for r in recs)
    good = nf.run_stimuli(adapter.Neurons(_FakeDevice("good")), nf.plan(27, 45, n=10), 27, 45, rest_s=1)
    assert len(good) == 20 and not any(r["flag"] for r in good)


def test_noise_filter_keeps_a_pulse_whose_window_the_stop_cut_short():
    import adapter
    import noisefilter as nf
    # the fake device stops at 13 s; rest 2.4 s puts the last pulse at 12.9 s, inside its 250 ms window
    recs = nf.run_stimuli(adapter.Neurons(_FakeDevice("good")), nf.plan(27, 45, n=10), 27, 45, rest_s=2.4)
    assert len(recs) == 20 and recs[-1]["i"] == 19 and "response window cut short" in recs[-1]["flag"]
    assert not any(r["flag"] for r in recs[:-1])
    r = nf.run(adapter.Neurons(_FakeDevice("good")), n=10, rest_s=3.0, a=27, b=45)   # pulses never sent: no verdict
    assert "error" in r and "verdict" not in r and r["error"].startswith("run stopped after")
    cut = nf.run(adapter.Neurons(_FakeDevice("good")), n=10, rest_s=2.4, a=27, b=45)   # all 20 kept, the last one cut short
    assert "verdict" not in cut and cut["error"] == "run stopped after 19 of 20 pulses: no result"
    assert nf.fmt(cut) == "EXPLORATORY noise filter: run stopped after 19 of 20 pulses: no result"   # the CLI prints it, no KeyError


def test_noise_filter_window_locks_to_the_echo_not_the_expected_pulse():
    """Atlas T-1027 #2: a pulse that lands 2 ms late opens its 20-250 ms window from where it landed."""
    import adapter
    import noisefilter as nf
    dry = _FakeDevice("good", echo_shift=2 * nf.MS)
    nf.run_stimuli(adapter.Neurons(dry), nf.plan(27, 45, n=10), 27, 45, rest_s=1)
    e0, e1 = (int(e.timestamp) for e in dry.echoes[:2])
    ms = nf.MS
    spikes = [(e0 + 19 * ms, 1), (e0 + 21 * ms, 1), (e1 + 249 * ms, 1)]   # expected time + 21, + 23 and + 251 ms
    recs = nf.run_stimuli(adapter.Neurons(_FakeDevice("good", spikes=spikes, echo_shift=2 * ms)), nf.plan(27, 45, n=10), 27, 45, rest_s=1)
    assert [r["y"] for r in recs[:2]] == [1, 1] and not any(r["flag"] for r in recs) and "at" not in recs[0]


def _pure_js():
    html = _html()
    import re
    head = "\n".join(re.search(rf"const {n} = [^\n]+;", html).group(0) for n in ("PROV", "provOf", "resultTail"))
    return head + html[html.index("// ---- pure chart and result code"):html.index("// ---- end pure ----")]


def _node(js):
    import shutil, subprocess
    if not shutil.which("node"):
        pytest.skip("node not installed")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr[-600:]
    return json.loads(out.stdout)


def test_noise_filter_card_says_exploratory_and_shows_no_p_value():
    import re
    html = _html()
    card = re.search(r'<section class="card" id="ncard".*?</section>', html, re.S).group(0)
    assert "Noise filter (exploratory)" in card and 'id="ntag"' in card and "Idea from Eytan, Brenner &amp; Marom 2003: in rat cortical networks" in card and "The index is Ulanovsky's (2003). Exploratory." in card and "dishes like this" not in card
    assert html.index('id="pcard"') < html.index('id="ncard"')            # below the memory test
    r = dict(verdict="none", mean=0.022, ci=[-0.034, 0.077], n=300, flagged=dict(n=0))
    head, small = _node(_pure_js() + f"console.log(JSON.stringify(nfResult({json.dumps(r)}, 'cl-sdk simulator')));")
    assert head == "No filtering detected" and small.startswith("Simulator running live · LIF spiking network. Exploratory")
    assert small.endswith("The conditions run in a fixed order, so drift or carry-over can make this contrast; the interval resamples pulses only.")
    head, _ = _node(_pure_js() + f"console.log(JSON.stringify(nfResult({json.dumps(dict(r, verdict='positive', mean=0.14, ci=[0.05, 0.2]))}, 'mock')));")
    assert head == "Knocks drew relatively more than the rumble (positive SSA contrast)" and "filter" not in head
    assert "95% CI" in small and not re.search(r"\bp\s*[=<>]|p-value", small)
    fn = html[html.index("function drawNoise(s)"):html.index('$("nrun").onclick')]
    assert not re.search(r"\.p\b|\bp\s*=\s*\$|p-value", fn)
    assert 's.phase !== "idle" || s.listen === "ear";' in fn          # the server refuses it in ear mode: say so, don't no-op
    flat = [dict(i=i, cond=1, rare=False, y=100) for i in range(10)]   # a flat line: a bbox gradient would draw nothing
    _, g = _node(_pure_js() + f"console.log(JSON.stringify(nfPlot({json.dumps(flat)}, 27, 45, 10, 400)));")
    assert 'gradientUnits="userSpaceOnUse"' in g and 'stroke="url(#fade)"' in g


def test_artefact_chart_renders_three_bars_and_chance_with_the_trap_off_the_axis():
    c = dict(acc=0.61, shuffled=0.49, timing_only=0.97, trap=0.95)
    H, svg = _node(_pure_js() + f"console.log(JSON.stringify(artefactChart({json.dumps(c)}, 'cl-sdk simulator', 'Software on timing alone', 600)));")
    for s in ("LIF spiking network", "stimulation window excluded", "Shuffled labels", "Software on timing alone", "chance 50%",
              "Simulator running live · LIF spiking network", ">61%<", ">49%<", ">97%<", 'stroke-dasharray="4 3"'):
        assert s in svg, s
    assert "trap" not in svg.lower() and ">95%<" not in svg and "hatch" not in svg     # never on the shared axis
    _, empty = _node(_pure_js() + "console.log(JSON.stringify(artefactChart(null, 'CL1', 'x', 400)));")
    assert "Live: CL1" in empty and ">–<" in empty
    assert ">held out, cross-validated</text>" in empty and "· held out" not in empty   # a phone gets two lines, not an overflow
    line = _node(_pure_js() + f"console.log(JSON.stringify(trapLine({json.dumps(c)})));")
    assert line == "Artefact trap, never counted: the input electrode's own trace scores 95%. That is what the pulses alone give away."
    html = _html()
    assert 'draw("achart", artefactChart, c, s.source, soft);' in html and '$("trap").textContent = trapLine(c);' in html
    assert '<p class="trapline" id="trap">' in html


def test_calibration_headline_compares_with_shuffled_labels_not_chance():
    js = lambda c: _node(_pure_js() + f"console.log(JSON.stringify(calVerdict({json.dumps(c)})));")
    assert js(dict(acc=0.55, shuffled=0.49, shuffled_sd=0.04, n=80)) == ["No signal beyond luck", "55% · shuffled labels 49%"]
    assert js(dict(acc=0.70, shuffled=0.49, shuffled_sd=0.04, n=80))[0] == "Above shuffled labels"
    assert js(dict(acc=0.60, shuffled=0.49, n=80))[0] == "No signal beyond luck"          # no sd: binomial SE 0.056, 2 SE = 0.11
    html = _html()
    assert "vs 50% chance" not in html and '$("acc").textContent = words;' in html


def test_minutes_say_on_the_chip_only_on_a_cl1_and_piles_get_the_readers_note():
    js = lambda e: _node(_pure_js() + f"console.log(JSON.stringify({e}));")
    assert js("chipTime(180, 'CL1')") == "about 3 min on the chip"
    assert js("chipTime(180, 'mock')") == "about 3 min (simulator)"
    assert js("chipTime(180, 'cl-sdk simulator')") == "about 3 min (simulator)"
    assert js("pileNote({acc: 0.55, shuffled: 0.49, shuffled_sd: 0.04, n: 80})") == "Decoder 55% on 80 held-out sequences · shuffled labels 49% · near chance"
    assert js("pileNote({acc: 0.70, shuffled: 0.49, shuffled_sd: 0.04, n: 80})").endswith("· above chance")
    html = _html()
    assert "min on the chip`" not in html and html.count("chipTime(") == 2   # the noise-filter button and the calibration line
    assert 'sceneSource(s.source), pileOpts(s));' in html
    assert "Dish responses · free knocks" in html and "What the dish heard" not in html


def test_noise_filter_takes_the_scans_two_strongest_drivers(monkeypatch):
    import noisefilter as nf
    monkeypatch.setattr(rhythm, "DRIVE_ORDER", None)
    assert nf.electrodes() == (rhythm.IN_CH, nf.DEFAULT_B)                  # before any scan: the default pair
    monkeypatch.setattr(rhythm, "IN_CH", 33)
    monkeypatch.setattr(rhythm, "DRIVE_ORDER", [33, 41, 12])
    assert nf.electrodes() == (33, 41)


# ---- T-1027 part 0: the kit layout and Windows MIME (Hermes's kit testing) -------------------------------------
_SPINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIVE_RHYTHM = "import bci; a = bci.parse_args(['--rhythm']); bci.parse_args = lambda *x: a\n"


def _py(code, **env):
    import subprocess
    e = {k: v for k, v in os.environ.items() if k not in ("RESCUE_SCENE", "HUB_DIR", "HUB_AGENT")}
    r = subprocess.run([sys.executable, "-c", code], cwd=_SPINE, env=dict(e, **env), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip()


def test_scene_dir_defaults_to_the_kits_sibling_folder(tmp_path):
    """In the kit the scene sits beside the spine (rescue-kit/rescue-scene); RESCUE_SCENE still overrides it. HOME
    moves so a home-relative default (a sibling checkout that happens to exist) cannot pass."""
    got = _py(_LIVE_RHYTHM + "import live; print(live.SCENE_DIR)", HOME=str(tmp_path))
    assert got == os.path.realpath(os.path.join(_SPINE, "..", "rescue-scene"))
    assert _py(_LIVE_RHYTHM + "import live; print(live.SCENE_DIR)", RESCUE_SCENE="/tmp") == os.path.realpath("/tmp")


def test_rescue_data_defaults_to_the_kits_sibling_folder(tmp_path):
    unset = "import os; os.environ.pop('RESCUE_DATA', None)\n"  # HOME=tmp_path: a home-relative default can't coincide
    assert _py(unset + "import ear; print(ear.DATA)", HOME=str(tmp_path)) == os.path.realpath(os.path.join(_SPINE, "..", "rescue-data"))
    assert _py("import ear; print(ear.DATA)", RESCUE_DATA="/tmp") == os.path.realpath("/tmp")


def test_scene_js_is_served_as_javascript_even_when_the_os_says_text_plain(tmp_path):
    """Windows' registry can map .js to text/plain, and Chrome then refuses the scene's module scripts."""
    (tmp_path / "main.js").write_text("export const x = 1;")
    code = ("import mimetypes; mimetypes.init(); mimetypes.add_type('text/plain', '.js')  # what the registry can do\n"
            + _LIVE_RHYTHM + "import live, threading, urllib.request\n"
            "from http.server import ThreadingHTTPServer\n"
            "s = ThreadingHTTPServer(('127.0.0.1', 0), live.H); threading.Thread(target=s.serve_forever, daemon=True).start()\n"
            "r = urllib.request.urlopen(f'http://127.0.0.1:{s.server_address[1]}/scene/main.js', timeout=10)\n"
            "print(r.headers['Content-Type'])")
    assert _py(code, RESCUE_SCENE=str(tmp_path)) == "text/javascript"


def test_cal_flag_halves_the_calibration_and_its_planned_chip_time():
    def fresh(*argv):
        code = (f"import bci; a = bci.parse_args(['--rhythm', *{list(argv)!r}]); bci.parse_args = lambda *x: a\n"
                "import json, live, rhythm; s = live.STATE\n"
                "print(json.dumps([s['cal_total'], s['cal_s'], len(rhythm.calibration(a.seed, a.cal)[0])]))")
        return json.loads(_py(code))
    (t40, s40, n40), (t20, s20, n20) = fresh(), fresh("--cal", "20")
    assert (t40, n40, t20, n20) == (80, 80, 40, 40)                    # the screen's cal_total and the sequences played
    assert 0.4 < s20 / s40 < 0.6 and 200 < s40 < 360                     # about half of about 4.5 min
    with pytest.raises(SystemExit):
        bci.parse_args(["--rhythm", "--cal", "19"])
    src, html = open(os.path.join(_SPINE, "live.py")).read(), _html()
    assert 'seqs, _rh["y"] = rhythm.calibration(ARGS.seed, ARGS.cal)' in src            # the run plays what the screen promises
    assert "${s.cal_total / 2} target" in html and "chipTime(s.cal_s, s.source)" in html and 'HOW.replaceAll("{k}", s.cal_total / 2)' in html


def test_finish_fixes_chance_line_on_the_bars_note_after_run_and_empty_states():
    import re
    c = dict(acc=0.55, shuffled=0.49, timing_only=0.97, trap=0.99)
    _, svg = _node(_pure_js() + f"console.log(JSON.stringify(artefactChart({json.dumps(c)}, 'mock', 'Software on timing alone', 330)));")
    dashes = re.findall(r'<line [^>]*y1="([\d.]+)" y2="([\d.]+)" stroke="var\(--ink\)" stroke-dasharray="4 3"', svg)
    bars = [float(y) for y in re.findall(r'<rect x="0" y="([\d.]+)" width="[\d.]+" height="12" rx="3" fill="var\(--grid\)"', svg)]
    assert len(dashes) == len(bars) == 3
    for (y1, y2), b in zip(dashes, bars):                          # each dash spans its bar, never the sublabel above it
        assert b - 6 <= float(y1) < b and b + 12 < float(y2) <= b + 18
    html = _html()
    card = re.search(r'<section class="card" id="ncard".*?</section>', html, re.S).group(0)
    assert card.index('id="nrun"') < card.index('class="srcnote"')     # the Run button above the fold, the source below
    assert ".srcnote { margin: 0; max-width: 70ch;" in html and "var(--sans)" in html[html.index(".srcnote {"):][:120]
    assert "Dish (calibration)<b" in html and "calibration CV<b" not in html
    assert "Knock on the pad or press Space; the dish's responses appear here." in html and "No knocks yet" not in html
    mount = html[html.index("function mountScene()"):html.index("const rescue = ")]
    assert "3D scene needs WebGL; the results here are unaffected" in mount and 'getContext("webgl")' in mount
    assert "if (s.scene && !sceneTried) mountScene();" in html                # tried once, not every tick
    assert html.count("getBoundingClientRect().width || 640") == 2 and "clientWidth || 640" not in html


def test_second_blind_check_one_vocabulary_one_primary_and_the_exploratory_result_smaller():
    """T-1027 Part 5: the cells only on a CL1, knock and rumble on screen, one filled button, no tracked caps,
    notes at most 75ch, per-electrode CIs, a unit on the plot, the stats in one column on a phone."""
    import re
    js = lambda e: _node(_pure_js() + f"console.log(JSON.stringify({e}));")
    who = lambda src: js(f"artefactChart(null, '{src}', 'x', 600)[1]")
    assert ">The cells<" in who("CL1") and ">Practice simulator<" in who("mock")
    assert ">LIF spiking network<" in who("cl-sdk simulator")
    _, g = _node(_pure_js() + "console.log(JSON.stringify(artefactChart({acc: 0.98, shuffled: 0.49, timing_only: 0.98}, 'mock', 'x', 600)));")
    assert re.findall(r'<text class="t" x="(\d+)"[^>]*text-anchor="end">98%', g) == ["596", "596"]   # clear of the edge
    _, p = _node(_pure_js() + "console.log(JSON.stringify(nfPlot([{i: 0, cond: 1, rare: false, y: 5}], 27, 45, 10, 400)));")
    assert "spikes per pulse" in p
    html = _html()
    css, body = html[:html.index("</style>")], html[html.index("</style>"):]
    assert "uppercase" not in css and not re.search(r"letter-spacing: 0\.0[4-9]em", css)
    assert re.search(r"#nresult \{ font-size: clamp\(1\.15rem, 2\.4vw, 1\.45rem\)", css)   # .result tops out at 2.4rem
    assert re.search(r"\.how \{[^}]*max-width: 75ch", css) and re.search(r"\.note \{[^}]*max-width: 75ch", css)
    assert "#nfacts { grid-template-columns: 1fr; }" in css
    filled = re.findall(r'<button class="b" id="(\w+)"', body)
    assert filled == ["prun", "bgo", "breveal"]   # bgo and breveal live on the blind card, shown alone
    card = re.search(r'<section class="card" id="ncard".*?</section>', body, re.S).group(0)
    assert not re.search(r"\b(rare|frequent)\b", card) and "knock vs rumble" in card
    fn = html[html.index("function drawNoise(s)"):html.index('$("nrun").onclick')]
    assert "r.ci_a[0]" in fn and "r.ci_b[0]" in fn and "Pulses so far: ${live.length} / ${2 * n}" in fn
    assert not re.search(r"`[^`]*\b(rare|frequent)\b[^`]*`", fn.replace('"_rare"', "").replace('"_frequent"', ""))


def test_port_flag_moves_the_screen_off_8777():
    assert bci.parse_args(["--rhythm"]).port == 8777 and bci.parse_args(["--rhythm", "--port", "8790"]).port == 8790
    src = open(os.path.join(_SPINE, "live.py")).read()
    assert 'ThreadingHTTPServer(("0.0.0.0", ARGS.port), H)' in src and 'f"http://localhost:{ARGS.port}' in src


def test_a_decoder_at_chance_sends_the_scene_no_target():
    """T-1027 Part 5: the scene never shows a green hit for a decoder no better than shuffled labels."""
    js = lambda e: _node(_pure_js() + f"console.log(JSON.stringify({e}));")
    luck, good = "{acc: 0.55, shuffled: 0.49, shuffled_sd: 0.04, n: 80}", "{acc: 0.80, shuffled: 0.49, shuffled_sd: 0.04, n: 80}"
    assert js(f"pileOpts({{targets: [2], calib: {luck}}})")["target"] is None
    assert js(f"pileOpts({{targets: [2], calib: {good}}})")["target"] == 2
    assert js(f"pileOpts({{targets: [1, 2], calib: {good}}})")["target"] is None and js("pileOpts({targets: [2]})") == {"target": None}
    import re   # "decoder" on screen: "reader" suggests the dish read the rhythm; only code comments keep the word
    assert not [ln for ln in _html().splitlines() if re.search(r"reader", ln.split("//")[0], re.I)]


_PAGE_JS = r"""
const calls = [], els = {}, queued = [], replies = __STATES__;
const el = () => { const cl = new Set(), at = {};
  return { classList: { add: c => cl.add(c), remove: c => cl.delete(c), toggle: (c, on) => (on ?? !cl.has(c)) ? cl.add(c) : cl.delete(c), contains: c => cl.has(c) },
    style: { setProperty() {} }, dataset: {}, setAttribute: (k, v) => at[k] = String(v), getAttribute: k => at[k] ?? null, addEventListener() {},
    getBoundingClientRect: () => ({ width: __W__ }), getContext: () => ({}), play() {},
    appendChild(c) { c.contentWindow = { rescue: new Proxy({}, { get: (_, fn) => (...a) => calls.push([fn, ...a]) }) }; c.onload && c.onload(); } }; };
globalThis.document = { getElementById: id => els[id] ??= el(), createElement: el, querySelectorAll: () => [], addEventListener() {} };
globalThis.addEventListener = () => {};
const wait = setTimeout;
globalThis.setTimeout = f => queued.push(f);   // the test steps the poll by hand
const hang = (_, o) => new Promise((_, no) => o && o.signal && o.signal.addEventListener("abort", () => no(o.signal.reason)));
globalThis.fetch = async (u, o) => { const s = replies.shift(); if (s === "hang") return hang(u, o); if (!s) throw new TypeError("Failed to fetch"); return { json: async () => s }; };
const look = () => ({ pill: els.pill.textContent, real: els.pill.classList.contains("real"), status: els.status.textContent, inert: !!els.app?.inert,
  ctag: els.ctag.textContent, vtag: els.vtag?.textContent, chart: els.achart.innerHTML, sound_off: !!els.me.disabled && els.nosnd.hidden === false, calls: calls.splice(0) });
(async () => {
  const seen = [];
  __PAGE__
  for (const st of __STATES__) {
    if (st === "hang") await new Promise(r => wait(r, 2400));   // past the page's 2 s deadline, in real time
    for (let k = 0; k < 3; k++) await new Promise(r => setImmediate(r)); seen.push(look()); const f = queued.shift(); if (f) f(); }
  console.log(JSON.stringify(seen));
})();
"""


def _page(states, width=600):
    """Runs rhythm.html's whole script in node on a stub DOM: each /state poll gets the next state; None is a failed
    fetch, "hang" a request that never answers."""
    html = _html()
    js = (_PAGE_JS.replace("__STATES__", json.dumps(states)).replace("__W__", str(width))
          .replace("__PAGE__", html[html.index("<script>") + 8:html.rindex("</script>")]))
    return _node(js)


def _live_state(**env):
    return json.loads(_py(_LIVE_RHYTHM + "import json, live; print(json.dumps(live.STATE))", **env))


_LUCK = dict(acc=0.5, shuffled=0.5, shuffled_sd=0.05, n=80, correct=40, timing_only=0.9, trap=0.6)
_HIT = dict(id=1, knocks=[0, 300, 600], verdict="survivor", confidence=0.8, timing_only=dict(verdict="debris", confidence=0.6),
            truth=None, clip=None, bed=None)


def test_a_free_knock_at_chance_shows_the_scene_no_hit_and_marks_nothing():
    """T-1027 final review #1: pileOpts gated the blind piles, but a free knock still sent the scene "survivor" and markSpot."""
    s = dict(_live_state(), phase="ready", scene=True, latest=_HIT)
    luck, = _page([dict(s, calib=_LUCK)])
    assert ["markSpot"] not in luck["calls"] and luck["calls"][0][:2] == ["setVerdict", "unsure"]
    good, = _page([dict(s, calib=dict(_LUCK, acc=0.8, correct=64))])
    assert good["calls"][:2] == [["setVerdict", "survivor", 0.8, "practice"], ["markSpot"]]
    nocal, = _page([dict(s, calib=None)])
    assert ["markSpot"] not in nocal["calls"] and nocal["calls"][0][:2] == ["setVerdict", "unsure"]


def test_a_failed_state_poll_drops_live_and_listening_and_recovers():
    """T-1027 final review #2: a failed /state poll left "Live: CL1", "Listening for knocks" and a live pad on screen."""
    live = dict(_live_state(), source="CL1", pill="CL1 · live neurons", phase="ready", calib=_LUCK)
    before, lost, back = _page([live, None, live])
    assert before["pill"] == "Live: CL1" and before["real"] and before["status"] == "Listening for knocks" and not before["inert"]
    assert lost["pill"] == "Connection lost · retrying" and not lost["real"] and lost["status"] == "No contact with the dish" and lost["inert"]
    assert back == dict(before, calls=[])
    before, hung, back = _page([live, "hang", live])   # Prometheus: a held-open request never reached the catch
    assert hung["pill"] == "Connection lost · retrying" and hung["status"] == "No contact with the dish" and hung["inert"]
    assert back == dict(before, calls=[])


def test_the_noise_plot_unit_wraps_on_a_phone():
    """T-1027 final review nit: at 390 the one-line unit ran off the right edge and cut "20–250 ms"."""
    import re
    for W, lines in ((342, ["spikes per pulse, all read electrodes,", "20–250 ms after it"]), (700, ["spikes per pulse, all read electrodes, 20–250 ms after it"])):
        _, g = _node(_pure_js() + f"console.log(JSON.stringify(nfPlot([{{i: 0, cond: 1, rare: false, y: 5}}], 27, 45, 10, {W})));")
        assert [t for t in re.findall(r"<text[^>]*>([^<]*)</text>", g) if "spikes" in t or "after it" in t] == lines
        assert len(max(lines, key=len)) * 7.3 + 36 < W   # 12px IBM Plex Mono advances 7.2px a character


@pytest.mark.parametrize("width", [600, 342])   # the chart's width at 1280 and at 390
def test_the_cells_off_control_keeps_its_name_on_pill_tag_and_chart(width):
    """T-1027 final review #3: NO_CELLS=1 said "simulator check, cells off" on the server, but the page rebuilt it as the practice dish."""
    s = _live_state(NO_CELLS="1")
    assert s["cells_off"] and "network off (control)" in s["pill"] and not _live_state()["cells_off"]
    got, = _page([dict(s, calib=_LUCK)], width)
    assert got["pill"].startswith("Practice simulator · network off (control)") and got["ctag"] == "Practice simulator · network off (control)"
    assert "Practice simulator · network off (control)" in got["chart"] and ">Network off (control)<" in got["chart"] and "LIF" not in got["chart"]


def test_the_simulator_path_says_lif_network_and_replay_says_replay():
    """Hackathon: on the simulator, pill, tags and chart say a live LIF spiking network: no dish, cells, neurons or CL1."""
    import re
    s = dict(_live_state(), phase="ready", calib=_LUCK, latest=dict(_HIT, clip="c0ffee", bed="concrete"))
    for src in ("cl-sdk simulator",):
        got, = _page([dict(s, source=src, scene=True)])
        words = " ".join([got["pill"], got["ctag"], got["vtag"], re.sub(r"<[^>]+>", " ", got["chart"])])
        assert got["pill"].startswith("Simulator running live · LIF spiking network") and got["ctag"] == "Simulator running live · LIF spiking network" and "Simulator running live · LIF spiking network" in got["chart"]
        assert got["calls"][0][-1] == ("simulator" if src == "cl-sdk simulator" else "practice")   # the scene's source
        assert got["vtag"] == "Replay" and not re.search(r"dish|cells?\b|neuron|CL1", words, re.I), words
    assert _node(_pure_js() + "console.log(JSON.stringify(provOf('replay')))") == "Replay"


def test_sound_is_off_without_the_rendered_wavs_and_a_failed_load_clears_running(monkeypatch, tmp_path):
    """T-1027 final review #4: a kit without the gitignored WAVs offered Sound, and Calibrate then wedged at running=True."""
    import shutil, ear
    if not os.path.isfile(os.path.join(ear.DATA, "sequences.json")):
        pytest.skip("rescue-data not found")
    shutil.copy(os.path.join(ear.DATA, "sequences.json"), tmp_path)   # the index, not the WAVs it names
    had = "MOCK_SPEED" in os.environ
    import live
    if not had:
        os.environ.pop("MOCK_SPEED", None)
    assert _live_state(RESCUE_DATA=str(tmp_path))["sound"] is False and _live_state()["sound"] is live._sound_ok()
    monkeypatch.setattr(ear, "DATA", str(tmp_path))
    monkeypatch.setattr(live, "STATE", dict(_live_state(RESCUE_DATA=str(tmp_path)), running=True))
    monkeypatch.setattr(live, "_listen", ["knock"])
    assert live._switch(b'{"listen": "ear"}') == (409, dict(error="Sound clips not in this kit")) and live._listen == ["knock"]
    monkeypatch.setattr(live, "_listen", ["ear"])
    monkeypatch.setattr(live, "_cal", {})
    monkeypatch.setattr(live, "_rh", dict(live._rh))
    with pytest.raises(OSError):
        live._run_rhythm()
    assert live.STATE["running"] is False and live.STATE["phase"] == "idle" and live.STATE["error"].startswith("Calibration failed")
    got, = _page([dict(live.STATE, running=False)])
    assert got["sound_off"] and got["status"].startswith("Calibration failed")
