# biocomp-spine: Wetware scaling law

The first scaling law for living neurons: information per stimulus, in bits, as the stimulation alphabet grows
2, 4, 8, 16. Each stage stimulates one electrode per trial from a nested alphabet (the first N of 16 electrodes
spread across the array), 20 times each, in a random order, one trial per second. A reader (random forest,
5-band power 10-410 ms after the pulse, stratified folds kept in time order) guesses which electrode was
stimulated. Information = plug-in mutual information of the held-out confusion matrix minus the mean of 20
label shuffles. Controls: one fresh shuffle (the grey line) and the same reader on the 400 ms before the pulse
(must sit near 0). Artefact trap: the same reader on a window that starts AT the pulse, so it includes the
stimulation. Bigger alphabets change the stimulation, so the artefact alone can draw a rising curve (a saline
dish scored 44% on MNIST in Loeffler). The screen shows the trap greyed beside the honest number, labelled
"includes stimulation: not counted". If the trap is high and the honest number is not, the curve is artefact. Pulse: 160 us at 2 uA biphasic, 0.32 nC (limit 3 nC), 1 Hz per channel (limit 200 Hz).
Excluded: grounded 0, 7, 56, 63 and reference 4.

## Run it
    uv sync
    uv run python bci.py                  # mock culture, prints the four points (about 2 min)
    uv run python bci.py --skip-ms 50     # readout starts 50 ms after the pulse instead of 10
    NO_CELLS=1 uv run python bci.py       # dead dish: every point on the floor
    uv run python bci.py --replay         # Wagenaar 2006 rat cortex recording 1-1-20 (seconds)
    uv run python bci.py --replay 2-1-24  # another recording from $REPLAY_DIR (default data/replay)
    uv run python live.py                 # screen at http://localhost:8777, press Start (mock: about 2 min at 5x)
    uv run python live.py --replay        # the same screen from the recording, labelled REPLAY

The organisers' LIF simulator (cl-sdk 1.0.0 + organoid_simulator, cloned from github.com/wbmills/biocompute-outlier-labs at a8e9988, Python 3.13;
live.py defaults CL_SDK_DATA_SOURCE to its LIF source with coupling 4.0 mV, seed unset):

    cd ../biocompute-outlier-labs && uv sync && CL_SDK_VISUALISATION=0 PYTHONPATH=../biocomp-spine \
        uv run --with scikit-learn python ../biocomp-spine/live.py --rhythm   # pill: Simulator running live · LIF spiking network
    uv run python artefact.py             # measure the stim artefact, suggests --skip-ms (about 12 s on the dish)
    uv run python probe.py                # the pre-registered probe-omission experiment, 120 trials (--short: 80)
    NO_CELLS=1 uv run python probe.py     # simulator check: cells off, D must sit near 0
    # the design's numbers (histories, probe time, windows, blocks, burst rule, read cap) live in probe_design.json
    uv run pytest -q                      # mock limits, adapter, scaling, rhythm, scan, ear, probe (about 4 min)

`--reps N` sets trials per pattern per stage (default 20: 320 trials at 16 patterns, 600 in all, 10 min on the
real dish; `--reps 12` is 360 in all, 6 min). `--skip-ms N` (default 10, never below 10) sets where the readout
window starts. Both flags work for `bci.py` and `live.py`. The replay holds about 50 repeats per electrode, so it
caps at 12 per stage.
`MOCK_SPEED=x` paces the mock and the replay on the screen (default 5; ignored by the real CL1).

Mock, seed 0, default 20 reps, stage 2/4/8/16: 0.75, 1.51, 2.18, 2.48 bits (shuffled -0.02, pre-stim 0.10,
trap 3.40 at 16); with `--skip-ms 50`, 2.31 at 16. Simulator check (cells off): -0.02 at 16, trap 3.40 (the mock's artefact alone).
Replay 1-1-20: 0.74, 1.75, 2.12, 2.45 bits, trap 3.03. The pre-stim control reads 0.5-1.0 bits because their
stimulus order was a fixed cycle, so the previous response gives the class away. With `--skip-ms 50` the
replay drops to 0.1-0.3 bits: its information sits in the first 50 ms after the pulse. The mock's numbers say
nothing about real neurons.

## Probe-omission experiment (pre-registered, the primary result)
The design was fixed on 2 Nov at 18:25 UTC and amended after the hostile review, before any real data
(the design is recorded in `probe_design.json`). The question: does the culture's
response to an identical pulse depend on the earlier stimulation history, beyond the activity that history leaves
behind?

History A = pulses at 0, 200, 400, 600 ms; history B = 0, 100, 500, 600 ms (one fixed pattern: same count, span and
gap to the probe). The probe pulse comes at 800 ms, or its slot is left silent. Four conditions, A+ A- B+ B-, one of
each per block in random order, trials at fixed 3 s onsets: 30 blocks, 120 trials, 6 min of chip time (`--short`:
20 blocks, 80 trials, 4 min). Before them, about 40 s, all label-blind: 5 s of quiet recording sets the
network-burst threshold (mean + 5 sd of active electrodes per 50 ms bin); the 30 s scan picks the input and the
readout electrodes by their response 40-250 ms after a pulse (so the nearest electrode's artefact cannot pick
them); 20 pulses on the input check the readout for sub-millisecond-jitter events 20-40 ms after the pulse
(electrical, not neural), and if any shows them the Y window starts at 40 ms instead of 20. All of it is written to
`frozen.json` before any A/B trial. Spikes are the device's own detections; nothing here adapts a threshold per
trial (whether the CL1's detector adapts is a question for the hosts). The pulse is the same everywhere
(negative-leading biphasic, 160 us per phase at 2 uA). Every trial counts: a pulse that would go out over 2 ms
late is never fired, and its trial is flagged, never dropped or re-run. So is a trial with a pulse the device
rejects, a pulse with no echo in `analysis.stims` within 5 ms of its time, a pulse whose echo shows it was
delivered over 2 ms off schedule, or skipped ticks. Spikes are binned by
absolute ms (each ring row remembers which ms it holds), so a tick gap or ticks off the ms grid neither keep a
stale spike nor lose a real one.

Y = the raw pooled readout spike count 20-250 ms after the scheduled probe time. Per block,
D = (Y[A+] - Y[A-]) - (Y[B+] - Y[B-]). The omission trials subtract whatever the history leaves behind, so D
measures only the history x probe interaction, and a purely additive response (y = h + p) gives D = 0 exactly (a
test checks this through the same code). Reported: mean D with a 95% interval from the same sign flips (the test
inverted), the two-sided sign-flip p over blocks (the rule: p < 0.05), median D, the share of blocks with D > 0,
the minimum detectable effect at 80% power (about 0.53 sd of D at 30 blocks), the four condition means, bursts,
baseline rate, flagged trials, and the first half against the second. A null reads "No history effect larger than
X spikes either way in this dish tonight", X the larger end of the interval in size (the claim is two-sided). Sensitivity, fixed in advance: without
trials that had a burst in the 1 s before history onset; and Y minus the same electrodes' count in a matched 230 ms
window before history onset. Descriptive, not tests: Y[A-] against Y[B-] (the leftover activity alone), and D in
the 20-60 and 60-250 ms sub-windows. Secondary, labelled as such: a logistic decoder of A vs B from the probe
trials' per-electrode counts, trained on the first half of the session and scored once on the second. Spike counts
only: band power of an additive voltage makes an interaction by itself (p^2 + 2hp). Without a cell-free chip,
electrical-interface effects are not excluded; on the mock, the simulator check (cells off) is that control.

    uv run python probe.py [--short] [--seed 0]
    uv run python live.py --rhythm [--short] [--cal 20] [--port 8777]   # "Run the experiment", before Calibrate

Every number above lives in `probe_design.json` (histories, probe time, onset period, blocks, Y window and its
electrical fallback, sub-windows, selection window, jitter check, quiet time, burst rule, sign flips, readout cap)
and is copied into `frozen.json` with each run. On the screen it is the top card: the four conditions' live
rasters with the probe time marked (dashed where the slot is silent) and the read window shaded, then the four
means and D with its interval. A null shows as a null.

Practice-dish results (mock, simulated; says nothing about real neurons), 30 blocks of 4, 0 flagged trials,
two-sided sign flips, D in pooled spikes:

| run | mean D | 95% CI | p | median D | D > 0 | MDE (80%) |
|---|---|---|---|---|---|---|
| seed 0 | +0.13 | -20.1 to +20.6 | 0.99 | +3.0 | 53% | 28.8 |
| seed 1 | +2.93 | -15.7 to +21.6 | 0.75 | +1.5 | 50% | 26.5 |
| seed 2 | -13.30 | -37.1 to +10.6 | 0.26 | -41.5 | 33% | 33.7 |
| seed 3 | +6.97 | -8.7 to +22.6 | 0.37 | +3.5 | 53% | 22.2 |
| cells off | +0.07 | -1.4 to +1.5 | 0.97 | +0.5 | 50% | 2.0 |
| --short, seed 0 (20 blocks) | -3.60 | -26.8 to +19.2 | 0.75 | -3.0 | 50% | 32.0 |

No history effect on any seed; the cells-off D sits at 0 as it must. Not corrected for the many secondary
lines: seed 2's baseline-subtracted D is -32.4 (p = 0.029) and its Y[A-] - Y[B-] is +20.3 (p = 0.011); seed 3's
20-60 ms D is +3.6 (p = 0.029). The decoder scored 47-57% on the second half everywhere. Logs: `uv run python
probe.py --seed N` prints every line.

Earlier the same day, a first design (decode A vs B from the probe response alone, calibration block, frozen
decoder, test block) failed on every practice-dish seed:
30 + 30 per block, test block: seed 0 27/60 (45%, 95% CI 33-58%, p = 0.82), seed 1 27/60 (45%, p = 0.82), seed 2 33/60 (55%, CI 42-67%, p = 0.26),
seed 3 23/60 (38%, CI 27-51%, p = 0.97); simulator check (cells off) 29/60 (48%, p = 0.65). All FAIL, 0 dropped
trials, baseline 6-7 Hz per read electrode (cells off 0.2-0.3 Hz). With band power as a third candidate: seed 0
chose it (calibration CV 62%) and scored 33/60 (55%, p = 0.26), with the control also 55%; seeds 1-3 chose counts.
The mock predicts this: its short-term plasticity (U = 0.6, recovery 0.4 s) leaves the probe's release factor at
0.806 after A and 0.800 after B (sd 0.018) at a 500 ms final gap, under 1%, too small for 60 trials to see.

## Rescue Listener: knock-rhythm mode
The judge knocks a rhythm (space bar, the on-screen pad, or knocks near the laptop mic). Each knock becomes one
pulse (the same safe pulse) on input electrode 27 at the knocked timing. The dish's short-term memory (paired-pulse
facilitation and depression on 50-500 ms gaps) makes its response to each pulse depend on the gaps before it, so
the rhythm is readable from the neurons. Survivor = SOS (3 fast, 3 slow, 3 fast), any repeated 3-knock motif, or five_tap_reply (5 even taps 0.4-0.7 s
apart, the reply to a rescuer's "tap five times"; `FIVE_TAP` in `rhythm.py`);
debris = the same number of knocks at random times over the same span. Calibration plays 40 of each (about 3.5 s
apiece with the rest, about 5 min on the dish). Reader: spikes 10-50 ms after each pulse on the 24 electrodes
within 2 of the input, summarised over the sequence (stats, lag 1-3 autocorrelation, the first 12 in order), plus
5-band power after the last pulse; random forest, folds by sequence in time order.

    uv run python rhythm.py               # calibrate and print the tally (or: uv run python bci.py --rhythm)
    NO_CELLS=1 uv run python rhythm.py    # simulator check, cells off: chance
    uv run python live.py --rhythm        # screen: Calibrate (about 1 min on the mock at 5x), then knock
    # on the screen, Scan (about 30 s) picks the input electrode and the 24 readouts by evoked response first

Controls: shuffled labels, the simulator check (cells off), and the timing-only baseline (the same reader on the knock timings alone).
Software reads rhythm from timing at 99%: the screen shows that number so "couldn't software do it?" is asked
and answered. The claim is that living neurons can read it too, from their own memory, not that they beat software.
Mock, five_tap_reply on (the default), seeds 0-3: 55%, 68%, 50%, 72% cross-validated (a calibration diagnostic, not a prospective test), mean 61% (seed 3: p = 4e-5;
seed 0: p = 0.22, not significant); shuffled 49-51%; timing-only 94-98%; simulator check (cells off) 45% (seed 0). Without five-tap
(`FIVE_TAP = False`), seed 0: 54/80 (68%, p = 0.001). One seed of 80 trials swings by 10-20 points, so single-seed
numbers are not comparable. Memory off (the mock's paired-pulse facilitation and depression patched to 1.0,
five-tap on), seeds 0-3: 57%, 38%, 57%, 38%, mean 47%. On average the score comes from the memory, but seeds 0
and 2 score about the same either way: four seeds of 80 trials are a weak test of that claim. These numbers are
after the post-pulse power stopped including the stimulated electrode (it now averages the read set only).
Earlier, before the mock moved its pulses to iteration_timestamp + 80 us like the real API, seed 0 read 75%;
shifting the pulse by -25, -23 or -50 frames gave 76, 62, 60% at seed 0, so that was spread, not a timing effect.

The judge's verdicts are in `/state`: `latest` and the last 20 in `verdicts`, each
`{id, knocks (ms), verdict: "survivor"|"debris", confidence, timing_only: {verdict, confidence}}`; `calib` holds the
tally. The screen shows `survivor` as "Target rhythm" and `debris` as "Matched irregular rhythm": the output is
"target rhythm detected", and rescue is the application. The keys stay as they are because the scene reads them. The 3D rubble scene mounts in `#scene-slot` on `rhythm.html` and polls `/state`. `POST /knock
{"knocks": [ms, ...]}` queues a rhythm (cleaned: sorted, knocks under 60 ms apart merged, at most 20 knocks and
4 s; at least 3); it answers 409 until calibration is done. `POST /hear {"pattern": "five_tap_reply"}` (or `{}` for any)
plays a random rescue-data trial of that pattern: in knock mode its knock times, in ear mode its sound (never a
calibration clip); the page's "Play a five-tap reply" button sends it, and the WAV plays for the room. `--rhythm` with `--replay` is refused: a recording
cannot be stimulated.

## Blind piles
Shows the system never knew where the person was. Three 6 s listening windows, Pile 1, 2 and 3. The judge picks
0 to 3 target piles in secret, taps the target rhythm (the five-tap reply) on those, and taps like rubble falling
(4-7 random taps) on the rest. Every window is live taps, so the code cannot know which piles were chosen. Each
window goes to the dish exactly as in knock mode, and a pile is flagged when its P(target) is above the threshold
frozen at calibration (the 95th percentile of cross-validated P(target) on the irregular trials, in the `knock`
section of `frozen.json`). Piles are flagged one by one, so a tie between piles needs no rule: both are flagged or
neither. Until Reveal nothing hints at a pile: no tap timeline, no tap count, no verdict card, no free-knock
panel, and `/hear`, `/mode` and free `/knock` answer 409. The judge marks their targets and presses Reveal; then
the screen shows the true target, the dish's one-line outcome ("Dish flagged Pile 1", "Dish flagged no pile", or
"Dish unsure: no score for Pile N" when a pile can't be scored), and a table: flagged piles, hits, false flags and
misses for the dish, for the software timing rule, and for chance (a fair coin per pile, expected numbers; it gets
a whole round right 1 time in 8). The three bars are in a neutral colour with a tick at the frozen threshold
(flagged piles in the flag colour; a pile with no score says why, e.g. under 3 taps). One round is a demo, not evidence.

In `/state`: `piles_phase` (`listening:N`, `judging`, `heard`, `done`); once done, `piles` (`[{id, taps,
p_survivor, dish_verdict, software_verdict, software_p_survivor}]`), `targets`, `flagged` and `software_flagged`
(lists of pile numbers), `outcome` (that phrase), `threshold`, `software_threshold`, `tally` and `software_tally` (`{hits, false_flags,
misses}`). `POST /piles` starts a round; `POST /knock {"knocks": [...], "pile": N}` is window N; `POST /reveal
{"targets": [1, 3]}` publishes the result (409 before all three are heard, 400 for anything but distinct piles 1-3).

    uv run python live.py --rhythm        # Calibrate, then Blind piles, then Listen to pile 1, 2, 3, then Reveal
    uv run python piles.py                # 100 simulated rounds on the practice dish (about 20 min on the mock)

Simulated on the mock, 100 rounds, seed 0 (28 rounds with no target; chance is per pile: a coin that flags a share
q of piles flags q of targets and q of the rest, a fair coin gets 12.5% of rounds exactly right, never flagging gets
28%): the dish (threshold 0.70, from a 55% calibration) flagged 1% of target piles and 2% of the rest, so its 27%
of rounds right (29% against debris trials) is the never-flag rate. Software on timing (threshold 0.35) flagged
100% of targets, 44% of human rubble piles and 13% of debris trials; rounds right 49% and 81%. On the mock the dish
does not find the piles; software finds them and over-flags human taps.

## Rescue Listener: ear mode
The dish hears the sound, not knock times. Each WAV trial from `../rescue-data` beside the spine (read only; `RESCUE_DATA` overrides) is cut into 50 ms
frames with power in 16 log-spaced bands (100 Hz-7 kHz) laid tonotopically on electrodes 24-39 (rows 3-4 as a
snake, neighbouring pitches on neighbouring electrodes). A frame sounds when its 100-1400 Hz bands (where a knock
is) average 9 dB over the clip's median; its 3 loudest bands that are 6 dB up each get one pulse. The reader reads
rows 1, 2, 5 and 6; controls as knock mode (shuffled, simulator check with cells off, software on the same pulse timings).

    uv run python ear.py [--snr 10|0|-10] # calibrate on 40 + 40 clips and print the tally (default +10 dB)
    NO_CELLS=1 uv run python ear.py       # dead dish: chance
    EAR_SNR=10 uv run python live.py --rhythm   # screen: the Knocks | Sound switch, Calibrate, then Play a clip

Honest state: the beds are full of debris impacts, so the knock rhythm is hard to hear in 50 ms frames. Software
on the true knock times reads 99%; software on this encoding's pulse times reads 67% at +10 dB and chance at 0
and -10 dB (400 trials, 5-fold). Calibration now draws 40 + 40 at random over all 534 trials (every
pattern and both beds). Mock, +10 dB, seeds 0-3 (mock seed and forest seed both): dish 61%, 45%, 48%,
54%, mean 52%; shuffled 48-50%; software on the same pulse timings 55%, 56%, 54%, 49%; simulator check (cells off) 46% (seed 0);
about 28 pulses per clip. Ear mode sits at about chance on the mock; the 61-62% at seed 0 is a lucky draw.

## On the real CL1
1. `uv add cl-sdk` (Python >= 3.12; on the device the system build is used, do not pin it).
2. No code edit: `adapter.py` is the only module that imports `cl` (`try: import cl except ImportError:
   import cl_mock`); every other file goes through it. Lines marked `# CHECK:` are not confirmed yet. Check with
   `uv run python -c "import cl; print(cl.__file__, cl.is_simulator())"`. The screen's pill then reads
   "CL1 · live neurons" (or "cl-sdk simulator" if `is_simulator()` is true).
3. `uv run python bci.py` for the table, or `uv run python live.py` for the screen.
4. `uv run python artefact.py` first: if it suggests more than 10, pass that `--skip-ms`.
5. Dead-dish control: a saline or cell-free dish. A custom SDK simulator may respond to stimulation
   (the organisers' LIF simulator does), so simulator mode alone is not a dead-dish control.

## The three knobs (top of `bci.py`)
- **Stim pattern**: `ALPHABET` (16 electrodes; put the first ones near live units), `STIM` (current), `PULSES`/`PULSE_HZ`.
- **Window**: `SKIP_MS` (`--skip-ms`, never below 10, the artefact; try 50) and `WINDOW_MS`; `TRIAL_S` must stay
  longer than both.
- **Decoder**: `TREES` and the classifier in `info()`.
Change one at a time and keep the printed table of every run.

## Known limitations (every `shortcut:` in the code; check these first on arrival)
- `grep -n "CHECK:" *.py`: the raw-to-uV factor 0.195, re-iterating the Loop after `recover_from_jitter()`, and
  that the stim echo shows in `tick.analysis.stims` on the device (windows lock to it; without it they keep the
  expected time, iteration_timestamp + 80 us).
- Safety limits (<= 3 uA, <= 3 nC per phase, biphasic alternating, <= 200 Hz per channel, lead >= 80 us, never
  electrodes 0, 7, 56, 63 or reference 4) are enforced in `adapter.py` and the mock raises on any breach.
- `record()` is a no-op in the mock.
- Features are one FFT per 400 ms window, not Loeffler's 10 ms bins.
- Rhythm: input 27 and its neighbours by default; on the dish press Scan, which picks both by evoked response.
  The ordered part of the rhythm features assumes about 9-12 knocks like calibration; longer rhythms are cut.
  The mic onset detector is a level jump over an adapting room level: test it in the room, and fall back to the
  pad. The mic needs https or localhost.
- Replay features are spike counts per channel, 10-290 ms (the files hold spikes, and the next pulse comes at 301 ms); its channels are drawn on the grid in index order, not the true MEA layout.
- Live estimates run in a separate process (a forest in a thread pushed the tick p99 to 3.5 ms). With it, tick
  p99 (wall) is about 0.3 ms on the mock; the tests check the p99 of the body's own CPU time, which a busy VM does
  not inflate. Wall p99.9 is about 4.5 ms. A `TimeoutError` from the device is recovered with
  `loop.recover_from_jitter()` (up to 20 times a run, counted and printed); past that, use `bci.py`.

## Offline
Fonts are vendored in `static/` (Space Grotesk and IBM Plex Mono, latin subset, both SIL OFL 1.1). The page loads nothing from the network.
