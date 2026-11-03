# Rescue Listener scene

A drone over the January 2010 airborne LiDAR of Port-au-Prince, around the collapsed
National Palace. Plain static three.js (vendored), runs fully offline.

```bash
uv run --no-project python -m http.server 8791 --bind 127.0.0.1   # then open http://127.0.0.1:8791/
```

## Driving it: `window.rescue`

| Call | What happens |
|---|---|
| `rescue.listen(on = true, source)` | The drone hovers over the pile and scans (fast-forwards the flight if still in transit). `source` (`'live'`, `'practice'`, `'simulator'` or `'manual'`) sets the caption naming where the knocks go, e.g. "Knocks are sent to the practice dish (simulated) at base." Throws on any other source. |
| `rescue.dropNodes(piles = PILES, source)` | **The main ending.** The drone sets one listening node (a microphone) on each of three rubble piles (or pass three `[x, z]` in scene metres), lands clear of them, and its rotors spin down to a full stop. All three nodes look, pulse and are framed the same. `source` as for `listen`. |
| `rescue.listenPile(i)` | `i` is 1 to 3. Sent before the nodes' flight starts, it is held and applied then. That node's ring pulses, the others dim, and the HUD says "Pile i listening". Every pile gets the same ring, pulse timing and camera framing, so nothing hints at the answer before `setPileVerdicts`. |
| `rescue.setPileVerdicts(list, flagged = [], source = 'manual', { target, note } = {})` | `list` is three `{ p_survivor }` (0 to 1, or `null` when unscored, in pile order); `flagged` is a list of distinct pile numbers (`[2]`, `[1, 3]`, `[]`). No score or percentage is shown anywhere. `target` is the pile the survivor is really under, when the caller knows it: a flagged target goes green, "Target rhythm flagged", with a ripple; a false flag is amber, "Flagged, no survivor"; a missed target is outlined, "Missed". `target: null` (the spine sends this whenever the reader is at chance) gives a neutral ink flag, never green: "Practice · Pile 2 flagged" on the practice dish. `note`, if given, must be a string; it shows in small ink under the verdict, e.g. "Reader 55% on held-out sequences · shuffled labels 49%". Every verdict on the practice dish reads "Practice · ...". Throws on a wrong length, a bad `p_survivor`, a bad pile number or target, a non-string note, or an unknown source. Sent before `dropNodes()` or while the drone is still en route, it is checked at once, held (the latest call wins) and applied when the nodes' flight starts, never lost. `setVerdict` and `markSpot` are held the same way while a drop is pending. A second `dropNodes()` that replaces a drop still pending drops the held calls; calls made before the very first drop still apply to it. |
| `rescue.dropNode(source)` | Single-pile fallback: one node on the heap, then land and stop. |
| `rescue.setVerdict(kind, confidence, source = 'manual')` | Single-pile verdict. `kind` `'survivor'` shows "Target rhythm flagged" (green), `'debris'` "No target rhythm" (grey), `'unsure'` "Unclear". `confidence` must be 0 to 1 but is never shown. Throws on anything else. |
| `rescue.markSpot()` | Secondary ending. A marker drops from the drone onto the target and a short beam rises from it. A later `setVerdict` other than `'survivor'` takes the mark down; any new `setVerdict` clears an earlier pile verdict. |
| `rescue.audioLevel(x)` | Optional: feed a live level 0 to 1 to the verdict-card waveform. Without it the waveform shows rotor hum in flight and a neutral floor while listening. It is never drawn from a verdict, so it is the same on every pile. |
| `rescue.reset()` | Back to the start of the flight. |
| `rescue.seek(seconds)` | Jump along the flight (it arrives at 42 s). |
| `rescue.state` | `{ t, arrived, listening, verdict, source, marked, nodes, node, pile, pileVerdicts, dronePx, fps }`. |

`source` is `'live'`, `'practice'`, `'simulator'` or `'manual'` and defaults to `'manual'`; it shows on the
always-visible line "Flight: scripted · Verdict: …". The dish bridge must pass `'live'`. `'simulator'` (the spine's LIF
spiking network) shows "Simulator running live · LIF spiking network" and prefixes each verdict "Simulated · …".

Keys (hint row shown with `?keys`): **L** listen, **S** target-rhythm verdict, **D** no-rhythm verdict (both source manual), **M** mark spot, **N** drop node, **R** reset.

Main ending: `dropNodes()`, `listenPile(1..3)` in turn, then `setPileVerdicts([{ p_survivor: .18 }, { p_survivor: .91 }, { p_survivor: .22 }], [2], 'live')`.
Fallbacks: `dropNode()` then `setVerdict('survivor', .87, 'live')`; or `listen() → setVerdict(...) → markSpot()`.

Stills and fps: `uv run --no-project --with playwright python -I scripts/shoot.py <out_dir>`
(page served on :8791). `?capture` takes big time steps for software GL.

`?lite=1` drops the AO pass, renders at DPR 1 with a 1024 shadow map and a third of the haze,
smoke and dust. It also switches on by itself when the first 60 frames average under 40 fps
(never under `?capture`). `rescue.state.lite` says which mode is running.

## The crop on Google Maps

UTM 18N (EPSG:32618) 780757–781057 E, 2052199–2052499 N: a 300 m square, drawn as the
circle inscribed in it.

- NW: [18.544472, -72.340395](https://www.google.com/maps?q=18.544472,-72.340395)
- NE: [18.544432, -72.337556](https://www.google.com/maps?q=18.544432,-72.337556)
- SE: [18.541724, -72.337598](https://www.google.com/maps?q=18.541724,-72.337598)
- SW: [18.541764, -72.340437](https://www.google.com/maps?q=18.541764,-72.340437)

LiDAR: Haiti 2010 earthquake airborne LiDAR, OpenTopography, DOI 10.5069/G96Q1V50, public domain.

Aerial: World Bank 2010, RIT WASP, via OpenAerialMap (CC BY 4.0). The 21 Jan 2010 photo,
reprojected to exactly the crop (`data/aerial.json` has the source, licence and processing);
its scattered black nodata pixels are filled from the nearest valid pixel. It lines up with
the LiDAR to about 1 m; the palace front block and east wing lean 3-6 m south in the photo.

Sky and light: Poly Haven, "Kloofendal 48d Partly Cloudy (Pure Sky)", 2k HDR, CC0
(https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky), vendored in `vendor/hdri/`.
The sun is placed where it sits in that file (azimuth 124°, 48° up), which matches the shadows
in the aerial photo.

Drone model: "Quadrocopter 2" by yanix, CC BY 4.0
(https://sketchfab.com/3d-models/quadrocopter-2-6885c5f1a68148e1b2b3a4f69b8b3d70; licence
http://creativecommons.org/licenses/by/4.0/), vendored as `vendor/models/quadrocopter2.glb`
from the Objaverse copy of that upload, unmodified. Scaled to 1.0 m across the rotors; its four
propeller nodes spin and stop on landing.

three.js r179 and its addons are vendored in `vendor/three/` (MIT).
