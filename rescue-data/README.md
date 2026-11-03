# Rescue Listener data

Audio and figures for the Rescue Listener demo, where living neurons on a Cortical Labs CL1 must tell a trapped survivor's knocking rhythm from random falling debris. `knocks/` holds 20 clean single door knocks (16 kHz mono, CC BY 4.0, see CREDITS.md); `synth/` holds labelled synthetic debris, drone-rotor and distant-machinery beds; `sequences.json` defines 267 survivor trials (SOS, a repeated 3-knock motif, 2-2 pairs, and `five_tap_reply`: 5 even taps about 0.4-0.7 s apart, the reply to a rescuer's "tap five times") and 267 debris trials matched one-to-one on knock count, span and bed, with knock times in ms, class, pattern, seed and `bed`; two beds exist, `rotor` (debris + grit + hovering-drone hum, used by the first 400 trials and the first three demo clips) and `rotor_off` (debris + grit + distant machinery, no rotor hum: the default for new renders, for the node a drone lands on the rubble before switching its rotors off); `sequences/snr+10|snr0|snr-10/` renders each trial as a 4 s WAV with the knocks mixed into the synthetic bed (these renders are gitignored and regenerate deterministically); `demo/` holds five 8 s showcase clips, of which `node_five_tap_reply_snr0.wav` (five taps, a pause, five taps) and `node_debris_only.wav` use the rotor-off bed; `survival.json` gives sourced, checked earthquake time-to-rescue figures; `manifest.json` lists every file with sha256, licence, source and a synthetic flag (`"mixed"` means real knocks over a synthetic bed). Both classes use the same knocks and gains, so only timing tells them apart.

Regenerate everything except survival.json (hand-checked against its sources) and these docs:

```bash
uv run python scripts/build.py
```
