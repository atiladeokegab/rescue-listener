# rescue-listener

The biocomp hack kit in one repo:

| Folder | What it is |
|---|---|
| `biocomp-spine/` | The closed-loop spine (`bci.py`, `rhythm.py`, live screen). Runs on the real CL1 / cl-sdk if `cl` imports, otherwise on `cl_mock` (simulated dish). |
| `rescue-scene/` | 3D rubble scene (offline three.js, vendored). The demo screen shows it in its slot; nothing to run separately. |
| `rescue-data/` | Knock, debris and survivor audio plus sources (`CREDITS.md`, `survival.json`). |

## Laptop setup

Windows, PowerShell. The demo runs on the organisers' LIF simulator. Steps 1 to 4 need a network: do them at home.
Put both repos in the same folder, side by side.

1. **Clone the kit:**
   `git clone https://github.com/atiladeokegab/rescue-listener`
2. **Clone the simulator** next to it, at the pinned commit:
   `git clone https://github.com/wbmills/biocompute-outlier-labs; cd biocompute-outlier-labs; git checkout a8e9988`
3. **Install uv:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, then open a new PowerShell window.
4. **Sync, and fetch everything once while online.** In `biocompute-outlier-labs`, run `uv sync` (it fetches Python 3.13 and cl-sdk). Then run step 5 once, so uv caches scikit-learn, and stop it with Ctrl+C.
5. **Start**, from the `biocompute-outlier-labs` folder:
   ```powershell
   $env:CL_SDK_VISUALISATION = "0"; $env:PYTHONPATH = "..\rescue-listener\biocomp-spine"
   uv run --with scikit-learn python ..\rescue-listener\biocomp-spine\live.py --rhythm
   ```
   The PowerShell window prints `http://localhost:8777  (cl-sdk simulator · no cells)`: that confirms it is on the organisers' simulator. Open http://localhost:8777.
   The two `$env:` settings last only for that PowerShell window. In a new window, set them again.
   For the night, add `--short` for the short memory test: it runs 20 blocks instead of 30 (about 4 min instead of 6).
   If chip time is under about 10 minutes, add `--cal 20` (calibration in about 2.3 min instead of 4.6).
6. **Check:**
   - the 3D rubble scene shows in its slot;
   - press **Calibrate** and wait for **Calibrated**;
   - knock 5 even taps on the pad (or Space), then pause 1.2 s; all 5 taps appear;
   - turn Wi-Fi off, stop and start it again, and reload: it still works.
   - Sound mode is not used on the night: stay in Knocks.
7. **If something goes wrong:**
   - *Port taken* (`WinError 10048`): another `live.py` is running. Close its window, or start with `--port 8778` and open localhost:8778.
   - *uv missing* (`'uv' is not recognized`): open a new PowerShell window after step 3; if it still fails, run step 3 again.
   - *The window prints "Practice dish"* instead of "cl-sdk simulator": cl-sdk did not load: you started it from the wrong folder. Go back to step 5 and start it from `biocompute-outlier-labs`. That folder also selects the LIF network, through its `.env`.
   - *`No module named sklearn` or a download error with Wi-Fi off*: step 4's online run was skipped. Run it with Wi-Fi on.
   - *Scene not loading* (empty slot): press Ctrl+F5. Then check that `rescue-scene` still sits inside `rescue-listener`.
   - *Simulator won't start at all*: fall back to the built-in practice dish. Run `cd ..\rescue-listener\biocomp-spine; uv sync; uv run python live.py --rhythm`.
   - *Windows firewall prompt* on start: Allow or Cancel both work; localhost does not need it.

## Setup

**Run setup at home, before you leave: it downloads Python 3.14 and the packages, so it needs a network. After that everything runs offline.**

You need only [uv](https://docs.astral.sh/uv/) and git. uv fetches Python 3.14 on its own (pinned in each piece's `.python-version`).

### Windows (PowerShell)

```powershell
git clone https://github.com/atiladeokegab/rescue-listener.git; cd rescue-listener
powershell -ExecutionPolicy Bypass -File setup.ps1     # uv sync in biocomp-spine and rescue-data
```

Run it:

```powershell
cd biocomp-spine; uv run pytest -q                    # spine tests on the simulated dish
uv run python live.py --rhythm                        # demo screen at http://localhost:8777
```

### Mac and Linux

```bash
git clone https://github.com/atiladeokegab/rescue-listener.git && cd rescue-listener
./setup.sh                                             # uv sync in biocomp-spine and rescue-data
./smoke.sh                                             # spine tests + headless demo-screen load, no outside network
cd biocomp-spine && uv run python live.py --rhythm     # demo screen at http://localhost:8777
```

Each Python piece keeps its own environment and `uv.lock` (both pinned to Python 3.14, the version they were tested on).
`scripts/build.py` in rescue-data renders the trial audio in `sequences/snr*/`. That audio is not in git, so the kit leaves it out.
The script calls `curl` and `timeout`, so it is Mac/Linux only. Knock mode does not need the audio.

`DROPPED.txt` lists anything left out at assembly (`.venv`, `__pycache__`, files over 50 MB).
