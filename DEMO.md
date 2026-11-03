# Run the live demo on the organisers' LIF simulator (Windows)

Windows, PowerShell (5.1 or 7). Steps 1 to 3 need a network.

## 1. Clone both repos, side by side

```powershell
mkdir C:\demo; cd C:\demo
git clone https://github.com/atiladeokegab/rescue-listener
git clone https://github.com/wbmills/biocompute-outlier-labs
cd biocompute-outlier-labs; git checkout a8e9988
```

No uv yet? `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, then open a new PowerShell window and `cd C:\demo\biocompute-outlier-labs`.

## 2. Sync the simulator (Python 3.13 and cl-sdk)

```powershell
uv sync
```

## 3. Set the environment (this window only)

```powershell
$env:CL_SDK_DATA_SOURCE        = 'organoid_simulator.lif_data_source:make_lif_source'
$env:CL_SDK_DATA_SOURCE_CONFIG = '{"coupling_gain_mV":4.0}'
$env:CL_SDK_VISUALISATION      = '0'
$env:PYTHONPATH                = 'C:\demo\rescue-listener\biocomp-spine'
```

Use single quotes, exactly as written, so PowerShell passes the JSON unchanged. Check it with `$env:CL_SDK_DATA_SOURCE_CONFIG`, which must print `{"coupling_gain_mV":4.0}`.
Do not use `set` or `setx`: `set` is cmd syntax, and `setx` only affects new windows.
Leave `CL_SDK_ACCELERATED_TIME` unset: the demo must run in real time.

## 4. Start, from `C:\demo\biocompute-outlier-labs`

```powershell
uv run --with scikit-learn python C:\demo\rescue-listener\biocomp-spine\live.py --rhythm --short
```

The page's pill reads `Simulator running live · LIF spiking network`. If it says `Practice simulator`, cl-sdk did not load: check that you are in `biocompute-outlier-labs`.

## 5. Open the page

Open http://localhost:8777 in the browser.

## 6. The 3D scene

Nothing to start. `live.py` serves the scene itself, in the slot on the same page (`/scene/`), from `C:\demo\rescue-listener\rescue-scene`. `setup.ps1` only runs `uv sync`; it does not start the scene.
If the slot is empty, press Ctrl+F5.

## 7. Check before the judges arrive

1. The scene shows in its slot.
2. Press **Calibrate** and wait for **Calibrated**.
3. Knock 5 even taps on the pad (or Space), then pause 1.2 s. All 5 taps appear.
4. Stay in Knocks: Sound mode is not used tonight.

## If something goes wrong

- **Port taken** (`WinError 10048`): close the other `live.py` window, or add `--port 8778` and open localhost:8778.
- **`'uv' is not recognized`**: open a new PowerShell window after installing uv.
- **`No module named sklearn` with no network**: run step 4 once while online, so uv caches scikit-learn.
- **Simulator won't start**: fall back to the practice simulator. Run `cd C:\demo\rescue-listener\biocomp-spine; uv sync; uv run python live.py --rhythm`. The pill reads `Practice simulator`, so say "practice simulator" in the demo.
- **Firewall prompt**: Allow or Cancel both work; localhost does not need it.

## For a local coding agent setting this up

Follow steps 1 to 5 in order, run in PowerShell. Success means http://localhost:8777 loads, the pill reads
`Simulator running live · LIF spiking network`, and Calibrate reaches **Calibrated**. Do not set
`CL_SDK_ACCELERATED_TIME`. If `uv` is missing, install it as step 1 says. Report the pill text and any error.
On macOS or Linux, the same steps work with `export VAR='...'` in place of `$env:VAR = '...'` and `/` paths.
