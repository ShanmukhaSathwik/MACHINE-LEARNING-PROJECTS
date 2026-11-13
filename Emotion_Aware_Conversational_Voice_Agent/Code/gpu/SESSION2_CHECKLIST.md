# GPU session 2 — checklist

Step by step, in order. All commands are **PowerShell**, run from the `Code\` folder unless a step says otherwise.
`SETUP_GPU.md` has the background for each step (venv, token, USB copies).
In the commands below the USB drive is `E:`. Change the letter if yours differs.

Preset: session 1 trained T3 with **`gpu_24gb`** (the run is called `gpu_24gb_20261112_1500`). Session 1's
`check_env.py` reported an NVIDIA RTX 5000 Ada Generation with 31.99 GB VRAM total, which is in the
"22 GB or more" row of the table in `SETUP_GPU.md` section 2a. Use the same preset, unless `check_env.py` now
reports less VRAM. In that case pick the preset from that table and use it in every command below.

On the GPU machine, `GPU_OUTPUTS_DIR` is **not set**. So every `gpu\outputs\...` path below is relative to `Code\`
there (session 1's report: outputs = `...\Code\gpu\outputs`).

---

## A. Before leaving home

### A1. Re-record the room noise (audio enhancements OFF)
1. Windows **Settings → System → Sound → Input** → pick the microphone you will use → **Audio enhancements: Off**.
   Also set the input volume you will use for the agent, and keep the mic where it will be.
2. Record 10 seconds with **nobody talking**. Press Enter, then stay silent:
   ```powershell
   cd F:\projects\Final_project\Emotion_Aware_Conversational_Voice_Agent\Code
   .\.venv312\Scripts\Activate.ps1
   python scripts/record.py room_noise --seconds 10 --out data/room_noise
   ```
   - `record.py` warns that the clip is almost silent. For this recording that is expected.
   - It refuses to overwrite an existing `room_noise.wav`. If one exists from an earlier try, rename the old
     file first (`Rename-Item data\room_noise\room_noise.wav room_noise_old.wav`), then record again.
3. Measure:
   ```powershell
   python scripts/measure_silence_gate.py
   ```

### A2. Apply the gate only if the script's own check passes
- **Check passed:** the script prints a line `gate = <value> dBFS (noise ... + ... dB margin | ... dB margin + ... quietest speech)`.
  Put that value into `gpu\gpu_config.yaml`, at `asr: silence_dbfs:` (in `base:`), and change its comment to
  say "measured by scripts/measure_silence_gate.py, results/metrics/silence_gate.csv".
- **Check failed:** the script prints `NO GATE separates them`. Leave `silence_dbfs` unchanged. Move the mic or
  cut the noise, then redo A1. Do not pick a number by hand.
- Keep in mind: the speech side comes from the corpora's own recordings, while the noise comes from your mic.
- Then run the tests:
  ```powershell
  python -m pytest -q
  ```

### A3. Push the repo state
Steps 8b, 9a and 9b are committed already. Step 10 (`gpu/bench_gpu.py`, `gpu/bench_turns.yaml`,
`tests/test_bench_gpu.py`, `gpu/SESSION2_CHECKLIST.md`, `gpu/gpu_config.yaml`) and any gate change from A2 still
need committing.
```powershell
git status                      # nothing left to commit for Code\ when you are done
git add gpu/bench_gpu.py gpu/bench_turns.yaml gpu/SESSION2_CHECKLIST.md gpu/gpu_config.yaml tests/test_bench_gpu.py results/metrics/silence_gate.csv
git commit -m "Step 10: bench + session 2 checklist"
git push
```
Optional plumbing check at home (no models, no network; its CSV stays in the bench folder, not in `results\metrics\`):
```powershell
python gpu/bench_gpu.py --preset smoke_cpu --stub --quiet
```

### A4. Copy to the USB
```powershell
$USB = "E:\session2"
New-Item -ItemType Directory -Force $USB | Out-Null

# T3 checkpoint (best\ only, about 1.2 GB): config.json, id2label.json, model.safetensors, preprocessor_config.json, config_used.yaml
robocopy $env:GPU_OUTPUTS_DIR\ser_finetune\gpu_24gb_20261112_1500\best $USB\best /E

# CSM voice reference clips (CALM, ENCOURAGE, NEUTRAL, VALIDATE .wav) + their record (also in git)
robocopy $env:GPU_OUTPUTS_DIR\voice_refs $USB\voice_refs *.wav
Copy-Item gpu\voice_refs.json $USB\
```
If the GPU machine's work folder was cleaned since session 1, also bring the training audio (the bench uses
test-split clips from `data\raw`). Use the command from `SETUP_GPU.md` section 2a:
```powershell
robocopy data\raw E:\ser_audio\raw *.wav /S /XD .git
```
robocopy exit code 1 means "files copied". 8 or higher is a failure.

---

## B. On the GPU machine

### B1. Open the session
Session 1 used a **full clone** on this machine (paths from session 1's `check_env.py` report):
`C:\shan\MACHINE-LEARNING-PROJECTS\Emotion_Aware_Conversational_Voice_Agent\Code`, venv `.venv_gpu`.

**First check that the folder is still there:**
```powershell
$CODE = "C:\shan\MACHINE-LEARNING-PROJECTS\Emotion_Aware_Conversational_Voice_Agent\Code"
Test-Path $CODE\.venv_gpu\Scripts\Activate.ps1        # True -> continue below. False -> see the note under this block
```
- **True:** open the session.
  ```powershell
  cd $CODE
  git pull
  .\.venv_gpu\Scripts\Activate.ps1
  $env:HF_HOME = "C:\shan\hf_cache"     # model downloads go here, for this window only
  $env:GPU_OUTPUTS_DIR                  # must print NOTHING here (everything goes to gpu\outputs\)
  git log --oneline -1                  # must show the commit you pushed in A3
  ```
- **False** (folder or venv gone): do the fresh setup in `SETUP_GPU.md` section 1 first. It ends inside the
  new `Code\` folder with the venv active. Point `$CODE` there, then continue at B2:
  ```powershell
  $CODE = (Get-Location).Path           # run this inside the new Code\ folder
  ```

### B2. Install the two new pins
```powershell
pip install -r gpu\requirements_gpu.txt          # adds sounddevice==0.5.6 and pyttsx3==2.99; the rest is already installed
python -c "import sounddevice, pyttsx3; print('sounddevice', sounddevice.__version__, '| pyttsx3 ok')"
```

### B3. Put T3 and the voice clips in place
Run this from `Code\` (`cd $CODE`). `GPU_OUTPUTS_DIR` is unset here, so the agent looks for T3 at
`Code\gpu\outputs\ser_finetune\gpu_24gb_20261112_1500\best` (`ser.checkpoint` in `gpu_config.yaml`).
Anywhere else, the agent falls back to the pretrained model.
```powershell
cd $CODE
robocopy E:\session2\best gpu\outputs\ser_finetune\gpu_24gb_20261112_1500\best /E
robocopy E:\session2\voice_refs gpu\outputs\voice_refs *.wav
Test-Path gpu\outputs\ser_finetune\gpu_24gb_20261112_1500\best\id2label.json     # True
(Get-ChildItem gpu\outputs\voice_refs -Filter *.wav).Name                       # CALM, ENCOURAGE, NEUTRAL, VALIDATE
```
If you brought the audio, copy and count it as in `SETUP_GPU.md` section 2a (expect 1440 RAVDESS and 7442 CREMA-D wavs).

### B4. Token for this window only, then the environment check
```powershell
$env:HF_TOKEN = Read-Host "Paste HF token"       # needed for sesame/csm-1b; never written to disk
python -u gpu\check_env.py | Tee-Object -FilePath session2_check_env.txt
```
Expect `CUDA available: True`, the GPU, the library versions, and `logged in as <you>`.
(`check_env.py` does not list sounddevice or pyttsx3; B2 checked those.)

### B5. Tests, in this order
Save every run's screen output with `Tee-Object`, as below. Errors go to the screen only, so photograph them.

**1. Qwen demo rerun** (prompt fixes from commit 6474ab2):
```powershell
python -u gpu\llm_gpu.py --preset gpu_24gb --demo | Tee-Object -FilePath session2_llm_demo.txt
```
- **CALM** must give exactly ONE next step. In session 1 it gave two ("list one small thing you can do right now.
  Maybe grab a cup of water or stretch your arms").
- **NEUTRAL** must not state a fact that was not given. In session 1 it said "The weather's nice today".
- Every row should say `backend qwen`. A new `results\metrics\llm_demo_qwen_Qwen3-8B_<time>.csv` is written; the old one is kept.

**2. CSM `--speak`, one per strategy** (each command loads CSM, speaks, saves a wav, prints time_to_first_audio and the real-time factor):
```powershell
python -u gpu\tts_gpu.py --speak "I'm really sorry you're going through this. That sounds exhausting." --strategy VALIDATE --preset gpu_24gb --no-play | Tee-Object -FilePath session2_tts_speak.txt
python -u gpu\tts_gpu.py --speak "Let's take this one step at a time. Just start with the first chapter tonight." --strategy CALM --preset gpu_24gb --no-play | Tee-Object -FilePath session2_tts_speak.txt -Append
python -u gpu\tts_gpu.py --speak "You did it! All that studying really paid off." --strategy ENCOURAGE --preset gpu_24gb --no-play | Tee-Object -FilePath session2_tts_speak.txt -Append
python -u gpu\tts_gpu.py --speak "Hey, good to hear from you. How was your day?" --strategy NEUTRAL --preset gpu_24gb --no-play | Tee-Object -FilePath session2_tts_speak.txt -Append
python -u gpu\tts_gpu.py --speak "I'm really glad you told me. You don't have to go through it alone." --strategy VALIDATE --crisis --preset gpu_24gb --no-play | Tee-Object -FilePath session2_tts_speak.txt -Append
```
- The last command is a crisis turn. Its output must say `reference: NEUTRAL  (crisis -> NEUTRAL)`.
- Each run writes `results\metrics\tts_speak_csm*.csv` (a timestamp is added after the first) and a wav in `gpu\outputs\tts_speak\`.
- With speakers or headphones, drop `--no-play` to hear it live. Otherwise listen to the wavs.

**3. Live agent, one turn from a wav** (asks for consent the first time: type `yes`):
```powershell
python -u gpu\live_agent_gpu.py --wav data\raw\Actor_16\03-01-04-01-01-01-16.wav --user session2 --preset gpu_24gb --no-play | Tee-Object -FilePath session2_live_wav.txt
```
- The header must show `tau_conf 0.64` and an emotion model path ending in `...gpu_24gb_20261112_1500\best`,
  **without** `[FALLBACK]`.
- `ser model` must show `'fallback': False`, and `voice` must show `csm`.

**4. Benchmark** (does not play audio; bench users, memory and logs go to `gpu\outputs\bench\`):
```powershell
python -u gpu\bench_gpu.py --preset gpu_24gb | Tee-Object -FilePath session2_bench.txt
```
- At the end: the summary table, `GPU peak memory: after load ... GB, after run ... GB`, and the saved CSV
  `results\metrics\bench_gpu_24gb_<time>.csv`.
- If it prints `!!! FALLBACK USED`, note which part fell back (T3, Qwen or CSM) and photograph the screen.

**5. Push-to-talk with a real mic** (only if this machine has a microphone, ideally a headset):
```powershell
python -u gpu\live_agent_gpu.py --live --user session2 --preset gpu_24gb
```
- Enter starts recording, Enter stops it, q quits. After each turn, check the debug view (transcript, abstained,
  confidence, mood, strategy, voice, time to first audio).
- Try "delete all my data", then say "no". The agent must answer "Okay, nothing was deleted."

### B6. Bring results home (before cleanup)
```powershell
$OUT = "E:\session2_results"
New-Item -ItemType Directory -Force $OUT | Out-Null
Copy-Item results\metrics\bench_gpu_24gb_*.csv, results\metrics\tts_speak_csm*.csv, results\metrics\llm_demo_qwen_Qwen3-8B*.csv $OUT\
Copy-Item session2_*.txt $OUT\
Copy-Item gpu\outputs\env_report_*.json $OUT\
robocopy gpu\outputs\tts_speak $OUT\tts_speak *.wav
robocopy gpu\outputs\bench $OUT\bench /E
robocopy results\logs $OUT\logs session_session2_*.jsonl
```

### B7. Remove the session-2 user's data from this machine, then clean up
Only do this after B6. It deletes `session2`'s memories, consent record and session logs on THIS machine.
It asks you to type yes.
```powershell
python scripts\memory_manager.py --user session2 --delete-all
deactivate
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
```
- `gpu\outputs\bench\` holds only the bench users (`bench_warmup`, `bench_p1` ...), no real person's data. It goes away
  with the work folder.
- Close PowerShell.
- If you signed in to GitHub: **Credential Manager → Windows Credentials** → remove `git:https://github.com`.
- Ask whether `C:\shan\MACHINE-LEARNING-PROJECTS` and `C:\shan\hf_cache` can stay until the next session
  (`SETUP_GPU.md` section 4).

---

## C. Back at home
```powershell
cd F:\projects\Final_project\Emotion_Aware_Conversational_Voice_Agent\Code
Copy-Item E:\session2_results\*.csv results\metrics\
Copy-Item E:\session2_results\env_report_*.json gpu\outputs\
robocopy E:\session2_results $env:GPU_OUTPUTS_DIR\session2 /E /XF *.csv *.json
```
Then add the measured numbers to `ROADMAP.md` ("GPU session 2 — measured"), copied from the files only.
