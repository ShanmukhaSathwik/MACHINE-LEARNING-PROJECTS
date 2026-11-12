# GPU setup guide (professor's Windows machine)

Copy-paste steps for each GPU session. All commands are **PowerShell**.
Nothing here touches the professor's own Python or files: everything lives in
one work folder that you can delete at the end.

---

## 0. Once, at home (before session 1)

1. **Hugging Face account + token**
   - huggingface.co → Settings → Access Tokens → *Create new token* → **Fine-grained**.
   - Permissions: *Read access to contents of all public gated repos you can access*.
     (Write access to your private results repo gets added in a later step.)
   - Copy the token (`hf_...`) into a password manager / your phone notes. Never commit it.
2. **Accept the CSM-1B license**: open the `sesame/csm-1b` model page while logged in → *Agree and access*.
3. **Install the same libraries in your home venv** (CPU, for smoke tests):
   ```powershell
   cd F:\projects\Final_project\Emotion_Aware_Conversational_Voice_Agent\Code
   .\.venv312\Scripts\Activate.ps1
   pip install -r gpu\requirements_gpu.txt
   python gpu\check_env.py      # libraries should now show versions, CUDA still False
   ```
   Your main `requirements.txt` is not changed.
4. **Token at home:** scripts read `HF_TOKEN` from `Code\.env` (gitignored) automatically.
   `check_env.py` should then say `logged in as <you> (token from: .env)`.
5. **Large outputs on C: (home PC only).** F: is small, so run folders, checkpoints and reports go to C:.
   Set the variable once, for your user (no admin needed):
   ```powershell
   [Environment]::SetEnvironmentVariable("GPU_OUTPUTS_DIR", "C:\emotion-agent\gpu_outputs", "User")
   ```
   Then **fully restart VS Code** (close every window), so its terminals pick the variable up. Check in a new terminal:
   ```powershell
   $env:GPU_OUTPUTS_DIR          # -> C:\emotion-agent\gpu_outputs
   python gpu\check_env.py       # the "outputs" line shows the same folder (created if missing)
   ```
   With it set, run folders, checkpoints and logs go there instead of `gpu\outputs\`. Two things stay in the repo:
   `results\metrics\*.csv`, and the env reports (`check_env.py` always saves `env_report_*.json` to the repo's
   `gpu\outputs\`, because git tracks them). Old runs already in `gpu\outputs\` stay where they are.
   To `--resume` one of them, give its full path (`F:\...\Code\gpu\outputs\ser_finetune\<run>`).
   **The GPU machine does not set this variable**, so there everything stays in `gpu\outputs\`.

---

## 1. First session: setup (~20–30 min)

### 1.1 Check the machine has Python 3.12 and git
```powershell
py -3.12 --version     # expect Python 3.12.x
git --version
nvidia-smi             # shows GPU name, driver, and "CUDA Version" (max the driver supports)
```
- No Python 3.12 → install from python.org (Windows installer, *Install for me only*, no admin needed)
  or `winget install Python.Python.3.12`.
- No git → `winget install Git.Git`, then reopen PowerShell.
- Write down the GPU name and the **CUDA Version** from `nvidia-smi`.

### 1.2 Pick a work folder on a drive with 40 GB+ free
Ask the professor which drive to use. Example uses `D:`:
```powershell
$WORK = "D:\emotion-agent"
New-Item -ItemType Directory -Force $WORK | Out-Null
cd $WORK
```

### 1.3 Clone only the voice-agent code (sparse checkout)
```powershell
git clone --filter=blob:none --sparse https://github.com/ShanmukhaSathwik/MACHINE-LEARNING-PROJECTS.git repo
cd repo
git sparse-checkout set Emotion_Aware_Conversational_Voice_Agent/Code
cd Emotion_Aware_Conversational_Voice_Agent\Code
```
If the repo is private, git will ask you to sign in: use a GitHub fine-grained token
with read-only access to this repo, and remove it at cleanup (section 4).

### 1.4 Create a separate venv
```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (this window only).

### 1.5 Install the CUDA build of torch 2.15.0
1. Open pytorch.org → *Get Started* → choose **Stable, Windows, Pip, Python, CUDA**.
2. Pick the highest CUDA option that is **≤ the "CUDA Version" from `nvidia-smi`**.
3. Run the command it shows, but pin the version to match your home PC:
   ```powershell
   pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/<cuXXX>
   ```
   (`<cuXXX>` is the tag from the selector, e.g. the part after `/whl/`.)

### 1.6 Install the rest
```powershell
pip install -r gpu\requirements_gpu.txt
```

### 1.7 Point caches + token at this session only
These last only while this PowerShell window is open — nothing is saved on the machine.
```powershell
$env:HF_HOME  = "$WORK\hf_cache"        # model/dataset downloads go here, not C:
$env:HF_TOKEN = Read-Host "Paste HF token"   # OPTIONAL in session 1 (see below); not written to disk
```
**HF_TOKEN is optional for session 1:** `superb/hubert-large-superb-er` is not gated, so the
fine-tuning downloads it without a token. The token is only needed later, for `sesame/csm-1b`.
If you skip it, just don't run the `Read-Host` line.

Repeat 1.7 in every new PowerShell window.
Do **not** copy your `.env` to this machine: a typed-in session variable leaves nothing behind.

### 1.8 Verify
Console output is saved to a text file with `Tee-Object` (it still shows on screen), so you can bring it home:
```powershell
python -u gpu\check_env.py | Tee-Object -FilePath session1_check_env.txt
```
Expect: `CUDA available: True`, your GPU name and VRAM, all libraries with versions,
`Hugging Face: logged in as <you>` (or `not logged in` if you skipped the token, which is fine for session 1),
and a recommendation line.
The report is saved to `gpu\outputs\env_report_<computer>_<time>.json` — bring it home (section 3).

**If CUDA shows False:** torch is the CPU build. Run
`pip uninstall -y torch` and redo 1.5 with the CUDA index URL.

---

## 2. Every later session (~5 min)

```powershell
$WORK = "D:\emotion-agent"
cd $WORK\repo
git pull
cd Emotion_Aware_Conversational_Voice_Agent\Code
.\.venv-gpu\Scripts\Activate.ps1
$env:HF_HOME  = "$WORK\hf_cache"
$env:HF_TOKEN = Read-Host "Paste HF token"   # only needed once the session uses sesame/csm-1b
python -u gpu\check_env.py | Tee-Object -FilePath session<N>_check_env.txt   # <N> = session number
```
If the professor cleaned the folder, redo section 1 (downloads are the slow part).

---

## 2a. Bring the training audio

The audio is gitignored (`data/raw/`), so `git clone` does not bring it. The manifest
`data/manifests/ser_4class.csv` *is* in git and expects the files at the same paths under `Code\`.
Total about 1.2 GB (RAVDESS 0.57 GB + CREMA-D 0.59 GB).

**At home: copy to a USB drive** (example: USB is `E:`). Only `.wav` files are copied, so
`data\raw\crema_d\.git` (Git LFS store, another 0.6 GB) stays at home:
```powershell
cd F:\projects\Final_project\Emotion_Aware_Conversational_Voice_Agent\Code
robocopy data\raw E:\ser_audio\raw *.wav /S /XD .git
```

**On the GPU machine: copy into the same paths** (USB letter may differ there):
```powershell
# New PowerShell window? Set $WORK again first (same folder as in 1.2):
$WORK = "D:\emotion-agent"
cd $WORK\repo\Emotion_Aware_Conversational_Voice_Agent\Code
robocopy E:\ser_audio\raw data\raw *.wav /S
```

**Check the counts** (both must match before training):
```powershell
$rav   = (Get-ChildItem data\raw\Actor_* -Recurse -Filter *.wav).Count
$crema = (Get-ChildItem data\raw\crema_d\AudioWAV -Filter *.wav).Count
"RAVDESS wavs: $rav (expect 1440)   CREMA-D wavs: $crema (expect 7442)"
if ($rav -eq 1440 -and $crema -eq 7442) { "OK" } else { "MISSING FILES: copy again" }
```

**Run the fine-tuning.** Pick the preset from the `VRAM ... GB total` number on the GPU line of
`python gpu\check_env.py`:

| VRAM total | preset |
|---|---|
| under 14 GB | `gpu_12gb` |
| 14 to 22 GB (a "16 GB" card reports ~15-16) | `gpu_16gb` |
| 22 GB or more (a "24 GB" card reports ~23.5-24) | `gpu_24gb` |

**First, a 2-minute sanity run** (20 weight updates on 200 training clips, 50 validation clips):
```powershell
python -u gpu\finetune_ser.py --preset gpu_16gb --max-steps 20 --max-train-clips 200 --max-val-clips 50 | Tee-Object -FilePath session1_sanity.txt
```
(`Tee-Object` saves only normal output. A crash's `ERROR:` line and traceback go to stderr, so they show on
screen but not in the file: if something fails, copy the screen text or take a photo.)
Check in the output:
- the `device` line shows `cuda` and `mixed precision: bfloat16` (or `float16` on older GPUs), not `off`;
- the epoch line prints `peak VRAM ... GB`;
- `new best val macro-F1 ... -> saved ...\best` appears, and the run ends with `SERModel loads best/ as model_name: OK`.

If it fails inside the data loader (worker / pickling / "DataLoader worker exited" errors), rerun with
`--num-workers 0`. If it runs out of GPU memory (`CUDA out of memory`), use the next smaller preset
(`gpu_24gb` -> `gpu_16gb` -> `gpu_12gb`). The run folder ends in `_sanity`, so it can't be mistaken for a real run;
its test numbers mean nothing.

**Then the real run:**
```powershell
python -u gpu\finetune_ser.py --preset gpu_16gb | Tee-Object -FilePath session1_train.txt
```
If the sanity run needed `--num-workers 0`, add it here too (the real run then gets `_sanity` in its name as well;
or set `num_workers: 0` for that preset in `gpu\gpu_config.yaml`).
- Each run gets its own folder `gpu\outputs\ser_finetune\<preset>_<date_time>\` with `metrics.csv`
  (one line per epoch), `best\` (best checkpoint by validation macro-F1) and `last.pt` (for resuming).
  Budget about 5 GB of disk per hubert-large run at rest (`best\` ~1.3 GB + `last.pt` ~3.7 GB). Saving `last.pt`
  briefly needs about **8.7 GB**, because it is written to `last.pt.tmp` first and then replaces the old file.
  One whole session (run + ~2.5 GB model cache + ~1.2 GB audio + CUDA torch venv) needs about **15-18 GB**:
  ask for **20 GB** (the 40 GB+ in 1.2 also covers the later steps).
- If the session ends mid-run, this continues from the last finished epoch (`-Append` adds to the same log):
  ```powershell
  python -u gpu\finetune_ser.py --resume gpu\outputs\ser_finetune\<run folder> | Tee-Object -FilePath session1_train.txt -Append
  ```
- At the end it evaluates `best\` once on the test split and writes
  `results\metrics\ser_finetune_test_<run>.csv` (outside `gpu\outputs\`). Copy everything home with section 3.

---

## 3. Bringing results home

Nothing is uploaded automatically: `finetune_ser.py` pushes nothing anywhere, so the USB copy below is
the only way results get home. Do it before cleanup (section 4).

**On the GPU machine** (from `Code\`, venv not needed). The USB is `E:` here; adjust the letter.
Set `$RUN` to the real run folder name (from `dir gpu\outputs\ser_finetune`):
```powershell
$RUN = "gpu_16gb_20261112_1400"     # <- your real run folder
$USB = "E:\session1"
New-Item -ItemType Directory -Force $USB | Out-Null

# a) run folder WITHOUT last.pt: brings best\, metrics.csv, config_used.yaml, test_summary.json
robocopy gpu\outputs\ser_finetune\$RUN $USB\$RUN /E /XF last.pt last.pt.tmp

# b) per-clip test results (written to results\metrics\, NOT under gpu\outputs\)
Copy-Item results\metrics\ser_finetune_test_$RUN.csv $USB\

# c) session logs (Tee-Object files) + env report
Copy-Item session1_*.txt $USB\
Copy-Item gpu\outputs\env_report_*.json $USB\
```
robocopy exit code 1 means "files copied" (fine). 8 or higher is a failure.

`last.pt` (~3.7 GB) is only needed for `--resume`. If the professor lets the work folder stay until the
next session, leave it there. Once the run is finished, it isn't needed at all.

**At home** (from `Code\`, USB as `E:`). The run goes to `$env:GPU_OUTPUTS_DIR` on C: (section 0, step 5),
not to F:. The test CSV goes into the repo:
```powershell
cd F:\projects\Final_project\Emotion_Aware_Conversational_Voice_Agent\Code
$env:GPU_OUTPUTS_DIR                 # must print C:\emotion-agent\gpu_outputs; if empty, do section 0 step 5 first
$RUN = "gpu_16gb_20261112_1400"      # same name as above
robocopy E:\session1\$RUN $env:GPU_OUTPUTS_DIR\ser_finetune\$RUN /E
Copy-Item E:\session1\ser_finetune_test_$RUN.csv results\metrics\
Copy-Item E:\session1\session1_*.txt $env:GPU_OUTPUTS_DIR\
Copy-Item E:\session1\env_report_*.json gpu\outputs\
```
To use the model afterwards, its folder is `$env:GPU_OUTPUTS_DIR\ser_finetune\$RUN\best`.

So at home: the run and the session logs go to `$env:GPU_OUTPUTS_DIR`. The test CSV and the GPU machine's env report go
into the repo (`results\metrics\` and `gpu\outputs\`), next to the env reports `check_env.py` writes at home.
Env reports are the one output file git tracks (`gpu\.gitignore` ignores the rest). Commit them and the test CSV
from your PC if you want them in the repo.

---

## 4. End of every session: cleanup

```powershell
deactivate
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
```
- Close PowerShell (clears all session variables).
- If you signed in to GitHub: Windows search → **Credential Manager** → *Windows Credentials* →
  remove the `git:https://github.com` entry.
- Ask the professor whether `D:\emotion-agent` can stay until next session.
  If not: `Remove-Item -Recurse -Force D:\emotion-agent` (after results are copied out).

---

## Questions to ask the professor (first visit)

- Which drive/folder can I use, and can it stay between sessions?
- Is using a secure tunnel (e.g. Cloudflare Tunnel) OK on this network later for the phone demo?
- How long can each session be, and how often?
