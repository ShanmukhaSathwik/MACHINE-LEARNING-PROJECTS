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
$env:HF_TOKEN = Read-Host "Paste HF token"   # not written to disk
```
Repeat 1.7 in every new PowerShell window.
Do **not** copy your `.env` to this machine: a typed-in session variable leaves nothing behind.

### 1.8 Verify
```powershell
python gpu\check_env.py
```
Expect: `CUDA available: True`, your GPU name and VRAM, all libraries with versions,
`Hugging Face: logged in as <you>`, and a recommendation line.
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
$env:HF_TOKEN = Read-Host "Paste HF token"
python gpu\check_env.py
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
python gpu\finetune_ser.py --preset gpu_16gb --max-steps 20 --max-train-clips 200 --max-val-clips 50
```
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
python gpu\finetune_ser.py --preset gpu_16gb
```
If the sanity run needed `--num-workers 0`, add it here too (the real run then gets `_sanity` in its name as well;
or set `num_workers: 0` for that preset in `gpu\gpu_config.yaml`).
- Each run gets its own folder `gpu\outputs\ser_finetune\<preset>_<date_time>\` with `metrics.csv`
  (one line per epoch), `best\` (best checkpoint by validation macro-F1) and `last.pt` (for resuming).
  Budget about 5 GB of disk per hubert-large run.
- If the session ends mid-run: `python gpu\finetune_ser.py --resume gpu\outputs\ser_finetune\<run folder>`
  continues from the last finished epoch.
- At the end it evaluates `best\` once on the test split and writes
  `results\metrics\ser_finetune_test_<run>.csv`. Bring home that CSV, `metrics.csv`, `test_summary.json`
  and the `best\` folder (~1.3 GB): `last.pt` is not needed once the run is finished.

---

## 3. Bringing results home

- **Env report / small logs:** commit from your PC after copying, or upload to OneDrive.
- **Checkpoints, adapters, audio:** pushed to your private Hugging Face repo by the
  training scripts (added in a later step). USB copy of `gpu\outputs\` as backup.

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
