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
