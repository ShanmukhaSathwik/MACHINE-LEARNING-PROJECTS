"""
finetune_ser.py - fine-tune a pretrained speech emotion model on RAVDESS + CREMA-D.

Training T2. Settings come from the `finetune:` block of gpu/gpu_config.yaml (base + preset).

What it does:
  1. Reads data/manifests/ser_4class.csv, keeps in_4class rows, splits by the split column
     (speaker-independent; checked again here).
  2. Loads the base model (default superb/hubert-large-superb-er). Label ids follow the model's OWN
     id2label (mapped with ser_gpu.map_labels), never a hand-written order.
  3. Audio: wav_path (relative to Code/) -> ser_gpu preprocessing (mono, 16 kHz, trim, cut) ->
     augmentation on TRAIN ONLY (speed, pitch, noise; each with prob 0.5) -> padded per batch with attention masks.
  4. Loss: cross-entropy, class-weighted from the train counts. AdamW + linear warmup/decay,
     gradient clipping at 1.0, optional mixed precision (bf16/fp16 from device.py) and gradient checkpointing.
  5. After each epoch: val accuracy, val macro-F1, per-class F1, train loss, seconds, peak VRAM
     -> printed and appended to <run_dir>/metrics.csv. Best val macro-F1 -> <run_dir>/best/
     (model + feature extractor + id2label.json + config_used.yaml). Early stop on patience.
  6. At the end: reload <run_dir>/best/, evaluate ONCE on test, per corpus (accuracy, macro-F1, confusion),
     plus RAVDESS angry/happy/sad accuracy next to the off-the-shelf Check A number.
     Per-clip rows -> results/metrics/ser_finetune_test_<run_id>.csv.
  7. Checks that ser_gpu.SERModel loads <run_dir>/best/ as model_name (so the agent can use it).

Run folders: gpu/outputs/ser_finetune/<preset>_<YYYYmmdd_HHMM>/ (a suffix is added if it exists).
With $GPU_OUTPUTS_DIR set (home PC), gpu/outputs/ means that folder instead (device.outputs_dir());
the test CSV always goes to results/metrics/ in the repo. --resume also takes an absolute run folder path.
Inside a run, best/ and last.pt are updated as training goes; earlier runs are never touched.
last.pt holds model + optimizer + scheduler state for --resume (about 3x the model size: ~4 GB for hubert-large).

Usage (works from any folder):
    python gpu/finetune_ser.py --preset smoke_cpu                 # CPU smoke test: 2 steps, checks the loop only
    python gpu/finetune_ser.py --preset gpu_16gb --max-steps 20 --max-train-clips 200 --max-val-clips 50   # GPU sanity run
    python gpu/finetune_ser.py --preset gpu_16gb
    python gpu/finetune_ser.py --resume gpu/outputs/ser_finetune/gpu_16gb_20261112_1400
    python gpu/finetune_ser.py --resume C:/emotion-agent/gpu_outputs/ser_finetune/gpu_16gb_20261112_1400
"""

import argparse
import csv
import datetime
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import yaml

from device import StageTimer, code_path, outputs_dir, setup
from ser_gpu import CLASSES, SAMPLE_RATE, SERModel, map_labels

SPLITS = ["train", "val", "test"]
CORPORA = ["ravdess", "crema_d"]
THREE_CLASS = ["happy", "angry", "sad"]
OFF_THE_SHELF_REF = "off-the-shelf hubert-large (Check A): 45/96 = 46.9%"
MAX_GRAD_NORM = 1.0
METRIC_FIELDS = (["epoch", "global_step", "train_loss", "val_acc", "val_macro_f1"]
                 + [f"f1_{c}" for c in CLASSES] + ["lr", "seconds", "peak_vram_gb"])
TEST_FIELDS = ["utt_id", "corpus", "speaker_id", "true_label", "pred_label", "confidence"] + [f"p_{c}" for c in CLASSES]


# Data

def load_rows(manifest_path, in_4class_only=True) -> list:
    """Manifest rows as dicts (in_4class -> bool, duration -> float); only in_4class rows if asked."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["in_4class"] = str(r["in_4class"]).strip().lower() == "true"
        r["duration"] = float(r["duration"])
    return [r for r in rows if r["in_4class"]] if in_4class_only else rows


def split_rows(rows) -> dict:
    out = {s: [] for s in SPLITS}
    for r in rows:
        out[r["split"]].append(r)
    return out


def speaker_leaks(rows) -> dict:
    """{speaker: sorted splits} for every speaker found in more than one split (should be empty)."""
    seen = defaultdict(set)
    for r in rows:
        seen[r["speaker_id"]].add(r["split"])
    return {spk: sorted(s) for spk, s in seen.items() if len(s) > 1}


def subset(rows, n, seed):
    """All rows if n is None, else n random rows (fixed seed)."""
    if n is None or n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n)


def label_ids(id2label) -> dict:
    """{class: model output id}, taken from the model's own id2label. Raises unless it covers exactly CLASSES."""
    index_map = map_labels(id2label)
    if len(index_map) != len(CLASSES) or set(index_map.values()) != set(CLASSES):
        raise ValueError(f"Model labels {dict(id2label)} map to {index_map}, which is not exactly {CLASSES}")
    return {cls: idx for idx, cls in index_map.items()}


def class_weights(label_id_list, num_labels) -> np.ndarray:
    """Inverse-frequency weights: n_total / (num_labels * n_class). A class with no clips counts as 1 (stays finite)."""
    counts = np.maximum(np.bincount(np.asarray(label_id_list, dtype=int), minlength=num_labels), 1).astype(np.float64)
    return (counts.sum() / (num_labels * counts)).astype(np.float32)


def preprocess(y, sr, max_seconds):
    """ser_gpu's preprocessing (mono, 16 kHz, trim silence, cut), without loading a model:
    SERModel.preprocess only reads self.max_seconds."""
    return SERModel.preprocess(SimpleNamespace(max_seconds=max_seconds), y, sr)


def augment(y, aug, rng) -> np.ndarray:
    """Train-only augmentation on a 16 kHz mono clip. Each step runs with probability aug['prob']:
    speed (resample: length / speed, pitch moves with it), pitch shift (length kept), white noise at a random SNR."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if rng.random() < aug["prob"]:
        speed = rng.uniform(*aug["speed"])
        y = librosa.resample(y, orig_sr=SAMPLE_RATE * speed, target_sr=SAMPLE_RATE)
    if rng.random() < aug["prob"]:
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=rng.uniform(*aug["pitch_semitones"]))
    if rng.random() < aug["prob"]:
        power = float(np.mean(y ** 2))
        if power > 0:
            snr_db = rng.uniform(*aug["noise_snr_db"])
            y = y + rng.normal(0.0, math.sqrt(power / 10 ** (snr_db / 10)), size=y.shape)
    return np.ascontiguousarray(y, dtype=np.float32)


class SERDataset(torch.utils.data.Dataset):
    """(16 kHz float32 waveform, label id) per manifest row. Augments only if augment_cfg is given."""

    def __init__(self, rows, class_to_id, max_seconds, augment_cfg=None, seed=0):
        self.rows, self.class_to_id, self.max_seconds = rows, class_to_id, max_seconds
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        y, sr = sf.read(str(code_path(row["wav_path"])), dtype="float32")
        y = preprocess(y, sr, self.max_seconds)
        if self.augment_cfg:
            y = augment(y, self.augment_cfg, self.rng)
        return y, self.class_to_id[row["label"]]


def make_datasets(splits, class_to_id, max_seconds, augment_cfg, seed) -> dict:
    """Train gets augmentation; val and test never do."""
    return {s: SERDataset(splits[s], class_to_id, max_seconds, augment_cfg if s == "train" else None, seed)
            for s in SPLITS}


def _worker_init(_):
    """Each loader worker gets its own augmentation stream (torch gives every worker a different seed)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


class Collator:
    """Pads a batch with the model's feature extractor and returns attention masks + labels."""

    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, batch):
        ys, labels = zip(*batch)
        feats = self.extractor(list(ys), sampling_rate=SAMPLE_RATE, padding=True,
                               return_attention_mask=True, return_tensors="pt")
        out = {k: v for k, v in feats.items()}   # plain dict: safe for pin_memory and worker transfer
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out


# Metrics

def classification_metrics(true_ids, pred_ids, id_to_class) -> dict:
    """Accuracy, macro-F1 over CLASSES, per-class F1, confusion {true: {pred: n}}."""
    t = [id_to_class[i] for i in true_ids]
    p = [id_to_class[i] for i in pred_ids]
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for a, b in zip(t, p):
        conf[a][b] += 1
    f1 = {}
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in CLASSES if o != c)
        fn = sum(conf[c][o] for o in CLASSES if o != c)
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    n = len(t)
    return {"n": n, "correct": sum(a == b for a, b in zip(t, p)), "acc": (sum(a == b for a, b in zip(t, p)) / n) if n else 0.0,
            "macro_f1": float(np.mean([f1[c] for c in CLASSES])), "f1": f1, "confusion": conf}


def print_confusion(conf, indent="    "):
    print(indent + " " * 9 + "".join(f"{c:>9s}" for c in CLASSES))
    for a in CLASSES:
        if sum(conf[a].values()):
            print(indent + f"{a:>9s}" + "".join(f"{conf[a][b]:>9d}" for b in CLASSES))


# Training helpers

def _amp_context(use_amp, amp_dtype):
    return torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.autocast("cpu", enabled=False)


def evaluate(model, loader, device, use_amp, amp_dtype):
    """(true ids, predicted ids, softmax probs [n, num_labels]) over a loader."""
    model.eval()
    trues, preds, probs = [], [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with _amp_context(use_amp, amp_dtype):
                logits = model(**batch).logits
            pr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probs.append(pr)
            preds.extend(pr.argmax(axis=1).tolist())
            trues.extend(labels.tolist())
    return trues, preds, (np.concatenate(probs) if probs else np.zeros((0, len(CLASSES))))


def _free_path(path: Path) -> Path:
    """path if free, else path_2, path_3, ... (directories) or a timestamped name (files)."""
    if not path.exists():
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    i = 2
    while path.with_name(f"{path.name}_{i}").exists():
        i += 1
    return path.with_name(f"{path.name}_{i}")


OVERRIDES = {   # CLI flag (argparse dest) -> finetune config key; for a quick GPU sanity run
    "max_steps": "max_steps",
    "max_train_clips": "max_train_clips",
    "max_val_clips": "max_val_clips",
    "num_workers": "num_workers",
}
SANITY_SUFFIX = "_sanity"


def apply_overrides(ft, args) -> list:
    """Apply the CLI overrides that were given to the finetune config (in place).
    Returns [(key, old value, new value)] for every override applied."""
    applied = []
    for dest, key in OVERRIDES.items():
        value = getattr(args, dest, None)
        if value is not None:
            applied.append((key, ft.get(key), value))
            ft[key] = value
    return applied


def resume_dir(path) -> Path:
    """--resume argument -> run folder. Absolute paths are used as given (e.g. a run under C:\\);
    relative ones resolve from Code/, with gpu/outputs/... going through outputs_dir() (see code_path)."""
    path = Path(path)
    return path if path.is_absolute() else code_path(path)


def make_run_id(preset, when, sanity=False) -> str:
    """<preset>_<YYYYmmdd_HHMM>, plus "_sanity" for runs with CLI overrides."""
    return f"{preset}_{when:%Y%m%d_%H%M}" + (SANITY_SUFFIX if sanity else "")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=code_path(""), capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _append_metrics(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def print_counts(splits):
    print("\nClips per split x label x corpus (in_4class rows)")
    print(f"  {'split':6s} {'label':8s}" + "".join(f"{c:>9s}" for c in CORPORA) + f"{'total':>9s}")
    for s in SPLITS:
        for lab in CLASSES:
            n = [sum(1 for r in splits[s] if r["label"] == lab and r["corpus"] == c) for c in CORPORA]
            print(f"  {s:6s} {lab:8s}" + "".join(f"{x:>9d}" for x in n) + f"{sum(n):>9d}")
        n = [sum(1 for r in splits[s] if r["corpus"] == c) for c in CORPORA]
        print(f"  {s:6s} {'= all':8s}" + "".join(f"{x:>9d}" for x in n) + f"{sum(n):>9d}")


# Main

def run(args):
    from transformers import (AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification,
                              get_linear_schedule_with_warmup)

    # Config and run folder
    print(f"outputs    : {outputs_dir()}")
    if args.resume:
        run_dir = resume_dir(args.resume)
        saved = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
        if args.preset and args.preset != saved["preset"]:
            raise ValueError(f"--preset {args.preset} differs from the run's preset {saved['preset']}")
        cfg, device, dtype = setup(saved["preset"])
        cfg["finetune"] = saved["finetune"]
        run_id = run_dir.name
        overrides = []
    else:
        cfg, device, dtype = setup(args.preset)
        overrides = apply_overrides(cfg["finetune"], args)
        run_id = make_run_id(cfg["preset"], datetime.datetime.now(), sanity=bool(overrides))
        run_dir = _free_path(code_path(cfg["finetune"]["output_dir"]) / run_id)
        run_id = run_dir.name
        run_dir.mkdir(parents=True)
    for key, old, new in overrides:
        print(f"override   : finetune.{key} {old} -> {new}   (CLI; run marked {SANITY_SUFFIX})")
    ft = cfg["finetune"]
    seed = cfg["seed"]
    best_dir = run_dir / "best"
    use_amp = bool(ft["mixed_precision"]) and device == "cuda"
    if ft["mixed_precision"] and device != "cuda":
        raise ValueError("finetune.mixed_precision is true but the device is CPU; set it false for CPU presets")
    amp_dtype = dtype if use_amp else torch.float32

    # Data
    rows = load_rows(code_path(ft["manifest"]), ft["in_4class_only"])
    leaks = speaker_leaks(rows)
    if leaks:
        raise ValueError(f"Speakers in more than one split: {leaks}")
    splits = split_rows(rows)
    print_counts(splits)
    splits["train"] = subset(splits["train"], ft["max_train_clips"], seed)
    splits["val"] = subset(splits["val"], ft["max_val_clips"], seed)

    # Model + labels (ids follow the model's own id2label)
    extractor = AutoFeatureExtractor.from_pretrained(ft["base_model"])
    # The superb heads use a weighted sum over ALL layers (use_weighted_layer_sum). LayerDrop skips random layers in
    # training mode, which leaves fewer hidden states than layer weights and crashes the forward pass.
    # So LayerDrop is turned off for fine-tuning (inference never drops layers, so predictions are unaffected).
    base_cfg = AutoConfig.from_pretrained(ft["base_model"])
    model_kwargs = {}
    if getattr(base_cfg, "use_weighted_layer_sum", False) and getattr(base_cfg, "layerdrop", 0.0) > 0:
        model_kwargs["layerdrop"] = 0.0
        print(f"\nlayerdrop {base_cfg.layerdrop} -> 0.0 (incompatible with use_weighted_layer_sum during training)")
    model = AutoModelForAudioClassification.from_pretrained(ft["base_model"], **model_kwargs)
    class_to_id = label_ids(model.config.id2label)
    id_to_class = {i: c for c, i in class_to_id.items()}
    num_labels = model.config.num_labels
    order = [f"{i}={id_to_class[i]}({model.config.id2label[i]})" for i in sorted(id_to_class)]
    print(f"\nModel label ids (from its id2label): {', '.join(order)}"
          f"   same order as CLASSES: {[id_to_class[i] for i in sorted(id_to_class)] == CLASSES}")
    if ft["freeze_feature_encoder"]:
        model.freeze_feature_encoder()
    if ft["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.to(device)

    # Loaders
    datasets = make_datasets(splits, class_to_id, cfg["ser"]["max_seconds"], ft["augment"], seed)
    collate = Collator(extractor)
    workers = int(ft["num_workers"])
    loader_kw = {"collate_fn": collate, "num_workers": workers, "pin_memory": device == "cuda",
                 "worker_init_fn": _worker_init if workers else None, "persistent_workers": workers > 0}
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(datasets["train"], batch_size=ft["batch_size"], shuffle=True,
                                               generator=gen, **loader_kw)
    val_loader = torch.utils.data.DataLoader(datasets["val"], batch_size=ft["batch_size"] * 2, **loader_kw)
    test_loader = torch.utils.data.DataLoader(datasets["test"], batch_size=ft["batch_size"] * 2, **loader_kw)

    accum = int(ft["grad_accum"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * ft["max_epochs"]
    if ft["max_steps"]:
        total_steps = min(total_steps, int(ft["max_steps"]))
    warmup = int(round(ft["warmup_ratio"] * total_steps))

    # Loss, optimizer, schedule
    train_ids = [class_to_id[r["label"]] for r in splits["train"]]
    weights = class_weights(train_ids, num_labels) if ft["class_weighted_loss"] else np.ones(num_labels, np.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    no_decay = ("bias", "layer_norm", "layernorm", "LayerNorm")
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in params if not any(k in n for k in no_decay)], "weight_decay": ft["weight_decay"]},
        {"params": [p for n, p in params if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ], lr=float(ft["lr"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_epoch, global_step, best_f1, bad_epochs, stopped = 0, 0, -1.0, 0, False
    if args.resume:
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        best_f1, bad_epochs, stopped = state["best_f1"], state["bad_epochs"], state["stopped"]
        torch.set_rng_state(state["torch_rng"])
        datasets["train"].rng = np.random.default_rng(state["epoch"] + seed + 1)
        print(f"\nResumed {run_id} after epoch {state['epoch'] + 1} (step {global_step}, best val macro-F1 {best_f1:.4f})")
    else:
        used = {**cfg, "run_id": run_id, "device": device, "dtype": str(dtype).replace("torch.", ""),
                "label_ids": class_to_id, "git_commit": _git_commit(),
                "cli_overrides": {k: {"config": old, "used": new} for k, old, new in overrides},
                "n_clips": {s: len(splits[s]) for s in SPLITS}}
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(used, sort_keys=False), encoding="utf-8")

    print(f"\nrun        : {run_id}   ({run_dir})")
    print(f"device     : {device}   dtype {str(dtype).replace('torch.', '')}   "
          f"mixed precision: {str(amp_dtype).replace('torch.', '') if use_amp else 'off'}"
          f"   gradient checkpointing: {ft['gradient_checkpointing']}")
    print(f"model      : {ft['base_model']}   (feature encoder frozen: {ft['freeze_feature_encoder']})")
    print(f"batch      : {ft['batch_size']} x accum {accum} = effective {ft['batch_size'] * accum}   workers {workers}")
    print(f"clips      : train {len(splits['train'])}  val {len(splits['val'])}  test {len(splits['test'])}")
    print(f"steps      : {steps_per_epoch} per epoch, {total_steps} total (max_epochs {ft['max_epochs']}"
          f"{', max_steps ' + str(ft['max_steps']) if ft['max_steps'] else ''}), warmup {warmup}, lr {ft['lr']}")
    print(f"class wts  : " + "  ".join(f"{id_to_class[i]} {weights[i]:.3f}" for i in range(num_labels)))

    # Train
    metrics_path = run_dir / "metrics.csv"
    for epoch in range(start_epoch, ft["max_epochs"]):
        if stopped:
            break
        timer = StageTimer()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        with timer("epoch"):
            for i, batch in enumerate(train_loader):
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with _amp_context(use_amp, amp_dtype):
                    logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels)
                scaler.scale(loss / accum).backward()
                losses.append(loss.item())
                if (i + 1) % accum == 0 or i + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if ft["max_steps"] and global_step >= ft["max_steps"]:
                        stopped = True
                        break
            vt, vp, _ = evaluate(model, val_loader, device, use_amp, amp_dtype)
        m = classification_metrics(vt, vp, id_to_class)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else None
        row = {"epoch": epoch + 1, "global_step": global_step, "train_loss": round(float(np.mean(losses)), 5),
               "val_acc": round(m["acc"], 5), "val_macro_f1": round(m["macro_f1"], 5),
               **{f"f1_{c}": round(m["f1"][c], 5) for c in CLASSES},
               "lr": f"{scheduler.get_last_lr()[0]:.3e}", "seconds": round(timer.times["epoch"], 1),
               "peak_vram_gb": round(peak, 2) if peak is not None else ""}
        _append_metrics(metrics_path, row)
        print(f"\nepoch {epoch + 1}/{ft['max_epochs']}  step {global_step}  train loss {row['train_loss']:.4f}  "
              f"val acc {m['correct']}/{m['n']} = {m['acc']:.3f}  val macro-F1 {m['macro_f1']:.3f}  "
              f"({row['seconds']}s{', peak VRAM ' + str(row['peak_vram_gb']) + ' GB' if peak is not None else ''})")
        print("  val F1: " + "  ".join(f"{c} {m['f1'][c]:.3f}" for c in CLASSES))
        if m["macro_f1"] > best_f1:
            best_f1, bad_epochs = m["macro_f1"], 0
            model.save_pretrained(best_dir)
            extractor.save_pretrained(best_dir)
            (best_dir / "id2label.json").write_text(json.dumps(
                {str(i): {"class": id_to_class[i], "model_label": model.config.id2label[i]} for i in sorted(id_to_class)},
                indent=2), encoding="utf-8")
            (best_dir / "config_used.yaml").write_text((run_dir / "config_used.yaml").read_text(encoding="utf-8"),
                                                       encoding="utf-8")
            print(f"  new best val macro-F1 {best_f1:.4f} -> saved {best_dir}")
        else:
            bad_epochs += 1
            print(f"  no improvement ({bad_epochs}/{ft['early_stop_patience']}); best {best_f1:.4f}")
        if bad_epochs >= ft["early_stop_patience"]:
            print(f"  early stop: {bad_epochs} epochs without a better val macro-F1")
            stopped = True
        if stopped and ft["max_steps"] and global_step >= ft["max_steps"]:
            print(f"  reached max_steps {ft['max_steps']}")
        tmp = run_dir / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "global_step": global_step, "best_f1": best_f1, "bad_epochs": bad_epochs, "stopped": stopped,
                    "torch_rng": torch.get_rng_state()}, tmp)
        tmp.replace(run_dir / "last.pt")
        if stopped:
            break
    print(f"\nPer-epoch metrics: {metrics_path}")

    # Test: reload best, evaluate once
    del model, optimizer, scheduler, params
    if device == "cuda":
        torch.cuda.empty_cache()
    best =AutoModelForAudioClassification.from_pretrained(best_dir).to(device)
    tt, tp, probs = evaluate(best, test_loader, device, use_amp, amp_dtype)
    test_rows = splits["test"]
    out_rows = []
    for r, t, p, pr in zip(test_rows, tt, tp, probs):
        out_rows.append({"utt_id": r["utt_id"], "corpus": r["corpus"], "speaker_id": r["speaker_id"],
                         "true_label": id_to_class[t], "pred_label": id_to_class[p], "confidence": round(float(pr.max()), 4),
                         **{f"p_{c}": round(float(pr[class_to_id[c]]), 4) for c in CLASSES}})
    print(f"\n=== Test (best checkpoint, evaluated once): {best_dir} ===")
    summary = {}
    for corpus in CORPORA:
        idx = [i for i, r in enumerate(test_rows) if r["corpus"] == corpus]
        m = classification_metrics([tt[i] for i in idx], [tp[i] for i in idx], id_to_class)
        summary[corpus] = {k: m[k] for k in ("n", "correct", "acc", "macro_f1", "f1", "confusion")}
        print(f"\n[{corpus}] accuracy {m['correct']}/{m['n']} = {m['acc']:.3f}   macro-F1 {m['macro_f1']:.3f}")
        print("  F1: " + "  ".join(f"{c} {m['f1'][c]:.3f}" for c in CLASSES))
        print("  confusion (rows = true, cols = predicted):")
        print_confusion(m["confusion"])
        if corpus == "ravdess":
            three = [i for i in idx if id_to_class[tt[i]] in THREE_CLASS]
            c3 = sum(tt[i] == tp[i] for i in three)
            summary[corpus]["three_class"] = {"correct": c3, "n": len(three)}
            print(f"  RAVDESS angry/happy/sad clips: fine-tuned {c3}/{len(three)} = "
                  f"{(c3 / len(three) if three else 0):.1%}   vs   {OFF_THE_SHELF_REF}")
    test_csv = _free_path(code_path("results/metrics") / f"ser_finetune_test_{run_id}.csv")
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TEST_FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    (run_dir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPer-clip test rows: {test_csv}")

    # The agent loads models through SERModel: check it accepts the checkpoint folder
    del best
    ser = SERModel(cfg, device, dtype, model_name=str(best_dir))
    clip = test_rows[0]
    out = ser.predict_file(code_path(clip["wav_path"]))
    same = out["label"] == out_rows[0]["pred_label"]
    print(f"\nSERModel loads {best_dir.name}/ as model_name: OK   "
          f"({clip['utt_id']}: SERModel -> {out['label']} {out['confidence']:.3f}, "
          f"batched test -> {out_rows[0]['pred_label']} {out_rows[0]['confidence']:.3f}, same label: {same})")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained speech emotion model on ser_4class.csv.")
    parser.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    parser.add_argument("--resume", metavar="RUN_DIR",
                        help="continue a run from its last.pt (same folder, same settings); relative or absolute path")
    sanity = parser.add_argument_group("overrides for one quick sanity run (run_id gets the suffix _sanity)")
    sanity.add_argument("--max-steps", type=int, metavar="N", help="stop after N weight updates")
    sanity.add_argument("--max-train-clips", type=int, metavar="N", help="use N random training clips")
    sanity.add_argument("--max-val-clips", type=int, metavar="N", help="use N random validation clips")
    sanity.add_argument("--num-workers", type=int, metavar="N", help="data-loading worker processes (0 = main process)")
    args = parser.parse_args()
    if args.resume and any(getattr(args, d) is not None for d in OVERRIDES):
        parser.error("overrides can't be combined with --resume (a resumed run keeps its saved settings)")
    try:
        return run(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        if "needs a CUDA GPU" not in str(e):
            raise   # keep the traceback for real training errors (e.g. CUDA out of memory)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
