"""
memory_manager.py
Demo step 5 / Task D step D2: long-term memory of short facts about the user, with two backends:
    MemoryManager        - one JSON file (logging.memory_file, data/memories.json); word-overlap retrieval.
    ChromaMemoryManager  - a Chroma folder (memory.vector_store_dir, vector_store/); MiniLM meaning-match retrieval.
make_memory_manager(conf) picks one from memory.backend ("json" or "chroma"; json until step D4).
Brain only: no torch, nothing imported from gpu/ (the config is read straight from gpu/gpu_config.yaml).
chromadb is imported only inside ChromaMemoryManager, so importing this module never needs it.

Record (one per fact, same shape from both backends):
    {memory_id (uuid4 hex), user_id, timestamp (ISO-8601 UTC), source: "user_turn", turn_id,
     fact, mood_snapshot (dict), importance (0..1)}
    source is always "user_turn": facts only ever come from what the user said, never from the agent's reply.
    Chroma keeps fact as the document and the rest as metadata (mood_snapshot as a JSON string; a None turn_id
    is left out, because Chroma metadata cannot hold None).

Importance (importance_from_turn): intensity = 1 - p_neutral (0 on an abstained turn);
    importance = base + (1 - base) * intensity; spike = intensity >= spike_threshold.

Retrieval (retrieve_relevant, same signature and return keys {memory_id, fact, similarity, importance, score}):
    JSON:   similarity = Jaccard word overlap (lowercased, stopwords removed); only similarity > 0 is returned.
    Chroma: similarity = 1 - cosine distance between MiniLM embeddings (chromadb's DefaultEmbeddingFunction,
            all-MiniLM-L6-v2 on ONNX). Every one of this user's facts is scored; a fact with
            similarity < memory.min_similarity is dropped BEFORE scoring, so high importance can never pull an
            unrelated fact back in. min_similarity is a placeholder (0.0) until step D3 picks it from labelled pairs.
    Both:   score = w_similarity * similarity + w_importance * importance; best score first (newest on a tie), top_k.

Duplicates (both): a new fact whose normalized text equals, or whose word overlap is >= 0.8 with, one of this
    user's facts updates that record's timestamp, keeps the higher importance and returns the existing memory_id.

Privacy: users never see each other's memories. delete / reset_user_memory remove records for real.
delete_all_my_data (shared by both backends) also revokes consent (safety.revoke_consent) and removes that
user's session logs (session_<user_id>_<stamp>.jsonl, see session_log_filename) - but never a file tracked by git.
JSON writes are atomic (temp file + replace). A corrupt file raises MemoryFileError and is never replaced.
The memory file and vector_store/ are git-ignored.

CLI:  python scripts/memory_manager.py --user NAME --list              [--backend json|chroma]
      python scripts/memory_manager.py --user NAME --forget MEMORY_ID  [--backend json|chroma]
      python scripts/memory_manager.py --user NAME --delete-all        [--backend json|chroma]
"""

import argparse
import copy
import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from safety import DEFAULT_CONSENT_PATH, revoke_consent

CODE_DIR = Path(__file__).resolve().parent.parent          # Code/
CONFIG_PATH = CODE_DIR / "gpu" / "gpu_config.yaml"
SOURCE = "user_turn"
DUPLICATE_OVERLAP = 0.8      # word overlap at or above which a new fact updates an old one instead of adding
REQUIRED_KEYS = ("top_k", "w_similarity", "w_importance", "base_importance", "spike_threshold", "max_fact_words")
CHROMA_REQUIRED_KEYS = REQUIRED_KEYS + ("min_similarity",)
BACKENDS = ("json", "chroma")
COLLECTION = "memories"

# Words that carry no topic. "user" is here because every stored fact starts with "The user ...".
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if im ive id ill in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own really same she should so some such than that
thats the their theirs them themselves then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your yours yourself yourselves
dont didnt doesnt cant wont isnt arent wasnt werent also get got going gonna want wanna like just
user users
""".split())


class MemoryFileError(RuntimeError):
    """The memory file exists but is not valid. It was not changed; fix or move it by hand."""


# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------

def load_memory_config(config_path=CONFIG_PATH) -> dict:
    """{"memory": base.memory, "memory_file": Path, "log_dir": Path, "vector_store_dir": Path | None} from
    gpu_config.yaml (base only; no preset changes memory). Paths are resolved relative to Code/.
    vector_store_dir is None when memory.vector_store_dir is not set (only the chroma backend needs it)."""
    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)["base"]
    store = base["memory"].get("vector_store_dir")
    return {
        "memory": dict(base["memory"]),
        "memory_file": CODE_DIR / base["logging"]["memory_file"],
        "log_dir": CODE_DIR / base["logging"]["log_dir"],
        "vector_store_dir": CODE_DIR / store if store else None,
    }


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------

def normalize_text(text) -> str:
    """Lowercase, apostrophes removed ("I'm" -> "im"), other punctuation -> space, spaces collapsed."""
    t = str(text or "").lower().replace("’", "'").replace("‘", "'").replace("'", "")
    t = re.sub(r"[^\w\s]|_", " ", t)
    return " ".join(t.split())


def content_words(text) -> set:
    return {w for w in normalize_text(text).split() if w not in STOPWORDS}


def jaccard(a, b) -> float:
    """Word overlap |A & B| / |A | B| on content words. 0.0 if either side has none."""
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def importance_from_turn(probs, abstained, cfg) -> tuple:
    """(importance, spike). intensity = 1 - p_neutral, or 0 on an abstained turn;
    importance = base + (1 - base) * intensity; spike = intensity >= spike_threshold."""
    base = float(cfg["base_importance"])
    intensity = 0.0 if abstained else min(1.0, max(0.0, 1.0 - float(probs.get("neutral", 0.0))))
    return base + (1.0 - base) * intensity, intensity >= float(cfg["spike_threshold"])


def session_log_filename(user_id, when=None) -> str:
    """session_<user_id>_<YYYYmmddTHHMMSSZ>.jsonl. Step 9 names session logs with this, so
    delete_all_my_data can find them. The stamp has no underscore, so user "bob" never matches "bob_smith"."""
    when = when or datetime.now(timezone.utc)
    return f"session_{user_id}_{when.strftime('%Y%m%dT%H%M%SZ')}.jsonl"


def is_user_session_log(name, user_id) -> bool:
    """True for session_<user_id>_<stamp>.jsonl where <stamp> has no underscore."""
    return re.fullmatch(rf"session_{re.escape(user_id)}_[^_/\\]+\.jsonl", name) is not None


def git_tracked(path) -> "bool | None":
    """True if git tracks the file, False if not (or no repository / no git), None if git gave an unexpected error."""
    path = Path(path)
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", "--", path.name], cwd=path.parent,
                           capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return False                       # git not installed: nothing can be tracked
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1 or "not a git repository" in (r.stderr or "").lower():
        return False
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_user_id(user_id) -> None:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(f"user_id must be a non-empty string, got {user_id!r}")


def _check_cfg(cfg_memory, keys) -> None:
    missing = [k for k in keys if k not in cfg_memory]
    if missing:
        raise ValueError(f"memory config is missing {missing} (see base.memory in gpu/gpu_config.yaml)")


def _check_fact(fact, importance) -> tuple:
    """Shared store() validation. Returns (stripped fact, importance as float)."""
    if not isinstance(fact, str) or not fact.strip():
        raise ValueError("fact must be a non-empty string")
    importance = float(importance)
    if not 0.0 <= importance <= 1.0:
        raise ValueError(f"importance must be in [0, 1], got {importance}")
    return fact.strip(), importance


def _is_duplicate(old_fact, fact) -> bool:
    """Same normalized text, or word overlap >= DUPLICATE_OVERLAP."""
    return normalize_text(old_fact) == normalize_text(fact) or jaccard(old_fact, fact) >= DUPLICATE_OVERLAP


# ---------------------------------------------------------------------------------------------
# Shared by both backends
# ---------------------------------------------------------------------------------------------

class BaseMemoryManager:
    """What does not depend on where facts are stored. Subclasses provide reset_user_memory (and the rest)."""

    def reset_user_memory(self, user_id) -> int:          # pragma: no cover - always overridden
        raise NotImplementedError

    def delete_all_my_data(self, user_id, log_dir, consent_path=DEFAULT_CONSENT_PATH) -> dict:
        """reset_user_memory + safety.revoke_consent + remove this user's session logs in log_dir
        (session_<user_id>_<stamp>.jsonl). A log file tracked by git is never removed: it is skipped and reported."""
        _check_user_id(user_id)
        report = {
            "user_id": user_id,
            "memories_removed": self.reset_user_memory(user_id),
            "consent_revoked": revoke_consent(user_id, consent_path),
            "logs_removed": [],
            "logs_skipped_tracked": [],
            "logs_skipped_unchecked": [],
        }
        log_dir = Path(log_dir)
        if log_dir.is_dir():
            for f in sorted(log_dir.iterdir()):
                if not (f.is_file() and is_user_session_log(f.name, user_id)):
                    continue
                tracked = git_tracked(f)
                if tracked is True:
                    report["logs_skipped_tracked"].append(f.name)
                elif tracked is None:
                    report["logs_skipped_unchecked"].append(f.name)
                else:
                    f.unlink()
                    report["logs_removed"].append(f.name)
        return report

    def close(self) -> None:
        """Release open handles. Nothing to do for the JSON file; Chroma releases its folder."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------------------------

class MemoryManager(BaseMemoryManager):
    """JSON-file memory. Every call reads the file fresh, so two instances on one file stay consistent."""

    def __init__(self, path, cfg_memory: dict):
        _check_cfg(cfg_memory, REQUIRED_KEYS)
        self.path = Path(path)
        self.cfg = dict(cfg_memory)

    # -- file --

    def _load(self) -> list:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise MemoryFileError(f"Memory file {self.path} is corrupt ({e}). "
                                  "It was not changed; fix or move it by hand.") from e
        if not (isinstance(data, dict) and isinstance(data.get("memories"), list)
                and all(isinstance(r, dict) and "memory_id" in r and "user_id" in r for r in data["memories"])):
            raise MemoryFileError(f"Memory file {self.path} is not a {{\"memories\": [records]}} object. "
                                  "It was not changed; fix or move it by hand.")
        return data["memories"]

    def _save(self, records: list) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"memories": records}, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # -- write --

    def store(self, user_id, fact, turn_id, mood_snapshot, importance) -> str:
        """Add a fact; returns its memory_id. A near-duplicate of one of this user's facts (same normalized
        text, or word overlap >= 0.8) updates that record's timestamp, keeps the higher importance,
        and returns the existing memory_id instead."""
        _check_user_id(user_id)
        fact, importance = _check_fact(fact, importance)
        records = self._load()
        for r in records:
            if r["user_id"] != user_id:
                continue
            if _is_duplicate(r["fact"], fact):
                r["timestamp"] = _now_iso()
                r["importance"] = max(float(r["importance"]), importance)
                self._save(records)
                return r["memory_id"]
        record = {
            "memory_id": uuid.uuid4().hex,
            "user_id": user_id,
            "timestamp": _now_iso(),
            "source": SOURCE,
            "turn_id": turn_id,
            "fact": fact,
            "mood_snapshot": copy.deepcopy(dict(mood_snapshot or {})),
            "importance": importance,
        }
        records.append(record)
        self._save(records)
        return record["memory_id"]

    def delete(self, user_id, memory_id) -> bool:
        """Remove one record for real. False for an unknown id or another user's memory."""
        _check_user_id(user_id)
        records = self._load()
        keep = [r for r in records if not (r["memory_id"] == memory_id and r["user_id"] == user_id)]
        if len(keep) == len(records):
            return False
        self._save(keep)
        return True

    def reset_user_memory(self, user_id) -> int:
        """Remove all of this user's facts (no one else's). Returns how many were removed."""
        _check_user_id(user_id)
        records = self._load()
        keep = [r for r in records if r["user_id"] != user_id]
        removed = len(records) - len(keep)
        if removed:
            self._save(keep)
        return removed

    # delete_all_my_data: BaseMemoryManager

    # -- read --

    def list_user_facts(self, user_id) -> list:
        """This user's records, newest first (copies)."""
        _check_user_id(user_id)
        mine = [(i, r) for i, r in enumerate(self._load()) if r["user_id"] == user_id]
        mine.sort(key=lambda ir: (ir[1]["timestamp"], ir[0]), reverse=True)
        return [copy.deepcopy(r) for _, r in mine]

    def retrieve_relevant(self, user_id, query, top_k=None) -> list:
        """[{memory_id, fact, similarity, importance, score}], best score first, at most top_k (default
        cfg top_k). Only this user's facts, and only those with similarity > 0."""
        _check_user_id(user_id)
        k = int(self.cfg["top_k"] if top_k is None else top_k)
        w_sim, w_imp = float(self.cfg["w_similarity"]), float(self.cfg["w_importance"])
        hits = []
        for r in self._load():
            if r["user_id"] != user_id:
                continue
            sim = jaccard(query, r["fact"])
            if sim <= 0.0:
                continue
            imp = float(r["importance"])
            hits.append({"memory_id": r["memory_id"], "fact": r["fact"], "similarity": sim,
                         "importance": imp, "score": w_sim * sim + w_imp * imp, "_ts": r["timestamp"]})
        hits.sort(key=lambda h: (h["score"], h["_ts"]), reverse=True)
        for h in hits:
            del h["_ts"]
        return hits[:max(k, 0)]


# ---------------------------------------------------------------------------------------------
# ChromaMemoryManager
# ---------------------------------------------------------------------------------------------

class ChromaMemoryManager(BaseMemoryManager):
    """Chroma-folder memory: one collection "memories" (cosine distance) in store_dir.
    id = memory_id, document = fact, metadata = the other record fields (+ "seq", a per-user counter that orders
    records stored within the same clock tick; it is never returned). embedding_function defaults to chromadb's
    DefaultEmbeddingFunction (all-MiniLM-L6-v2 on ONNX, no torch); tests pass a fake one. Call close() (or use
    `with`) to release the folder - on Windows it stays locked while the client is open."""

    def __init__(self, store_dir, cfg_memory: dict, embedding_function=None):
        _check_cfg(cfg_memory, CHROMA_REQUIRED_KEYS)
        import chromadb                                   # here, not at module top: importing memory_manager needs no chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.path = Path(store_dir)
        self.cfg = dict(cfg_memory)
        self.path.mkdir(parents=True, exist_ok=True)
        self._embed = embedding_function if embedding_function is not None else DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(
            COLLECTION, embedding_function=self._embed, configuration={"hnsw": {"space": "cosine"}})

    def close(self) -> None:
        client, self._client, self._col = getattr(self, "_client", None), None, None
        if client is not None:
            client.close()

    # -- record <-> chroma --

    def _user_rows(self, user_id) -> list:
        """[(memory_id, fact, metadata)] for this user, unordered."""
        got = self._col.get(where={"user_id": user_id}, include=["documents", "metadatas"])
        return list(zip(got["ids"], got["documents"], got["metadatas"]))

    @staticmethod
    def _to_record(memory_id, fact, md) -> dict:
        return {
            "memory_id": memory_id,
            "user_id": md["user_id"],
            "timestamp": md["timestamp"],
            "source": md["source"],
            "turn_id": md.get("turn_id"),
            "fact": fact,
            "mood_snapshot": json.loads(md["mood_snapshot"]),
            "importance": float(md["importance"]),
        }

    # -- write --

    def store(self, user_id, fact, turn_id, mood_snapshot, importance) -> str:
        """Add a fact; returns its memory_id. Same validation and duplicate rule as MemoryManager.store."""
        _check_user_id(user_id)
        fact, importance = _check_fact(fact, importance)
        if turn_id is not None and not isinstance(turn_id, (str, int, float, bool)):
            raise ValueError(f"turn_id must be None, a string or a number for the chroma backend, got {turn_id!r}")
        rows = self._user_rows(user_id)
        seq = max((int(md.get("seq", 0)) for _, _, md in rows), default=0) + 1
        for memory_id, old_fact, md in rows:
            if _is_duplicate(old_fact, fact):
                self._col.update(ids=[memory_id], metadatas=[{
                    "timestamp": _now_iso(), "seq": seq,
                    "importance": max(float(md["importance"]), importance)}])
                return memory_id
        memory_id = uuid.uuid4().hex
        md = {"user_id": user_id, "timestamp": _now_iso(), "source": SOURCE, "importance": importance,
              "mood_snapshot": json.dumps(dict(mood_snapshot or {}), ensure_ascii=False), "seq": seq}
        if turn_id is not None:
            md["turn_id"] = turn_id
        self._col.add(ids=[memory_id], documents=[fact], metadatas=[md])
        return memory_id

    def delete(self, user_id, memory_id) -> bool:
        """Remove one record from Chroma for real. False for an unknown id or another user's memory."""
        _check_user_id(user_id)
        if not isinstance(memory_id, str) or not memory_id:
            return False
        got = self._col.get(ids=[memory_id], include=["metadatas"])
        if not got["ids"] or got["metadatas"][0].get("user_id") != user_id:
            return False
        self._col.delete(ids=[memory_id])
        return True

    def reset_user_memory(self, user_id) -> int:
        """Remove all of this user's facts (no one else's). Returns how many were removed."""
        _check_user_id(user_id)
        ids = self._col.get(where={"user_id": user_id}, include=[])["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    # -- read --

    def list_user_facts(self, user_id) -> list:
        """This user's records, newest first, same dict shape as MemoryManager (mood_snapshot as a dict)."""
        _check_user_id(user_id)
        rows = sorted(self._user_rows(user_id), key=lambda r: (r[2]["timestamp"], int(r[2].get("seq", 0))),
                      reverse=True)
        return [self._to_record(*r) for r in rows]

    def retrieve_relevant(self, user_id, query, top_k=None) -> list:
        """[{memory_id, fact, similarity, importance, score}], best score first, at most top_k (default cfg top_k).
        Scores ALL of this user's facts: similarity = 1 - cosine distance; similarity < min_similarity is dropped
        before scoring; score = w_similarity * similarity + w_importance * importance; ties -> newest first."""
        _check_user_id(user_id)
        k = int(self.cfg["top_k"] if top_k is None else top_k)
        w_sim, w_imp = float(self.cfg["w_similarity"]), float(self.cfg["w_importance"])
        min_sim = float(self.cfg["min_similarity"])
        n = len(self._col.get(where={"user_id": user_id}, include=[])["ids"])
        if n == 0 or not isinstance(query, str) or not query.strip():
            return []
        res = self._col.query(query_texts=[query], n_results=n, where={"user_id": user_id},
                              include=["documents", "metadatas", "distances"])
        hits = []
        for memory_id, fact, md, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                             res["distances"][0]):
            sim = 1.0 - float(dist)
            if sim < min_sim:
                continue
            imp = float(md["importance"])
            hits.append({"memory_id": memory_id, "fact": fact, "similarity": sim, "importance": imp,
                         "score": w_sim * sim + w_imp * imp, "_order": (md["timestamp"], int(md.get("seq", 0)))})
        hits.sort(key=lambda h: (h["score"], h["_order"]), reverse=True)
        for h in hits:
            del h["_order"]
        return hits[:max(k, 0)]


def make_memory_manager(conf: dict, embedding_function=None) -> BaseMemoryManager:
    """MemoryManager or ChromaMemoryManager from load_memory_config() output, by conf["memory"]["backend"]
    (missing -> "json"). embedding_function is passed to Chroma only (tests use a fake one)."""
    backend = conf["memory"].get("backend", "json")
    if backend == "json":
        return MemoryManager(conf["memory_file"], conf["memory"])
    if backend == "chroma":
        if not conf.get("vector_store_dir"):
            raise ValueError("memory.backend is 'chroma' but memory.vector_store_dir is not set in gpu/gpu_config.yaml")
        return ChromaMemoryManager(conf["vector_store_dir"], conf["memory"], embedding_function)
    raise ValueError(f"unknown memory.backend {backend!r}; expected one of {BACKENDS}")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main(argv=None, input_fn=input) -> int:
    ap = argparse.ArgumentParser(description="List, forget or delete a user's stored memories (demo step 5).")
    ap.add_argument("--user", required=True, help="user id")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="show this user's facts, newest first")
    mode.add_argument("--forget", metavar="MEMORY_ID", help="delete one memory")
    mode.add_argument("--delete-all", action="store_true",
                      help="delete all my data: memories, consent record and session logs (asks first)")
    ap.add_argument("--memory-file", help="memory file (default: logging.memory_file in gpu_config.yaml)")
    ap.add_argument("--backend", choices=BACKENDS, help="override memory.backend in gpu_config.yaml")
    args = ap.parse_args(argv)

    conf = dict(load_memory_config())
    conf["memory"] = dict(conf["memory"])
    if args.backend:
        conf["memory"]["backend"] = args.backend
    if args.memory_file:
        conf["memory_file"] = Path(args.memory_file)
    with make_memory_manager(conf) as mm:
        return _run_cli(args, mm, conf, input_fn)


def _run_cli(args, mm, conf, input_fn) -> int:
    if args.list:
        facts = mm.list_user_facts(args.user)
        print(f"{len(facts)} memories for {args.user!r} in {mm.path}")
        for r in facts:
            print(f"  {r['memory_id']}  {r['timestamp']}  importance {r['importance']:.2f}  {r['fact']}")
        return 0
    if args.forget:
        ok = mm.delete(args.user, args.forget)
        print("Deleted." if ok else f"No memory {args.forget!r} for user {args.user!r}; nothing deleted.")
        return 0 if ok else 1

    print(f"This deletes ALL stored data for {args.user!r}: memories, the consent record and session logs.")
    if input_fn("type yes to confirm: ").strip().lower() != "yes":
        print("Cancelled; nothing was deleted.")
        return 1
    report = mm.delete_all_my_data(args.user, conf["log_dir"], DEFAULT_CONSENT_PATH)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
