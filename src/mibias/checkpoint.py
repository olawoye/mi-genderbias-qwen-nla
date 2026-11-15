"""Save-and-resume helpers: a Colab disconnect should cost at most one chunk of work.

Design
------
* Everything lives under one run directory on Google Drive (survives runtime resets).
* Every write is atomic: write to a temp file, then `os.replace`. A crash mid-write
  never leaves a half-written file that later looks "done".
* `run_meta.json` records what the run is (model id, seed, ...). Re-opening the same
  directory with a different config raises `StaleCheckpointError` instead of silently
  mixing results from two different models.
* `cached(name, fn)`      - one-shot step: compute once, load on every later call.
* `map_chunks(name, items, fn)` - long loop split into chunks; finished chunks are
  skipped on resume. The item list is fingerprinted, so editing the prompts
  invalidates the old chunks instead of reusing them.

Results are JSON (lists/dicts) or dicts of numpy arrays (saved as .npz). Torch tensors
should be converted with `.float().cpu().numpy()` before returning. That keeps this module
torch-free and testable anywhere.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

Arrays = dict[str, np.ndarray]


class StaleCheckpointError(RuntimeError):
    """The run directory was created with a different config or different items."""


def _jsonable(o: Any) -> Any:
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def fingerprint(obj: Any) -> str:
    """Short stable hash of any JSON-able object (dataclasses included)."""
    blob = json.dumps(obj, sort_keys=True, default=_jsonable).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _dump_json(obj: Any) -> bytes:
    return json.dumps(obj, indent=1, default=_jsonable).encode()


def _dump_npz(arrays: Arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class Checkpointer:
    """One run directory = one (model, seed, config) combination."""

    def __init__(self, root: str | Path, meta: dict[str, Any], verbose: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.meta = json.loads(_dump_json(meta))  # normalise (tuples->lists, Paths->str)
        meta_path = self.root / "run_meta.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old != self.meta:
                diff = {k: (old.get(k), self.meta.get(k)) for k in set(old) | set(self.meta)
                        if old.get(k) != self.meta.get(k)}
                raise StaleCheckpointError(
                    f"{self.root} was created with a different config: {diff}. "
                    "Use a new run name, or delete the directory to start over.")
        else:
            atomic_write_bytes(meta_path, _dump_json(self.meta))

    # ---------- low level ----------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ckpt] {msg}")

    def _path(self, name: str, kind: str) -> Path:
        return self.root / f"{name}.{'json' if kind == 'json' else 'npz'}"

    def has(self, name: str, kind: str = "json") -> bool:
        return self._path(name, kind).exists()

    def save(self, name: str, obj: Any, kind: str = "json") -> Path:
        p = self._path(name, kind)
        atomic_write_bytes(p, _dump_json(obj) if kind == "json" else _dump_npz(obj))
        return p

    def load(self, name: str, kind: str = "json") -> Any:
        p = self._path(name, kind)
        if kind == "json":
            return json.loads(p.read_text())
        with np.load(p) as z:
            return {k: z[k] for k in z.files}

    # ---------- one-shot step ----------
    def cached(self, name: str, fn: Callable[[], Any], kind: str = "json") -> Any:
        if self.has(name, kind):
            self._log(f"{name}: loaded (already done)")
            return self.load(name, kind)
        t = time.time()
        out = fn()
        self.save(name, out, kind)
        self._log(f"{name}: computed and saved in {time.time() - t:.1f}s")
        return out

    # ---------- resumable loop ----------
    def map_chunks(self, name: str, items: Sequence[Any], fn: Callable[[Sequence[Any]], Any],
                   chunk_size: int = 16, kind: str = "json") -> Any:
        """Apply fn to items in chunks, saving each chunk. Re-running skips finished chunks.

        kind="json":   fn(chunk) -> list with len(chunk) entries;  returns the full list.
        kind="arrays": fn(chunk) -> dict of arrays, axis 0 == len(chunk); returns concatenated dict.
        """
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        spec = {"items": fingerprint(list(items)), "n": len(items), "chunk_size": chunk_size, "kind": kind}
        spec_path = d / "_spec.json"
        if spec_path.exists():
            old = json.loads(spec_path.read_text())
            if old != spec:
                raise StaleCheckpointError(
                    f"{d}: items or chunking changed since the last run ({old} -> {spec}). "
                    f"Delete {d} (or call reset('{name}')) to recompute.")
        else:
            atomic_write_bytes(spec_path, _dump_json(spec))

        ext = "json" if kind == "json" else "npz"
        n_chunks = (len(items) + chunk_size - 1) // chunk_size
        done = sum((d / f"chunk_{i:05d}.{ext}").exists() for i in range(n_chunks))
        self._log(f"{name}: {done}/{n_chunks} chunks already done")

        for i in range(n_chunks):
            p = d / f"chunk_{i:05d}.{ext}"
            if p.exists():
                continue
            chunk = items[i * chunk_size:(i + 1) * chunk_size]
            out = fn(chunk)
            n_out = len(out) if kind == "json" else len(next(iter(out.values())))
            if n_out != len(chunk):
                raise ValueError(f"fn returned {n_out} results for a chunk of {len(chunk)}")
            atomic_write_bytes(p, _dump_json(out) if kind == "json" else _dump_npz(out))
            if self.verbose and (i + 1) % max(1, n_chunks // 10) == 0:
                self._log(f"{name}: chunk {i + 1}/{n_chunks}")

        parts = []
        for i in range(n_chunks):
            p = d / f"chunk_{i:05d}.{ext}"
            if kind == "json":
                parts.append(json.loads(p.read_text()))
            else:
                with np.load(p) as z:
                    parts.append({k: z[k] for k in z.files})
        if kind == "json":
            return [x for part in parts for x in part]
        return {k: np.concatenate([part[k] for part in parts], axis=0) for k in parts[0]}

    # ---------- housekeeping ----------
    def reset(self, name: str) -> None:
        for kind in ("json", "arrays"):
            p = self._path(name, kind)
            if p.exists():
                p.unlink()
        if (self.root / name).is_dir():
            shutil.rmtree(self.root / name)
        self._log(f"{name}: reset")

    def status(self) -> list[str]:
        return sorted(p.stem if p.is_file() else p.name + "/"
                      for p in self.root.iterdir() if not p.name.startswith((".", "run_meta")))


def open_run(cfg, run: str = "v1") -> Checkpointer:
    """Checkpointer for this config. Bump `run` to start a fresh experiment."""
    meta = {"model_id": cfg.model_id, "seed": cfg.seed, "dtype": cfg.dtype, "run": run}
    return Checkpointer(Path(cfg.cache_dir) / run, meta)


def export_results(cfg, dest: str | Path | None = None) -> Path:
    """Copy final JSON/PNG results from Drive into the repo's results/ folder (for committing)."""
    from .config import REPO_ROOT
    dest = Path(dest or REPO_ROOT / "results")
    dest.mkdir(parents=True, exist_ok=True)
    for p in Path(cfg.results_dir).glob("*"):
        if p.is_file() and p.suffix in {".json", ".png", ".csv"}:
            shutil.copy2(p, dest / p.name)
    return dest
