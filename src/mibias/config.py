"""Single source of truth for model ids, layers and paths (Qwen + NLA repo)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = Path("/content/drive/MyDrive")


def on_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()


def _drive_or_local(env_var: str, sub: str, local: Path) -> Path:
    """$env_var if set, else a Drive path on Colab (survives runtime resets), else repo-local."""
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    return DRIVE_ROOT / sub if on_colab() else local


@dataclass(frozen=True)
class Config:
    # Target model
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    n_layers: int = 28
    d_model: int = 3584
    dtype: str = "bfloat16"

    # Tiny model with the same architecture family, for CPU smoke tests
    smoke_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # NLA checkpoints only read residual-stream layer 20 (hidden_states[20]).
    nla_layer: int = 20
    nla_av_id: str = "kitft/nla-qwen2.5-7b-L20-av"
    nla_ar_id: str = "kitft/nla-qwen2.5-7b-L20-ar"

    # Chosen from notebook 02 (activation patching), must be <= nla_layer; None until then.
    steer_layer: int | None = None

    seed: int = 0
    # Checkpoints + results go to Drive on Colab. `checkpoint.export_results` copies the
    # final JSON/PNGs into the repo's results/ folder for committing.
    results_dir: Path = field(default_factory=lambda: _drive_or_local("MIBIAS_RESULTS", "mibias_results/qwen", REPO_ROOT / "results"))
    cache_dir: Path = field(default_factory=lambda: _drive_or_local("MIBIAS_CACHE", "mibias_cache/qwen", REPO_ROOT / ".cache"))


def get_config(smoke: bool | None = None) -> Config:
    """smoke=True (or env MIBIAS_SMOKE=1) swaps in the tiny model for local tests."""
    cfg = Config()
    if smoke is None:
        smoke = os.environ.get("MIBIAS_SMOKE") == "1"
    if smoke:
        from dataclasses import replace
        cfg = replace(cfg, model_id=cfg.smoke_model_id, n_layers=24, d_model=896,
                      nla_layer=16, cache_dir=REPO_ROOT / ".cache" / "smoke",
                      results_dir=REPO_ROOT / ".cache" / "smoke_results")
    return cfg
