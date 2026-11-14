"""Single source of truth for model ids, layers and paths (Qwen + NLA repo)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")
    # On Colab, cache heavy activations on Drive so a runtime reset doesn't lose them.
    cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("MIBIAS_CACHE", "/content/drive/MyDrive/mibias_cache/qwen")
    ))


def get_config(smoke: bool | None = None) -> Config:
    """smoke=True (or env MIBIAS_SMOKE=1) swaps in the tiny model for local tests."""
    cfg = Config()
    if smoke is None:
        smoke = os.environ.get("MIBIAS_SMOKE") == "1"
    if smoke:
        from dataclasses import replace
        cfg = replace(cfg, model_id=cfg.smoke_model_id, n_layers=24, d_model=896,
                      nla_layer=16, cache_dir=REPO_ROOT / ".cache")
    return cfg
