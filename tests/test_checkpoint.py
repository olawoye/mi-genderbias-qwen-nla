"""Save/resume behaviour, including a simulated Colab crash mid-loop."""
import numpy as np
import pytest

from mibias import prompts
from mibias.checkpoint import Checkpointer, StaleCheckpointError, fingerprint

META = {"model_id": "m", "seed": 0}


def test_cached_runs_once(tmp_path):
    calls = []
    ck = Checkpointer(tmp_path, META, verbose=False)
    f = lambda: calls.append(1) or {"x": 1}
    assert ck.cached("step", f) == {"x": 1}
    assert Checkpointer(tmp_path, META, verbose=False).cached("step", f) == {"x": 1}
    assert len(calls) == 1


def test_map_chunks_resumes_after_crash(tmp_path):
    items = list(range(10))
    seen = []

    def flaky(chunk):
        if chunk[0] == 6 and not seen.count("crashed"):
            seen.append("crashed")
            raise RuntimeError("simulated Colab disconnect")
        seen.extend(chunk)
        return [x * 2 for x in chunk]

    ck = Checkpointer(tmp_path, META, verbose=False)
    with pytest.raises(RuntimeError):
        ck.map_chunks("loop", items, flaky, chunk_size=3)
    out = Checkpointer(tmp_path, META, verbose=False).map_chunks("loop", items, flaky, chunk_size=3)
    assert out == [x * 2 for x in items]
    assert [x for x in seen if x != "crashed"] == items  # every item computed exactly once
    assert not list(tmp_path.rglob("*.tmp")), "no half-written temp files left behind"


def test_map_chunks_arrays(tmp_path):
    ck = Checkpointer(tmp_path, META, verbose=False)
    fn = lambda c: {"acts": np.ones((len(c), 4)) * np.array(c)[:, None]}
    out = ck.map_chunks("acts", list(range(7)), fn, chunk_size=3, kind="arrays")
    assert out["acts"].shape == (7, 4) and out["acts"][6, 0] == 6


def test_stale_config_is_refused(tmp_path):
    Checkpointer(tmp_path, META, verbose=False)
    with pytest.raises(StaleCheckpointError):
        Checkpointer(tmp_path, {"model_id": "other", "seed": 0}, verbose=False)


def test_changed_items_are_refused(tmp_path):
    ck = Checkpointer(tmp_path, META, verbose=False)
    ck.map_chunks("loop", [1, 2, 3], lambda c: list(c))
    with pytest.raises(StaleCheckpointError):
        ck.map_chunks("loop", [1, 2, 4], lambda c: list(c))
    ck.reset("loop")
    assert ck.map_chunks("loop", [1, 2, 4], lambda c: list(c)) == [1, 2, 4]


def test_fingerprint_handles_prompt_dataclasses():
    a = fingerprint(prompts.pronoun_prompts())
    assert a == fingerprint(prompts.pronoun_prompts()) and len(a) == 12
