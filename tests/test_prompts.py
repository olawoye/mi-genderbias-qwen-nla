"""CPU-only tests: no model download needed. Run with `pytest -q`."""
from mibias import prompts
from mibias.config import get_config


def test_occupations_load():
    occs = prompts.load_occupations()
    assert len(occs) >= 40
    assert all(0 <= o.pct_female <= 100 for o in occs)
    assert len({o.name for o in occs}) == len(occs), "duplicate occupation"


def test_pronoun_prompts_end_before_pronoun():
    ps = prompts.pronoun_prompts()
    assert len(ps) == len(prompts.load_occupations()) * len(prompts.PRONOUN_TEMPLATES)
    assert all(not p.text.endswith(" ") for p in ps), "trailing space breaks ' he'/' she' tokenisation"


def test_resume_prompts_are_balanced():
    rs = prompts.resume_prompts()
    males = [r for r in rs if r.gender == "male"]
    females = [r for r in rs if r.gender == "female"]
    assert len(males) == len(females)
    # Each gender sees the male role in slot A exactly half the time.
    assert sum(r.male_role_letter == "A" for r in males) * 2 == len(males)


def test_contrast_pairs_have_no_occupations():
    # The gender direction must not be built from occupation words (avoids circularity).
    occ_words = {w.lower() for o in prompts.load_occupations() for w in o.name.split()}
    for m, f in prompts.contrast_pairs():
        words = set((m + " " + f).lower().replace(".", "").split())
        assert not (words & occ_words), (m, f)


def test_config_smoke_switch():
    assert get_config(smoke=True).model_id != get_config(smoke=False).model_id
