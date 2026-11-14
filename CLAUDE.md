# CLAUDE.md: mi-genderbias-qwen-nla

## What this repo is
A small, learning-oriented mechanistic interpretability (MI) demo for an independent research paper. It finds, explains and removes **occupational gender bias** in **Qwen2.5-7B-Instruct**. The explanation step uses Anthropic's **Natural Language Autoencoders** (NLA: activation verbalizer AV + activation reconstructor AR, checkpoints `kitft/nla-qwen2.5-7b-L20-{av,ar}`).
The sister repo `olawoye/mi-genderbias-mistral-steer` runs the same pipeline on Mistral-7B, using a linear probe instead of the NLA. Keep the shared files (`prompts.py`, `metrics.py`, `hooks.py`, `locate.py`, `steer.py`, `data/`) **identical** across both repos.

## Pipeline (one notebook per step)
00 setup check → 01 baseline bias (she−he logit diff, BLS correlation, résumé A/B task) → 02 locate (logit lens, activation patching per layer) → 03 explain (NLA on layer-20 activations) → 04 steer + controls (diff-of-means direction, projection ablation, random-direction control, perplexity, NLA re-read).

## Rules
- Use plain PyTorch forward hooks on HF `transformers` models, not TransformerLens or nnsight. Writing the hooks by hand is the point.
- All logic lives in `src/mibias/`; notebooks only call functions and plot.
- The NLA reads `hidden_states[20]` (the residual stream after block 20). Steering must act at layer ≤ 20 for the NLA to see it.
- Load NLA scale factors and the prompt template from each checkpoint's `nla_meta.yaml`. Never hardcode them.
- Notebook 03: extract activations → save parquet → free the target model → launch the SGLang AV server. Never hold two 7B models at once.
- Build the gender direction only from `prompts.contrast_pairs()`. These contain no occupation words, which keeps the evaluation non-circular.
- Every result is written to `results/` as JSON + PNG, with seed and model id recorded.
- Local machine is CPU-only: run `MIBIAS_SMOKE=1` (Qwen2.5-0.5B) for quick checks. `pytest -q` must pass without downloads.

## Known TODOs
- `data/occupations.csv` %-female values are **approximate**. Verify them against the BLS CPS Table 11 (latest year) before quoting in the paper.
