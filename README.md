# Finding and fixing gender bias in Qwen2.5-7B with Natural Language Autoencoders

A small, reproducible mechanistic interpretability demo:

1. **Measure:** does the model assume a nurse is "she" and an engineer is "he"? Does it steer equally qualified CVs into different roles depending on the candidate's name?
2. **Locate:** at which layer does gender information enter the residual stream (logit lens, activation patching)?
3. **Explain:** read the layer-20 activations in plain English with Anthropic's [Natural Language Autoencoders](https://www.transformer-circuits.pub/2026/nla/index.html) (released AV/AR checkpoints for Qwen2.5-7B).
4. **Fix:** remove a difference-of-means "gender direction" and show the bias shrinks, against a random-direction control, without breaking the model.

Sister repo (same pipeline, Mistral-7B, linear probe instead of the NLA): [mi-genderbias-mistral-steer](https://github.com/olawoye/mi-genderbias-mistral-steer).

## Notebooks

| # | Notebook | GPU | Open |
|---|---|---|---|
| 00 | Setup check | any | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olawoye/mi-genderbias-qwen-nla/blob/main/notebooks/00_colab_setup.ipynb) |
| 01 | Baseline bias | L4 | _coming next_ |
| 02 | Locate | L4 | _coming_ |
| 03 | Explain with NLA | A100 | _coming_ |
| 04 | Steer + controls | L4 / A100 | _coming_ |

## Run locally (CPU, for development only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

## Status
Work in progress, accompanying an independent research paper on mechanistic interpretability.

## License
MIT. Model and NLA checkpoints keep their own licences.
