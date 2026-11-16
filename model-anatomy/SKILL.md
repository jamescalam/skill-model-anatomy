---
name: model-anatomy
description: Use when someone wants to see a neural network in 3D, to visualise or explore a model's architecture or weights, embed a model diagram in an article, or make a "model explorer" page from a safetensors or PyTorch checkpoint. It builds a static page where every tensor is drawn at its real size with its real weights, in named groups with data-flow links and a hover trace.
license: MIT
---

# Model Anatomy

You turn a model checkpoint into a static web page: a 3D model where every tensor is a slab at its true shape,
coloured by its real weights, outlined in groups the reader understands ("Embeddings", "24 layers", "Output head").
It is a static folder (HTML, JS, one `scene.json`, one `atlas.png`), so it hosts anywhere and embeds with an iframe.

## First command

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy pyyaml pillow huggingface_hub
hf download <org>/<model> model.safetensors config.json --local-dir ckpt
python3 <this skill's folder>/assets/init_groups.py ckpt groups.yaml --name "Model name"
```

- **No `python3`, or no `pip` (NixOS, some Debian):** use [uv](https://docs.astral.sh/uv/). Skip the first two
  lines, write `uv run --with numpy,pyyaml,pillow python` wherever this file says `python3`, and
  `uvx --from huggingface_hub hf download …` for `hf download …`.
- **The model's real id:** Hugging Face moved many short names under an organisation: `distilgpt2` is
  `distilbert/distilgpt2`, `gpt2` is `openai-community/gpt2`, `bert-base-uncased` is `google-bert/bert-base-uncased`.
  The model's page shows its id at the top; `hf download` says "not found" for a wrong one.
- **Which files:** name them. A big model's weights are sharded: list the shards and `model.safetensors.index.json`
  (the model page's "Files" tab shows them). Only PyTorch weights? Fetch `pytorch_model.bin` and install torch.
  `config.json` lets the draft see a tied output head. (`huggingface-cli` is the old name of `hf`.)
- `CHECKPOINT` is that folder, or one `.safetensors`, `.bin`, `.pt`, `.pth` or `.ckpt` file. `init_groups.py`
  prints the groups it found.

## What to do

1. **Get the checkpoint, not the code.** Only the weights file's header and a sample of each tensor are read, memory-mapped:
   a 7B model builds on a laptop. Never commit or publish the weights; the page holds only a sampled picture of them.
2. **Draft the groups:** `python3 <this skill's folder>/assets/init_groups.py CHECKPOINT groups.yaml`. It finds the repeated block
   (`layers.{i}`, `h.{i}`, `layer.{i}`, `block.{i}`), orders one copy in the usual data flow, and puts the rest in
   "Embeddings" and "Output".
3. **Make the groups true and readable.** Read the model's code or paper, then edit `groups.yaml`
   ([references/groups.md](references/groups.md)):
   - name groups in the reader's words, not code names;
   - give each group a `role` (`embed`, `attn_q`, `classifier`, `other`…): the page takes colours and where it marks
     Input and Output from roles, never from group names;
   - check each layer's rows follow the real order (pre-norm vs post-norm, parallel attention and MLP);
   - mark parts the use case never runs as `unused: true` (they draw grey);
   - add `flow` between groups and `links` for anything that skips (a head reading a middle layer).
4. **Build:** `python3 <this skill's folder>/assets/build.py CHECKPOINT groups.yaml site/`. It prints each group's tensor and weight
   counts. "N tensors in no group, drawn under Other" means the groups file missed some: place them or leave them.
   The build checks every weight is drawn exactly once.
5. **Look at it:** `python3 -m http.server -d site 8000` (any free port), open `http://localhost:8000/`. Check that the groups read
   in data-flow order, hovering a tensor traces what feeds it and what reads it, and the phone width (390 px) works.
   `?selftest` runs the page's own checks in the console.
6. **Optional, real values:** `python3 <this skill's folder>/assets/capture_values.py MODEL site/values.json "an example input" --scene site/scene.json`
   runs one input through the model (transformers + torch) and shows each stage's output beside it. Rebuild with it as
   the fourth argument to keep it on rebuilds.
7. **Host and embed** ([references/hosting.md](references/hosting.md), [references/embedding.md](references/embedding.md)):
   any static host works; on Spacesheep, `npx -y spacesheep@latest deploy site --title "…"`.
8. **Optional, shared view:** `python3 <this skill's folder>/assets/bundle_shared.py site shared/` makes a version where whoever turns,
   zooms or taps the model moves it for everyone who has it open (a Spacesheep worker space; see hosting.md).

## Reads, and what it does with the rest

| Format | Read with | Notes |
|---|---|---|
| `.safetensors`, one or sharded | numpy (memory-mapped) | F32, F16, BF16, F64, integer and bool; fp8 draws grey at its true size |
| PyTorch `.bin` `.pt` `.pth` `.ckpt` | torch, `weights_only=True` | a state dict, or one wrapped in `state_dict` / `model` |
| GGUF, ONNX, Keras `.h5`, Flax `.msgpack`, TF | none | stops and says to convert ([references/checkpoints.md](references/checkpoints.md)) |

## Rules

- **Every tensor is drawn.** Nothing is hidden: tensors no group names go under "Other". Say so if you leave them.
- **Real sizes only.** Don't rescale one part to look better. A table longer than 80 units is drawn shortened and its
  label says the true size: that is the only exception.
- **The reader's words.** Group and stage names a non-specialist understands; code names stay in the hover card.
- **Nothing private in the page:** no weights files, keys, or local paths. The atlas is a sampled, 8-bit picture.
- **Big models:** if the textures don't fit one 4096-pixel atlas, the build halves them and says so. For a very
  large model, set `texture: 64` and `scale: 256` in the groups file.
- **Report what you checked:** the build's counts, the browser widths you looked at, and what you didn't check.

## Files

- `assets/build.py` checkpoint + groups file → static site. `assets/init_groups.py` drafts the groups file.
- `assets/capture_values.py` one real run's stage outputs. `assets/bundle_shared.py` the shared-view version.
- `assets/web/` the page (`index.html`, `explorer.js`, three.js 0.160.0 vendored). `explorer-module.js` mounts the
  model in someone else's three.js scene, `module-demo.html` tries it ([references/module.md](references/module.md)).
- `assets/example-bge-small.yaml` and `assets/example-gliner2-base.yaml`: complete hand-written groups files (a BERT
  embedding model; base GLiNER, a DeBERTa encoder with heads).
