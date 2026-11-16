---
name: 3d-models-explorer
description: Turns a neural network checkpoint (safetensors or PyTorch) into an interactive 3D explorer web page. Every tensor is drawn at its real size and textured with its real weights, laid out in named groups, with data-flow links, a hover trace of inputs and outputs, an embed mode for articles, and an optional shared view where everyone sees the same camera. Use when someone asks to visualise, explore or show a model's architecture or weights in 3D, embed a model diagram in a blog post or article, or make a "model explorer" page for any transformer or other network.
license: MIT
---

# 3D models explorer

You turn a model checkpoint into a static web page: a 3D model where every tensor is a slab at its true shape,
coloured by its real weights, outlined in groups the reader understands ("Embeddings", "24 layers", "Output head").
It is a static folder (HTML, JS, one `scene.json`, one `atlas.png`), so it hosts anywhere and embeds with an iframe.

## First command

```bash
pip install numpy pyyaml pillow        # torch too, only for a PyTorch .bin/.pt checkpoint
python assets/init_groups.py CHECKPOINT groups.yaml --name "Model name"
```

`CHECKPOINT` is a folder in the Hugging Face layout (`model.safetensors`, sharded `model.safetensors.index.json`,
or `pytorch_model.bin`), or one `.safetensors`, `.bin`, `.pt`, `.pth` or `.ckpt` file. It prints the groups it
found. To get a Hugging Face model: `huggingface-cli download <id> --include "*.safetensors" "*.json" --local-dir ckpt`.

## What to do

1. **Get the checkpoint, not the code.** Only the weights file's header and a sample of each tensor are read, memory-mapped:
   a 7B model builds on a laptop. Never commit or publish the weights; the page holds only a sampled picture of them.
2. **Draft the groups:** `python assets/init_groups.py CHECKPOINT groups.yaml`. It finds the repeated block
   (`layers.{i}`, `h.{i}`, `layer.{i}`, `block.{i}`), orders one copy in the usual data flow, and puts the rest in
   "Embeddings" and "Output".
3. **Make the groups true and readable.** Read the model's code or paper, then edit `groups.yaml`
   ([references/groups.md](references/groups.md)):
   - name groups in the reader's words, not code names;
   - check each layer's rows follow the real order (pre-norm vs post-norm, parallel attention and MLP);
   - mark parts the use case never runs as `unused: true` (they draw grey);
   - add `flow` between groups and `links` for anything that skips (a head reading a middle layer).
4. **Build:** `python assets/build.py CHECKPOINT groups.yaml site/`. It prints each group's tensor and weight
   counts. "N tensors in no group, drawn under Other" means the groups file missed some: place them or leave them.
   The build checks every weight is drawn exactly once.
5. **Look at it:** `python -m http.server -d site 8000`, open `http://localhost:8000/`. Check that the groups read
   in data-flow order, hovering a tensor traces what feeds it and what reads it, and the phone width (390 px) works.
   `?selftest` runs the page's own checks in the console.
6. **Optional, real values:** `python assets/capture_values.py MODEL site/values.json "an example input" --scene site/scene.json`
   runs one input through the model (transformers + torch) and shows each stage's output beside it. Rebuild with it as
   the fourth argument to keep it on rebuilds.
7. **Host and embed** ([references/hosting.md](references/hosting.md), [references/embedding.md](references/embedding.md)):
   any static host works; on Spacesheep, `npx -y spacesheep@latest deploy site --title "…"`.
8. **Optional, shared view:** `python assets/bundle_shared.py site shared/` makes a version where whoever turns,
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
  model in someone else's three.js scene ([references/module.md](references/module.md)).
- `assets/example-gliner2-base.yaml` a complete hand-written groups file (base GLiNER, a DeBERTa encoder with heads).
