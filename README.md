# Model Anatomy

An agent skill that turns a model checkpoint into an interactive 3D page: every tensor at its real size, textured
with its real weights, laid out in named groups with data-flow links, a hover trace, an article embed mode and an
optional shared view. It works for any safetensors or PyTorch checkpoint, not one architecture.

The skill is the [`model-anatomy/`](model-anatomy) folder in the [Agent Skills](https://agentskills.io/specification)
format. Its [`SKILL.md`](model-anatomy/SKILL.md) is what an agent reads.

## Try it by hand

```bash
pip install numpy pyyaml pillow
huggingface-cli download openai-community/gpt2 --include "*.safetensors" "*.json" --local-dir ckpt
python model-anatomy/assets/init_groups.py ckpt groups.yaml --name "GPT-2 small"
python model-anatomy/assets/build.py ckpt groups.yaml site
python -m http.server -d site 8000      # http://localhost:8000/
```

## What's in it

| Path | What it is |
|---|---|
| `model-anatomy/SKILL.md` | the instructions an agent reads |
| `model-anatomy/assets/build.py` | checkpoint + groups file → static site (`scene.json`, `atlas.png`, the page) |
| `model-anatomy/assets/init_groups.py` | drafts a groups file for any checkpoint |
| `model-anatomy/assets/capture_values.py` | optional: one real input's output at each stage |
| `model-anatomy/assets/bundle_shared.py` + `shared.js` | optional: the shared-view version (a Spacesheep worker space) |
| `model-anatomy/assets/web/` | the page, the embeddable module and its demo pages, three.js r160 |
| `model-anatomy/assets/example-gliner2-base.yaml` | a complete hand-written groups file |
| `model-anatomy/references/` | groups file format, checkpoints, embedding, hosting, the module API |
| `tests/` | the scripts on small made-up checkpoints: `pip install numpy pyyaml pillow pytest && pytest tests` |
| `THIRD_PARTY.md` | every third-party file and its licence |

## Check and release

`npx -y spacesheep@latest skills lint` checks the folder the way Spacesheep does; CI runs it and the tests on every
push and pull request. A tag `v<major>.<minor>.<patch>` makes a release with that version's `CHANGELOG.md` notes.
Never move a tag.

To try the skill in Claude Code, copy `model-anatomy/` into `~/.claude/skills/`.
