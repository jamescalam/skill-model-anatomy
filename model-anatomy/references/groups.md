# The groups file

A YAML file that says what each part of the checkpoint is, in the order data flows through it. `init_groups.py`
drafts one; [../assets/example-gliner2-base.yaml](../assets/example-gliner2-base.yaml) is a complete hand-made one.

```yaml
model: GPT-2 small            # the title on the page
scale: 64                     # weights per unit of length (bigger model: raise it, 128 or 256)
texture: 128                  # image pixels on a grid's long side (bigger model: lower it, 64)
flow:                         # what feeds what between groups
  - [Embeddings, 12 layers]
  - [12 layers, Output]
links:                        # optional: extra links between modules, by exact module name
  - ["@12 layers", lm_head]   #   "@Group" is that group's output (its last row)
groups:
  - name: Embeddings
    rows:
      - [{m: wte, lie: flat, texture: 1024}, wpe]
  - name: 12 layers
    repeat: h.{i}             # every h.0, h.1, … copy, stacked back in depth
    skip: {in: 2, pairs: [[2, 5]]}   # optional: residual adds (row indexes, from 0): see "Residual links" below
    rows:
      - [ln_1]
      - [attn.c_attn]
      - [attn.c_proj]
      - [ln_2]
      - [mlp.c_fc]
      - [mlp.c_proj]
  - name: Output
    role: classifier          # optional: the group's colour
    rows:
      - [ln_f]
```

## Keys

| Key | Where | What it does |
|---|---|---|
| `name` | group | its outline label; a group without one only arranges its children |
| `children` | group | other groups inside it |
| `arrange` | group | `down` (default) or `across`, for its children |
| `rows` | group | top to bottom; each row is a list of modules side by side (they run in parallel) |
| `repeat` | group | a name with `{i}`; `rows` names are then relative to it, one copy per number found |
| `skip` | repeated group | residual adds, pointing forward from what is carried to where it's added: `in: k` the layer's input to row k; `pairs: [[a, b]]` row a's output to row b |
| `chain` | group | `false` stops rows linking to the row below |
| `unused` | group | `true` draws it grey and faded; the reader can hide it |
| `role` | group or module | what it does: `embed`, `attn_q`, `attn_k`, `attn_v`, `attn_out`, `ff_up`, `ff_down`, `norm`, `classifier`, `other`. It sets the colour, and the page marks Input at the first `embed` group and Output at the last running `classifier` group. Without one, a group's role is the one holding most of its weights |
| `m` | module | a tensor name or a prefix: every tensor under it is drawn (a weight as a grid, its bias as a strip beside it) |
| `lie: flat` | module | lays a grid flat, for a table too long to stand up (a vocabulary) |
| `texture` | module or top | image pixels on the long side (a vocabulary table wants 1024 or 2048) |
| `title` | module | its name on the card, instead of the one build.py guesses |
| `flow` | top | `[from group, to group, (side), (rows)]`; `side` links to every copy of a repeated block |
| `links` | top | `[from module, to module, (kind)]`; kind `flow`, `side` or `skip` |

## Residual links (`skip`)

A skip link points **forward, from the value that is carried round to the row where it's added back in**. Rows count
from 0, top to bottom, in one copy of the repeated block.

- `in: k`: the layer's input (the previous copy's last row) is added at row k.
- `pairs: [[a, b]]`: row a's output is added at row b.

| Layout | Rows | `skip` |
|---|---|---|
| Pre-norm (GPT-2, Llama) | norm, qkv, attention out, norm, expand, shrink | `{in: 2, pairs: [[2, 5]]}`: added after attention out, then after shrink |
| Post-norm (BERT) | qkv, attention out, norm, expand, shrink, norm | `{in: 2, pairs: [[2, 5]]}`: added at each half's norm (it adds, then normalises) |
| GPT-2 with its mask row | norm, qkv, mask, attention out, norm, expand, shrink | `{in: 3, pairs: [[3, 6]]}` |

`init_groups.py` drafts `skip` this way; check it against the model's forward pass.

## Values that aren't learned

Some checkpoints save buffers beside the weights: GPT-2's causal mask (`attn.bias`, 1 x 1 x 1024 x 1024 per layer),
position numbers (`position_ids`), rotary frequencies, a norm's running statistics. The build draws them like any
tensor (leading 1s dropped, so the mask is a 1024 x 1024 grid) but counts them apart: the page says "N learned
weights, plus M fixed values". Place them in the layer like any module (`attn.bias` as its own row, between qkv and
attention out); `init_groups.py` does.

## Tied output heads

Most causal language models (GPT-2, many Llamas) score their last hidden state against the word table itself: the
checkpoint has no head tensor. With `config.json` beside the weights, `init_groups.py` names the last group
"Output (head is the word table)" and links its final norm to the word table. Without it, add both yourself.

## Making it true

- **Read the model's forward pass** before trusting the draft. Pre-norm models (GPT-2, Llama) put the norm first;
  post-norm ones (BERT) put it after. Parallel blocks (GPT-J, Falcon) run attention and MLP in one row.
- **Mixture-of-experts:** put the router first, then each expert's modules in one row (`[experts.0, experts.1, …]`).
- **Encoder-decoder (T5, BART):** two repeated groups, `arrange: across` in a parent, and a `side` flow from the
  encoder to every decoder layer (cross-attention).
- **Tied weights** (an output head that reuses the embedding table) are one tensor in the file: draw it once and
  add a `links` entry from the last layer to it.
- **A tensor in two groups** stops the build: each tensor lands in exactly one.
- Anything no group names is drawn under "Other". That is fine for buffers; name the rest.
