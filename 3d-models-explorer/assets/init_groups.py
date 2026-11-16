"""Draft a groups file for any checkpoint: what comes before the repeated blocks, the blocks, and what comes after.

    python init_groups.py CHECKPOINT OUT.yaml [--name "Model name"]

It finds the repeated block (the `layers.{i}` / `h.{i}` / `layer.{i}` prefix with the most copies), orders one
copy's modules in the usual data flow (norm, query/key/value side by side, attention output, norm, feed-forward
expand, shrink), and puts everything else in "Embeddings" (before) or "Output" (after). It is a draft: read it,
rename the groups in the reader's words, mark unused parts, and fix any row it guessed wrong (references/groups.md).
Every tensor a draft leaves out is still drawn, under "Other", so nothing is hidden by mistake.
"""
import re, sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import describe, open_checkpoint   # noqa: E402

BLOCK = re.compile(r"^(.*?(?:^|\.)(?:layer|layers|h|block|blocks|resblocks))\.(\d+)\.")
# Where a module of one layer sits in the flow; modules with the same rank share a row (they run side by side).
RANK = {"attention: norm": 0, "attention: query, key and value": 1, "attention: query": 1, "attention: key": 1,
        "attention: value": 1, "attention: output": 2, "feed-forward: norm before it": 3, "feed-forward: gate": 4,
        "feed-forward: expand": 4, "feed-forward: shrink": 5, "feed-forward: norm": 6}
SUFFIX = re.compile(r"\.(weight|weight_g|weight_v|weight_orig|parametrizations\.weight\.original[01]|bias|scale|g|b|gamma|beta|inv_freq|running_mean|running_var|num_batches_tracked)$")


def module_of(name):
    """A tensor's module: its name without the last part when that part is a weight, bias or a norm's scale."""
    return SUFFIX.sub("", name) if SUFFIX.search(name) else name


def dedupe(mods):
    """Drop a module that is a prefix of another (build.py draws every tensor under a prefix): keep the longest."""
    out = []
    for m in mods:
        if not any(o != m and (o.startswith(m + ".")) for o in mods):
            out.append(m)
    return out


def entry(ck, mod, names):
    """A module as the groups file writes it: a long table (a vocabulary) lies flat with a bigger texture."""
    shapes = [ck.shape(n) for n in names if n == mod or n.startswith(mod + ".")]
    big = [s for s in shapes if len(s) >= 2 and s[0] >= 8 * max(1, s[-1]) and s[0] >= 4096]
    return {"m": mod, "lie": "flat", "texture": 1024} if big else mod


def draft(ck, model_name=None):
    names = ck.names()
    copies = {}
    for n in names:
        m = BLOCK.match(n)
        if m:
            copies.setdefault(m.group(1), set()).add(int(m.group(2)))
    copies = {p: sorted(c) for p, c in copies.items() if len(c) > 1}
    groups, flow = [], []
    if copies:
        prefix = max(copies, key=lambda p: (len(copies[p]), -len(p)))
        n_layers, first = len(copies[prefix]), copies[prefix][0]
        pat = f"{prefix}.{{i}}"
        inside = re.compile("^" + re.escape(prefix) + r"\.\d+\.")
        one = f"{prefix}.{first}."
        layer_mods = dedupe(sorted({module_of(n)[len(one):] for n in names if n.startswith(one)}))
        by_rank = {}
        for m in layer_mods:
            t = describe(f"{one}{m}")["title"].split(" · ", 1)[-1]
            r = 3 if m.endswith("attention.output.LayerNorm") else RANK.get(t, 7)   # BERT's norm comes after attention
            by_rank.setdefault(r, []).append(m)
        rows = [sorted(by_rank[r]) for r in sorted(by_rank)]
        outside = [n for n in names if not inside.match(n)]
        # before or after the blocks: the embedding tables (and a norm beside them) come before; the rest after
        role = {n: describe(module_of(n))["role"] for n in outside}
        parent = lambda n: module_of(n).rpartition(".")[0]
        embed_parents = {parent(n) for n in outside if role[n] == "embed"} - {""}
        before = [n for n in outside if role[n] == "embed" or (role[n] == "norm" and parent(n) in embed_parents)]
        after = [n for n in outside if n not in before]
        if before:
            emb = sorted(dedupe(sorted({module_of(n) for n in before})), key=lambda m: (describe(m)["role"] != "embed", m))   # tables, then norms
            groups.append({"name": "Embeddings", "role": "embed", "rows": [[entry(ck, m, names)] for m in emb]})
        block = {"name": f"{n_layers} layers", "role": "attn_q", "repeat": pat, "rows": rows}
        groups.append(block)
        if after:
            head = any(describe(module_of(n))["role"] == "classifier" for n in after)
            groups.append({"name": "Output", "role": "classifier" if head else "other",
                           "rows": [[entry(ck, m, names)] for m in dedupe(sorted({module_of(n) for n in after}))]})
        if before:
            flow.append(["Embeddings", block["name"]])
        if after:
            flow.append([block["name"], "Output"])
    else:   # no repeated block (a small or unusual model): one group, each module a row, in the file's order
        mods = dedupe(list(dict.fromkeys(module_of(n) for n in names)))
        groups.append({"name": "Model", "role": "other", "rows": [[entry(ck, m, names)] for m in mods]})
    spec = {"model": model_name or "Model", "scale": 64, "texture": 128}
    if flow:
        spec["flow"] = flow
    spec["groups"] = groups
    return spec


HEAD = """# A draft from init_groups.py: check it against the model's code before anyone sees it (references/groups.md).
# groups: each is outlined and labelled, and holds `children` (other groups) or `rows` (what it draws).
#   repeat: a name with {i}, for the repeated blocks; their `rows` names are relative to it.
#   rows: top to bottom; modules in one row run side by side. {m: name, lie: flat} lays a long table flat.
#   unused: true greys a part the model doesn't run.
#   role: what the group does, which sets its colour and where the page marks input and output:
#         embed (the input end), attn_q (a stack of layers), ff_up, norm, classifier (the output end), other.
# flow: [from group, to group]: what feeds what between groups.  scale: weights per unit.  texture: image pixels.
"""


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else None
    ck = open_checkpoint(sys.argv[1])
    spec = draft(ck, name)
    Path(sys.argv[2]).write_text(HEAD + yaml.safe_dump(spec, sort_keys=False, width=120))
    g = spec["groups"]
    print(f"{len(ck.names())} tensors -> {len(g)} groups: " + ", ".join(x["name"] for x in g) + f"; wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
