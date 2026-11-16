"""Draft a groups file for any checkpoint: what comes before the repeated blocks, the blocks, and what comes after.

    python init_groups.py CHECKPOINT OUT.yaml [--name "Model name"]

It finds the repeated block (the `layers.{i}` / `h.{i}` / `layer.{i}` prefix with the most copies), orders one
copy's modules in the usual data flow (norm, query/key/value side by side, attention output, norm, feed-forward
expand, shrink), and puts everything else in "Embeddings" (before) or "Output" (after). It is a draft: read it,
rename the groups in the reader's words, mark unused parts, and fix any row it guessed wrong (references/groups.md).
Every tensor a draft leaves out is still drawn, under "Other", so nothing is hidden by mistake.
"""
import json, re, sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import describe, is_fixed, open_checkpoint   # noqa: E402

BLOCK = re.compile(r"^(.*?(?:^|\.)(?:layer|layers|h|block|blocks|resblocks))\.(\d+)\.")
# Where a module of one layer sits in the flow; modules with the same rank share a row (they run side by side).
RANK = {"attention: norm": 0, "attention: query, key and value": 1, "attention: query": 1, "attention: key": 1,
        "attention: value": 1, "attention: causal mask (fixed, not learned)": 1.5, "attention: output": 2, "feed-forward: norm before it": 3, "feed-forward: gate": 4,
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


def tied_head(path):
    """True when the model's config says its output head reuses the word table (GPT-2, most causal LMs): the checkpoint
    then has no head tensor of its own."""
    p = Path(path)
    cfg = (p if p.is_dir() else p.parent) / "config.json"
    try:
        c = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return False
    lm = any(re.search(r"LMHead|ForCausalLM|ForMaskedLM|ForConditionalGeneration", a) for a in c.get("architectures") or [])
    return lm and c.get("tie_word_embeddings", True) is not False


def draft(ck, model_name=None, path=None):
    names = ck.names()
    # a tensor's module; a fixed buffer (GPT-2's attn.bias mask) stays whole, so it isn't read as some module's bias
    mod = lambda n: n if is_fixed(n, ck.shape(n), ck.dtype(n)) else module_of(n)
    copies = {}
    for n in names:
        m = BLOCK.match(n)
        if m:
            copies.setdefault(m.group(1), set()).add(int(m.group(2)))
    copies = {p: sorted(c) for p, c in copies.items() if len(c) > 1}
    groups, flow, links = [], [], []
    if copies:
        prefix = max(copies, key=lambda p: (len(copies[p]), -len(p)))
        n_layers, first = len(copies[prefix]), copies[prefix][0]
        pat = f"{prefix}.{{i}}"
        inside = re.compile("^" + re.escape(prefix) + r"\.\d+\.")
        one = f"{prefix}.{first}."
        layer_mods = dedupe(sorted({mod(n)[len(one):] for n in names if n.startswith(one)}))
        by_rank = {}
        for m in layer_mods:
            t = describe(f"{one}{m}")["title"].split(" · ", 1)[-1]
            r = 3 if m.endswith("attention.output.LayerNorm") else RANK.get(t, 7)   # BERT's norm comes after attention
            by_rank.setdefault(r, []).append(m)
        rows = [sorted(by_rank[r]) for r in sorted(by_rank)]
        # the residual adds: where the layer's input is added back (after attention), then where that sum is added back
        # (after the feed-forward). Post-norm models (BERT) add just before the norm that follows each half.
        def row_of(test):
            return next((i for i, row in enumerate(rows) for m in row if test(m, describe(f"{one}{m}")["title"].split(" · ", 1)[-1])), None)
        add_attn = row_of(lambda m, t: m.endswith("attention.output.LayerNorm"))
        add_attn = row_of(lambda m, t: t == "attention: output") if add_attn is None else add_attn
        add_ff = row_of(lambda m, t: t == "feed-forward: norm")
        add_ff = row_of(lambda m, t: t == "feed-forward: shrink") if add_ff is None else add_ff
        outside = [n for n in names if not inside.match(n)]
        # before or after the blocks: the embedding tables (and a norm beside them) come before; the rest after
        role = {n: describe(mod(n))["role"] for n in outside}
        parent = lambda n: mod(n).rpartition(".")[0]
        # a norm sits with the tables only inside their own module (BERT's embeddings.LayerNorm); one beside the blocks
        # (GPT-2's transformer.ln_f, Llama's model.norm) is the final norm, after them
        embed_parents = {parent(n) for n in outside if role[n] == "embed"} - {"", prefix.rpartition(".")[0]}
        before = [n for n in outside if role[n] == "embed" or (role[n] == "norm" and parent(n) in embed_parents)]
        after = [n for n in outside if n not in before]
        if before:
            emb = sorted(dedupe(sorted({mod(n) for n in before})), key=lambda m: (describe(m)["role"] != "embed", m))   # tables, then norms
            groups.append({"name": "Embeddings", "role": "embed", "rows": [[entry(ck, m, names)] for m in emb]})
        block = {"name": f"{n_layers} layers", "role": "attn_q", "repeat": pat, "rows": rows}
        if add_attn is not None and add_ff is not None and add_attn < add_ff:
            block["skip"] = {"in": add_attn, "pairs": [[add_attn, add_ff]]}
        groups.append(block)
        out_name = "Output"
        if after:
            head = any(describe(mod(n))["role"] == "classifier" for n in after)
            out_mods = sorted(dedupe(sorted({mod(n) for n in after})), key=lambda m: (describe(m)["role"] != "norm", m))   # norms, then heads
            rows_of = lambda m: max([ck.shape(n)[0] for n in names if (n == m or n.startswith(m + ".")) and len(ck.shape(n)) == 2] or [0])
            tables = sorted([m for m in (emb if before else []) if describe(m)["role"] == "embed" and rows_of(m)], key=rows_of, reverse=True)   # the word table: the most rows
            tied = not head and tables and tied_head(path) if path else False
            if tied:   # no head tensor: the last hidden state is scored against the word table itself
                out_name = "Output (head is the word table)"
                links.append([out_mods[-1], tables[0]])
            groups.append({"name": out_name, "role": "classifier" if head or tied else "other",
                           "rows": [[entry(ck, m, names)] for m in out_mods]})
        if before:
            flow.append(["Embeddings", block["name"]])
        if after:
            flow.append([block["name"], out_name])
    else:   # no repeated block (a small or unusual model): one group, each module a row, in the file's order
        mods = dedupe(list(dict.fromkeys(module_of(n) for n in names)))
        groups.append({"name": "Model", "role": "other", "rows": [[entry(ck, m, names)] for m in mods]})
    spec = {"model": model_name or "Model", "scale": 64, "texture": 128}
    if flow:
        spec["flow"] = flow
    if links:
        spec["links"] = links
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
    spec = draft(ck, name, sys.argv[1])
    Path(sys.argv[2]).write_text(HEAD + yaml.safe_dump(spec, sort_keys=False, width=120))
    g = spec["groups"]
    print(f"{len(ck.names())} tensors -> {len(g)} groups: " + ", ".join(x["name"] for x in g) + f"; wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
