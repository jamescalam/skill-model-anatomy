"""Optional: run one real input through the model and record what each stage outputs, shown beside it in the explorer.

    python capture_values.py MODEL OUT/values.json "the input text" [--scene OUT/scene.json] [--note "..."]

MODEL is a Hugging Face id or folder that transformers' AutoModel can load (needs torch and transformers). Each
module that owns weights gets a hook; its output's shape, mean, std, min, max and first 8 values are kept. fp32 and
eval() (dropout off), so another machine can re-run it and compare. With --scene, names are matched to the scene's
modules when the loaded model drops or adds a prefix (AutoModel loads Llama's "model.layers.0" as "layers.0").
For a model AutoModel can't run (a custom architecture), copy this file and replace load() and run() with its own
loading and forward call: the hooks and the output format stay the same. Pass the result to build.py as VALUES.json.
"""
import json, sys

import torch


def load(name):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval()
    return model, tok


def run(model, tok, text, dev):
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        model(**ids)
    return ids["input_ids"][0].tolist(), tok.convert_ids_to_tokens(ids["input_ids"][0])


def main():
    args = [a for a in sys.argv[1:]]
    opt = lambda k: args[args.index(k) + 1] if k in args else None
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if len(pos) < 3:
        sys.exit(__doc__)
    name, out, text = pos[:3]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(name)
    model.to(dev)
    stages = {}

    def hook(n):
        def f(mod, a, o):
            o = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(o) and n not in stages:
                t = o.detach().float().reshape(-1)
                stages[n] = dict(shape=list(o.shape), mean=float(t.mean()), std=float(t.std()) if t.numel() > 1 else 0.0,
                                 min=float(t.min()), max=float(t.max()), first=[float(x) for x in t[:8]])
        return f
    for n, mod in model.named_modules():
        if n and any(True for _ in mod.parameters(recurse=False)):
            mod.register_forward_hook(hook(n))
    ids, toks = run(model, tok, text, dev)
    if opt("--scene"):   # match the loaded model's module names to the checkpoint's
        names = [m["name"] for m in json.load(open(opt("--scene")))["modules"]]
        have = set(names)
        fixed = {}
        for n, st in stages.items():
            hit = n if n in have else next((s for s in names if s.endswith("." + n)), None) \
                or next((s for s in names if n.endswith("." + s)), None)
            if hit:
                fixed[hit] = st
        print(f"{len(fixed)} of {len(stages)} stages matched the scene's {len(names)} modules")
        stages = fixed
    json.dump(dict(model=name, dtype="float32", mode="eval", device=dev, torch=torch.__version__, text=text,
                   input_ids=ids, tokens=toks, note=opt("--note"), stages=stages),
              open(out, "w"), separators=(",", ":"))
    print(len(stages), "stages;", len(ids), "tokens ->", out)


if __name__ == "__main__":
    main()
