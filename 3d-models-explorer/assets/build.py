"""Build a 3D explorer of a checkpoint: every tensor at its real size, laid out by a groups file.

    python build.py CHECKPOINT GROUPS.yaml OUT_DIR [VALUES.json]

CHECKPOINT is a folder or a file:
  - a folder with model.safetensors, or a sharded model.safetensors.index.json (Hugging Face layout);
  - a folder with pytorch_model.bin (or its sharded index), read with torch if it is installed;
  - one .safetensors file, or one PyTorch file (.bin, .pt, .pth, .ckpt) holding a state dict.
Anything else (GGUF, ONNX, Keras .h5, Flax .msgpack) stops with what to convert it to.

OUT_DIR gets a static site: index.html, the viewer, scene.json (the layout) and atlas.png (each tensor's real values,
sampled every k-th cell, scaled by 3 standard deviations, packed in one greyscale image). Copy OUT_DIR anywhere and
embed index.html in an iframe. Needs numpy, pyyaml and pillow.
"""
import json, math, re, shutil, struct, sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

STRIP = 0.6      # a bias or a norm's width, in units
GAP = 4.0        # between modules in a row, and between rows
CHILD_GAP = 10   # between groups
LAYER_GAP = 8    # between copies of a repeated block, in depth
PAD = 2.5        # a group's outline round what it holds
THICK = 2.0      # every tensor is a cube: this deep (a vector's or matrix's third dimension is one value deep)
FLAT = THICK     # the thickness given to a grid laid flat
CUT = 80.0       # a table deeper than this (in units) is drawn shortened, with its true size in its label
PADPX = 2        # edge padding round each image in the atlas, so mipmaps don't bleed between tensors
ATLAS_W = 4096
DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "I32": np.int32,
          "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
         "F8_E4M3": 1, "F8_E5M2": 1}


class AtlasFull(Exception):
    pass


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


# ---- plain names and colour roles, for any architecture -------------------------------------------------------
# A repeated block's copy number: BERT/DeBERTa "layer.3.", Llama/Mistral "layers.3.", GPT-2 "h.3.", ViT/T5 "block.3.".
LAYER = re.compile(r"(?:^|\.)(?:layer|layers|h|block|blocks|resblocks)\.(\d+)\.(.+)$")
# The rest of a layer's module name -> (title, role, part). Checked in order; the first pattern found wins.
PARTS = [
    # BERT / DeBERTa (post-norm): their names first, as base GLiNER's example draws them
    (r"attention\.output\.dense$", ("attention: output", "attn_out", "attention")),
    (r"attention\.output\.LayerNorm$", ("attention: norm", "norm", "attention")),
    (r"intermediate\.dense$", ("feed-forward: expand", "ff_up", "feed-forward")),
    (r"(^|\.)output\.dense$", ("feed-forward: shrink", "ff_down", "feed-forward")),
    (r"(^|\.)output\.LayerNorm$|(^|\.)(post_feedforward_layernorm|output_layer_norm)$", ("feed-forward: norm", "norm", "feed-forward")),
    # GPT-2, Llama, Mistral, Qwen, Gemma, Phi, Falcon, GPT-NeoX, T5, ViT, DistilBERT, Meta's own Llama names
    (r"(^|\.)(c_attn|qkv|query_key_value|qkv_proj|Wqkv|in_proj)$", ("attention: query, key and value", "attn_q", "attention")),
    (r"(^|\.)(query_proj|query|q_proj|q|wq|q_lin)$", ("attention: query", "attn_q", "attention")),
    (r"(^|\.)(key_proj|key|k_proj|k|wk|k_lin)$", ("attention: key", "attn_k", "attention")),
    (r"(^|\.)(value_proj|value|v_proj|v|wv|v_lin)$", ("attention: value", "attn_v", "attention")),
    (r"(^|\.)(o_proj|out_proj|o|out_lin)$|attn\.c_proj$|attn\.proj$|attention\.wo$|attention\.dense$|attn\.dense$",
     ("attention: output", "attn_out", "attention")),
    (r"(^|\.)(post_attention_layernorm|ln_2|ffn_norm|norm2|sa_layer_norm|layernorm_after)$",
     ("feed-forward: norm before it", "norm", "feed-forward")),
    (r"(^|\.)(input_layernorm|ln_1|attention_norm|norm1|layernorm_before|layer_norm|LayerNorm|norm)$",
     ("attention: norm", "norm", "attention")),
    (r"(^|\.)(gate_proj|w1|wi_0)$", ("feed-forward: gate", "ff_up", "feed-forward")),
    (r"(^|\.)(up_proj|c_fc|fc1|w3|wi|wi_1|lin1|dense_h_to_4h|fc_in)$", ("feed-forward: expand", "ff_up", "feed-forward")),
    (r"mlp\.c_proj$|(^|\.)(down_proj|fc2|w2|wo|lin2|dense_4h_to_h|fc_out)$", ("feed-forward: shrink", "ff_down", "feed-forward")),
]
PARTS = [(re.compile(p), v) for p, v in PARTS]
WHOLE = [  # a name outside the repeated blocks -> role
    (r"norm|ln_f|LayerNorm|layernorm", "norm"),   # first: "embeddings.LayerNorm" is a norm, not a table
    (r"embed|wte|wpe|word_embeddings|position|tok_embeddings|patch_embed|cls_token|pos_embed", "embed"),
    (r"lm_head|classifier|score|(^|\.)head|pooler|qa_outputs|proj_out|output_layer", "classifier"),
]
WHOLE = [(re.compile(p), r) for p, r in WHOLE]

def describe(name):
    """A plain title, a colour role and the part of a layer for a module's name (the code name stays in `name`)."""
    m = LAYER.search(name)
    if m:
        for rx, (t, role, part) in PARTS:
            if rx.search(m.group(2)):
                return dict(title=f"Layer {m.group(1)} · {t}", role=role, part=part, copy=int(m.group(1)))
        return dict(title=f"Layer {m.group(1)} · {m.group(2)}", role="other", part="other", copy=int(m.group(1)))
    role = next((r for rx, r in WHOLE if rx.search(name)), "other")
    return dict(title=name, role=role, part="other", copy=None)


# ---- reading checkpoints ------------------------------------------------------------------------------------
class Checkpoint:
    """One or more safetensors files, read in place (memory-mapped): nothing is loaded until a tensor is sampled."""
    def __init__(self, paths):
        self.files, self.where = [], {}
        for p in paths:
            h, base = read_header(p)
            self.files.append((np.memmap(p, np.uint8, mode="r"), base))
            for n, t in h.items():
                self.where[n] = (len(self.files) - 1, t)

    def names(self):
        return list(self.where)

    def shape(self, name):
        return list(self.where[name][1]["shape"])

    def array(self, name):
        fi, t = self.where[name]
        mm, base = self.files[fi]
        a, b = t["data_offsets"]
        raw = mm[base + a:base + b]
        dt = t["dtype"]
        if dt not in DTYPES:   # fp8 and other packed types: drawn at their size, with flat (zero) values
            print(f"  {name}: dtype {dt} has no reader, drawn grey", file=sys.stderr)
            return np.zeros(t["shape"] or [1], np.float32)
        arr = raw.view(DTYPES[dt])
        if dt == "BF16":  # bfloat16: the top half of a float32
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.reshape(t["shape"]) if t["shape"] else arr.reshape(1)


class TorchCheckpoint:
    """A PyTorch state dict (.bin, .pt, .pth, .ckpt), loaded once with torch on the CPU."""
    def __init__(self, paths):
        try:
            import torch
        except ImportError:
            sys.exit("this checkpoint is a PyTorch file: install torch (pip install torch), or convert it to "
                     "safetensors first (references/checkpoints.md)")
        self.t = {}
        for p in paths:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            for key in ("state_dict", "model", "module"):   # training checkpoints wrap the weights
                if isinstance(sd, dict) and isinstance(sd.get(key), dict):
                    sd = sd[key]
            if not isinstance(sd, dict):
                sys.exit(f"{p} holds a {type(sd).__name__}, not a state dict of tensors")
            def walk(d, prefix):   # nested dicts (one state dict per sub-model) become dotted names
                for n, v in d.items():
                    if torch.is_tensor(v):
                        self.t[prefix + str(n)] = v
                    elif isinstance(v, dict):
                        walk(v, prefix + str(n) + ".")
            walk(sd, "")
        if not self.t:
            sys.exit("no tensors found in " + ", ".join(map(str, paths)))

    def names(self):
        return list(self.t)

    def shape(self, name):
        return list(self.t[name].shape)

    def array(self, name):
        v = self.t[name].detach()
        v = v.float() if v.is_floating_point() else v
        a = v.numpy()
        return a if a.ndim else a.reshape(1)


TORCH = (".bin", ".pt", ".pth", ".ckpt")
NOPE = {".gguf": "GGUF (llama.cpp)", ".onnx": "ONNX", ".h5": "Keras/TF HDF5", ".keras": "Keras", ".msgpack": "Flax",
        ".tflite": "TF Lite", ".pb": "TensorFlow", ".npz": "NumPy archive"}


def open_checkpoint(path):
    """A folder (Hugging Face layout) or one file -> a reader with names(), shape(name) and array(name)."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"{path} does not exist")
    if path.is_dir():
        for idx, single, cls in (("model.safetensors.index.json", "model.safetensors", Checkpoint),
                                 ("pytorch_model.bin.index.json", "pytorch_model.bin", TorchCheckpoint)):
            if (path / single).exists():
                return cls([path / single])
            if (path / idx).exists():
                shards = sorted(set(json.loads((path / idx).read_text())["weight_map"].values()))
                return cls([path / s for s in shards])
        st = sorted(path.glob("*.safetensors"))
        if st:
            return Checkpoint(st)
        pt = sorted(p for p in path.iterdir() if p.suffix in TORCH)
        if pt:
            return TorchCheckpoint(pt)
        other = sorted({NOPE[p.suffix] for p in path.rglob("*") if p.suffix in NOPE})
        sys.exit(f"no safetensors or PyTorch weights in {path}" + (f" (found {', '.join(other)}: convert it, "
                 "references/checkpoints.md)" if other else ""))
    if path.suffix == ".safetensors":
        return Checkpoint([path])
    if path.suffix in TORCH:
        return TorchCheckpoint([path])
    kind = NOPE.get(path.suffix, f"a {path.suffix or 'suffix-less'} file")
    sys.exit(f"can't read {kind}: give a .safetensors or PyTorch checkpoint (references/checkpoints.md says how to convert)")


class Builder:
    def __init__(self, ck, spec, out):
        self.ck, self.spec, self.out = ck, spec, out
        self.s = float(spec.get("scale", 64))
        self.tex_default = int(spec.get("texture", 128))
        self.used = {}           # tensor name -> group index
        self.items, self.groups = [], []
        self.mods, self.edges, self.gm = [], [], {}   # modules (a weight and its bias...), data-flow links, per-group row modules
        self.last_mid, self.row_mods = None, []
        self.texs = []   # each tensor's sampled values as a small greyscale image, packed into one atlas at the end

    # ---- tensors to boxes -------------------------------------------------------------------------------
    def module_tensors(self, name):
        names = [n for n in self.ck.names() if n == name or n.startswith(name + ".")]
        if not names:
            sys.exit(f"groups file names {name!r}, which matches no tensor")
        return names

    def texture(self, name, rows, cols, long_side, flip=False):
        """Sample every k-th cell into a greyscale image: 128 is zero, 0 and 255 are -/+3 std."""
        a = self.ck.array(name)
        a = a.reshape(rows, cols)
        k = max(1, math.ceil(max(rows, cols) / long_side))
        s = np.asarray(a[::k, ::k], np.float32)
        sd = float(s.std()) or 1.0
        img = np.clip(s / (3 * sd), -1, 1)
        img = ((img + 1) * 127.5).round().astype(np.uint8)
        if flip:
            img = img[::-1]
        self.texs.append(img)
        return len(self.texs) - 1, float(s.mean()), sd

    def module(self, entry, prefix, gi):
        """One module (a weight and its bias, a norm's pair): items placed from its top-left-front corner."""
        if isinstance(entry, str):
            entry = {"m": entry}
        name = prefix + entry["m"]
        flat = entry.get("lie") == "flat"
        T = int(entry.get("texture", self.tex_default))
        names = self.module_tensors(name)
        for n in names:
            if n in self.used:
                sys.exit(f"{n} is in two groups")
            self.used[n] = gi
        grids = [n for n in names if len(self.ck.shape(n)) >= 2]
        lines = [n for n in names if len(self.ck.shape(n)) < 2]
        placed, x, h, d = [], 0.0, 0.0, 0.0
        m0 = len(self.items)
        for n in grids:
            shp = self.ck.shape(n)
            r, c = shp[0], int(np.prod(shp[1:]))
            w, ht = max(c / self.s, STRIP), max(r / self.s, STRIP)  # a 1-row grid still shows (its true shape is in its label)
            fn, mean, sd = self.texture(n, r, c, T, flip=flat)
            it = dict(name=n, shape=shp, params=r * c, kind="grid", flat=flat, group=gi, tex=fn, std=sd)
            if flat:
                if ht > CUT:   # the 128k-row vocabulary table is 2,000 units deep: draw it shortened
                    it["cut"] = dict(rows=r, drawn=CUT * self.s); ht = CUT
                it.update(pos=[x + w / 2, -FLAT / 2, -ht / 2], size=[w, ht])
                d, h = max(d, ht), max(h, FLAT)
            else:
                it.update(pos=[x + w / 2, -ht / 2, 0], size=[w, ht])
                h = max(h, ht)
            self.items.append(it); placed.append(it)
            x += w
            # its bias beside it: a strip as tall as its rows
            b = n[:-len("weight")] + "bias" if n.endswith("weight") else None
            if b in lines and self.ck.shape(b)[0] == r and not flat:
                fnb, _, sdb = self.texture_line(b, T, vertical=True)
                self.items.append(dict(name=b, shape=self.ck.shape(b), params=r, kind="strip", flat=False, group=gi,
                                       tex=fnb, std=sdb, pos=[x + 0.4 + STRIP / 2, -ht / 2, 0], size=[STRIP, ht]))
                lines.remove(b)
                x += 0.4 + STRIP
            x += GAP
        x = max(0.0, x - GAP) if grids else 0.0
        y = -h - (GAP / 2 if grids and lines else 0)
        for n in lines:  # norms, and biases with no grid beside them: strips across
            L = int(np.prod(self.ck.shape(n)) or 1)
            w = L / self.s
            fn, _, sd = self.texture_line(n, T, vertical=False)
            self.items.append(dict(name=n, shape=self.ck.shape(n), params=L, kind="strip", flat=False, group=gi, tex=fn,
                                   std=sd, pos=[w / 2, y - STRIP / 2, 0], size=[w, STRIP]))
            x = max(x, w)
            y -= STRIP + 0.3
        d = max(d, THICK)
        self.last_mid = len(self.mods)
        d0 = describe(name)
        gname = self.groups[gi].get("name")
        if d0["title"] == name and gname:   # no plain name for it: its group's name and the code name's last parts
            d0["title"] = f"{gname}: {'.'.join(name.split('.')[-2:])}"
        for k in ("title", "role"):   # a groups file may name a module's title or colour role itself
            if k in entry:
                d0[k] = entry[k]
        self.mods.append(dict(name=name, items=[m0, len(self.items)], group=gi, **d0))
        for it in self.items[m0:]:
            it["mod"] = self.last_mid
        return x, -y if lines else h, d

    def texture_line(self, name, T, vertical):
        a = np.asarray(self.ck.array(name), np.float32).reshape(-1)
        k = max(1, math.ceil(len(a) / T))
        s = a[::k]
        sd = float(s.std()) or 1.0
        v = ((np.clip(s / (3 * sd), -1, 1) + 1) * 127.5).round().astype(np.uint8)
        img = v.reshape(-1, 1) if vertical else v.reshape(1, -1)
        self.texs.append(img)
        return len(self.texs) - 1, float(s.mean()), sd

    # ---- groups to boxes ---------------------------------------------------------------------------------
    def shift(self, first, dx, dy, dz, last=None):
        for it in self.items[first:last]:
            it["pos"] = [it["pos"][0] + dx, it["pos"][1] + dy, it["pos"][2] + dz]

    def rows(self, rows, prefix, gi):
        """Rows top to bottom, each centred on the widest."""
        spans, y, W, D = [], 0.0, 0.0, 0.0
        rm = []
        for row in rows:
            start, x, h, d = len(self.items), 0.0, 0.0, 0.0
            rm.append([])
            for entry in row:
                m0 = len(self.items)
                w, hh, dd = self.module(entry, prefix, gi)
                rm[-1].append(self.last_mid)
                self.shift(m0, x, y, 0)
                x += w + GAP; h = max(h, hh); d = max(d, dd)
            x -= GAP
            spans.append((start, len(self.items), x))
            W, D = max(W, x), max(D, d)
            y -= h + GAP
        for s, e, w in spans:
            for it in self.items[s:e]:
                it["pos"][0] += (W - w) / 2
        self.row_mods = rm
        return W, -(y + GAP), D

    def group(self, node, path, parent_unused=False):
        """Lay out one group from its top-left-front corner; returns its size (w, h, d)."""
        name = node.get("name")
        unused = bool(node.get("unused", parent_unused))
        gi = len(self.groups)
        g = dict(name=name, path=path + [name] if name else path, unused=unused, outline=bool(name), parent=None)
        if node.get("role"):
            g["role"] = node["role"]
        self.groups.append(g)
        first = len(self.items)
        if "rows" in node:
            if "repeat" in node:
                pat = node["repeat"]
                rx = re.compile("^" + re.escape(pat).replace(r"\{i\}", r"(\d+)") + r"\.")
                copies = sorted({int(m.group(1)) for n in self.ck.names() if (m := rx.match(n))})
                if not copies:
                    sys.exit(f"repeat {pat!r} matches nothing")
                W = H = dl = 0.0
                copy_rows = []
                for k, i in enumerate(copies):
                    c0 = len(self.items)
                    w, h, dl = self.rows(node["rows"], pat.replace("{i}", str(i)) + ".", gi)
                    copy_rows.append(self.row_mods)
                    self.shift(c0, 0, 0, -k * LAYER_GAP)
                    for it in self.items[c0:]:
                        it["copy"] = i
                    W, H = max(W, w), max(H, h)
                D = (len(copies) - 1) * LAYER_GAP + dl
                self.gm[gi] = dict(first=copy_rows[0][0], last=copy_rows[-1][-1], copies=copy_rows)
                self.chain(copy_rows, node.get("skip"), unused)
                g["copies"] = len(copies)
            else:
                W, H, D = self.rows(node["rows"], "", gi)
                self.gm[gi] = dict(first=self.row_mods[0], last=self.row_mods[-1], copies=[self.row_mods])
                if node.get("chain", True):
                    self.chain([self.row_mods], None, unused)
        else:
            across = node.get("arrange", "down") == "across"
            sizes, starts = [], []
            for ch in node.get("children", []):
                c0 = len(self.items); cg = len(self.groups)
                sizes.append(self.group(ch, g["path"], unused))
                starts.append((c0, len(self.items), cg, len(self.groups)))
                self.groups[cg]["parent"] = gi
            W = H = D = 0.0
            if across:
                x = 0.0
                for (c0, c1, cg, cg1), (w, h, d) in zip(starts, sizes):
                    self.shift(c0, x, 0, 0, c1); self.move_groups(cg, x, 0, 0, cg1)
                    x += w + CHILD_GAP; H = max(H, h); D = max(D, d)
                W = x - CHILD_GAP
            else:
                y = 0.0
                W = max(w for w, _, _ in sizes)
                for (c0, c1, cg, cg1), (w, h, d) in zip(starts, sizes):
                    dx = (W - w) / 2
                    self.shift(c0, dx, y, 0, c1); self.move_groups(cg, dx, y, 0, cg1)
                    y -= h + CHILD_GAP; D = max(D, d)
                H = -(y + CHILD_GAP)
        p = PAD if name else 0
        if p:  # make room for the outline: everything in it moves in by PAD
            self.shift(first, p, -p, -p / 2)
            self.move_groups(gi + 1, p, -p, -p / 2)
            W, H, D = W + 2 * p, H + 2 * p, D + p
        g["box"] = [0.0, 0.0, 0.0, W, H, D]  # x, y, z of its top-left-front corner, then w, h, d
        g["tensors"] = sum(1 for it in self.items[first:])
        g["params"] = int(sum(it["params"] for it in self.items[first:]))
        return W, H, D

    def link(self, A, B, kind, unused=False):
        """A data-flow link from each module of A to the module(s) of B (pairwise when the counts match)."""
        pairs = list(zip(A, B)) if len(A) == len(B) and len(A) > 1 else [(a, b) for a in A for b in B]
        for a, b in pairs:
            self.edges.append(dict(a=a, b=b, k=kind, u=bool(unused)))

    def chain(self, copies, skip, unused):
        """Rows feed the row below; a repeated block's last row feeds the next copy's first; `skip` adds the residual links."""
        for k, rl in enumerate(copies):
            for r in range(len(rl) - 1):
                self.link(rl[r], rl[r + 1], "flow", unused)
            if k:
                self.link(copies[k - 1][-1], rl[0], "flow", unused)
                if skip and "in" in skip:
                    self.link(copies[k - 1][-1], rl[skip["in"]], "skip", unused)
            for a, b in (skip or {}).get("pairs", []):
                self.link(rl[a], rl[b], "skip", unused)

    def pack(self):
        """Shelf-pack every tensor's image into one atlas.png; each item gets `uv` = [u0, t0, u1, t1] (t counts from the top)."""
        order = sorted(range(len(self.texs)), key=lambda i: -self.texs[i].shape[0])
        x = y = shelf = 0
        place = {}
        for i in order:
            h, w = self.texs[i].shape
            bw, bh = w + 2 * PADPX, h + 2 * PADPX
            if x + bw > ATLAS_W:
                x, y, shelf = 0, y + shelf, 0
            place[i] = (x + PADPX, y + PADPX)
            x += bw; shelf = max(shelf, bh)
        W, H = ATLAS_W, y + shelf
        if H > 4096:
            raise AtlasFull(f"the atlas needs {W}x{H}")
        atlas = np.full((H, W), 128, np.uint8)
        for i, (px, py) in place.items():
            img = self.texs[i]; h, w = img.shape
            atlas[py - PADPX:py + h + PADPX, px - PADPX:px + w + PADPX] = np.pad(img, PADPX, mode="edge")
        Image.fromarray(atlas, "L").save(self.out / "atlas.png", optimize=True)
        # a half-size copy (2x2 averaged) for phones and low memory: the same uv rectangles work on it, so scene.json does not change
        h2, w2 = H // 2, W // 2
        half = atlas[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).astype(np.float32).mean(axis=(1, 3)).round().astype(np.uint8)
        Image.fromarray(half, "L").save(self.out / "atlas-half.png", optimize=True)
        self.atlas = dict(file="atlas.png", w=W, h=H)
        for it in self.items:
            px, py = place[it["tex"]]; h, w = self.texs[it["tex"]].shape
            it["uv"] = [round(px / W, 6), round(py / H, 6), round((px + w) / W, 6), round((py + h) / H, 6)]
            del it["tex"]

    def colour_groups(self):
        """Each outlined group's colour role: its own `role`, else the role holding most of its weights (norms aside)."""
        kids = {}
        for gi, g in enumerate(self.groups):
            p = g.get("parent")
            while p is not None:
                kids.setdefault(p, set()).add(gi); p = self.groups[p].get("parent")
        for gi, g in enumerate(self.groups):
            if not g["outline"] or g.get("role"):
                continue
            mine, weight = {gi} | kids.get(gi, set()), {}
            for m in self.mods:
                if m["group"] in mine and m["role"] not in ("norm", "other"):
                    a, b = m["items"]
                    weight[m["role"]] = weight.get(m["role"], 0) + sum(it["params"] for it in self.items[a:b])
            if any(r.startswith("attn") or r.startswith("ff") for r in weight):   # a stack of layers reads as attention
                g["role"] = "attn_q"
            else:
                g["role"] = max(weight, key=weight.get) if weight else "other"

    def move_groups(self, from_gi, dx, dy, dz, last=None):
        for g in self.groups[from_gi:last]:
            if "box" in g:
                b = g["box"]; g["box"] = [b[0] + dx, b[1] + dy, b[2] + dz] + b[3:]

    def build(self):
        root = {"children": self.spec["groups"]}
        W, H, D = self.group(root, [])
        byname = {g["name"]: i for i, g in enumerate(self.groups) if g.get("name")}
        skips = {}
        def find(node):
            for ch in node.get("children", []):
                if ch.get("name"): skips[ch["name"]] = ch.get("skip")
                find(ch)
        find(root)
        for a, b, *rest_ in self.spec.get("flow", []):
            kind = rest_[0] if rest_ else "flow"
            only = rest_[1] if len(rest_) > 1 else None   # which modules of the target's first row it feeds
            ga, gb = self.gm[byname[a]], self.gm[byname[b]]
            un = self.groups[byname[b]]["unused"]
            if kind == "side":
                for rl in gb["copies"]:
                    self.link(ga["last"], [rl[0][i] for i in only] if only else rl[0], "side", un)
            else:
                self.link(ga["last"], gb["first"], "flow", un)
                sk = skips.get(b)
                if sk and "in" in sk:
                    self.link(ga["last"], gb["copies"][0][sk["in"]], "skip", un)
        mid = {m["name"]: i for i, m in enumerate(self.mods)}
        for a, b, *kind in self.spec.get("links", []):   # explicit links between named modules; "@Group" is a group's output
            ends = []
            for x in (a, b):
                if x.startswith("@"):
                    ends.append(self.gm[byname[x[1:]]]["last"])
                else:
                    if x not in mid:
                        sys.exit(f"links names {x!r}, which is no module")
                    ends.append([mid[x]])
            self.link(ends[0], ends[1], kind[0] if kind else "flow", self.mods[ends[1][0]]["group"] is not None and self.groups[self.mods[ends[1][0]]["group"]]["unused"])
        rest = [n for n in self.ck.names() if n not in self.used]
        if rest:  # nothing hidden by mistake: what no group names goes in Other, below the rest
            print(f"{len(rest)} tensors in no group, drawn under Other:", *rest[:20], sep="\n  ")
            gi = len(self.groups); first = len(self.items)
            self.groups.append(dict(name="Other", path=["Other"], unused=False, outline=True, parent=0))
            w, h, d = self.rows([[n] for n in rest], "", gi)
            self.shift(first, PAD, -H - CHILD_GAP - PAD, -PAD / 2)
            self.groups[gi].update(box=[0, -H - CHILD_GAP, 0, w + 2 * PAD, h + 2 * PAD, d + PAD],
                                   tensors=len(rest), params=int(sum(it["params"] for it in self.items[first:])))
            H += CHILD_GAP + h + 2 * PAD; W = max(W, w + 2 * PAD)
        self.colour_groups()
        self.pack()
        total = sum(int(np.prod(self.ck.shape(n))) for n in self.ck.names())
        drawn = sum(it["params"] for it in self.items)
        assert drawn == total, (drawn, total)
        return dict(model=self.spec.get("model", ""), scale=self.s, total=total, tensors=len(self.ck.names()),
                    size=[W, H, D], thick=THICK, atlas=self.atlas, groups=self.groups, items=self.items,
                    modules=self.mods, edges=self.edges)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    ck_path, groups, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = yaml.safe_load(groups.read_text())
    ck = open_checkpoint(ck_path)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "tex", ignore_errors=True)   # an older build's per-tensor images
    for attempt in range(6):   # a big model's images don't fit one 4096-wide atlas: halve the texture sizes until they do
        try:
            scene = Builder(ck, spec, out).build()
            break
        except AtlasFull as e:
            shrink(spec)
            print(f"{e}: halving every texture size and building again", file=sys.stderr)
    else:
        sys.exit("the atlas still does not fit: lower `texture` in the groups file")
    (out / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    if len(sys.argv) > 4:   # a values.json from capture_values.py: one real input's outputs, shown beside each stage
        shutil.copy2(sys.argv[4], out / "values.json")
    web = Path(__file__).resolve().parent / "web"
    for f in web.rglob("*"):
        if f.is_file() and not f.name.startswith("module-"):   # the module's test pages stay out of a build
            dst = out / f.relative_to(web); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    named = [g for g in scene["groups"] if g["outline"]]
    print(f"{scene['tensors']} tensors, {scene['total']:,} weights, {len(scene['items'])} drawn -> {out}")
    for g in named:
        print(f"  {' / '.join(g['path']):40s} {g['tensors']:4d} tensors {g['params']:>13,}{'  (unused)' if g['unused'] else ''}")


def shrink(spec):
    """Halve `texture` everywhere in a groups file (its default and any module's own), at least 8 pixels."""
    spec["texture"] = max(8, int(spec.get("texture", 128)) // 2)
    def walk(node):
        if isinstance(node, dict):
            if "texture" in node and "m" in node:
                node["texture"] = max(8, int(node["texture"]) // 2)
            for v in node.values(): walk(v)
        elif isinstance(node, list):
            for v in node: walk(v)
    walk(spec.get("groups", []))


if __name__ == "__main__":
    main()
