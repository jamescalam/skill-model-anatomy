"""The skill's scripts on small made-up checkpoints (no downloads, no weights in the repo).

    pip install numpy pyyaml pillow pytest && pytest tests
"""
import json, struct, subprocess, sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ASSETS = Path(__file__).resolve().parent.parent / "3d-models-explorer" / "assets"
sys.path.insert(0, str(ASSETS))
import build  # noqa: E402
import init_groups  # noqa: E402

RNG = np.random.default_rng(0)


def write_safetensors(path, tensors, bf16=False):
    """A minimal safetensors writer, so the tests need no torch or safetensors package."""
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        if bf16 and a.dtype == np.float32:
            raw, dt = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes(), "BF16"
        else:
            raw, dt = a.tobytes(), {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("int64"): "I64"}[a.dtype]
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw); off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def w(*shape):
    return RNG.standard_normal(shape).astype(np.float32) * 0.02


def gpt2(layers=3, d=32, vocab=500):
    t = {"wte.weight": w(vocab, d), "wpe.weight": w(64, d)}
    for i in range(layers):
        p = f"h.{i}."
        t.update({p + "ln_1.weight": w(d), p + "ln_1.bias": w(d), p + "attn.c_attn.weight": w(d, 3 * d), p + "attn.c_attn.bias": w(3 * d),
                  p + "attn.c_proj.weight": w(d, d), p + "attn.c_proj.bias": w(d), p + "ln_2.weight": w(d), p + "ln_2.bias": w(d),
                  p + "mlp.c_fc.weight": w(d, 4 * d), p + "mlp.c_fc.bias": w(4 * d), p + "mlp.c_proj.weight": w(4 * d, d),
                  p + "mlp.c_proj.bias": w(d)})
    t.update({"ln_f.weight": w(d), "ln_f.bias": w(d)})
    return t


def llama(layers=2, d=32, vocab=5000):
    t = {"model.embed_tokens.weight": w(vocab, d)}
    for i in range(layers):
        p = f"model.layers.{i}."
        t.update({p + "input_layernorm.weight": w(d), p + "self_attn.q_proj.weight": w(d, d), p + "self_attn.k_proj.weight": w(d // 4, d),
                  p + "self_attn.v_proj.weight": w(d // 4, d), p + "self_attn.o_proj.weight": w(d, d),
                  p + "post_attention_layernorm.weight": w(d), p + "mlp.gate_proj.weight": w(3 * d, d),
                  p + "mlp.up_proj.weight": w(3 * d, d), p + "mlp.down_proj.weight": w(d, 3 * d)})
    t.update({"model.norm.weight": w(d), "lm_head.weight": w(vocab, d)})
    return t


def run(script, *args):
    r = subprocess.run([sys.executable, str(ASSETS / script), *map(str, args)], capture_output=True, text=True)
    return r


def build_and_check(ck, tmp, expect_groups, n_params):
    g = tmp / "groups.yaml"
    r = run("init_groups.py", ck, g, "--name", "Test model")
    assert r.returncode == 0, r.stderr
    spec = yaml.safe_load(g.read_text())
    assert [x["name"] for x in spec["groups"]] == expect_groups
    out = tmp / "site"
    r = run("build.py", ck, g, out)
    assert r.returncode == 0, r.stderr
    assert "in no group" not in r.stdout   # the draft placed every tensor
    scene = json.loads((out / "scene.json").read_text())
    assert scene["total"] == n_params
    assert sum(it["params"] for it in scene["items"]) == n_params
    for f in ("index.html", "explorer.js", "atlas.png", "atlas-half.png", "vendor/three.module.min.js"):
        assert (out / f).exists(), f
    assert not list(out.glob("module-*.html"))
    return spec, scene


def test_gpt2_style(tmp_path):
    t = gpt2(); ck = tmp_path / "gpt2"; ck.mkdir()
    write_safetensors(ck / "model.safetensors", t)
    spec, scene = build_and_check(ck, tmp_path, ["Embeddings", "3 layers", "Output"], sum(a.size for a in t.values()))
    layer = spec["groups"][1]
    assert layer["repeat"] == "h.{i}"
    assert layer["rows"] == [["ln_1"], ["attn.c_attn"], ["attn.c_proj"], ["ln_2"], ["mlp.c_fc"], ["mlp.c_proj"]]
    roles = {m["name"]: m["role"] for m in scene["modules"]}
    assert roles["h.0.attn.c_attn"] == "attn_q" and roles["h.1.mlp.c_fc"] == "ff_up" and roles["h.2.mlp.c_proj"] == "ff_down"
    assert roles["wte"] == "embed" and roles["ln_f"] == "norm"
    tops = {g["name"]: g["role"] for g in scene["groups"] if g["outline"]}
    assert tops == {"Embeddings": "embed", "3 layers": "attn_q", "Output": "other"}
    assert any(e["k"] == "flow" for e in scene["edges"])


def test_llama_style_sharded_bf16(tmp_path):
    t = llama(); ck = tmp_path / "llama"; ck.mkdir()
    names = list(t); half = len(names) // 2
    write_safetensors(ck / "model-00001-of-00002.safetensors", {n: t[n] for n in names[:half]}, bf16=True)
    write_safetensors(ck / "model-00002-of-00002.safetensors", {n: t[n] for n in names[half:]}, bf16=True)
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        n: "model-0000%d-of-00002.safetensors" % (1 if i < half else 2) for i, n in enumerate(names)}}))
    spec, scene = build_and_check(ck, tmp_path, ["Embeddings", "2 layers", "Output"], sum(a.size for a in t.values()))
    rows = spec["groups"][1]["rows"]
    assert rows[0] == ["input_layernorm"]
    assert rows[1] == ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]
    assert rows[-1] == ["mlp.down_proj"]
    assert spec["groups"][0]["rows"][0][0] == {"m": "model.embed_tokens", "lie": "flat", "texture": 1024}
    ck2 = build.open_checkpoint(ck)
    a = ck2.array("model.layers.0.self_attn.q_proj.weight")
    assert a.dtype == np.float32 and np.allclose(a, t["model.layers.0.self_attn.q_proj.weight"], atol=1e-3)


def test_no_repeated_block(tmp_path):
    t = {"encoder.0.weight": w(16, 8), "encoder.0.bias": w(16), "decoder.weight": w(8, 16)}
    f = tmp_path / "tiny.safetensors"; write_safetensors(f, t)
    spec, scene = build_and_check(f, tmp_path, ["Model"], sum(a.size for a in t.values()))


def test_leftovers_go_under_other(tmp_path):
    t = gpt2(layers=2); ck = tmp_path / "m.safetensors"; write_safetensors(ck, t)
    g = tmp_path / "g.yaml"
    g.write_text(yaml.safe_dump({"model": "Partial", "groups": [{"name": "2 layers", "repeat": "h.{i}", "rows": [["attn.c_attn"]]}]}))
    r = run("build.py", ck, g, tmp_path / "site")
    assert r.returncode == 0, r.stderr
    scene = json.loads((tmp_path / "site" / "scene.json").read_text())
    assert any(x["name"] == "Other" for x in scene["groups"])
    assert sum(it["params"] for it in scene["items"]) == scene["total"]


def test_big_textures_shrink_to_fit(tmp_path):
    t = gpt2(layers=32, d=256); ck = tmp_path / "m.safetensors"; write_safetensors(ck, t)
    g = tmp_path / "g.yaml"; run("init_groups.py", ck, g)
    spec = yaml.safe_load(g.read_text()); spec["texture"] = 2048; g.write_text(yaml.safe_dump(spec))
    r = run("build.py", ck, g, tmp_path / "site")
    assert r.returncode == 0, r.stderr
    assert "halving" in r.stderr


@pytest.mark.parametrize("name", ["model.gguf", "model.onnx", "weights.xyz"])
def test_unreadable_formats_say_what_to_do(tmp_path, name):
    f = tmp_path / name; f.write_bytes(b"\0" * 16)
    r = run("init_groups.py", f, tmp_path / "g.yaml")
    assert r.returncode != 0 and "can't read" in r.stderr and "safetensors" in r.stderr


def test_pytorch_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    t = gpt2(layers=2)
    f = tmp_path / "pytorch_model.bin"
    torch.save({"state_dict": {k: torch.from_numpy(v) for k, v in t.items()}}, f)
    build_and_check(f, tmp_path, ["Embeddings", "2 layers", "Output"], sum(a.size for a in t.values()))


def test_shared_bundle(tmp_path):
    t = gpt2(layers=2); ck = tmp_path / "m.safetensors"; write_safetensors(ck, t)
    g = tmp_path / "g.yaml"; run("init_groups.py", ck, g); run("build.py", ck, g, tmp_path / "site")
    r = run("bundle_shared.py", tmp_path / "site", tmp_path / "shared")
    assert r.returncode == 0, r.stderr
    files = sorted(p.name for p in (tmp_path / "shared").iterdir())
    assert files == ["index.html", "shared.js", "worker.js"]
    worker = (tmp_path / "shared" / "worker.js").read_text()
    assert '"/scene.json"' in worker and '"/atlas.png"' in worker and "api/ws" in worker
    html = (tmp_path / "shared" / "index.html").read_text()
    assert 'id="who"' in html and "shared.js" in html and "<!--SS_VIEWER-->" in html
    if subprocess.run(["node", "--version"], capture_output=True).returncode == 0:   # the worker parses as a module
        (tmp_path / "w.mjs").write_text(worker)
        r = subprocess.run(["node", "--check", str(tmp_path / "w.mjs")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_no_private_paths_in_the_skill():
    root = ASSETS.parent.parent
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts and p.suffix in {".md", ".py", ".js", ".html", ".yaml", ".yml", ".json"}:
            s = p.read_text(errors="ignore")
            assert ("/ho" + "me/") not in s and ("/Us" + "ers/") not in s, p
