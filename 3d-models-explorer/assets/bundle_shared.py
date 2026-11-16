"""Turn a built explorer into the shared-view version: everyone who has it open sees the same view.

    python bundle_shared.py SITE_DIR OUT_DIR [--title "Model explorer · shared view"]

SITE_DIR is what build.py wrote. OUT_DIR gets three files for a Spacesheep worker space:
  index.html  the explorer's page plus the "who's here" pill and shared.js
  shared.js   sends this reader's turns, zooms and taps through the space's room, and follows everyone else's
  worker.js   serves every file of the site (binaries as base64) and relays api/ws to the space's room (env.ROOM)
Deploy OUT_DIR as one space (references/hosting.md). The room keeps nothing: once everyone leaves, the next reader
starts at the default view. Theme and the Details panel stay each reader's own.
"""
import base64, json, mimetypes, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILL_CSS = """  /* the shared view's pill: who else is here, and who is moving the model right now */
  #who { right: 14px; bottom: 14px; padding: 5px 10px; font-size: 12px; color: var(--muted); display: flex; gap: 7px; align-items: center; max-width: min(340px, calc(100% - 28px)); }
  #who[hidden] { display: none; }
  #who .d { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex: none; }
  #who.on .d { background: var(--reads); }
  #who .t { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  @media (max-width: 560px), (max-height: 520px) { #who { right: 8px; } }
</style>"""
PILL = '<div id="who" class="panel" hidden><span class="d"></span><span class="t">Shared view</span></div>\n<!--SS_VIEWER-->\n'
TEXT = {".html", ".js", ".json", ".css", ".svg", ".txt"}
TYPES = {".js": "text/javascript; charset=utf-8", ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
         ".png": "image/png", ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}

WORKER_TAIL = r"""};
const bytes = new Map();   // decoded once per isolate

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.endsWith("/api/ws")) {   // the relay room: one per space, shared by every tab
      if ((request.headers.get("Upgrade") || "").toLowerCase() !== "websocket") return new Response("WebSocket only", { status: 426 });
      return env.ROOM.get(env.ROOM.idFromName(env.SPACE_ID + "/view")).fetch(request);
    }
    let path = url.pathname.replace(/^\/s\/[^/]+/, "");
    if (path === "" || path.endsWith("/")) path += "index.html";
    const f = FILES[path];
    if (!f) return new Response("Not found", { status: 404 });
    if (path === "/index.html") {   // the page knows its reader's name without a round trip
      let who = null;
      try { who = JSON.parse(decodeURIComponent(request.headers.get("x-spacesheep-viewer") || "null")); } catch {}
      const tag = "<script>window.SS_VIEWER=" + JSON.stringify(who).replace(/</g, "\\u003c") + "</script>";
      return new Response(f[1].replace("<!--SS_VIEWER-->", tag), { headers: { "Content-Type": f[0], "Cache-Control": "no-store" } });
    }
    let body = f[1];
    if (f[2]) {
      if (!bytes.has(path)) { const s = atob(f[1]), b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); bytes.set(path, b); }
      body = bytes.get(path);
    }
    return new Response(body, { headers: { "Content-Type": f[0], "Cache-Control": "private, max-age=300" } });
  },
};
"""


def shared_index(html, title):
    if 'id="who"' in html:
        return html
    html = html.replace("</style>", PILL_CSS, 1)
    html = html.replace('<script type="importmap">', PILL + '<script type="importmap">', 1)
    html = html.replace("</body>", '<script type="module" src="./shared.js"></script>\n</body>', 1)
    if title:
        html = html.replace("<title>Model explorer</title>", f"<title>{title}</title>", 1)
    return html


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    site, out = Path(sys.argv[1]), Path(sys.argv[2])
    title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else "Model explorer · shared view"
    if not (site / "scene.json").exists():
        sys.exit(f"{site} has no scene.json: run build.py first")
    out.mkdir(parents=True, exist_ok=True)
    index = shared_index((site / "index.html").read_text(), title)
    shared = (HERE / "shared.js").read_text()
    files = {"/index.html": index, "/shared.js": shared}
    lines = ["// A model explorer with a shared view. Made by bundle_shared.py from a build.py site; rebuild, don't edit.",
             "// Serves the explorer's files, and api/ws: the space's relay room, where each tab's moves reach the others.",
             "const FILES = {"]
    for p in sorted(site.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        key = "/" + p.relative_to(site).as_posix()
        kind = TYPES.get(p.suffix) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if key in files:
            body, b64 = files.pop(key), False
        elif p.suffix in TEXT:
            body, b64 = p.read_text(), False
        else:
            body, b64 = base64.b64encode(p.read_bytes()).decode(), True
        lines.append(f"  {json.dumps(key)}: {json.dumps([kind, body] + ([True] if b64 else []))},")
    for key, body in files.items():
        lines.append(f"  {json.dumps(key)}: {json.dumps([TYPES[Path(key).suffix], body])},")
    (out / "worker.js").write_text("\n".join(lines) + "\n" + WORKER_TAIL)
    (out / "index.html").write_text(index)
    (out / "shared.js").write_text(shared)
    mb = (out / "worker.js").stat().st_size / 1e6
    print(f"wrote {out}/index.html, shared.js and worker.js ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
