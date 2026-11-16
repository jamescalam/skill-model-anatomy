# Hosting

The build is a static folder. Anything that serves files works; open it over http, not `file://` (modules and
`fetch` need a server).

| Where | How |
|---|---|
| Locally | `python -m http.server -d site 8000` → http://localhost:8000/ |
| Spacesheep | `npx -y spacesheep@latest deploy site --title "Model in 3D" --emoji 🧊 --json`, private until shared (`spacesheep share <url> --visibility public`) |
| GitHub Pages | commit `site/` to a `gh-pages` branch or a `docs/` folder |
| Any CDN / bucket | upload the folder as is; serve `.js` as `text/javascript` |

Spacesheep: update the same page with `deploy site --space <url> -m "what changed"`.

## The shared view

`python assets/bundle_shared.py site shared/` writes `index.html`, `shared.js` and `worker.js`. Deployed as one
Spacesheep space, its `worker.js` serves the site and relays `api/ws` to the space's room (`env.ROOM`, one per space):

```bash
npx -y spacesheep@latest deploy shared --title "Model in 3D · shared view" --json
```

- Turn, zoom and pan stream at up to 20 updates a second and are eased into; the pinned stage, colour mode,
  connections and unused parts follow at once. A late joiner asks the oldest tab for the current view.
- A pill in the corner says who is here and who is moving it. Theme and the Details panel stay each reader's own.
- The room keeps nothing: once everyone leaves, the next reader starts at the default view.
- `?ssdebug` shows a log of the room's messages; `?ssdebug&ssspin` turns the model for 3 s, to test a second tab follows.
- `worker.js` holds the whole site (the atlas as base64), so it is about 1.3× the site's size.

Elsewhere, the same `shared.js` works with any WebSocket relay at `api/ws` that broadcasts each message to the other
tabs as `{from, data}`, sends `{to, data}` to one tab, and greets each tab with `{sys: "welcome", id, host}`.
