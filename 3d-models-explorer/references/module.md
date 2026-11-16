# The model inside your own three.js scene

`assets/web/explorer-module.js` draws the same model as one object in a host page's scene: one instanced mesh and
one atlas (about 1 draw call, ~2 more for links). It uses the host's renderer, camera and loop; no panels or labels.

```js
import * as THREE from "three";                 // 0.160.0 is what it is built on
import { createExplorer } from "./explorer-module.js";
const ex = createExplorer({ scene, camera, base: "./site/", theme: "dark", height: 100 });
await ex.load();                                // fetches scene.json and atlas.png from `base`
ex.highlight("12 layers", "#6cff8f", 1);         // glow a group (0..1), optional boost above 1 for bloom
ex.dim(["Output"], 0.9);                         // darken groups (0..1)
ex.lift(["Output"], 0.5);                        // slide groups up (0..1); a dashed outline stays
const p = ex.anchor("12 layers", "top");         // a point for a label, in scene coordinates
```

| Call | Does |
|---|---|
| `load()` | fetch and build; calls made before it finishes are kept and replayed |
| `highlight(group, colour, amount, boost?)` / `dim([groups], amount)` / `clear()` | per-group light (one name for highlight, a list for dim) |
| `highlightLayer(i, colour, amount, boost)` / `dimLayer(i, amount)` / `layerBox(i)` / `layerCount` (a property) | one copy of the repeated block |
| `layerSlab(i, colour, amount, boost)` | a translucent pane round one layer |
| `setRim(colour, amount, {width, boost, front})` | an outline round every cube, optionally only behind a moving front |
| `lift([groups], t)` | slide groups straight up |
| `box(name)` / `anchor(part, where)` | a group's Box3, a point on it |
| `setPalette({...})` / `setTheme(name)` / `setDetail(0)` / `dispose()` | colours; low detail (`lowMemory: true` loads `atlas-half.png`) |

The host's `scene.fog` is respected. The `module-*.html` pages in `assets/web/` each show one call; they are left
out of a build, so copy one beside a built site to try it.
