# Embedding it in a page

```html
<iframe src="https://YOUR-HOST/explorer/index.html?embed" title="The model in 3D"
        style="width:100%;aspect-ratio:16/10;border:0" loading="lazy"></iframe>
```

`?embed` gives a smaller legend and a one-line card. On a touch screen a vertical swipe over the model still
scrolls the article until the reader taps it ("Tap and pinch"); Done gives the touch back. The model loads when the
frame scrolls into view and stops drawing when it leaves.

## URL options

| Option | Does |
|---|---|
| `embed` | article mode (above) |
| `theme=light` / `theme=dark` | fixes the theme; otherwise it follows the reader's system |
| `focus=<group name>` | opens framed on that group |
| `trace=<part of a tensor name>` | opens with that stage pinned and its inputs and outputs traced |
| `cam=x,y,z,tx,ty,tz` | opens at that camera place and target (link to a view) |
| `colour=weights` | opens coloured by weight sign and size instead of by part |
| `unused=show` | shows groups marked `unused` (hidden unless this is set; the reader can toggle them) |
| `scene=<path>` / `values=<path>` | another scene.json or values.json |
| `selftest` / `perf` | the page's own interaction checks / frame timings, for testing |

A parent page can switch the theme without reloading:

```js
iframe.contentWindow.postMessage({ type: "explorer-theme", value: "light" }, "*");
```

## Sizes

- 16:10 suits a desktop article column; on a phone the legend folds to the bottom.
- A 0.5B model builds to about 3–4 MB (most of it the atlas). Phones load `atlas-half.png` when `lowMemory` is set
  in the module; the page itself uses the full atlas.
