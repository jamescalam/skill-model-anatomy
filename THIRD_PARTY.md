# Third-party files

| File | From | Licence |
|---|---|---|
| `model-anatomy/assets/web/vendor/three.module.min.js` | [three.js](https://github.com/mrdoob/three.js) r160 (`build/three.module.min.js`), unchanged | MIT, `vendor/LICENSE-three.js.txt` |
| `model-anatomy/assets/web/vendor/OrbitControls.js` | three.js r160 `examples/jsm/controls/OrbitControls.js` | MIT (same) |
| `model-anatomy/assets/web/vendor/CSS2DRenderer.js` | three.js r160 `examples/jsm/renderers/CSS2DRenderer.js` | MIT (same) |

Everything else is this repo's own code, under its MIT licence. No code, shaders or assets are copied from
[llm-viz](https://github.com/bbycroft/llm-viz) or any other visualiser; the idea of drawing a model's tensors as
blocks in 3D was the inspiration.

The build's Python needs numpy (BSD-3), PyYAML (MIT) and Pillow (MIT-CMU); `torch` (BSD-3) and `transformers`
(Apache-2.0) are optional and installed by the user, never bundled.
