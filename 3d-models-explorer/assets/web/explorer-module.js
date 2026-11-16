// The explorer as a module for someone else's WebGL scene. Every tensor of the model is one cube, all of them
// one instanced mesh with one picture atlas, so adding it to a scene costs 1 draw call (and ~2 more for the main-path links).
//
//   import { createExplorer } from "./explorer-module.js";
//   const ex = createExplorer({ scene, camera, base: "./", theme: "dark", height: 100 });
//   await ex.load();                                   // fetch scene.json + atlas.png (call it when the drop is near)
//   ex.highlight("12 layers", "#6cff8f", 1);           // glow a part (0..1)
//   ex.dim(["Output"], 0.9);                          // darken parts (0..1)
//   ex.lift(["Output"], t);                           // slide parts straight up, t 0..1, a dashed outline stays behind
//   ex.anchor("12 layers", "top")                       // a point on a part (or the whole model) in scene coordinates, for a label
//
// It draws with the host's renderer and loop: no renderer, loop, controls, label or panel in here. It imports the bare
// "three" (0.160.0 is what the explorer was built on); the host's import map or bundler decides which copy that is.
//
// Coordinates: ex.root is centred on (0,0,0), y up, z toward the camera, scaled so the tallest side is `height`.
// ex.box(name) is a part's Box3 in root space (before any transform you give the root).
import * as THREE from "three";

const THEMES = {
  dark: { path: "#a8b3cc", pathUnused: "#515764",
    roles: { embed: "#d1a67a", norm: "#8e96a5", attn_q: "#7c86ff", attn_k: "#9d82ff", attn_v: "#bd84ff", attn_out: "#2dd4bf", ff_up: "#fbd23b", ff_down: "#e8b13d", classifier: "#fb7185", other: "#6e7480" } },
  light: { path: "#566683", pathUnused: "#a9acb3",
    roles: { embed: "#a7763f", norm: "#69727f", attn_q: "#4a54e0", attn_k: "#7650e0", attn_v: "#a24fd6", attn_out: "#0d9488", ff_up: "#d99a00", ff_down: "#a8790f", classifier: "#e11d48", other: "#9aa0aa" } },
};
const ROLE_ORDER = ["embed", "attn_q", "attn_k", "attn_v", "attn_out", "ff_up", "ff_down", "norm", "classifier", "other"];

const vert = `
  attribute vec4 iUv; attribute float iRole; attribute vec4 iGlow; attribute float iDim; attribute float iBoost; attribute float iLift;
  varying vec2 vUv; varying float vFront; varying vec4 vRect; varying float vRole; varying vec4 vGlow; varying float vDim; varying float vBoost; varying vec2 vSize; varying vec3 vWorld;
  #include <fog_pars_vertex>
  void main() {
    vUv = uv; vFront = step(0.5, normal.z); vRect = iUv; vRole = iRole; vGlow = iGlow; vDim = iDim; vBoost = iBoost;
    vSize = vec2(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz));
    vWorld = (modelMatrix * (instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0) + vec4(0.0, iLift, 0.0, 0.0))).xyz;   // this cube's centre in world space (for the rim's front)
    vec4 mvPosition = modelViewMatrix * (instanceMatrix * vec4(position, 1.0) + vec4(0.0, iLift, 0.0, 0.0));   // iLift: lift() slides a part straight up
    gl_Position = projectionMatrix * mvPosition;
    #include <fog_vertex>
  }`;
const frag = `
  uniform sampler2D uAtlas; uniform vec3 uRoles[10]; uniform vec3 uBase; uniform float uUseBase; uniform float uSideMul;
  varying vec2 vUv; varying float vFront; varying vec4 vRect; varying float vRole; varying vec4 vGlow; varying float vDim; varying float vBoost; varying vec2 vSize; varying vec3 vWorld;
  uniform vec3 uRimColor; uniform float uRimAmount; uniform float uRimWidth; uniform float uRimFront; uniform vec3 uRimN; uniform float uRimOffset; uniform float uRimSoft;
  #include <fog_pars_fragment>
  void main() {
    float v = texture2D(uAtlas, vec2(mix(vRect.x, vRect.z, vUv.x), mix(vRect.w, vRect.y, vUv.y))).r * 2.0 - 1.0;   // this tensor's real weight, sampled
    vec3 col = uUseBase > 0.5 ? uBase : uRoles[int(vRole + 0.5)];
    vec3 c = col * (0.80 + 0.28 * (v * 0.5 + 0.5));   // the grain of the weights, over the part's colour
    if (vFront < 0.5) c = col * uSideMul;
    vec3 g = vGlow.rgb * (vFront > 0.5 ? 0.78 + 0.22 * (v * 0.5 + 0.5) : 0.55);   // a glowing cube keeps its grain and its sides
    // glow: tint toward the colour (its hue stays), then add light on top; boost > 1 adds more of that light, so the colour
    // overshoots 1.0 for a bloom pass without the tint itself saturating to white
    c = mix(c, g, vGlow.a * 0.85) + vGlow.rgb * vGlow.a * (0.12 + 0.35 * (vBoost - 1.0));
    if (uRimAmount > 0.0) {   // the rim: an outline round each cube's front face, a faint tint on its sides; only on cubes the front has passed
      float passed = uRimFront > 0.5 ? 1.0 - smoothstep(0.0, max(uRimSoft, 0.001), dot(uRimN, vWorld) - uRimOffset) : 1.0;
      vec2 d = min(vUv, 1.0 - vUv) * vSize;
      float edge = min(d.x, d.y);
      float on = uRimAmount * passed;
      if (vFront > 0.5) c = mix(c, uRimColor, on * (1.0 - smoothstep(uRimWidth * 0.6, uRimWidth, edge)));
      else c = mix(c, uRimColor, on * 0.3);
    }
    c *= 1.0 - 0.93 * vDim;                                               // dim: toward black
    gl_FragColor = linearToOutputTexel(vec4(c, 1.0));
    #include <fog_fragment>   // the host's scene.fog, if it has one: distance darkens the cubes
  }`;

function ribbon(acc, curve, radius, colour, arrow) {
  const S = 6, base = acc.pos.length / 3, up = new THREE.Vector3(0, 0, 1), n = new THREE.Vector3(), b = new THREE.Vector3(), tan = new THREE.Vector3();
  // The head is a cone 7 radii long and 3 wide at its base, so near its tip it is thinner than the line: the line
  // stops at the cone's base (one cone length back along the curve, by arc length), never reaching the tip.
  // A link shorter than that gets a smaller head (at most 45% of the link) instead of a flipped one.
  const len = curve.getLength(), N = Math.max(22, Math.min(64, Math.ceil(len / 5))), L = arrow ? Math.min(radius * 7, len * 0.45) : 0, uEnd = len > 0 ? 1 - L / len : 1;
  const pts = []; for (let i = 0; i <= N; i++) pts.push(curve.getPointAt(uEnd * i / N));
  const push = (p) => { acc.pos.push(p.x, p.y, p.z); acc.col.push(colour.r, colour.g, colour.b); };
  for (let i = 0; i <= N; i++) {
    tan.copy(curve.getTangentAt(uEnd * i / N));
    n.crossVectors(tan, up); if (n.lengthSq() < 1e-6) n.set(1, 0, 0); n.normalize(); b.crossVectors(tan, n).normalize();
    for (let s = 0; s < S; s++) { const a = (s / S) * Math.PI * 2; push(pts[i].clone().addScaledVector(n, Math.cos(a) * radius).addScaledVector(b, Math.sin(a) * radius)); }
  }
  for (let i = 0; i < N; i++) for (let s = 0; s < S; s++) {
    const a = base + i * S + s, c = base + i * S + (s + 1) % S;
    acc.idx.push(a, a + S, c, c, a + S, c + S);
  }
  if (arrow) {   // the cone, from the line's end to the link's end
    const end = curve.getPointAt(1), back = pts[N], t = end.clone().sub(back);
    if (t.lengthSq() < 1e-12) t.copy(curve.getTangentAt(1)); t.normalize();
    const R = L * 3 / 7, ring = acc.pos.length / 3;
    n.crossVectors(t, up); if (n.lengthSq() < 1e-6) n.set(1, 0, 0); n.normalize(); b.crossVectors(t, n).normalize();
    for (let s = 0; s < S; s++) { const a = (s / S) * Math.PI * 2; push(back.clone().addScaledVector(n, Math.cos(a) * R).addScaledVector(b, Math.sin(a) * R)); }
    push(end);
    for (let s = 0; s < S; s++) acc.idx.push(ring + s, ring + (s + 1) % S, ring + S);
  }
}
const SIDE = -1;   // which side of the stack a climbing link bows to
// A link that ends higher than it starts (layer N's bottom to layer N+1's top: 135 units up) would climb straight past the head
// it is about to reach. It bows out to the side in x instead, clears the stack, and comes down into the head from above.
// Start and end stay where they are.
function climbCurve(p0, p1, A, B, side) {
  const rise = p1.y - p0.y, w = Math.max(A.max.x - A.min.x, B.max.x - B.min.x), off = side * (w / 2 + 6), h = 10, V = (x, y, z) => new THREE.Vector3(x, y, z);
  return new THREE.CatmullRomCurve3([p0, V(p0.x, p0.y - 6, p0.z), V(p0.x + off, p0.y + 0.3 * rise, p0.z), V(p1.x + off, p0.y + 0.8 * rise, p1.z), V(p1.x + off * 0.35, p1.y + h * 1.5, p1.z), V(p1.x, p1.y + h, p1.z), p1], false, "centripetal");
}
function curveBetween(A, B) {
  const p0 = new THREE.Vector3((A.min.x + A.max.x) / 2, A.min.y, (A.min.z + A.max.z) / 2), p1 = new THREE.Vector3((B.min.x + B.max.x) / 2, B.max.y, (B.min.z + B.max.z) / 2);
  const dy = Math.max(3, Math.abs(p0.y - p1.y) * 0.45);
  if (p1.y - p0.y > 20) return climbCurve(p0, p1, A, B, SIDE);
  return new THREE.CubicBezierCurve3(p0, p0.clone().add(new THREE.Vector3(0, -dy, 0)), p1.clone().add(new THREE.Vector3(0, dy, 0)), p1);
}

// lowMemory: true loads atlas-half.png (a quarter of the texture memory, a little softer grain) instead of atlas.png; setDetail(0) called
// before load() does the same. If atlas-half.png is missing the full atlas is used.
export function createExplorer({ scene, camera = null, base = "./", theme = "dark", height = 100, lowMemory = false } = {}) {
  if (!scene) throw new Error("createExplorer needs { scene }");
  const root = new THREE.Group(); root.name = "explorer"; root.visible = false;   // until load() resolves
  const scaler = new THREE.Group(), inner = new THREE.Group();                   // root > scaler (the size) > inner (the centring)
  scaler.add(inner); root.add(scaler); scene.add(root);

  let themeName = THEMES[theme] ? theme : "dark", palette = { base: null, roles: null }, detail = 1, model = null, s = 1, loaded = false, disposed = false, loading = null;
  const parts = new Map();     // name -> { items: number[], glow: THREE.Color|null, amount, boost, dimAmount, liftT, liftDist, unused }
  const ghosts = new Map();    // name -> the dashed outline left where a lifted part was
  const boxes = new Map();     // name -> Box3 in root space
  const slabs = new Map();   // layer number -> { fill, edges } (made the first time a layer's slab is asked for)
  const layers = [], layerBoxes = [];   // layer number -> { items, glow, amount, boost, dimAmount } and its Box3 in root space
  let cubes = null, glowArr, dimArr, boostArr, liftArr, linkMeshes = [], atlas = null;
  const queued = [];   // highlight / dim / clear calls made before load() finished: replayed, in order, when it does
  const U = { uAtlas: { value: null }, uRoles: { value: ROLE_ORDER.map(() => new THREE.Color()) }, uBase: { value: new THREE.Color() }, uUseBase: { value: 0 }, uSideMul: { value: 0.55 },
    uRimColor: { value: new THREE.Color("#6cff8f") }, uRimAmount: { value: 0 }, uRimWidth: { value: 0.5 }, uRimFront: { value: 0 }, uRimN: { value: new THREE.Vector3(0, 0, 1) }, uRimOffset: { value: 0 }, uRimSoft: { value: 1 } };

  function applyColours() {
    const T = THEMES[themeName], roles = { ...T.roles, ...(palette.roles || {}) };
    ROLE_ORDER.forEach((r, i) => U.uRoles.value[i].set(roles[r]));
    U.uBase.value.set(palette.base || "#808080"); U.uUseBase.value = palette.base ? 1 : 0;
    for (const gh of ghosts.values()) gh.material.color.set(T.path);
    for (const m of linkMeshes) { const c = new THREE.Color(m.userData.unused ? T.pathUnused : T.path), col = m.geometry.attributes.color; for (let i = 0; i < col.count; i++) col.setXYZ(i, c.r, c.g, c.b); col.needsUpdate = true; }
  }
  function writePart(p) {   // one part's glow and dim into the per-cube buffers
    const g = p.glow || new THREE.Color(0, 0, 0), a = p.glow ? p.amount : 0;
    const lf = (p.liftT || 0) * (p.liftDist || 0) / s;   // in the cubes' own (unscaled) units
    for (const k of p.items) { glowArr.setXYZW(k, g.r, g.g, g.b, a); dimArr.setX(k, p.dimAmount); boostArr.setX(k, p.boost); liftArr.setX(k, lf); }
    glowArr.needsUpdate = true; dimArr.needsUpdate = true; boostArr.needsUpdate = true; liftArr.needsUpdate = true;
  }

  const api = {
    root,
    get parts() { return [...parts.keys()]; },
    get loaded() { return loaded; },
    get atlasSize() { return atlas && atlas.image ? [atlas.image.width, atlas.image.height] : null; },   // [2048, 1244] on the low-memory atlas, [4096, 2488] otherwise
    get size() { return model ? new THREE.Vector3(...model.size).multiplyScalar(s) : new THREE.Vector3(); },

    load() {
      if (loading) return loading;
      loading = (async () => {
        const r = await fetch(base + "scene.json"); if (!r.ok) throw new Error("scene.json: " + r.status);
        model = await r.json(); if (disposed) return api;
        const [W, H, D] = model.size, T3 = model.thick || 2;
        s = height / Math.max(W, H, D); scaler.scale.setScalar(s);
        const cx = W / 2, cy = -H / 2, cz = -D / 2;   // scene: x right from 0, y DOWN (negative), z BACK (negative); the root is centred, y up, z toward the camera
        inner.position.set(-cx, -cy, -cz);
        const loadAtlas = (file) => new Promise((res, rej) => new THREE.TextureLoader().load(base + file, res, undefined, rej));
        const low = lowMemory || detail === 0;
        try { atlas = await loadAtlas(low ? model.atlas.file.replace(/\.png$/, "-half.png") : model.atlas.file); }
        catch (e) { if (!low) throw e; atlas = await loadAtlas(model.atlas.file); }   // no half-size atlas next to scene.json: use the full one
        atlas.flipY = false; atlas.magFilter = THREE.NearestFilter; atlas.minFilter = THREE.LinearMipmapLinearFilter; atlas.colorSpace = THREE.NoColorSpace; atlas.anisotropy = 4;
        U.uAtlas.value = atlas;
        const n = model.items.length, geo = new THREE.BoxGeometry(1, 1, 1);
        const uv = new Float32Array(n * 4), role = new Float32Array(n), gl4 = new Float32Array(n * 4), dm = new Float32Array(n), bo = new Float32Array(n).fill(1);
        const cubesMesh = new THREE.InstancedMesh(geo, new THREE.ShaderMaterial({ uniforms: { ...U, ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog) }, vertexShader: vert, fragmentShader: frag, fog: true }), n);
        const m4 = new THREE.Matrix4(), p3 = new THREE.Vector3(), s3 = new THREE.Vector3(), flatQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)), idQ = new THREE.Quaternion();
        model.items.forEach((it, k) => {
          p3.set(it.pos[0], it.pos[1], it.flat ? it.pos[2] : it.pos[2] - T3 / 2); s3.set(it.size[0], it.size[1], T3);
          m4.compose(p3, it.flat ? flatQ : idQ, s3); cubesMesh.setMatrixAt(k, m4);
          uv.set(it.uv, k * 4); role[k] = Math.max(0, ROLE_ORDER.indexOf(model.modules[it.mod].role));
        });
        geo.setAttribute("iUv", new THREE.InstancedBufferAttribute(uv, 4)); geo.setAttribute("iRole", new THREE.InstancedBufferAttribute(role, 1));
        glowArr = new THREE.InstancedBufferAttribute(gl4, 4); glowArr.setUsage(THREE.DynamicDrawUsage); geo.setAttribute("iGlow", glowArr);
        dimArr = new THREE.InstancedBufferAttribute(dm, 1); dimArr.setUsage(THREE.DynamicDrawUsage); geo.setAttribute("iDim", dimArr);
        liftArr = new THREE.InstancedBufferAttribute(new Float32Array(n), 1); liftArr.setUsage(THREE.DynamicDrawUsage); geo.setAttribute("iLift", liftArr);
        boostArr = new THREE.InstancedBufferAttribute(bo, 1); boostArr.setUsage(THREE.DynamicDrawUsage); geo.setAttribute("iBoost", boostArr);
        cubesMesh.frustumCulled = false; inner.add(cubesMesh); cubes = cubesMesh;
        // parts: every named group, with every cube inside it (a cube's group path lists all the groups round it)
        for (const g of model.groups) if (g.outline && g.name && !parts.has(g.name)) {
          const [x, y, z, w, h, d] = g.box, c = new THREE.Vector3(cx, cy, cz);
          parts.set(g.name, { items: [], glow: null, amount: 0, boost: 1, dimAmount: 0, liftT: 0, liftDist: 0, unused: !!g.unused });
          boxes.set(g.name, new THREE.Box3(new THREE.Vector3(x, y - h, z - d).sub(c).multiplyScalar(s), new THREE.Vector3(x + w, y, z).sub(c).multiplyScalar(s)));
        }
        model.items.forEach((it, k) => { for (const name of model.groups[it.group].path) if (parts.has(name)) parts.get(name).items.push(k); });
        // the main path: group to group, layer to layer, as a few thick arrows
        const modBox = model.modules.map((m) => {
          const bb = new THREE.Box3();
          for (let k = m.items[0]; k < m.items[1]; k++) {
            const it = model.items[k], [w2, h2] = it.size;
            const lo = it.flat ? [it.pos[0] - w2 / 2, it.pos[1] - T3 / 2, it.pos[2] - h2 / 2] : [it.pos[0] - w2 / 2, it.pos[1] - h2 / 2, it.pos[2] - T3];
            const hi = it.flat ? [it.pos[0] + w2 / 2, it.pos[1] + T3 / 2, it.pos[2] + h2 / 2] : [it.pos[0] + w2 / 2, it.pos[1] + h2 / 2, it.pos[2]];
            bb.expandByPoint(new THREE.Vector3(...lo)); bb.expandByPoint(new THREE.Vector3(...hi));
          }
          return bb;
        });
        // the repeated block, copy by copy (a module's `copy` is its layer number)
        const cc = new THREE.Vector3(cx, cy, cz);
        model.modules.forEach((m, mi) => {
          if (m.copy == null) return;
          if (!layers[m.copy]) { layers[m.copy] = { items: [], glow: null, amount: 0, boost: 1, dimAmount: 0 }; layerBoxes[m.copy] = new THREE.Box3(); }
          for (let k = m.items[0]; k < m.items[1]; k++) layers[m.copy].items.push(k);
          layerBoxes[m.copy].union(modBox[mi]);
        });
        layerBoxes.forEach((b, i) => { layerBoxes[i] = new THREE.Box3(b.min.clone().sub(cc).multiplyScalar(s), b.max.clone().sub(cc).multiplyScalar(s)); });
        const groups = new Map();
        for (const e of model.edges) {
          const a = model.modules[e.a], b = model.modules[e.b];
          if (e.k !== "flow" || (a.group === b.group && a.copy === b.copy)) continue;
          const key = `${e.a}|${b.copy ?? "g" + b.group}|${e.u ? 1 : 0}`;
          if (!groups.has(key)) groups.set(key, { a: e.a, bs: [], u: e.u });
          groups.get(key).bs.push(e.b);
        }
        const acc = [{ pos: [], col: [], idx: [] }, { pos: [], col: [], idx: [] }];
        for (const g of groups.values()) { const B = new THREE.Box3(); g.bs.forEach((i) => B.union(modBox[i])); ribbon(acc[g.u ? 1 : 0], curveBetween(modBox[g.a], B), 0.5, new THREE.Color("#ffffff"), true); }
        acc.forEach((a, i) => {
          if (!a.pos.length) return;
          const gg = new THREE.BufferGeometry(); gg.setAttribute("position", new THREE.Float32BufferAttribute(a.pos, 3)); gg.setAttribute("color", new THREE.Float32BufferAttribute(a.col, 3)); gg.setIndex(a.idx);
          const m = new THREE.Mesh(gg, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true })); m.frustumCulled = false; m.userData.unused = i === 1; inner.add(m); linkMeshes.push(m);
        });
        applyColours();
        loaded = true; root.visible = true; api.setDetail(detail);
        for (const [fn, args] of queued.splice(0)) api[fn](...args);   // calls made before the load finished
        return api;
      })();
      return loading;
    },

    get layerCount() { return layers.length; },   // 24 (after load)
    layerBox(i) { const b = layerBoxes[i]; return b ? b.clone() : null; },   // layer i's Box3 in root space
    // Glow one layer (0..23): the same colour, amount and boost as highlight(). It writes the same per-cube buffers as a part's
    // highlight, so when both touch a cube the later call wins.
    highlightLayer(i, colour = "#6cff8f", amount = 1, boost = 1) {
      if (!loaded) { queued.push(["highlightLayer", [i, colour, amount, boost]]); return api; }
      const p = layers[i]; if (!p) return api;
      p.glow = new THREE.Color(colour); p.amount = amount; p.boost = boost; writePart(p); return api;
    },
    // One shape for a layer: a translucent pane the size of the layer's box, with a bright outline, so the cubes (its ~10 tensors) read
    // as lying in one slab. amount 0..1 (0 hides it); `boost` > 1 pushes the outline above 1.0 for a bloom pass. Two draw calls while it is on.
    layerSlab(i, colour = "#6cff8f", amount = 1, boost = 1) {
      if (!loaded) { queued.push(["layerSlab", [i, colour, amount, boost]]); return api; }
      const lb = layerBoxes[i]; if (!lb) return api;
      let sl = slabs.get(i);
      if (!sl) {
        const size = lb.getSize(new THREE.Vector3()).add(new THREE.Vector3(1.2, 1.2, 0.8)), centre = lb.getCenter(new THREE.Vector3());
        const g = new THREE.BoxGeometry(size.x, size.y, size.z);
        const fill = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, opacity: 0 }));
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ transparent: true, depthWrite: false }));
        fill.position.copy(centre); edges.position.copy(centre); fill.renderOrder = 3; edges.renderOrder = 4;
        root.add(fill); root.add(edges); sl = { fill, edges }; slabs.set(i, sl);
      }
      const on = amount > 0.001;
      sl.fill.visible = sl.edges.visible = on;
      sl.fill.material.color.set(colour); sl.fill.material.opacity = 0.22 * amount;
      sl.edges.material.color.set(colour).multiplyScalar(boost); sl.edges.material.opacity = Math.min(1, amount);
      return api;
    },
    dimLayer(i, amount = 0.8) {
      if (!loaded) { queued.push(["dimLayer", [i, amount]]); return api; }
      const p = layers[i]; if (p) { p.dimAmount = amount; writePart(p); } return api;
    },

    box(name) { const b = boxes.get(name); return b ? b.clone() : null; },   // root space, before any transform you put on the root

    // boost > 1 pushes the glow above 1.0, for a bloom pass in an HDR target to catch (default 1: the colour itself)
    highlight(name, colour = "#6cff8f", amount = 1, boost = 1) {
      if (!loaded) { queued.push(["highlight", [name, colour, amount, boost]]); return api; }
      const p = parts.get(name); if (!p) return api;
      p.glow = new THREE.Color(colour); p.amount = amount; p.boost = boost; if (loaded) writePart(p); return api;
    },
    // A point on a part, for a label: where = top (default) | bottom | left | right | front | back | centre. part = a part's name, or
    // nothing / "model" for the whole model. In scene coordinates (the root's transform included), following any lift(), so the host
    // can project it with its own camera. null until load() has finished.
    anchor(part, where = "top") {
      if (!loaded) return null;
      let b;
      if (part == null || part === "model") { b = new THREE.Box3(); for (const [n, bx] of boxes) { const p = parts.get(n), up = (p.liftT || 0) * (p.liftDist || 0); b.union(up ? bx.clone().translate(new THREE.Vector3(0, up, 0)) : bx); } }
      else { const bx = boxes.get(part), p = parts.get(part); if (!bx) return null; const up = (p.liftT || 0) * (p.liftDist || 0); b = up ? bx.clone().translate(new THREE.Vector3(0, up, 0)) : bx.clone(); }
      const c = b.getCenter(new THREE.Vector3()), v = c.clone();
      if (where === "top") v.y = b.max.y; else if (where === "bottom") v.y = b.min.y;
      else if (where === "left") v.x = b.min.x; else if (where === "right") v.x = b.max.x;
      else if (where === "front") v.z = b.max.z; else if (where === "back") v.z = b.min.z;
      root.updateWorldMatrix(true, false); return root.localToWorld(v);
    },
    // Slide parts straight up and away, t from 0 to 1 (scrub it both ways; 0 draws exactly what it did before). distance is in root units
    // (default: 30% of the model's height). A dashed outline stays where each part was (fading in with t), the lifted cubes stay visible
    // (use dim() to ghost them), and links into parts that never run (unused groups) fade out with it.
    lift(names, t = 1, opts = {}) {
      if (!loaded) { queued.push(["lift", [names, t, opts]]); return api; }
      t = Math.max(0, Math.min(1, t));
      const dist = opts.distance != null ? opts.distance : 0.3 * api.size.y;
      for (const nme of [].concat(names)) {
        const p = parts.get(nme); if (!p) continue;
        p.liftT = t; p.liftDist = dist; writePart(p);
        let gh = ghosts.get(nme);
        if (!gh) {
          const bx = boxes.get(nme), a = bx.min, z = bx.max, V = (x, y, w) => new THREE.Vector3(x, y, w), pts = [];
          const c = [V(a.x, a.y, a.z), V(z.x, a.y, a.z), V(z.x, z.y, a.z), V(a.x, z.y, a.z), V(a.x, a.y, z.z), V(z.x, a.y, z.z), V(z.x, z.y, z.z), V(a.x, z.y, z.z)];
          for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) pts.push(c[i], c[j]);
          const g = new THREE.BufferGeometry().setFromPoints(pts), u = Math.max(0.5, 0.012 * api.size.y);
          gh = new THREE.LineSegments(g, new THREE.LineDashedMaterial({ color: 0xa8b3cc, dashSize: u, gapSize: u, transparent: true, opacity: 0, depthWrite: false }));
          gh.computeLineDistances(); gh.frustumCulled = false; root.add(gh); ghosts.set(nme, gh);
        }
        gh.material.color.set(THEMES[themeName].path); gh.material.opacity = t; gh.visible = t > 0;
      }
      let gone = 0; for (const p of parts.values()) if (p.unused) gone = Math.max(gone, p.liftT || 0);
      for (const m of linkMeshes) if (m.userData.unused) m.material.opacity = 1 - gone;
      return api;
    },
    dim(names, amount = 0.8) {
      if (!loaded) { queued.push(["dim", [names, amount]]); return api; }
      for (const nme of [].concat(names)) { const p = parts.get(nme); if (p) { p.dimAmount = amount; if (loaded) writePart(p); } }
      return api;
    },
    clear() { if (!loaded) { queued.push(["clear", []]); return api; } for (const sl of slabs.values()) sl.fill.visible = sl.edges.visible = false; for (const p of [...parts.values(), ...layers]) { if (!p) continue; p.glow = null; p.amount = 0; p.boost = 1; p.dimAmount = 0; p.liftT = 0; p.liftDist = 0; writePart(p); } for (const gh of ghosts.values()) { gh.visible = false; gh.material.opacity = 0; } for (const m of linkMeshes) m.material.opacity = 1; return api; },
    setTheme(name) { if (THEMES[name]) { themeName = name; if (loaded) applyColours(); } return api; },
    // An outline round every cube, for the whole model at once; `amount` 0..1 (0 = off), `width` in model units (0.5 = a thin line; the colour
    // may be pushed above 1 with `boost`). With `front` ({ normal: Vector3, offset, soft }, in WORLD space) only cubes where
    // dot(normal, centre) < offset get it, fading over `soft`: move `offset` and the rims light as a front passes. Independent of highlight().
    setRim(colour = "#6cff8f", amount = 1, opts = {}) {
      if (typeof opts === "number") opts = { width: opts };   // setRim(colour, amount, width) works too
      const { width = 0.5, boost = 1, front = null } = opts;
      U.uRimColor.value.set(colour).multiplyScalar(boost); U.uRimAmount.value = amount; U.uRimWidth.value = width;
      U.uRimFront.value = front ? 1 : 0;
      if (front) { U.uRimN.value.copy(front.normal).normalize(); U.uRimOffset.value = front.offset; U.uRimSoft.value = front.soft ?? 1; }
      return api;
    },
    // { base: "#hex" } makes every cube one colour; { roles: { attn_q: "#hex", ... } } overrides a role's colour
    setPalette(p = {}) { palette = { base: p.base || null, roles: p.roles || null }; if (loaded) applyColours(); return api; },
    // 1 = everything; below 0.5 the main-path links go (the cubes never thin out)
    setDetail(d) { detail = Math.max(0, Math.min(1, d)); for (const m of linkMeshes) m.visible = detail >= 0.5; return api; },
    dispose() {
      disposed = true; scene.remove(root);
      if (cubes) { cubes.geometry.dispose(); cubes.material.dispose(); }
      for (const m of linkMeshes) { m.geometry.dispose(); m.material.dispose(); }
      for (const gh of ghosts.values()) { gh.geometry.dispose(); gh.material.dispose(); } ghosts.clear();
      for (const sl of slabs.values()) { sl.fill.geometry.dispose(); sl.fill.material.dispose(); sl.edges.geometry.dispose(); sl.edges.material.dispose(); } slabs.clear();
      if (atlas) atlas.dispose(); parts.clear(); boxes.clear(); layers.length = 0; layerBoxes.length = 0; linkMeshes = [];
    },
  };
  return api;
}
