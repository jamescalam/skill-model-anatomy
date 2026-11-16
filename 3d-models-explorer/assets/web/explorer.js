// The model explorer: draws scene.json (from build.py) with three.js.
// Every tensor is one cube in a single instanced mesh; the weights' pictures are one atlas.
// URL options: ?embed, ?theme=light|dark, ?colour=weights, ?trace=<part of a tensor name>, ?focus=<group name>,
//   ?unused=show, ?cam=x,y,z,tx,ty,tz, ?scene=<path>, ?values=<path>. A parent page can switch the theme with
//   postMessage({ type: "explorer-theme", value: "light" | "dark" }).
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "./vendor/CSS2DRenderer.js";

const q = new URLSearchParams(location.search);
if (q.has("embed")) document.body.classList.add("embed");
const view = document.getElementById("view");
const $ = (id) => document.getElementById(id);

// ---- one colour table: the CSS, the shader, the lines and the outlines all read from it ----------------------
// Blue and orange mean negative and positive weights, and nothing else. Parts get their own hues; "reads from"
// and "feeds" get two more (lime and magenta) that no part uses.
const THEMES = {
  dark: {
    bg: "#0e1014", panel: "rgba(22,25,32,0.92)", text: "#e8eaf0", muted: "#a3abb8", line: "#2e3441",
    zero: "#55607a", pos: "#f59e42", neg: "#4aa3f0", side: "#2f3647", greyA: "#2a2e37", greyB: "#555b68", fadeTo: "#4a5266", ring: "#0e1014",
    path: "#a8b3cc", pathUnused: "#515764", reads: "#b7f542", feeds: "#ff5cf0", self: "#ffffff",
    roles: { embed: "#d1a67a", norm: "#8e96a5", attn_q: "#7c86ff", attn_k: "#9d82ff", attn_v: "#bd84ff", attn_out: "#2dd4bf",
      ff_up: "#fbd23b", ff_down: "#e8b13d", classifier: "#fb7185", span: "#6e7480", count: "#6e7480", other: "#6e7480" },
  },
  light: {
    bg: "#f6f5f1", panel: "rgba(255,255,255,0.94)", text: "#1a1d25", muted: "#4d5766", line: "#d6d6ce",
    zero: "#b4bac7", pos: "#d9561a", neg: "#1f6fd1", side: "#8c94a5", greyA: "#e1e1db", greyB: "#b0b2ab", fadeTo: "#d4d6dc", ring: "#ffffff",
    path: "#566683", pathUnused: "#a9acb3", reads: "#3f7d00", feeds: "#a21caf", self: "#111111",
    roles: { embed: "#a7763f", norm: "#69727f", attn_q: "#4a54e0", attn_k: "#7650e0", attn_v: "#a24fd6", attn_out: "#0d9488",
      ff_up: "#d99a00", ff_down: "#a8790f", classifier: "#e11d48", span: "#9aa0aa", count: "#9aa0aa", other: "#9aa0aa" },
  },
};
const ROLE_ORDER = ["embed", "attn_q", "attn_k", "attn_v", "attn_out", "ff_up", "ff_down", "norm", "classifier", "span", "count", "other"];
const ROLE_NAME = { embed: "Word and position tables", attn_q: "Attention: query", attn_k: "Attention: key", attn_v: "Attention: value",
  attn_out: "Attention: output", ff_up: "Feed-forward: expand", ff_down: "Feed-forward: shrink", norm: "Normalisation",
  classifier: "Output head", span: "Span head", count: "Count head", other: "Other" };
let themeName = "dark", T = THEMES.dark;
const fmt = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n));
const col = (h) => new THREE.Color(h);

// ---- renderer -----------------------------------------------------------------------------------------------
const coarse = matchMedia("(pointer: coarse)").matches;
const renderer = new THREE.WebGLRenderer({ antialias: !(coarse && devicePixelRatio >= 2), powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, coarse ? 1.5 : 2));
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:1";   // one stacking layer, below the panels
view.appendChild(labels.domElement);
const scene3 = new THREE.Scene();
scene3.background = col(T.bg);
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 40000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.screenSpacePanning = true; controls.zoomToCursor = true;

// Draw only when something changed, and only while the frame is on screen.
let needsDraw = true, raf = 0, onScreen = true;
function request() { needsDraw = true; if (!raf && onScreen) raf = requestAnimationFrame(frame); }
controls.addEventListener("change", request);

// ---- the cubes: one instanced mesh for the used parts, one (see-through) for the unused --------------------
const vert = `
  attribute vec4 iUv; attribute float iRole; attribute vec4 iState; attribute float iId;
  varying vec2 vUv; varying float vFront; varying vec4 vRect; varying float vRole; varying vec4 vState; varying vec2 vSize; varying float vId;
  void main() {
    vUv = uv; vFront = step(0.5, normal.z); vRect = iUv; vRole = iRole; vState = iState; vId = iId;
    vSize = vec2(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz));
    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  }`;
const frag = `
  uniform sampler2D uAtlas; uniform vec3 uZero, uPos, uNeg, uSide, uGreyA, uGreyB, uBg, uReads, uFeeds, uSelf, uFadeTo, uRing;
  uniform vec3 uRoles[12]; uniform float uMode; uniform float uCellId; uniform vec2 uCell; uniform float uAlpha;
  varying vec2 vUv; varying float vFront; varying vec4 vRect; varying float vRole; varying vec4 vState; varying vec2 vSize; varying float vId;
  void main() {
    // the picture of this tensor's weights: u across, t down the atlas (t counts from the top of the image)
    float v = texture2D(uAtlas, vec2(mix(vRect.x, vRect.z, vUv.x), mix(vRect.w, vRect.y, vUv.y))).r * 2.0 - 1.0;
    vec3 role = uRoles[int(vRole + 0.5)];
    vec3 weights = v < 0.0 ? mix(uZero, uNeg, -v) : mix(uZero, uPos, v);
    vec3 front = uMode > 0.5 ? weights : role * (0.80 + 0.28 * (v * 0.5 + 0.5));
    vec3 sideC = uMode > 0.5 ? uSide : role * 0.55;
    vec3 grey = mix(uGreyA, uGreyB, abs(v));
    front = mix(front, grey, vState.x); sideC = mix(sideC, mix(uGreyA, uGreyB, 0.25), vState.x);
    vec3 c = vFront > 0.5 ? front : sideC;
    // what is not the selected stage or beside it goes grey, so the selection stands out
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(c, mix(vec3(l) * 0.6, uFadeTo, 0.55), vState.y);
    c = mix(c, vec3(1.0), vState.w * 0.18);
    float kind = vState.z;   // 1 reads from, 2 feeds, 3 the selected stage: an outline round the face, a tint on the sides
    if (kind > 0.5) {
      vec3 halo = kind < 1.5 ? uReads : kind < 2.5 ? uFeeds : uSelf;
      vec2 d = min(vUv, 1.0 - vUv) * vSize;
      float edge = min(d.x, d.y);
      float w = kind > 2.5 ? 1.0 : 0.8;   // a ring that stands off any cube colour: contrast line outside, colour inside
      if (vFront > 0.5 && edge < w) c = edge < 0.3 ? uRing : halo; else if (vFront < 0.5) c = mix(c, halo, kind > 2.5 ? 0.2 : 0.55);
    }
    if (uCellId == vId && vFront > 0.5) {   // the hovered weight: its row and its column
      vec2 px = max(vec2(0.003), 0.5 / (vSize * 14.0));
      float onRow = 1.0 - step(px.y, abs(vUv.y - uCell.y)), onCol = 1.0 - step(px.x, abs(vUv.x - uCell.x));
      c = mix(c, vec3(1.0), max(onRow, onCol) * 0.6);
    }
    gl_FragColor = linearToOutputTexel(vec4(c, mix(1.0, uAlpha, vState.x)));
  }`;
const U = {   // uniforms every cube shares
  uAtlas: { value: null }, uZero: { value: col(T.zero) }, uPos: { value: col(T.pos) }, uNeg: { value: col(T.neg) }, uSide: { value: col(T.side) },
  uGreyA: { value: col(T.greyA) }, uGreyB: { value: col(T.greyB) }, uBg: { value: col(T.bg) }, uReads: { value: col(T.reads) }, uFadeTo: { value: col(T.fadeTo) }, uRing: { value: col(T.ring) },
  uFeeds: { value: col(T.feeds) }, uSelf: { value: col(T.self) }, uRoles: { value: ROLE_ORDER.map((r) => col(T.roles[r])) },
  uMode: { value: q.get("colour") === "weights" ? 1 : 0 }, uCellId: { value: -1 }, uCell: { value: new THREE.Vector2() }, uAlpha: { value: 0.5 },
};
const cubeMaterial = (unused) => new THREE.ShaderMaterial({ uniforms: U, vertexShader: vert, fragmentShader: frag,
  transparent: unused, depthWrite: !unused });

let model, meshes = [], items = [];    // meshes[0] used, meshes[1] unused; items[g] = the item at each instance
const where = [];                      // item index -> [mesh, instance]
const modBox = [];                     // module -> its Box3
const groupObjs = [], layerLabels = [], extraLabels = [];
let showUnused = true, showLinks = true, values = null;
const stateArr = [];                   // per mesh: the Float32Array behind iState

function buildCubes(data) {
  const atlas = new THREE.TextureLoader().load(data.atlas.file, request);
  atlas.flipY = false; atlas.magFilter = THREE.NearestFilter; atlas.minFilter = THREE.LinearMipmapLinearFilter;
  atlas.colorSpace = THREE.NoColorSpace; atlas.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
  U.uAtlas.value = atlas;
  const T3 = data.thick || 2;
  const groupsOf = [[], []];
  data.items.forEach((it, k) => groupsOf[data.groups[it.group].unused ? 1 : 0].push(k));
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), p3 = new THREE.Vector3(), s3 = new THREE.Vector3();
  const flatQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)), idQ = new THREE.Quaternion();
  groupsOf.forEach((list, mi) => {
    const g = geo.clone();
    const uv = new Float32Array(list.length * 4), role = new Float32Array(list.length), st = new Float32Array(list.length * 4), id = new Float32Array(list.length);
    const mesh = new THREE.InstancedMesh(g, cubeMaterial(mi === 1), list.length);
    items[mi] = list.map((k) => data.items[k]);
    list.forEach((k, j) => {
      const it = data.items[k];
      p3.set(it.pos[0], it.pos[1], it.flat ? it.pos[2] : it.pos[2] - T3 / 2);   // a standing grid's front face stays at its z
      s3.set(it.size[0], it.size[1], T3);
      m4.compose(p3, it.flat ? flatQ : idQ, s3); mesh.setMatrixAt(j, m4);
      uv.set(it.uv, j * 4);
      role[j] = Math.max(0, ROLE_ORDER.indexOf(data.modules[it.mod].role)); st[j * 4] = mi; id[j] = k;   // state.x = 1 for the unused
      where[k] = [mi, j];
    });
    g.setAttribute("iUv", new THREE.InstancedBufferAttribute(uv, 4));
    g.setAttribute("iRole", new THREE.InstancedBufferAttribute(role, 1));
    const sa = new THREE.InstancedBufferAttribute(st, 4); sa.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("iState", sa); stateArr[mi] = sa;
    g.setAttribute("iId", new THREE.InstancedBufferAttribute(id, 1));
    mesh.frustumCulled = false; mesh.renderOrder = mi;
    mesh.userData.unused = mi === 1;
    meshes[mi] = mesh; scene3.add(mesh);
  });
  // each module's box, from its items (a flat table lies on its back)
  data.modules.forEach((m, i) => {
    const b = new THREE.Box3();
    for (let k = m.items[0]; k < m.items[1]; k++) {
      const it = data.items[k], [w, h] = it.size;
      const lo = it.flat ? [it.pos[0] - w / 2, it.pos[1] - T3 / 2, it.pos[2] - h / 2] : [it.pos[0] - w / 2, it.pos[1] - h / 2, it.pos[2] - T3];
      const hi = it.flat ? [it.pos[0] + w / 2, it.pos[1] + T3 / 2, it.pos[2] + h / 2] : [it.pos[0] + w / 2, it.pos[1] + h / 2, it.pos[2]];
      b.expandByPoint(new THREE.Vector3(...lo)); b.expandByPoint(new THREE.Vector3(...hi));
    }
    modBox[i] = b;
  });
}

// ---- data flow --------------------------------------------------------------------------------------------
// At rest only the main path shows: group to group and layer to layer, as a few thick lines. A stage's own links
// (reads from, feeds, the residual) appear when it is hovered or pinned.
const hidden = (e) => e.u && !showUnused;   // a link into an unused part goes with it
let restMeshes = [], traceMesh = null, inE = [], outE = [];
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
function ribbonMesh(acc, extra) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(acc.pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(acc.col, 3));
  g.setIndex(acc.idx);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, ...extra }));
  m.frustumCulled = false; return m;
}
const SIDE = -1;   // which side of the stack a climbing link bows to
// A link that ends higher than it starts (layer N's bottom to layer N+1's top: 135 units up) would climb straight past the head
// it is about to reach. It bows out to the side in x instead, clears the stack, and comes down into the head from above.
// Start and end stay where they are.
function climbCurve(p0, p1, A, B, side) {
  const rise = p1.y - p0.y, w = Math.max(A.max.x - A.min.x, B.max.x - B.min.x), off = side * (w / 2 + 6), h = 10, V = (x, y, z) => new THREE.Vector3(x, y, z);
  return new THREE.CatmullRomCurve3([p0, V(p0.x, p0.y - 6, p0.z), V(p0.x + off, p0.y + 0.3 * rise, p0.z), V(p1.x + off, p0.y + 0.8 * rise, p1.z), V(p1.x + off * 0.35, p1.y + h * 1.5, p1.z), V(p1.x, p1.y + h, p1.z), p1], false, "centripetal");
}
function curveBetween(A, B, kind) {
  const p0 = new THREE.Vector3((A.min.x + A.max.x) / 2, A.min.y, (A.min.z + A.max.z) / 2);
  const p1 = new THREE.Vector3((B.min.x + B.max.x) / 2, B.max.y, (B.min.z + B.max.z) / 2);
  const dy = Math.max(3, Math.abs(p0.y - p1.y) * 0.45);
  let c0 = p0.clone().add(new THREE.Vector3(0, -dy, 0)), c1 = p1.clone().add(new THREE.Vector3(0, dy, 0));
  if (kind === "skip") {   // the residual stream bows out to the side
    const x = Math.max(A.max.x, B.max.x) + 4 + Math.abs(p0.y - p1.y) * 0.1;
    c0 = new THREE.Vector3(x, p0.y - dy, p0.z); c1 = new THREE.Vector3(x, p1.y + dy, p1.z);
    p0.x = A.max.x; p1.x = B.max.x; p0.y = (A.min.y + A.max.y) / 2; p1.y = (B.min.y + B.max.y) / 2;
  }
  if (kind !== "skip" && p1.y - p0.y > 20) return climbCurve(p0, p1, A, B, SIDE);
  return new THREE.CubicBezierCurve3(p0, c0, c1, p1);
}
function prepareLinks(data) {
  inE = data.modules.map(() => []); outE = data.modules.map(() => []);
  data.edges.forEach((e, k) => { outE[e.a].push(k); inE[e.b].push(k); });
  buildRestPath();
}
function buildRestPath() {
  restMeshes.forEach((m) => { scene3.remove(m); m.geometry.dispose(); m.material.dispose(); });
  restMeshes = [];
  const groups = new Map();   // one line per (source stage, target layer or group): to the box round all the stages it feeds there
  for (const e of model.edges) {
    const a = model.modules[e.a], b = model.modules[e.b];
    if (e.k !== "flow" || (a.group === b.group && a.copy === b.copy)) continue;
    const key = `${e.a}|${b.copy ?? "g" + b.group}|${e.u ? 1 : 0}`;
    if (!groups.has(key)) groups.set(key, { a: e.a, bs: [], u: e.u });
    groups.get(key).bs.push(e.b);
  }
  const acc = [{ pos: [], col: [], idx: [] }, { pos: [], col: [], idx: [] }];
  for (const g of groups.values()) {
    const B = new THREE.Box3(); g.bs.forEach((i) => B.union(modBox[i]));
    ribbon(acc[g.u ? 1 : 0], curveBetween(modBox[g.a], B, "flow"), 0.5, col(g.u ? T.pathUnused : T.path), true);
  }
  acc.forEach((a, i) => { if (!a.pos.length) return; const m = ribbonMesh(a); m.userData.unused = i === 1; m.renderOrder = 2; scene3.add(m); restMeshes.push(m); });
  applyVisibility();
}
function applyVisibility() {
  for (const m of restMeshes) m.visible = showLinks && (!m.userData.unused || showUnused);
  if (traceMesh) traceMesh.visible = showLinks;
  if (meshes[1]) meshes[1].visible = showUnused;
  for (const o of groupObjs) if (o.g.unused) { o.edges.visible = showUnused; if (!o.sub) o.lab.visible = showUnused; }
  request();
}

// What is traced: the pinned stage if there is one, else the one under the mouse. A tap pins (or unpins) a stage.
let pinned = -1, hoverMod = -1, traced = -1, hoverItem = -1;
const target = () => (pinned >= 0 ? pinned : hoverMod);
function refresh() { traced = -2; trace(target()); if (embedTouch) $("hint").hidden = controls.enabled || pinned >= 0; }   // on a touch embed the hint shows only while the model isn't taken and no card is pinned
function trace(mi) {
  if (mi === traced) return;
  traced = mi;
  const kindOf = new Map();   // module -> 1 reads from, 2 feeds
  const links = [];
  if (mi >= 0) {
    for (const k of inE[mi]) { const e = model.edges[k]; if (hidden(e)) continue; kindOf.set(e.a, 1); links.push([k, 1]); }
    for (const k of outE[mi]) { const e = model.edges[k]; if (hidden(e)) continue; if (!kindOf.has(e.b)) kindOf.set(e.b, 2); links.push([k, 2]); }
  }
  for (let k = 0; k < model.items.length; k++) {
    const [mm, j] = where[k], mod = model.items[k].mod, st = stateArr[mm].array;
    const kind = mod === mi ? 3 : kindOf.get(mod) || 0;
    st[j * 4 + 1] = mi >= 0 && kind === 0 ? 0.7 : 0;   // grey
    st[j * 4 + 2] = kind;                              // outline
    st[j * 4 + 3] = k === hoverItem ? 1 : 0;           // hover lift
  }
  stateArr.forEach((s) => { s.needsUpdate = true; });
  if (traceMesh) { scene3.remove(traceMesh); traceMesh.geometry.dispose(); traceMesh.material.dispose(); traceMesh = null; }
  if (links.length) {
    const acc = { pos: [], col: [], idx: [] };
    for (const [k, dir] of links) {
      const e = model.edges[k];
      ribbon(acc, curveBetween(modBox[e.a], modBox[e.b], e.k), 0.32, col(dir === 1 ? T.reads : T.feeds), true);
    }
    traceMesh = ribbonMesh(acc, { depthTest: false, transparent: true }); traceMesh.renderOrder = 10; traceMesh.visible = showLinks; scene3.add(traceMesh);
  }
  card(mi); request();
}
// the hovered item's lift without rebuilding the trace
function setHover(k) {
  if (k === hoverItem) return;
  for (const [i, v] of [[hoverItem, 0], [k, 1]]) if (i >= 0) { const [mm, j] = where[i]; stateArr[mm].array[j * 4 + 3] = v; stateArr[mm].needsUpdate = true; }
  hoverItem = k; request();
}

const word = (s) => s.replace(/\./g, ".​");   // a code name breaks at its dots, never mid-word
const titleOf = (i) => model.modules[i].title;
function listOf(set) { const a = [...set].map(titleOf); return a.slice(0, 3).join(", ") + (a.length > 3 ? ` +${a.length - 3} more` : ""); }
function flowText(mi) {
  const ups = new Set(), downs = new Set(), skipIn = new Set();
  for (const k of inE[mi]) { const e = model.edges[k]; if (hidden(e)) continue; (e.k === "skip" ? skipIn : ups).add(e.a); }
  for (const k of outE[mi]) { const e = model.edges[k]; if (hidden(e) || e.k === "skip") continue; downs.add(e.b); }
  return `<div class="f up"><i></i>reads from: ${ups.size ? listOf(ups) : "the input"}</div>` +
    (skipIn.size ? `<div class="f up"><i></i>plus the residual from: ${listOf(skipIn)}</div>` : "") +
    (downs.size ? `<div class="f down"><i></i>feeds: ${listOf(downs)}</div>` : "");
}
const num = (x) => (Math.abs(x) >= 100 || (Math.abs(x) < 0.01 && x !== 0) ? x.toExponential(2) : x.toPrecision(3));
function valuesText(mi) {   // what this stage produced for the example input: a plain sentence, the numbers behind a tap
  if (!values || mi < 0) return "";
  const m = model.modules[mi], st = values.stages[m.name];
  if (!st) return model.groups[m.group].unused ? `<div class="f v sum">On the example input: not run (this part is unused).</div>` : "";
  const sh = st.shape, toks = sh.length === 3 ? sh[1] : null;
  const what = sh.length === 3 ? `${sh[2].toLocaleString()} numbers for each of the ${toks} tokens` : `${sh.join(" × ")} numbers`;
  return `<div class="f v sum">On the example input it outputs ${what}, values centre near ${num(st.mean)} and spread about ±${num(st.std)}.</div>` +
    `<div class="f v num">shape ${sh.join(" × ")} · mean ${num(st.mean)} · std ${num(st.std)} · from ${num(st.min)} to ${num(st.max)} · first cells ${st.first.slice(0, 4).map(num).join(", ")}…</div>`;
}
const shapeText = (mi) => { const m = model.modules[mi], st = values && values.stages[m.name]; return (st ? st.shape : model.items[m.items[0]].shape).join(" × "); };
function card(mi) {   // one line by default (the stage and its shape); a tap opens what it reads, feeds and output
  const el = $("trace");
  if (mi < 0) { el.hidden = true; return; }
  el.hidden = false;
  if (el.dataset.mi !== String(mi)) { el.classList.remove("open"); el.dataset.mi = mi; }
  el.onclick = () => el.classList.toggle("open");
  const m = model.modules[mi];
  el.innerHTML = `<div class="t"><b></b><span class="shape"></span><span class="pin"></span></div><div class="det"><div class="code"></div>${flowText(mi)}${valuesText(mi)}` +
    `<div class="k"><span class="o up"></span>input to it <span class="o down"></span>reads its output <span class="o self"></span>this stage</div></div>`;
  el.querySelector("b").textContent = m.title;
  el.querySelector(".shape").textContent = shapeText(mi);
  el.querySelector(".code").innerHTML = word(m.name.replace(/&/g, "&amp;").replace(/</g, "&lt;"));
  el.querySelector(".pin").textContent = mi === pinned ? "pinned · tap again to clear" : "tap to pin";
}

// ---- groups, labels ---------------------------------------------------------------------------------------
const roleOfGroup = (name, unused) => (unused ? "other" : /Tokeniz/.test(name) ? "embed" : /Encoder/.test(name) ? "attn_q" : /Classifier/.test(name) ? "classifier" : "other");
function groupColour(g, topName) {   // the top group's role from scene.json (build.py sets it); older scenes guess from the name
  if (g.unused) return T.pathUnused;
  const top = model.groups.find((x) => x.outline && x.name === topName);
  return T.roles[(top && top.role) || roleOfGroup(topName, false)] || T.roles.other;
}
function topOf(gi) {
  let g = model.groups[gi], top = g.outline ? gi : -1;
  while (g.parent !== null && g.parent !== undefined) { if (model.groups[g.parent].outline) top = g.parent; g = model.groups[g.parent]; }
  return top;
}
function buildGroups(data) {
  data.groups.forEach((g, gi) => {
    if (!g.outline) return;
    const [x, y, z, w, h, d] = g.box, top = topOf(gi);
    const box = new THREE.Box3(new THREE.Vector3(x, y - h, z - d), new THREE.Vector3(x + w, y, z));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, Math.max(d, 0.01))),
      g.unused ? new THREE.LineDashedMaterial({ dashSize: 2, gapSize: 1.5, transparent: true, opacity: 0.8 })
        : new THREE.LineBasicMaterial({ transparent: true, opacity: g.path.length === 1 ? 0.95 : 0.5 }));
    edges.computeLineDistances(); box.getCenter(edges.position); scene3.add(edges);
    const el = document.createElement("div");
    el.className = "label" + (g.unused ? " unused" : "");
    el.textContent = g.name + (g.unused ? " · unused" : "");
    el.onclick = () => focus(gi);
    const lab = new CSS2DObject(el);
    if (g.path.length === 1) { lab.position.set(x, y, z); lab.center.set(0, 0); }
    else { lab.position.set(x + 0.8, y - 0.8, z); lab.center.set(0, 1); el.classList.add("sub"); }
    scene3.add(lab);
    groupObjs.push({ gi, g, box, edges, lab, top, topName: data.groups[top] ? data.groups[top].name : g.name, sub: g.path.length > 1 });
    if (g.path.length > 1) lab.visible = false;   // part names inside a group appear only when close
  });
}
function addLabel(text, pos, cls, center) {
  const el = document.createElement("div"); el.className = "label " + cls; el.textContent = text;
  const lab = new CSS2DObject(el); lab.position.copy(pos); lab.center.set(...(center || [0, 0])); scene3.add(lab);
  return { el, lab };
}
function buildDetailLabels(data) {
  // layer numbers and the attention / feed-forward parts appear as the camera comes close
  const byCopy = new Map(), byPart = new Map(), spine = new THREE.Box3();   // the labels line up along the left edge of the whole stack
  data.modules.forEach((m, i) => {
    if (m.copy == null) return;
    for (const [map, key] of [[byCopy, m.copy], [byPart, m.copy + "|" + m.part]]) {
      if (!map.has(key)) map.set(key, new THREE.Box3()); map.get(key).union(modBox[i]);
    }
    spine.union(modBox[i]);
  });
  for (const [c, b] of byCopy) { const l = addLabel(`Layer ${c}`, new THREE.Vector3(spine.min.x - 3, b.max.y, b.max.z), "layer", [1, 0]); layerLabels.push({ ...l, level: 1, copy: +c }); }
  for (const [k, b] of byPart) {
    const [c, part] = k.split("|");
    const l = addLabel(part === "attention" ? "attention" : "feed-forward", new THREE.Vector3(spine.min.x - 3, (b.min.y + b.max.y) / 2, b.max.z), "part", [1, 0.5]);
    layerLabels.push({ ...l, level: 2, copy: +c });
  }
  layerLabels.forEach((l) => { l.lab.visible = false; });
  // where a post goes in and where the decision comes out
  const emb = groupObjs.find((o) => /Tokeniz/.test(o.g.name || "")), cls = groupObjs.find((o) => /Classifier/.test(o.g.name || ""));
  if (emb) extraLabels.push(addLabel("Post in ↓", new THREE.Vector3(emb.box.min.x - 2, emb.box.max.y + 3, emb.box.max.z), "io", [1, 1]));
  if (cls) extraLabels.push(addLabel("Decision out ↓", new THREE.Vector3((cls.box.min.x + cls.box.max.x) / 2, cls.box.min.y - 2, cls.box.max.z), "io", [0.5, 0]));
}
let homeDist = 0;
function updateDetail() {   // which detail labels may show: the selected layer's, and a few when close; the culling in declutter() does the rest
  if (!homeDist) return;
  const r = camera.position.distanceTo(controls.target) / homeDist;
  const sel = target() >= 0 ? model.modules[target()].copy : null;
  for (const l of layerLabels) {
    if (l.level === 1) { const mine = sel != null && sel === l.copy, sparse = r < 0.5 && (l.copy === 0 || l.copy === 11 || l.copy === 23); l.lab.visible = mine || sparse; l.rank = mine ? 3 : 4; }
    else l.lab.visible = sel != null && sel === l.copy;
  }
  for (const o of groupObjs) if (o.sub) o.lab.visible = r < 0.5 && (!o.g.unused || showUnused);
}

// ---- theme --------------------------------------------------------------------------------------------------
function applyTheme(name) {
  themeName = name; T = THEMES[name];
  const root = document.documentElement; root.dataset.theme = name;
  for (const k of ["bg", "panel", "text", "muted", "line", "reads", "feeds", "pos", "neg", "zero", "path", "self"]) root.style.setProperty("--" + k, T[k]);
  scene3.background.set(T.bg);
  U.uZero.value.set(T.zero); U.uPos.value.set(T.pos); U.uNeg.value.set(T.neg); U.uSide.value.set(T.side); U.uGreyA.value.set(T.greyA);
  U.uGreyB.value.set(T.greyB); U.uBg.value.set(T.bg); U.uReads.value.set(T.reads); U.uFeeds.value.set(T.feeds); U.uSelf.value.set(T.self); U.uFadeTo.value.set(T.fadeTo); U.uRing.value.set(T.ring);
  ROLE_ORDER.forEach((r, i) => U.uRoles.value[i].set(T.roles[r]));
  if (model) {
    for (const o of groupObjs) { const c = o.g.unused ? T.pathUnused : groupColour(o.g, o.topName); o.edges.material.color.set(c); o.lab.element.style.borderColor = c; o.colour = c; }
    buildRestPath(); refresh(); legendColours(); renderKey();
  }
  const tb = $("themebtn"); if (tb) tb.textContent = name === "dark" ? "Light mode" : "Dark mode";
  request();
}
const mq = matchMedia("(prefers-color-scheme: light)");
const explicit = q.get("theme") === "light" || q.get("theme") === "dark";
applyTheme(explicit ? q.get("theme") : mq.matches ? "light" : "dark");
mq.addEventListener("change", () => { if (!explicit && !userChose) applyTheme(mq.matches ? "light" : "dark"); });
let userChose = false;
addEventListener("message", (e) => {   // an article's own light/dark toggle
  const d = e.data;
  if (d && d.type === "explorer-theme" && (d.value === "light" || d.value === "dark")) { userChose = true; applyTheme(d.value); }
});

// ---- panels -----------------------------------------------------------------------------------------------
function legend() {   // collapsed: the parts' colours and names only; "Details" opens the rest
  const el = $("legend"); el.hidden = false; el.classList.remove("open");
  el.innerHTML = `<h1>${model.model}</h1><div class="sub">${model.total.toLocaleString()} weights in ${model.tensors} tensors. Drag to turn, scroll to zoom, tap a part to fly to it.</div>`;
  const tops = []; model.groups.forEach((g, gi) => { if (g.outline && g.path.length === 1) tops.push(gi); });
  for (const gi of tops) {
    const g = model.groups[gi], b = document.createElement("button");
    b.className = "g" + (g.unused ? " unused" : ""); b.dataset.gi = gi;
    b.innerHTML = `<span class="sw"></span><span class="n"></span><span class="p">${fmt(g.params)} · ${(100 * g.params / model.total).toFixed(1)}%</span>`;
    b.querySelector(".n").textContent = g.name; b.onclick = () => focus(gi); el.appendChild(b);
  }
  const row = document.createElement("div"); row.className = "row";
  row.innerHTML = `<label><input type="checkbox"> Show unused parts</label><label class="lk"><input type="checkbox" checked> Connections</label>` +
    `<span class="seg" role="group" aria-label="Colour by"><button data-m="0">Parts</button><button data-m="1">Weights</button></span>` +
    `<button class="reset" id="themebtn"></button><button class="reset" id="homebtn">Whole model</button>`;
  row.querySelector("input").onchange = (e) => setUnused(e.target.checked);
  row.querySelector(".lk input").onchange = (e) => { showLinks = e.target.checked; applyVisibility(); };
  row.querySelectorAll(".seg button").forEach((b) => { b.onclick = () => { U.uMode.value = +b.dataset.m; segState(); renderKey(); request(); }; });
  row.querySelector("#themebtn").onclick = () => { userChose = true; applyTheme(themeName === "dark" ? "light" : "dark"); };
  row.querySelector("#homebtn").onclick = () => home();
  el.appendChild(row);
  const kb = document.createElement("div"); kb.id = "keybox"; el.appendChild(kb);
  const tog = document.createElement("button"); tog.className = "tog"; tog.textContent = "Details ▾"; tog.setAttribute("aria-expanded", "false");
  tog.onclick = () => { const o = el.classList.toggle("open"); tog.textContent = o ? "Hide ▴" : "Details ▾"; tog.setAttribute("aria-expanded", String(o)); };
  el.appendChild(tog);
  segState(); legendColours();
  $("themebtn").textContent = themeName === "dark" ? "Light mode" : "Dark mode";
}
function segState() { document.querySelectorAll("#legend .seg button").forEach((b) => b.classList.toggle("on", +b.dataset.m === U.uMode.value)); }
function legendColours() {
  document.querySelectorAll("#legend button.g").forEach((b) => {
    const g = model.groups[+b.dataset.gi], o = groupObjs.find((o) => o.gi === +b.dataset.gi);
    b.querySelector(".sw").style.background = g.unused ? T.pathUnused : o ? o.colour || groupColour(g, g.name) : "#888";
  });
}
const litCss = (hex) => { const c = new THREE.Color(hex).multiplyScalar(0.94); return "#" + c.getHexString(); };   // the colour a cube's front face really shows: role x the grain's average
function renderKey() {   // built from the same colours the shader uses
  const el = $("keybox");
  const weights = U.uMode.value > 0.5;
  const lines = `<div class="kl"><span class="o up"></span>input to the selected stage<br><span class="o down"></span>reads the selected stage's output<br><span class="line"></span>main path through the model</div>`;
  if (weights) {
    el.innerHTML = `<b>Each cell is one weight (sampled)</b><div class="bar" style="background:linear-gradient(90deg,${T.neg},${T.zero} 50%,${T.pos})"></div>` +
      `<div class="ends"><span>negative</span><span>0</span><span>positive</span></div>${lines}`;
  } else {
    const used = new Set(model.modules.map((m) => m.role));
    el.innerHTML = `<b>Colour shows what a part does</b>` + ROLE_ORDER.filter((r) => used.has(r) && r !== "other").map((r) =>
      `<div class="rk"><span class="sw" style="background:${litCss(T.roles[r])}"></span>${ROLE_NAME[r]}</div>`).join("") +
      `<div class="sub">The grain in each cube is its real weights.</div>${lines}`;
  }
}
function showValuesNote() {
  document.querySelectorAll("#legend .vnote").forEach((n) => n.remove());
  const p = document.createElement("details"); p.className = "vnote";
  p.innerHTML = `<summary>${values.title || "About the example input"}</summary><div></div>`;
  p.querySelector("div").textContent = values.note || `A real run of this model (${values.dtype}, ${values.device}) on “${values.text}”${values.tokens ? ", " + values.tokens.length + " tokens" : ""}. Hover a stage for what it output.`;
  const el = $("legend"); el.insertBefore(p, el.querySelector(".row"));
}

function setUnused(show) {
  showUnused = show;
  const cb = document.querySelector("#legend input"); if (cb) cb.checked = show;
  if (!show) {   // a selection inside a part that has just been hidden goes with it
    const gone = (mi) => mi >= 0 && model.groups[model.modules[mi].group].unused;
    if (gone(pinned)) pinned = -1;
    if (gone(hoverMod)) hoverMod = -1;
    if (hoverItem >= 0 && model.groups[model.items[hoverItem].group].unused) hoverItem = -1;
  }
  applyVisibility(); refresh();
}

// ---- camera ----------------------------------------------------------------------------------------------
let fly = null;
const compact = () => view.clientWidth <= 560 || view.clientHeight <= 520;   // a phone, or a short frame such as an article embed
function reserve() {   // what the panels take: the model is framed in what is left, so nothing sits over it
  const W = view.clientWidth, H = view.clientHeight, lg = $("legend");
  let left = 14, right = 14, top = embedTouch ? 92 : 60, bottom = 14;   // top: the one-line trace card (and the Done button on a touch screen)
  if (lg && !lg.hidden) { if (compact()) bottom = lg.offsetHeight + 18; else left = lg.offsetWidth + 30; }
  return { left, right, top, bottom, W, H };
}
function fit(box, instant) {
  const s = box.getSize(new THREE.Vector3());
  const front = box.clone();
  front.min.z = Math.max(box.min.z, box.max.z - Math.max(s.x, s.y) * 0.8);   // a very deep part is framed by its front
  const target = front.getCenter(new THREE.Vector3());
  const dir = (camera.aspect < 1 ? new THREE.Vector3(0.3, 0.25, 1) : new THREE.Vector3(0.55, 0.32, 1)).normalize();
  const right = new THREE.Vector3(0, 1, 0).cross(dir).normalize(), up = dir.clone().cross(right);
  const R = reserve(), uW = Math.max(0.35, (R.W - R.left - R.right) / R.W), uH = Math.max(0.35, (R.H - R.top - R.bottom) / R.H);
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * uH, tanH = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * camera.aspect * uW;   // the free area, not the whole screen
  let dist = 0;
  for (let i = 0; i < 8; i++) {
    const p = new THREE.Vector3(i & 1 ? front.max.x : front.min.x, i & 2 ? front.max.y : front.min.y, i & 4 ? front.max.z : front.min.z).sub(target);
    dist = Math.max(dist, p.dot(dir) + Math.max(Math.abs(p.dot(up)) / tanV, Math.abs(p.dot(right)) / tanH));
  }
  dist *= 1.04;
  const pos = target.clone().addScaledVector(dir, dist);
  if (instant) { camera.position.copy(pos); controls.target.copy(target); controls.update(); request(); return dist; }
  fly = { t0: performance.now(), p0: camera.position.clone(), q0: controls.target.clone(), p1: pos, q1: target }; request();
  return dist;
}
function home(instant) {   // the used parts, on every screen; the unused heads are one tick away
  const b = new THREE.Box3();
  for (const o of groupObjs) if (o.g.path.length === 1 && o.edges.visible && !(o.g.unused && !showUnused)) b.union(o.box);
  const d = fit(b, instant); if (!homeDist) homeDist = d; highlight(-1);
}
function focus(gi, instant) {
  const o = groupObjs.find((o) => o.gi === gi);
  if (!o) return;
  if (o.g.unused && !showUnused) setUnused(true);
  fit(o.box, instant); highlight(gi);
}
function highlight(gi) { document.querySelectorAll("#legend button.g").forEach((b) => b.classList.toggle("on", +b.dataset.gi === gi)); }

// ---- embedded in an article: the page keeps its scroll until the reader clicks into the model ---------------
// On a touch screen the model does not take the finger until the reader taps it: until then a vertical swipe over it scrolls the
// article (touch-action: pan-y). A tap takes the touch for turning and pinching; "Done" gives scrolling back.
const embedTouch = q.has("embed") && matchMedia("(pointer: coarse)").matches;
let ownTouch = () => {};
if (q.has("embed")) {
  controls.enableZoom = false;
  const hint = $("hint"), done = $("done"), canvas = renderer.domElement;
  hint.hidden = false;
  hint.textContent = embedTouch ? "Tap the model to explore: drag to turn, pinch to zoom" : "Click to explore: drag to turn, scroll to zoom";
  if (embedTouch) {
    controls.enabled = false; canvas.style.touchAction = "pan-y";
    ownTouch = () => { controls.enabled = true; controls.enableZoom = true; canvas.style.touchAction = "none"; hint.hidden = true; done.hidden = false; };
    done.onclick = () => { controls.enabled = false; controls.enableZoom = false; canvas.style.touchAction = "pan-y"; done.hidden = true; hint.hidden = pinned >= 0; };   // not over a pinned card
  } else canvas.addEventListener("pointerdown", () => { controls.enableZoom = true; hint.hidden = true; }, { once: true });
}

// ---- the first frame after the view has sat still is slow (the GPU has gone quiet: 10-48 ms on an M1 in a headless run).
// Pay it when the pointer arrives, before the first movement, so a drag doesn't start with one late frame.
let lastDraw = 0;
function warm() {
  if (!model || performance.now() - lastDraw < 1500) return;
  renderer.render(scene3, camera); lastDraw = performance.now();
}
for (const ev of ["pointerenter", "pointerdown", "touchstart", "wheel"]) renderer.domElement.addEventListener(ev, warm, { passive: true });

// ---- pointer: one pick per animation frame -----------------------------------------------------------------
const tip = $("tip"), ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
let down = null, pendingMove = null, lastTip = "";
function rayHit(e) {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const h = ray.intersectObjects(showUnused ? meshes : [meshes[0]], false)[0];
  if (!h) return null;
  const mi = h.object === meshes[0] ? 0 : 1, item = items[mi][h.instanceId];
  return { item, k: model.items.indexOf(item), face: h.face, uv: h.uv, r };
}
function hoverAt(e) {
  const hit = rayHit(e);
  if (!hit) { setHover(-1); hoverMod = -1; trace(target()); tip.style.display = "none"; lastTip = ""; U.uCellId.value = -1; return; }
  const it = hit.item;
  setHover(hit.k);
  hoverMod = it.mod; trace(target());
  const g = model.groups[it.group], m = model.modules[it.mod];
  let cellTxt = "", cellKey = "";
  if (hit.face && hit.face.normal.z > 0.5 && it.kind === "grid" && !it.flat) {   // the cell under the pointer: its row is one output, its column one input
    const rows = it.shape[0], cols = it.params / rows, u = hit.uv.x, v = hit.uv.y;
    U.uCellId.value = hit.k; U.uCell.value.set(u, v); request();
    const r = Math.min(rows - 1, Math.floor((1 - v) * rows)), c = Math.min(cols - 1, Math.floor(u * cols));
    cellKey = r + "," + c;
    cellTxt = `<div class="f">cell: output ${r.toLocaleString()} of ${rows.toLocaleString()}, input ${c.toLocaleString()} of ${cols.toLocaleString()}: this weight scales that input in that output</div>`;
  } else U.uCellId.value = -1;
  const key = hit.k + "|" + cellKey + "|" + traced;
  if (key !== lastTip) {
    lastTip = key;
    tip.innerHTML = `<b></b><div class="code"></div><div class="m"></div>${cellTxt}${flowText(it.mod)}${valuesText(it.mod)}`;
    tip.querySelector("b").textContent = m.title;
    tip.querySelector(".code").innerHTML = word(it.name.replace(/&/g, "&amp;").replace(/</g, "&lt;"));
    tip.querySelector(".m").textContent = `${it.shape.join(" × ")}${it.shape.length < 2 ? " × 1" : ""} = ${it.params.toLocaleString()} weights · ` +
      g.path.filter(Boolean).join(" › ") + (g.unused ? " (unused)" : "") + (it.cut ? ` · drawn shortened (${it.cut.rows.toLocaleString()} rows in all)` : "") + ` · spread ±${it.std.toPrecision(2)}`;
  }
  tip.style.display = "block";
  const x = Math.min(e.clientX - hit.r.left + 14, hit.r.width - tip.offsetWidth - 8);
  const y = Math.min(e.clientY - hit.r.top + 14, hit.r.height - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
renderer.domElement.addEventListener("pointermove", (e) => {
  // a gesture that travels more than 6px at any point is a drag, even if it comes back
  if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) down.moved = true;
  if (e.pointerType === "mouse" || e.buttons === 0) { pendingMove = e; if (!raf) raf = requestAnimationFrame(frame); }
});
renderer.domElement.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now(), moved: false }; });
renderer.domElement.addEventListener("pointerup", (e) => {
  const d = down; down = null;
  if (!d || d.moved || performance.now() - d.t > 500) return;   // a drag only rotates
  pendingMove = null;
  const hit = rayHit(e), mod = hit ? hit.item.mod : -1;
  if (embedTouch && hit && !controls.enabled) ownTouch();   // the reader tapped the model: it takes the touch now
  pinned = mod >= 0 && mod !== pinned ? mod : -1;
  const mouseLike = e.pointerType === "mouse";
  hoverMod = mouseLike ? mod : -1;   // a finger doesn't hover: an unpin clears the trace
  if (!mouseLike) setHover(-1);
  refresh();
  if (!mouseLike) tip.style.display = "none";
});
renderer.domElement.addEventListener("pointercancel", () => { down = null; });
renderer.domElement.addEventListener("pointerleave", () => {
  pendingMove = null; tip.style.display = "none"; lastTip = ""; U.uCellId.value = -1; setHover(-1);
  hoverMod = -1; trace(target());
});

// ---- labels that land on top of each other on screen are pushed apart (the bigger group's label stays put) --
// Run once the camera has settled, not on every frame.
let settleTimer = 0;
function settleSoon() { clearTimeout(settleTimer); settleTimer = setTimeout(() => { updateDetail(); labels.render(scene3, camera); declutter(); }, 120); }
function declutter() {
  // Priority: Post in / Decision out, group names, the selected layer (and its parts), a few layer numbers, part names inside a group.
  // A label that touches another label or a panel is hidden, at most 8 show, and the group names may slide to find room.
  const cands = [...extraLabels.map((l) => ({ el: l.el, vis: true, rank: 1 })), ...groupObjs.map((o) => ({ el: o.lab.element, vis: o.lab.visible, rank: o.sub ? 5 : 2 })),
    ...layerLabels.map((l) => ({ el: l.el, vis: l.lab.visible, rank: l.level === 1 ? l.rank || 4 : 3 }))].filter((o) => o.vis).sort((a, b) => a.rank - b.rank);
  for (const o of cands) { o.el.style.marginTop = "0px"; o.el.style.marginLeft = "0px"; o.el.style.visibility = ""; }
  const rects = cands.map((o) => o.el.getBoundingClientRect());   // read every position first, then write
  const panels = ["legend", "trace"].map($).filter((e) => e && !e.hidden).map((e) => e.getBoundingClientRect());
  const W = view.clientWidth, H = view.clientHeight, g = 3, placed = [], writes = [];
  const clash = (a, list) => list.some((p) => a.left - g < p.right && a.right + g > p.left && a.top - g < p.bottom && a.bottom + g > p.top);
  cands.forEach((o, i) => {
    const r = rects[i]; let dx = 0, dy = 0;
    const at = () => ({ left: r.left + dx, right: r.right + dx, top: r.top + dy, bottom: r.bottom + dy });
    if (o.rank <= 2) {   // these may slide: sideways off a panel, then down off a neighbour
      for (const pr of panels) if (clash(at(), [pr])) dx = pr.right + g + 4 - r.left;
      if (r.right + dx > W - 4) dx = W - 4 - r.right; else if (r.left + dx < 4) dx = 4 - r.left;
      for (let k = 0; k < 14 && clash(at(), placed); k++) { const hit = placed.find((p) => clash(at(), [p])); dy = hit.bottom + g - r.top + 1; }
      if (r.bottom + dy > H - 4) dy = H - 4 - r.bottom;
    }
    const a = at(), bad = placed.length >= 8 || clash(a, placed) || clash(a, panels) || a.left < 0 || a.right > W || a.top < 0 || a.bottom > H;
    if (!bad) placed.push(a);
    writes.push([o.el, dx, dy, bad]);
  });
  for (const [el, dx, dy, bad] of writes) { el.style.marginLeft = dx + "px"; el.style.marginTop = dy + "px"; el.style.visibility = bad ? "hidden" : ""; }
}

// ---- loop ------------------------------------------------------------------------------------------------
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h); labels.setSize(w, h);
  camera.aspect = w / h;
  const R = reserve();   // look through a window shifted so the model centres in the free area
  camera.setViewOffset(w, h, -Math.round((R.left - R.right) / 2), -Math.round((R.top - R.bottom) / 2), w, h);
  camera.updateProjectionMatrix(); request(); settleSoon();
}
addEventListener("resize", resize);
resize();
function frame(now) {
  raf = 0;
  if (!onScreen) return;
  if (pendingMove) { const e = pendingMove; pendingMove = null; hoverAt(e); }
  if (fly) {
    const t = Math.min(1, (now - fly.t0) / 700), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    camera.position.lerpVectors(fly.p0, fly.p1, e); controls.target.lerpVectors(fly.q0, fly.q1, e);
    if (t >= 1) fly = null;
    needsDraw = true;
  }
  const moved = controls.update();
  if (needsDraw) {
    renderer.render(scene3, camera); lastDraw = performance.now(); labels.render(scene3, camera); needsDraw = false;
    if (!moved && !fly) { clearTimeout(settleTimer); updateDetail(); labels.render(scene3, camera); declutter(); }   // at rest: place the labels now
    else settleSoon();
  }
  if (fly || moved) raf = requestAnimationFrame(frame);
}

// ---- start --------------------------------------------------------------------------------------------------
function build(data) {
  performance.mark("build-start");
  model = data;
  buildCubes(data); buildGroups(data); prepareLinks(data); buildDetailLabels(data);
  legend(); renderKey(); if (values) showValuesNote();
  $("status").remove();
  const f = q.get("focus");
  const fi = f ? data.groups.findIndex((g) => g.name && g.name.toLowerCase() === f.toLowerCase()) : -1;
  applyTheme(themeName);   // outlines and links take the theme's colours
  setUnused(q.get("unused") === "show");
  resize();
  home(true);   // (sets how far the whole-model view is, for the detail labels)
  if (fi >= 0) focus(fi, true);
  performance.mark("build-end");
  const cam = q.get("cam");   // ?cam=x,y,z,tx,ty,tz: the camera's place and what it looks at (for linking to a view)
  if (cam) { const c = cam.split(",").map(Number); camera.position.set(c[0], c[1], c[2]); controls.target.set(c[3], c[4], c[5]); controls.update(); request(); }
  const tr = q.get("trace");   // ?trace=<part of a tensor name>: opens with that stage pinned
  if (tr) { const mi = data.modules.findIndex((m) => m.name.includes(tr)); if (mi >= 0) { pinned = mi; refresh(); } }
  updateDetail(); request();
}
const loadValues = fetch(q.get("values") || "values.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
loadValues.then((v) => { values = v; if (values && model) { showValuesNote(); refresh(); } });
function start() {
  fetch(q.get("scene") || "scene.json").then((r) => r.json()).then(build).catch((e) => { $("status").textContent = "Couldn't load the model: " + e.message; });
}
if (q.has("embed") && "IntersectionObserver" in window) {   // in an article: load when the frame scrolls into view, pause when it leaves
  let started = false;
  new IntersectionObserver((es) => {
    onScreen = es[es.length - 1].isIntersecting;
    if (onScreen) { if (!started) { started = true; start(); } request(); }
  }, { threshold: 0.05 }).observe(view);
} else start();
function screenBox() {   // the model's outline on screen (used groups), in view pixels
  const b = { l: 1e9, t: 1e9, r: -1e9, b: -1e9 }, v = new THREE.Vector3(), w = view.clientWidth, h = view.clientHeight;
  for (const o of groupObjs) {
    if (o.sub || (o.g.unused && !showUnused)) continue;
    for (let i = 0; i < 8; i++) {
      v.set(i & 1 ? o.box.max.x : o.box.min.x, i & 2 ? o.box.max.y : o.box.min.y, i & 4 ? o.box.max.z : o.box.min.z).project(camera);
      const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h;
      b.l = Math.min(b.l, x); b.r = Math.max(b.r, x); b.t = Math.min(b.t, y); b.b = Math.max(b.b, y);
    }
  }
  return b;
}
window.__explorer = { warm, screenBox, renderer, scene3, meshes: () => meshes, trace, refresh, THREE, camera, controls, focus, home, get model() { return model; }, get modBox() { return modBox; },
  state: () => ({ pinned, hoverMod, traced }), setUnused, pin: (mi) => { pinned = mi; refresh(); }, applyTheme, request };
if (q.has("selftest")) import("./selftest.js");
if (q.has("perf")) import("./perf.js");
