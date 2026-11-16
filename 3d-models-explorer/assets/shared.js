--- shared.js ---
// The shared view: everyone with this space open sees the same model. Whoever turns, zooms or taps it moves
// it for the rest, through the space's relay room (api/ws, worker.js). Camera frames go out at most 20 a second
// and are eased into on arrival; the pinned stage, unused parts, connections and colours follow at once.
// Late joiners ask the room's oldest tab for the view. Theme and the Details panel stay each reader's own.
const ME = window.SS_VIEWER || {};
const MY_NAME = ME.name || (ME.username && !ME.username_auto ? "@" + ME.username : "Someone");
const GRACE_MS = 1200;   // after a reader lets go, the camera's damped glide is still theirs

// ?ssdebug: a small log of the room's messages, for checking the sync in a headless capture.
const LOG = new URLSearchParams(location.search).has("ssdebug") ? Object.assign(document.body.appendChild(document.createElement("pre")),
  { style: "position:absolute;left:8px;bottom:8px;z-index:9;margin:0;padding:6px;font:11px/1.3 monospace;background:#000c;color:#9f9;max-width:60%;white-space:pre-wrap" }) : null;
// The headless capture waits while data-ss-math reads "loading": with ?ssdebug, hold it 4 s past joining the room.
if (LOG) document.documentElement.setAttribute("data-ss-math", "loading");
const log = (t) => { if (LOG) LOG.textContent = (LOG.textContent + "\n" + Math.round(performance.now()) + " " + t).split("\n").slice(-14).join("\n"); };

const wait = setInterval(() => { const X = window.__explorer; if (X && X.model) { clearInterval(wait); start(X); } }, 100);

function start(X) {
  const { camera, controls, THREE } = X;
  const $ = (s) => document.querySelector(s);
  const pill = $("#who"), names = new Map();   // peer id -> their name
  let ws = null, opened = false, retry = 1000, driving = false, mineUntil = 0, lastSent = 0, trail = 0;
  let goal = null, easing = false, leader = "", leaderUntil = 0, leaderTimer = 0;
  const now = () => performance.now();
  const mine = () => driving || now() < mineUntil;
  const r3 = (v) => [v.x, v.y, v.z].map((n) => Math.round(n * 1000) / 1000);
  const pose = () => ({ p: r3(camera.position), t: r3(controls.target) });
  const vec = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite) ? new THREE.Vector3(a[0], a[1], a[2]) : null;
  const send = (data, to) => { if (data.k !== "cam") log("send " + data.k + (to ? " to " + to : "")); if (ws && ws.readyState === 1) ws.send(JSON.stringify(to ? { to, data } : data)); };

  // ---- the camera: sent while it is this reader's, eased into when it is someone else's ----
  function sendCam() { lastSent = now(); trail = 0; send({ k: "cam", n: MY_NAME, ...pose() }); }
  controls.addEventListener("start", () => { driving = true; goal = null; });
  controls.addEventListener("end", () => { driving = false; mineUntil = now() + GRACE_MS; });
  $("#legend").addEventListener("click", (e) => {   // a part's button or "Whole model" flies the camera
    if (e.isTrusted && e.target.closest("button.g, #homebtn")) { mineUntil = now() + GRACE_MS; goal = null; }
  }, true);
  controls.addEventListener("change", () => {
    if (!mine()) return;
    const gap = 50 - (now() - lastSent);
    if (gap <= 0) sendCam(); else if (!trail) trail = setTimeout(sendCam, gap);
  });
  function ease() {
    if (!goal || mine()) { easing = false; return; }
    camera.position.lerp(goal.p, 0.3); controls.target.lerp(goal.t, 0.3); controls.update();
    if (camera.position.distanceToSquared(goal.p) < 1e-4 && controls.target.distanceToSquared(goal.t) < 1e-4) goal = null;
    requestAnimationFrame(ease);
  }
  function follow(p, t, who) {
    goal = { p, t }; leader = who; leaderUntil = now() + 1500;
    clearTimeout(leaderTimer); leaderTimer = setTimeout(paint, 1600);
    if (!easing) { easing = true; requestAnimationFrame(ease); }
  }

  // ---- the rest of the shared state: read from the explorer, sent when it changes ----
  const ui = () => {
    const seg = $("#legend .seg button.on"), un = $("#legend .row input"), lk = $("#legend .lk input");
    return { pin: X.state().pinned, unused: !!(un && un.checked), links: !lk || lk.checked, mode: seg ? +seg.dataset.m : 0 };
  };
  let lastUi = JSON.stringify(ui());
  setInterval(() => { const s = JSON.stringify(ui()); if (s !== lastUi) { lastUi = s; send({ k: "ui", n: MY_NAME, ...JSON.parse(s) }); } }, 150);
  function applyUi(d) {
    const cur = ui();
    if (typeof d.unused === "boolean" && d.unused !== cur.unused) X.setUnused(d.unused);   // before the pin: hiding may unpin
    if (typeof d.links === "boolean" && d.links !== cur.links) { const c = $("#legend .lk input"); if (c) { c.checked = d.links; c.dispatchEvent(new Event("change")); } }
    if ((d.mode === 0 || d.mode === 1) && d.mode !== cur.mode) { const b = $(`#legend .seg button[data-m="${d.mode}"]`); if (b) b.click(); }
    if (Number.isInteger(d.pin) && d.pin >= -1 && d.pin < X.model.modules.length && d.pin !== X.state().pinned) X.pin(d.pin);
    lastUi = JSON.stringify(ui());
  }

  // ---- the room ----
  function onData(from, d) {
    if (!d || typeof d !== "object") return;
    if (typeof d.n === "string" && d.n) names.set(from, d.n.slice(0, 40));
    const who = names.get(from) || "Someone";
    if (d.k === "hi" && !d.re) send({ k: "hi", n: MY_NAME, re: 1 }, from);
    else if (d.k === "ask") send({ k: "state", n: MY_NAME, cam: pose(), ui: ui() }, from);
    else if (d.k === "state") {
      const p = d.cam && vec(d.cam.p), t = d.cam && vec(d.cam.t);
      log("state: cam " + !!(p && t) + ", mine " + mine() + ", ui " + JSON.stringify(d.ui));
      if (p && t && !mine()) { camera.position.copy(p); controls.target.copy(t); controls.update(); X.request(); }
      if (d.ui) applyUi(d.ui);
    } else if (d.k === "cam") {
      const p = vec(d.p), t = vec(d.t);
      if (p && t && !mine()) follow(p, t, who);
      if (LOG) { clearTimeout(window.__camLog); window.__camLog = setTimeout(() => log("last cam " + JSON.stringify({ p: d.p, t: d.t }) + " now " + JSON.stringify(pose())), 600); }
    } else if (d.k === "ui") { applyUi(d); leader = who; leaderUntil = now() + 1500; clearTimeout(leaderTimer); leaderTimer = setTimeout(paint, 1600); }
    paint();
  }
  function connect() {
    const u = new URL("api/ws", location.href);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(u);
    ws.onmessage = (e) => {
      let m; try { m = JSON.parse(e.data); } catch { return; }
      log("got " + (m.sys ? m.sys + " id=" + m.id + (m.host ? " host=" + m.host : "") : (m.data && m.data.k) + " from " + m.from));
      if (m.sys === "welcome") {
        retry = 1000; send({ k: "hi", n: MY_NAME });
        if (LOG) setTimeout(() => document.documentElement.removeAttribute("data-ss-math"), +new URLSearchParams(location.search).get("ssdebug") || 4000);
        if (LOG && new URLSearchParams(location.search).has("ssspin")) setTimeout(spin, 500);
        if (m.host && m.host !== m.id) send({ k: "ask" }, m.host);   // the room's oldest tab has the view
      } else if (m.sys === "leave") names.delete(m.id);
      else if (m.from && m.from !== "server") onData(m.from, m.data);
      paint();
    };
    ws.onopen = () => { opened = true; paint(); };
    ws.onclose = () => { names.clear(); paint(); setTimeout(connect, retry); retry = Math.min(retry * 2, 15000); };
  }

  function spin() {   // ?ssdebug&ssspin: turn the model for 3 s as a drag would, to check that the others follow
    const t0 = now(); driving = true;
    (function step() {
      const off = camera.position.clone().sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.02);
      camera.position.copy(controls.target).add(off); controls.update();
      if (now() - t0 < 3000) requestAnimationFrame(step); else { driving = false; mineUntil = now() + GRACE_MS; log("spin done " + JSON.stringify(pose())); }
    })();
  }

  // ---- who is here, and who is moving it ----
  function paint() {
    pill.hidden = false;
    const live = ws && ws.readyState === 1, others = [...names.values()];
    pill.classList.toggle("on", live);
    const lead = live && now() < leaderUntil ? leader : "";
    pill.querySelector(".t").textContent = !live ? (opened ? "Shared view: reconnecting…" : "Shared view: connecting…")
      : lead ? `${lead} is moving it`
      : others.length ? `Shared view · you and ${others.length === 1 ? others[0] : others.length + " others"}`
      : "Shared view · just you so far";
    pill.title = live ? "Whoever turns, zooms or taps the model moves it for everyone who has this space open." +
      (others.length ? "\nHere: " + others.join(", ") : "") : "";
  }
  function place() {   // on a phone the legend sits at the bottom: stay just above it
    const lg = $("#legend"), narrow = innerWidth <= 560 || innerHeight <= 520;
    pill.style.bottom = narrow && lg && !lg.hidden ? lg.offsetHeight + 16 + "px" : "";
  }
  new ResizeObserver(place).observe($("#legend")); addEventListener("resize", place);
  place(); paint(); connect();
}
