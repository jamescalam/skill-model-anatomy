// ?selftest runs touch and mouse sequences against the real page and writes the results into #selftest (for headless checks).
const E = window.__explorer, canvas = document.querySelector("#view canvas");
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const ev = (type, x, y, o = {}) => canvas.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, pointerId: 1, bubbles: true, pointerType: o.pt || "touch", buttons: type === "pointerup" ? 0 : 1, ...o }));
const frame = () => wait(80);
const R = canvas.getBoundingClientRect();
const card = () => document.getElementById("trace");
const state = () => ({ pinned: E.state().pinned, cardHidden: card().hidden, card: card().hidden ? "" : card().querySelector(".pin").textContent });
const tap = async (x, y, pt = "touch") => { ev("pointerdown", x, y, { pt }); ev("pointerup", x, y, { pt }); await frame(); };
const pass = (ok) => (ok ? "PASS" : "FAIL");
(async () => {
  await wait(1500);
  // a point on the focused layers, and a point with nothing under it
  const EMPTY = { x: R.left + R.width - 3, y: R.top + R.height * 0.5 };
  // a point on the model, for any layout: try a grid over the model's outline on screen until a tap pins a stage
  let P = { x: R.left + R.width * 0.55, y: R.top + R.height * 0.5 };
  const sb = E.screenBox();
  search: for (const fy of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]) for (const fx of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]) {
    const c = { x: R.left + sb.l + (sb.r - sb.l) * fx, y: R.top + sb.t + (sb.b - sb.t) * fy };
    await tap(c.x, c.y); const hit = state().pinned >= 0; await tap(EMPTY.x, EMPTY.y);
    if (hit) { P = c; break search; }
  }
  const out = [];
  const add = (name, ok, s) => out.push(`${pass(ok)}  ${name}  ${JSON.stringify(s)}`);
  await tap(P.x, P.y); let s = state(); const hit = s.pinned;
  add("first tap pins a stage and the card says pinned", hit >= 0 && !s.cardHidden && /^pinned/.test(s.card), s);
  await tap(P.x, P.y); s = state(); add("tap the same stage again (touch): unpinned and the card is gone", s.pinned === -1 && s.cardHidden, s);
  await tap(P.x, P.y); await tap(EMPTY.x, EMPTY.y); s = state(); add("pin, then tap empty space: cleared", s.pinned === -1 && s.cardHidden, s);
  ev("pointerdown", P.x, P.y); ev("pointermove", P.x + 30, P.y, { buttons: 1 }); ev("pointermove", P.x, P.y, { buttons: 1 }); ev("pointerup", P.x, P.y); await frame();
  s = state(); add("drag 30px out and back, release: not pinned", s.pinned === -1 && s.cardHidden, s);
  ev("pointerdown", P.x, P.y); ev("pointermove", P.x + 20, P.y, { buttons: 1 }); ev("pointerup", P.x + 20, P.y); await frame();
  s = state(); add("drag ending 20px away: not pinned", s.pinned === -1 && s.cardHidden, s);
  await tap(P.x, P.y); const used = state().pinned;   // (the drags above turned the camera a little, so this may be another stage)
  E.setUnused(true); E.setUnused(false); await frame();
  s = state(); add("a pinned used stage survives hiding the unused parts", used >= 0 && s.pinned === used, { used, ...s });
  await tap(EMPTY.x, EMPTY.y);
  const unusedMod = E.model.modules.findIndex((m) => E.model.groups[m.group] && E.model.groups[m.group].unused);
  if (unusedMod >= 0) {   // only for a model with a part marked unused
    E.setUnused(true); E.pin(unusedMod); await frame();
    const before = state().pinned; E.setUnused(false); await frame();
    s = state(); add("hiding the unused parts clears a pinned unused stage", before >= 0 && s.pinned === -1 && s.cardHidden, { before, ...s });
  } else out.push("SKIP  hiding the unused parts clears a pinned unused stage  (no part is marked unused)");
  await tap(P.x, P.y, "mouse"); s = state(); const mp = s.pinned;
  await tap(P.x, P.y, "mouse"); s = state();
  add("mouse: first click pins, second unpins (hover trace stays while the pointer is over it)", mp >= 0 && s.pinned === -1, { pinnedAfterFirst: mp, ...s });
  const pre = document.createElement("pre"); pre.id = "selftest"; pre.textContent = out.join("\n"); document.body.appendChild(pre);
})();
