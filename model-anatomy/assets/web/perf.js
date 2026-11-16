// ?perf measures load and frame costs in the real page and writes them into #perf (for headless runs).
const E = window.__explorer, canvas = document.querySelector("#view canvas");
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const t = (f, n) => { const a = performance.now(); for (let i = 0; i < n; i++) f(i); return (performance.now() - a) / n; };
(async () => {
  await wait(4000);   // let the textures arrive
  const res = performance.getEntriesByType("resource");
  const png = res.filter((r) => r.name.endsWith(".png"));
  const m0 = performance.getEntriesByName("build-start")[0], m1 = performance.getEntriesByName("build-end")[0];
  const R = canvas.getBoundingClientRect();
  const move = (i) => canvas.dispatchEvent(new PointerEvent("pointermove", { clientX: R.left + 200 + (i * 7) % 700, clientY: R.top + 150 + (i * 11) % 500, pointerType: "mouse", buttons: 0, bubbles: true }));
  const info = E.renderer.info;
  E.renderer.render(E.scene3, E.camera);
  const out = {
    requests: res.length, pngRequests: png.length, bytesKB: Math.round(res.reduce((a, r) => a + (r.encodedBodySize || 0), 0) / 1024),
    buildMs: m0 && m1 ? +(m1.startTime - m0.startTime).toFixed(1) : null,
    drawCalls: info.render.calls, triangles: info.render.triangles, textures: info.memory.textures, programs: info.programs ? info.programs.length : null,
    renderMsAvg: +t(() => { E.renderer.render(E.scene3, E.camera); E.renderer.getContext().finish(); }, 20).toFixed(2),
    hoverMsAvg: +t(move, 200).toFixed(3),
    traceSwitchMsAvg: +t((i) => { E.pin(i % 150); }, 40).toFixed(2),
  };
  const pre = document.createElement("pre"); pre.id = "perf"; pre.textContent = JSON.stringify(out); document.body.appendChild(pre);
})();
