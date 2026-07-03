/* URBAN-MTMC M0 dashboard — pure consumer of the schema-v1 event stream.
   Contract: CONTRACT.md §web + src/mtmc/events.py. Plain ES6, canvas 2D,
   zero external requests. Relies only on contract fields: global_id may be
   null/absent; floor_xy may be null (observation counted, dot skipped). */
"use strict";

(() => {
  // ---- constants ----------------------------------------------------------
  const PALETTE = [           // per-camera colors, assigned by plan order
    "#4da6ff", "#ffb454", "#7bd88f", "#ff6188",
    "#b48cff", "#ffd866", "#5ad4e6",
  ];
  const STALE_S = 3;          // drop a tracklet dot unseen this long (sim s)
  const TRAIL_S = 4;          // fading trail window (sim s)
  const GT_PATH_S = 8;        // ground-truth path window per global_id (sim s)
  const MARGIN_PX = 26;       // letterbox margin around the plan
  const HOVER_PX = 12;        // hit-test radius for the hover label
  const LERP_RATE = 9;        // 1/s; dots close ~83% of the gap per 200 ms tick
  const RAD = Math.PI / 180;
  const TAU = Math.PI * 2;
  const FONT_S = '9px ui-monospace, "SF Mono", Menlo, Consolas, monospace';
  const FONT_M = '10px ui-monospace, "SF Mono", Menlo, Consolas, monospace';
  const FONT_L = '11px ui-monospace, "SF Mono", Menlo, Consolas, monospace';

  // ---- dom ----------------------------------------------------------------
  const stage = document.getElementById("stage");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const elChip = document.getElementById("chip");
  const elClock = document.getElementById("clock");
  const elRate = document.getElementById("rate");
  const elTotal = document.getElementById("total");
  const elCams = document.getElementById("cams");

  // ---- state --------------------------------------------------------------
  let plan = null;
  let view = null;            // {scale, ox, oy, w, h} in CSS px
  let dpr = 1;
  let staticLayer = null;     // offscreen canvas: floor plan + camera cones
  const camColor = new Map(); // camera id -> palette color
  const camCountEls = new Map();
  const markers = new Map();  // "camera:track_id" -> marker (the pool)
  const seenKeys = new Set(); // every (camera, track_id) ever observed
  const gtPaths = new Map();  // global_id -> [{x, y, t}] true recent path
  const gtColors = new Map();
  const rateWindow = [];      // {t: wall ms, n: obs in tick} for obs/s
  let latestTs = 0;           // newest sim timestamp seen (shared clock)
  let gtOn = false;
  let mouse = null;           // {x, y} CSS px or null
  let lastFrameMs = 0;
  let lastStatsMs = 0;

  const px = (wx) => view.ox + wx * view.scale;
  const py = (wy) => view.oy + wy * view.scale;

  // ---- static layer: plan rendered to scale, rebuilt on resize ------------
  function resize() {
    const r = stage.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    if (!plan) return;
    const [pw, ph] = plan.size_m;
    const scale = Math.max(
      1e-3,
      Math.min((r.width - 2 * MARGIN_PX) / pw, (r.height - 2 * MARGIN_PX) / ph),
    );
    view = {
      scale,
      ox: (r.width - pw * scale) / 2,
      oy: (r.height - ph * scale) / 2,
      w: r.width,
      h: r.height,
    };
    buildStaticLayer();
  }

  function buildStaticLayer() {
    staticLayer = document.createElement("canvas");
    staticLayer.width = canvas.width;
    staticLayer.height = canvas.height;
    const s = staticLayer.getContext("2d");
    s.setTransform(dpr, 0, 0, dpr, 0, 0);

    s.fillStyle = "#060a10";
    s.fillRect(0, 0, view.w, view.h);

    // plan boundary
    const [pw, ph] = plan.size_m;
    s.strokeStyle = "#1c2836";
    s.lineWidth = 1;
    s.strokeRect(px(0), py(0), pw * view.scale, ph * view.scale);

    // walkable areas: slightly lighter than bg
    s.fillStyle = "#0c141f";
    for (const a of plan.walkable ?? []) {
      s.fillRect(px(a.x), py(a.y), a.w * view.scale, a.h * view.scale);
    }

    // shops: darker outlined rects with tiny labels
    s.font = FONT_S;
    for (const sh of plan.shops ?? []) {
      const x = px(sh.x), y = py(sh.y);
      const w = sh.w * view.scale, h = sh.h * view.scale;
      s.fillStyle = "#080d14";
      s.fillRect(x, y, w, h);
      s.strokeStyle = "#1c2836";
      s.strokeRect(x, y, w, h);
      s.fillStyle = "#40536a";
      s.fillText(sh.name ?? "", x + 4, y + 11);
    }

    // waypoint graph: very subtle dotted edges + faint nodes
    const wps = plan.waypoints ?? {};
    s.strokeStyle = "rgba(216,226,234,0.09)";
    s.lineWidth = 1;
    s.setLineDash([2, 5]);
    s.beginPath();
    for (const [a, b] of plan.edges ?? []) {
      const pa = wps[a], pb = wps[b];
      if (!pa || !pb) continue;
      s.moveTo(px(pa[0]), py(pa[1]));
      s.lineTo(px(pb[0]), py(pb[1]));
    }
    s.stroke();
    s.setLineDash([]);
    s.fillStyle = "rgba(216,226,234,0.12)";
    for (const p of Object.values(wps)) {
      s.beginPath();
      s.arc(px(p[0]), py(p[1]), 1.5, 0, TAU);
      s.fill();
    }

    // cameras: translucent wedge cone + position dot + id label.
    // yaw_deg (0 = +x, positive clockwise) matches canvas angles exactly
    // because the y axis points down in both systems.
    s.font = FONT_M;
    for (const cam of plan.cameras ?? []) {
      const color = camColor.get(cam.id) ?? "#d8e2ea";
      const cx = px(cam.pos[0]), cy = py(cam.pos[1]);
      const a0 = (cam.yaw_deg - cam.fov_deg / 2) * RAD;
      const a1 = (cam.yaw_deg + cam.fov_deg / 2) * RAD;
      s.beginPath();
      s.moveTo(cx, cy);
      s.arc(cx, cy, cam.range_m * view.scale, a0, a1);
      s.closePath();
      s.fillStyle = color;
      s.globalAlpha = 0.08;
      s.fill();
      s.globalAlpha = 0.3;
      s.strokeStyle = color;
      s.stroke();
      s.globalAlpha = 1;
      s.beginPath();
      s.arc(cx, cy, 3.5, 0, TAU);
      s.fillStyle = color;
      s.fill();
      s.strokeStyle = "#060a10";
      s.stroke();
      s.fillStyle = color;
      s.fillText(cam.id, cx + 6, cy - 6);
    }
  }

  // ---- websocket with backoff ---------------------------------------------
  let everConnected = false;
  let retryMs = 500;
  let reconnectTimer = null;

  function setChip(text, live) {
    elChip.textContent = text;
    elChip.classList.toggle("live", live);
    elChip.classList.toggle("down", !live && everConnected);
  }

  function scheduleReconnect() {
    if (reconnectTimer !== null) return;
    reconnectTimer = setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 8000);
  }

  function connect() {
    reconnectTimer = null;
    setChip(everConnected ? "RECONNECTING" : "CONNECTING", false);
    let ws;
    try {
      const proto = location.protocol === "https:" ? "wss://" : "ws://";
      ws = new WebSocket(proto + location.host + "/ws");
    } catch {
      scheduleReconnect();
      return;
    }
    ws.onopen = () => {
      everConnected = true;
      retryMs = 500;
      setChip("LIVE", true);
    };
    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg && msg.type === "tick") onTick(msg);
    };
    ws.onclose = () => {
      setChip("RECONNECTING", false);
      scheduleReconnect();
    };
  }

  // ---- tick ingest: move existing dots, never recreate ---------------------
  function onTick(msg) {
    const obs = Array.isArray(msg.observations) ? msg.observations : [];
    if (typeof msg.ts_s === "number") latestTs = Math.max(latestTs, msg.ts_s);
    rateWindow.push({ t: performance.now(), n: obs.length });

    const gtTick = new Map(); // global_id -> mean position this tick
    for (const o of obs) {
      if (!o || typeof o.camera !== "string" || typeof o.track_id !== "number") continue;
      const key = o.camera + ":" + o.track_id;
      seenKeys.add(key);
      if (!Array.isArray(o.floor_xy)) continue; // invariant 3: no geometry -> no dot
      const wx = o.floor_xy[0], wy = o.floor_xy[1];
      const ts = typeof o.ts_s === "number" ? o.ts_s : latestTs;
      let m = markers.get(key);
      if (!m) {
        m = { camera: o.camera, trackId: o.track_id, globalId: null,
              x: wx, y: wy, tx: wx, ty: wy, trail: [], lastTs: ts };
        markers.set(key, m);
      }
      m.tx = wx;
      m.ty = wy;
      m.lastTs = ts;
      m.globalId = o.global_id ?? null; // invariant 4: debug-only, may be null
      m.trail.push({ x: wx, y: wy, t: ts });
      if (m.globalId !== null) {
        const g = gtTick.get(m.globalId) ?? { sx: 0, sy: 0, n: 0, t: ts };
        g.sx += wx; g.sy += wy; g.n += 1;
        if (ts > g.t) g.t = ts;
        gtTick.set(m.globalId, g);
      }
    }
    // one averaged point per person per tick -> the true continuous path,
    // even where two cameras hold two separate noisy tracklets
    for (const [gid, g] of gtTick) {
      let path = gtPaths.get(gid);
      if (!path) { path = []; gtPaths.set(gid, path); }
      path.push({ x: g.sx / g.n, y: g.sy / g.n, t: g.t });
    }
  }

  // ---- render loop ---------------------------------------------------------
  function prune() {
    for (const [key, m] of markers) {
      if (latestTs - m.lastTs > STALE_S) { markers.delete(key); continue; }
      while (m.trail.length && latestTs - m.trail[0].t > TRAIL_S) m.trail.shift();
    }
    for (const [gid, path] of gtPaths) {
      while (path.length && latestTs - path[0].t > GT_PATH_S) path.shift();
      if (path.length === 0) gtPaths.delete(gid);
    }
  }

  function gtColor(gid) {
    let c = gtColors.get(gid);
    if (!c) {
      c = `hsl(${(gid * 137.508) % 360} 75% 62%)`; // golden-angle hue spread
      gtColors.set(gid, c);
    }
    return c;
  }

  function colorFor(m) {
    if (gtOn && m.globalId !== null) return gtColor(m.globalId);
    return camColor.get(m.camera) ?? "#d8e2ea";
  }

  function drawGtPaths() {
    ctx.lineWidth = 1.25;
    for (const [gid, path] of gtPaths) {
      if (path.length < 2) continue;
      ctx.strokeStyle = gtColor(gid);
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(px(path[0].x), py(path[0].y));
      for (let i = 1; i < path.length; i++) ctx.lineTo(px(path[i].x), py(path[i].y));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawTrails() {
    ctx.lineWidth = 1;
    for (const m of markers.values()) {
      const tr = m.trail;
      if (tr.length < 2) continue;
      ctx.strokeStyle = colorFor(m);
      for (let i = 1; i < tr.length; i++) {
        const age = latestTs - tr[i].t;
        ctx.globalAlpha = Math.max(0, 1 - age / TRAIL_S) * 0.45;
        ctx.beginPath();
        ctx.moveTo(px(tr[i - 1].x), py(tr[i - 1].y));
        ctx.lineTo(px(tr[i].x), py(tr[i].y));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawDots() {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(216,226,234,0.55)";
    for (const m of markers.values()) {
      ctx.beginPath();
      ctx.arc(px(m.x), py(m.y), 4, 0, TAU);
      ctx.fillStyle = colorFor(m);
      ctx.fill();
      ctx.stroke();
    }
  }

  function drawHover() {
    if (!mouse) return;
    let best = null;
    let bestD2 = HOVER_PX * HOVER_PX;
    for (const m of markers.values()) {
      const dx = px(m.x) - mouse.x, dy = py(m.y) - mouse.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = m; }
    }
    if (!best) return;
    const x = px(best.x), y = py(best.y);
    let label = best.camera + ":" + best.trackId;
    if (gtOn && best.globalId !== null) label += "  g" + best.globalId;
    ctx.font = FONT_L;
    const w = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(6,10,16,0.88)";
    ctx.fillRect(x + 8, y - 23, w + 10, 16);
    ctx.strokeStyle = "#1c2836";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 8, y - 23, w + 10, 16);
    ctx.fillStyle = "#d8e2ea";
    ctx.fillText(label, x + 13, y - 11);
  }

  function updateStats() {
    const now = performance.now();
    while (rateWindow.length && now - rateWindow[0].t > 1000) rateWindow.shift();
    let rate = 0;
    for (const e of rateWindow) rate += e.n;
    elRate.textContent = String(rate);
    const t = Math.floor(latestTs);
    elClock.textContent =
      String(Math.floor(t / 60)).padStart(2, "0") + ":" + String(t % 60).padStart(2, "0");
    elTotal.textContent = String(seenKeys.size);
    const counts = new Map();
    for (const m of markers.values()) counts.set(m.camera, (counts.get(m.camera) ?? 0) + 1);
    for (const [id, el] of camCountEls) el.textContent = String(counts.get(id) ?? 0);
  }

  function frame(nowMs) {
    requestAnimationFrame(frame);
    const dt = Math.min((nowMs - lastFrameMs) / 1000, 0.1);
    lastFrameMs = nowMs;
    if (!view || !staticLayer) return;

    prune();
    const k = 1 - Math.exp(-LERP_RATE * dt); // frame-rate-independent glide
    for (const m of markers.values()) {
      m.x += (m.tx - m.x) * k;
      m.y += (m.ty - m.y) * k;
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(staticLayer, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (gtOn) drawGtPaths();
    drawTrails();
    drawDots();
    drawHover();

    if (nowMs - lastStatsMs > 250) {
      lastStatsMs = nowMs;
      updateStats();
    }
  }

  // ---- sidebar --------------------------------------------------------------
  function buildSidebar() {
    for (const cam of plan.cameras ?? []) {
      const row = document.createElement("div");
      row.className = "cam-row";
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = camColor.get(cam.id);
      const name = document.createElement("span");
      name.textContent = cam.id;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = "0";
      row.append(sw, name, count);
      elCams.append(row);
      camCountEls.set(cam.id, count);
    }
  }

  // ---- boot -----------------------------------------------------------------
  async function loadPlan() {
    for (;;) {
      try {
        const r = await fetch("/api/plan");
        if (r.ok) return await r.json();
      } catch { /* server not up yet */ }
      await new Promise((res) => setTimeout(res, 1000));
    }
  }

  async function main() {
    plan = await loadPlan();
    (plan.cameras ?? []).forEach((c, i) => camColor.set(c.id, PALETTE[i % PALETTE.length]));
    buildSidebar();
    resize();
    new ResizeObserver(resize).observe(stage);
    window.addEventListener("resize", resize); // also fires on zoom / dpr change

    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    });
    canvas.addEventListener("mouseleave", () => { mouse = null; });
    document.getElementById("gt").addEventListener("change", (e) => {
      gtOn = e.target.checked;
    });

    connect();
    requestAnimationFrame((t) => {
      lastFrameMs = t;
      requestAnimationFrame(frame);
    });
  }

  main();
})();
