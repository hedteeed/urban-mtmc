/* URBAN-MTMC M0 dashboard — pure consumer of the schema-v1 event stream.
   Contract: CONTRACT.md §web + src/mtmc/events.py. Plain ES6, canvas 2D,
   zero external requests. Plan schema v2 (multi-floor): every floor renders
   as its own to-scale panel; an observation's floor is derived from its
   camera via the plan (events carry no floor field). Relies only on contract
   fields: global_id may be null/absent; floor_xy may be null (observation
   counted, dot skipped). */
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
  let stageW = 0, stageH = 0; // canvas size in CSS px
  const views = new Map();    // floor id -> {scale, ox, oy, panel:{x,y,w,h}}
  let floorById = new Map();  // floor id -> floor object from the plan
  const camFloor = new Map(); // camera id -> floor id: routes obs to panels
  let dpr = 1;
  let staticLayer = null;     // ONE offscreen canvas: all floor panels + cones
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

  // world metres -> canvas CSS px, in a given floor's view transform
  const fx = (v, wx) => v.ox + wx * v.scale;
  const fy = (v, wy) => v.oy + wy * v.scale;

  // ---- panel layout + static layer: rebuilt on resize -----------------------
  // Every floor renders as its own to-scale panel: side by side when the
  // canvas is wide, stacked when narrow — pick whichever arrangement lets
  // the floors render larger.
  function fitScale(f, pw, ph) {
    return Math.min(
      (pw - 2 * MARGIN_PX) / f.size_m[0],
      (ph - 2 * MARGIN_PX) / f.size_m[1],
    );
  }

  function layoutPanels() {
    const floors = plan.floors ?? [];
    const n = Math.max(1, floors.length);
    let row = Infinity, col = Infinity;
    for (const f of floors) {
      row = Math.min(row, fitScale(f, stageW / n, stageH));
      col = Math.min(col, fitScale(f, stageW, stageH / n));
    }
    const asRow = row >= col;
    views.clear();
    floors.forEach((f, i) => {
      const panel = asRow
        ? { x: (i * stageW) / n, y: 0, w: stageW / n, h: stageH }
        : { x: 0, y: (i * stageH) / n, w: stageW, h: stageH / n };
      const scale = Math.max(1e-3, fitScale(f, panel.w, panel.h));
      views.set(f.id, {
        scale,
        ox: panel.x + (panel.w - f.size_m[0] * scale) / 2,
        oy: panel.y + (panel.h - f.size_m[1] * scale) / 2,
        panel,
      });
    });
  }

  function resize() {
    const r = stage.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    if (!plan) return;
    stageW = r.width;
    stageH = r.height;
    layoutPanels();
    buildStaticLayer();
  }

  function buildStaticLayer() {
    staticLayer = document.createElement("canvas");
    staticLayer.width = canvas.width;
    staticLayer.height = canvas.height;
    const s = staticLayer.getContext("2d");
    s.setTransform(dpr, 0, 0, dpr, 0, 0);

    s.fillStyle = "#060a10";
    s.fillRect(0, 0, stageW, stageH);

    const floors = plan.floors ?? [];
    for (const f of floors) drawFloorPanel(s, f, views.get(f.id));

    // hairline dividers between panels
    s.strokeStyle = "#131c28";
    s.lineWidth = 1;
    s.beginPath();
    for (const f of floors) {
      const p = views.get(f.id).panel;
      if (p.x > 0) { s.moveTo(p.x + 0.5, 0); s.lineTo(p.x + 0.5, stageH); }
      if (p.y > 0) { s.moveTo(0, p.y + 0.5); s.lineTo(stageW, p.y + 0.5); }
    }
    s.stroke();
  }

  // stairs footprint: distinct fill, step lines across the long axis, label
  function drawStairs(s, v, st) {
    const x = fx(v, st.x), y = fy(v, st.y);
    const w = st.w * v.scale, h = st.h * v.scale;
    s.fillStyle = "#0a1220";
    s.fillRect(x, y, w, h);
    s.strokeStyle = "#22354a";
    s.lineWidth = 1;
    s.strokeRect(x, y, w, h);
    s.strokeStyle = "rgba(216,226,234,0.14)";
    s.beginPath();
    if (h >= w) {
      const n = Math.max(2, Math.round(st.h / 0.35)); // one tread per ~0.35 m
      for (let i = 1; i < n; i++) {
        s.moveTo(x + 1, y + (h * i) / n);
        s.lineTo(x + w - 1, y + (h * i) / n);
      }
    } else {
      const n = Math.max(2, Math.round(st.w / 0.35));
      for (let i = 1; i < n; i++) {
        s.moveTo(x + (w * i) / n, y + 1);
        s.lineTo(x + (w * i) / n, y + h - 1);
      }
    }
    s.stroke();
    s.font = FONT_S;
    s.fillStyle = "#5a7089";
    s.fillText("STAIRS", x + 4, y + 11);
  }

  function drawFloorPanel(s, f, v) {
    const [fw, fh] = f.size_m;
    s.save();
    s.beginPath();
    s.rect(v.panel.x, v.panel.y, v.panel.w, v.panel.h);
    s.clip(); // camera cones near edges never bleed into a neighbouring panel

    // panel label, centred above the floor boundary
    s.font = FONT_L;
    s.letterSpacing = "3px"; // silent no-op where unsupported
    s.textAlign = "center";
    s.fillStyle = "#66788c";
    s.fillText(f.name ?? f.id, v.ox + (fw * v.scale) / 2, v.oy - 10);
    s.textAlign = "left";
    s.letterSpacing = "0px";

    // floor boundary
    s.strokeStyle = "#1c2836";
    s.lineWidth = 1;
    s.strokeRect(fx(v, 0), fy(v, 0), fw * v.scale, fh * v.scale);

    // walkable areas: slightly lighter than bg
    s.fillStyle = "#0c141f";
    for (const a of f.walkable ?? []) {
      s.fillRect(fx(v, a.x), fy(v, a.y), a.w * v.scale, a.h * v.scale);
    }

    // rooms are visual outlines OVER walkable space (schema v2): outline +
    // tiny label, no fill — a fill would hide the walkable lightening.
    s.font = FONT_S;
    const stairKeys = new Set(
      (f.stairs ?? []).map((t) => [t.x, t.y, t.w, t.h].join()),
    );
    for (const room of f.rooms ?? []) {
      if (stairKeys.has([room.x, room.y, room.w, room.h].join())) continue; // stairs label themselves
      const x = fx(v, room.x), y = fy(v, room.y);
      s.strokeStyle = "#1c2836";
      s.strokeRect(x, y, room.w * v.scale, room.h * v.scale);
      s.fillStyle = "#40536a";
      s.fillText(room.name ?? "", x + 4, y + 11);
    }

    // stairs footprints
    for (const st of f.stairs ?? []) drawStairs(s, v, st);

    // waypoint graph: very subtle dotted edges + faint nodes. Cross-floor
    // (stairs) edges would span panels and are left out of the static graph —
    // the stairs footprints on both panels carry that link visually.
    const wps = plan.waypoints ?? {};
    s.strokeStyle = "rgba(216,226,234,0.09)";
    s.lineWidth = 1;
    s.setLineDash([2, 5]);
    s.beginPath();
    for (const e of plan.edges ?? []) {
      const [a, b] = Array.isArray(e) ? e : [e.from, e.to];
      const pa = wps[a], pb = wps[b];
      if (!pa || !pb || pa.floor !== f.id || pb.floor !== f.id) continue;
      s.moveTo(fx(v, pa.xy[0]), fy(v, pa.xy[1]));
      s.lineTo(fx(v, pb.xy[0]), fy(v, pb.xy[1]));
    }
    s.stroke();
    s.setLineDash([]);
    s.fillStyle = "rgba(216,226,234,0.12)";
    for (const p of Object.values(wps)) {
      if (p.floor !== f.id) continue;
      s.beginPath();
      s.arc(fx(v, p.xy[0]), fy(v, p.xy[1]), 1.5, 0, TAU);
      s.fill();
    }

    // cameras on this floor: translucent wedge cone + position dot + label.
    // yaw_deg (0 = +x, positive clockwise) matches canvas angles exactly
    // because the y axis points down in both systems.
    s.font = FONT_M;
    for (const cam of plan.cameras ?? []) {
      if (cam.floor !== f.id) continue;
      const color = camColor.get(cam.id) ?? "#d8e2ea";
      const cx = fx(v, cam.pos[0]), cy = fy(v, cam.pos[1]);
      const a0 = (cam.yaw_deg - cam.fov_deg / 2) * RAD;
      const a1 = (cam.yaw_deg + cam.fov_deg / 2) * RAD;
      s.beginPath();
      s.moveTo(cx, cy);
      s.arc(cx, cy, cam.range_m * v.scale, a0, a1);
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
    s.restore();
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
      const floor = camFloor.get(o.camera);     // invariant 3: floor derives from camera
      if (floor === undefined) continue;        // camera not in plan -> no panel to route to
      const wx = o.floor_xy[0], wy = o.floor_xy[1];
      const ts = typeof o.ts_s === "number" ? o.ts_s : latestTs;
      let m = markers.get(key);
      if (!m) {
        m = { camera: o.camera, floor, trackId: o.track_id, globalId: null,
              x: wx, y: wy, tx: wx, ty: wy, trail: [], lastTs: ts };
        markers.set(key, m);
      }
      m.tx = wx;
      m.ty = wy;
      m.lastTs = ts;
      m.globalId = o.global_id ?? null; // invariant 4: debug-only, may be null
      m.trail.push({ x: wx, y: wy, t: ts });
      if (m.globalId !== null) {
        // Camera phase stagger means one tick frame CAN straddle a mid-stairs
        // floor flip — never blend positions from two per-floor frames. On a
        // floor mismatch keep whichever side the latest timestamp is on.
        const g = gtTick.get(m.globalId);
        if (!g || g.floor === floor) {
          const acc = g ?? { sx: 0, sy: 0, n: 0, t: ts, floor };
          acc.sx += wx; acc.sy += wy; acc.n += 1;
          if (ts > acc.t) acc.t = ts;
          gtTick.set(m.globalId, acc);
        } else if (ts >= g.t) {
          gtTick.set(m.globalId, { sx: wx, sy: wy, n: 1, t: ts, floor });
        }
      }
    }
    // one averaged point per person per tick -> the true continuous path,
    // even where two cameras hold two separate noisy tracklets
    for (const [gid, g] of gtTick) {
      let path = gtPaths.get(gid);
      if (!path) { path = []; gtPaths.set(gid, path); }
      path.push({ x: g.sx / g.n, y: g.sy / g.n, t: g.t, floor: g.floor });
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

  function stairsRect(floorId, toId) {
    for (const st of floorById.get(floorId)?.stairs ?? []) {
      if (st.to === toId) return st;
    }
    return null;
  }

  // GT handoff cue: when a true path crosses floors, a short dashed line in
  // that person's color joins the stairs footprint on one panel to the
  // matching footprint on the other.
  function drawStairConnector(fa, fb, gid) {
    const ra = stairsRect(fa, fb), rb = stairsRect(fb, fa);
    const va = views.get(fa), vb = views.get(fb);
    if (!ra || !rb || !va || !vb) return;
    const off = ((gid * 7) % 9) - 4; // fan out concurrent transits a little
    ctx.save();
    ctx.strokeStyle = gtColor(gid);
    ctx.globalAlpha = 0.65;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(fx(va, ra.x + ra.w / 2), fy(va, ra.y + ra.h / 2) + off);
    ctx.lineTo(fx(vb, rb.x + rb.w / 2), fy(vb, rb.y + rb.h / 2) + off);
    ctx.stroke();
    ctx.restore();
  }

  function drawGtPaths() {
    ctx.lineWidth = 1.25;
    for (const [gid, path] of gtPaths) {
      if (path.length < 2) continue;
      ctx.strokeStyle = gtColor(gid);
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      let prev = null; // per-floor polylines: break the stroke at floor changes
      for (const p of path) {
        const v = views.get(p.floor);
        if (!v) { prev = null; continue; }
        if (prev && prev.floor === p.floor) ctx.lineTo(fx(v, p.x), fy(v, p.y));
        else ctx.moveTo(fx(v, p.x), fy(v, p.y));
        prev = p;
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      const joined = new Set(); // one connector per floor pair per person
      for (let i = 1; i < path.length; i++) {
        const fa = path[i - 1].floor, fb = path[i].floor;
        if (fa === fb) continue;
        const key = fa < fb ? fa + "|" + fb : fb + "|" + fa;
        if (joined.has(key)) continue;
        joined.add(key);
        drawStairConnector(fa, fb, gid);
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawTrails() {
    ctx.lineWidth = 1;
    for (const m of markers.values()) {
      const v = views.get(m.floor);
      const tr = m.trail;
      if (!v || tr.length < 2) continue;
      ctx.strokeStyle = colorFor(m);
      for (let i = 1; i < tr.length; i++) {
        const age = latestTs - tr[i].t;
        ctx.globalAlpha = Math.max(0, 1 - age / TRAIL_S) * 0.45;
        ctx.beginPath();
        ctx.moveTo(fx(v, tr[i - 1].x), fy(v, tr[i - 1].y));
        ctx.lineTo(fx(v, tr[i].x), fy(v, tr[i].y));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawDots() {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(216,226,234,0.55)";
    for (const m of markers.values()) {
      const v = views.get(m.floor);
      if (!v) continue;
      ctx.beginPath();
      ctx.arc(fx(v, m.x), fy(v, m.y), 4, 0, TAU);
      ctx.fillStyle = colorFor(m);
      ctx.fill();
      ctx.stroke();
    }
  }

  function drawHover() {
    if (!mouse) return;
    let best = null, bestV = null;
    let bestD2 = HOVER_PX * HOVER_PX;
    for (const m of markers.values()) {
      const v = views.get(m.floor);
      if (!v) continue;
      const dx = fx(v, m.x) - mouse.x, dy = fy(v, m.y) - mouse.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = m; bestV = v; }
    }
    if (!best) return;
    const x = fx(bestV, best.x), y = fy(bestV, best.y);
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
    if (!staticLayer || views.size === 0) return;

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

  // ---- sidebar: per-camera counts grouped under floor headings ---------------
  function buildSidebar() {
    const addRow = (cam) => {
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
    };
    const grouped = new Set();
    for (const f of plan.floors ?? []) {
      const head = document.createElement("div");
      head.className = "floor-head";
      head.textContent = f.name ?? f.id;
      elCams.append(head);
      for (const cam of plan.cameras ?? []) {
        if (cam.floor !== f.id) continue;
        grouped.add(cam.id);
        addRow(cam);
      }
    }
    for (const cam of plan.cameras ?? []) {
      if (!grouped.has(cam.id)) addRow(cam); // defensive: floor id not in plan
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
    floorById = new Map((plan.floors ?? []).map((f) => [f.id, f]));
    for (const c of plan.cameras ?? []) camFloor.set(c.id, c.floor);
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
