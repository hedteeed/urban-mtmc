# urban-mtmc

Multi-target multi-camera (MTMC) people tracking for indoor spaces — many CCTV
cameras, every person detected and tracked with one anonymous ID across
cameras, rendered live on a single floor-plan dashboard. Built to replace a
guard watching 20 screens with a guard watching one map.

## Status

Pre-M0. Documentation complete; build starting.

## Docs

The project curriculum and build plan live in [`docs/`](docs/) (open in a browser):

| Doc | What |
|---|---|
| [`complete-course.html`](docs/complete-course.html) | **Start here.** All three volumes merged into one 34-step course |
| [`mantis-decoded.html`](docs/mantis-decoded.html) | Vol 1 — deep review of the mantis drone-ISR repo this project learns from |
| [`urban-mtmc-manual.html`](docs/urban-mtmc-manual.html) | Vol 2 — square-one builder's manual, milestones M0–M7 |
| [`mtmc-mechanics.html`](docs/mtmc-mechanics.html) | Vol 3 — every mechanism worked by hand + hackathon playbook |

## Architecture (planned)

Per camera: RTSP ingest → YOLOX person detector (Apache-2.0) → ByteTrack (MIT)
→ OSNet Re-ID embeddings (MIT) → homography to floor plan.
Central: space-time-gated tracklet clustering → global IDs → WebSocket →
floor-plan dashboard (React + Leaflet CRS.Simple).

Anonymous, session-scoped IDs by design. No face recognition.

## Milestones

- **M0** — skeleton + floor-plan simulator (synthetic people end to end)
- **M1** — live person detection, one camera
- **M2** — ByteTrack, stable single-camera IDs
- **M3** — homography calibration, dots on the floor plan
- **M5** — second camera, overlap fusion
- **M6** — cross-camera identity across a blind gap (the hard one)
- **M7** — harden toward pilot (auth, retention, health, soak)

## License notes

- Detector/tracker/Re-ID dependencies chosen for permissive licenses
  (Apache-2.0 / MIT). AGPL tooling (Ultralytics, boxmot) deliberately avoided.
- The mantis repo studied in Vol 1 is not open-source; this project reuses its
  *patterns*, never its code.
- Public research datasets (CrowdHuman, MOT, Market-1501) are research-only:
  usable for prototypes, never in shipped models. Provenance is tracked per
  training run.
