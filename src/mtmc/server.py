"""mtmc.server — M0 HTTP/WebSocket front-end (CONTRACT.md §mtmc.server).

Constraints honoured here:

* ONE shared ``Simulation`` per process, created lazily on first need; all
  websocket clients watch the same world.
* A single background task steps the sim at 10 Hz real-time pace and fans
  each non-empty tick out to every connected client. Client connects and
  disconnects never construct, pause, or reset the sim.
* A slow or dead client must not block the others: sends run concurrently
  with a per-send timeout, and a failed client is dropped from the set.
* ``mtmc.simulator`` is being written in parallel — it is imported only at
  runtime, so this module always imports cleanly on its own.
* ``web/`` and ``floorplan/`` are located by walking up from this file, so
  the server works from any cwd.
* M1: ``--source`` picks the mode. ``"sim"`` (default) keeps M0 behaviour;
  anything else ("0", "rtsp://…") streams a real camera through the SAME hub
  and wire format — consumers never branch on source (contract). The detector
  and ``mtmc.camera`` are imported lazily, only when camera mode starts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mtmc.events import Observation, tick_message

logger = logging.getLogger("mtmc.server")

TICK_DT_S = 0.1  # contract: step the sim at 10 Hz
SEND_TIMEOUT_S = 1.0  # a client this far behind is dropped, not waited on
CAMERA_POLL_TIMEOUT_S = 0.2  # blocking queue-read slice; bounds shutdown latency
MJPEG_FPS = 10.0  # /video.mjpg re-sends latest_jpeg at ~this rate
_MJPEG_BOUNDARY = b"mtmc-frame"

DEFAULT_PORT = 8100
DEFAULT_SEED = 42
DEFAULT_PEOPLE = 12
DEFAULT_SOURCE = "sim"
DEFAULT_CAM_FPS = 5.0


class SimulationLike(Protocol):
    """Structural stand-in for ``mtmc.simulator.Simulation`` (built in parallel)."""

    @property
    def clock_s(self) -> float: ...

    def step(self, dt_s: float) -> list[Observation]: ...


class CameraWorkerLike(Protocol):
    """Structural stand-in for ``mtmc.camera.CameraWorker`` (M1)."""

    camera_id: str

    @property
    def latest_jpeg(self) -> bytes | None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_batch(self, timeout: float = ...) -> list[Observation] | None: ...


def _find_repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor holding ``floorplan/plan.json`` — cwd-independent."""
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "floorplan" / "plan.json").is_file():
            return parent
    # src layout fallback: server.py -> mtmc -> src -> repo root.
    return here.parents[2] if len(here.parents) >= 3 else here.parent


def _default_sim_factory(plan_path: Path, seed: int, people: int) -> Callable[[], SimulationLike]:
    """Deferred import: the server must import cleanly even if the simulator
    module is broken or absent, failing only when a sim is actually needed."""

    def factory() -> SimulationLike:
        from mtmc.simulator import Simulation

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return Simulation(plan, seed=seed, n_people_target=people)

    return factory


def _default_worker_factory(source: str, cam_fps: float) -> Callable[[], CameraWorkerLike]:
    """Deferred imports again: onnxruntime/cv2 (and agent A's detector, built
    in parallel) are only touched when camera mode actually starts — sim mode
    and a bare ``import mtmc.server`` never pay for them."""

    def factory() -> CameraWorkerLike:
        from mtmc.camera import CameraWorker
        from mtmc.detector import PersonDetector  # raises clearly if the model is missing

        return CameraWorker(source, PersonDetector(), cam_fps=cam_fps)

    return factory


class _HubBase:
    """Client set + concurrent fan-out, shared by both source modes.

    Subclasses provide ``attach`` (source-specific setup) and ``_run`` (the
    ticker that produces frames to broadcast).
    """

    _task_name = "mtmc-ticker"

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._clients: set[WebSocket] = set()

    async def attach(self, ws: WebSocket) -> bool:
        raise NotImplementedError

    def detach(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=self._task_name)

    async def _run(self) -> None:
        raise NotImplementedError

    async def _broadcast(self, frame: str) -> None:
        if not self._clients:
            return
        clients = list(self._clients)
        results = await asyncio.gather(*(self._send(ws, frame) for ws in clients))
        for ws, ok in zip(clients, results):
            if not ok:
                self._clients.discard(ws)
                # Close the socket so the browser's onclose fires and its
                # reconnect/backoff takes over; without this a once-stalled
                # tab is dropped server-side but stays frozen client-side.
                asyncio.get_running_loop().create_task(self._close_quietly(ws))

    @staticmethod
    async def _close_quietly(ws: WebSocket) -> None:
        try:
            await ws.close(code=1011)
        except Exception:  # noqa: BLE001 — already-closed sockets are fine
            pass

    @staticmethod
    async def _send(ws: WebSocket, frame: str) -> bool:
        try:
            await asyncio.wait_for(ws.send_text(frame), timeout=SEND_TIMEOUT_S)
        except Exception:
            return False
        return True


class _SimHub(_HubBase):
    """Owns the process-wide sim, its 10 Hz ticker task, and the client set.

    The ticker keeps running with zero clients so that connect/disconnect
    never disturbs the world (contract: disconnect must not kill the sim).
    """

    _task_name = "mtmc-sim-ticker"

    def __init__(self, sim_factory: Callable[[], SimulationLike]) -> None:
        super().__init__()
        self._sim_factory = sim_factory
        self._sim: SimulationLike | None = None

    async def attach(self, ws: WebSocket) -> bool:
        """Accept and register a client; on sim-construction failure the
        socket is closed (1011) and False returned — other clients unaffected."""
        await ws.accept()
        try:
            self._ensure_sim()
        except Exception as exc:  # e.g. mtmc.simulator not written yet
            logger.exception("cannot create Simulation")
            await ws.close(code=1011, reason=f"simulation unavailable: {exc}"[:120])
            return False
        self._clients.add(ws)
        self._ensure_task()
        return True

    def _ensure_sim(self) -> SimulationLike:
        if self._sim is None:
            self._sim = self._sim_factory()
        return self._sim

    async def _run(self) -> None:
        sim = self._ensure_sim()
        loop = asyncio.get_running_loop()
        next_deadline = loop.time()
        try:
            while True:
                observations = sim.step(TICK_DT_S)
                if observations:  # contract: broadcast non-empty ticks only
                    await self._broadcast(tick_message(sim.clock_s, observations))
                next_deadline += TICK_DT_S
                delay = next_deadline - loop.time()
                if delay <= 0:  # fell behind: resync instead of bursting steps
                    next_deadline = loop.time()
                    delay = 0.0
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("simulation ticker crashed")


class _CameraHub(_HubBase):
    """Camera mode (M1): the ticker drains CameraWorker's thread-side queue
    (blocking ``get_batch`` bridged into asyncio via the default executor) and
    fans each batch out as a ``tick_message`` frame — identical wire shape to
    sim mode, so consumers never branch on source (contract)."""

    _task_name = "mtmc-camera-ticker"

    def __init__(self, worker_factory: Callable[[], CameraWorkerLike]) -> None:
        super().__init__()
        self._worker_factory = worker_factory
        self._worker: CameraWorkerLike | None = None

    def ensure_worker(self) -> CameraWorkerLike:
        if self._worker is None:
            self._worker = self._worker_factory()
        return self._worker

    def start(self) -> None:
        """Lifespan startup: capture thread + drain task begin immediately, so
        the queue stays fresh and /video.mjpg works with zero ws clients."""
        self.ensure_worker().start()
        self._ensure_task()

    async def attach(self, ws: WebSocket) -> bool:
        await ws.accept()
        self._clients.add(ws)
        self._ensure_task()
        return True

    async def aclose(self) -> None:
        await super().aclose()  # ticker first, then the worker it reads from
        if self._worker is not None:
            # stop() joins the capture thread — keep that block off the loop.
            await asyncio.get_running_loop().run_in_executor(None, self._worker.stop)

    async def _run(self) -> None:
        worker = self.ensure_worker()
        loop = asyncio.get_running_loop()
        try:
            while True:
                batch = await loop.run_in_executor(None, worker.get_batch, CAMERA_POLL_TIMEOUT_S)
                if batch:  # contract: broadcast non-empty ticks only
                    await self._broadcast(tick_message(batch[0].ts_s, batch))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("camera ticker crashed")


def create_app(
    *,
    root: Path | None = None,
    seed: int = DEFAULT_SEED,
    people: int = DEFAULT_PEOPLE,
    sim_factory: Callable[[], SimulationLike] | None = None,
    source: str = DEFAULT_SOURCE,
    cam_fps: float = DEFAULT_CAM_FPS,
    camera_worker_factory: Callable[[], CameraWorkerLike] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``root``, ``sim_factory`` and ``camera_worker_factory`` exist for tests:
    the default factories import ``mtmc.simulator`` / ``mtmc.camera`` at call
    time, never at module import. ``source`` != "sim" selects camera mode.
    """
    repo_root = (root or _find_repo_root()).resolve()
    web_dir = repo_root / "web"
    plan_path = repo_root / "floorplan" / "plan.json"

    mode = "sim" if source == DEFAULT_SOURCE else "camera"
    camera_hub: _CameraHub | None = None
    hub: _HubBase
    if mode == "camera":
        if camera_worker_factory is None:
            camera_worker_factory = _default_worker_factory(source, cam_fps)
        camera_hub = _CameraHub(camera_worker_factory)
        hub = camera_hub
    else:
        if sim_factory is None:
            sim_factory = _default_sim_factory(plan_path, seed, people)
        hub = _SimHub(sim_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if camera_hub is not None:
            # Fail fast and loudly here (e.g. missing model file): a camera
            # server that cannot detect is misconfiguration, not stream loss.
            camera_hub.start()
        yield
        await hub.aclose()

    app = FastAPI(title="urban-mtmc M0 server", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        index_html = web_dir / "index.html"
        if not index_html.is_file():  # web/ is owned by agent C, may not exist yet
            return JSONResponse({"detail": "web/index.html not found"}, status_code=404)
        return FileResponse(index_html, media_type="text/html")

    @app.get("/api/plan")
    async def api_plan() -> JSONResponse:
        return JSONResponse(json.loads(plan_path.read_text(encoding="utf-8")))

    @app.get("/api/source")
    async def api_source() -> JSONResponse:
        # Contract: {"mode": "sim"|"camera", "camera_id": "live0"|null}.
        camera_id = camera_hub.ensure_worker().camera_id if camera_hub is not None else None
        return JSONResponse({"mode": mode, "camera_id": camera_id})

    @app.get("/video.mjpg")
    async def video_mjpg() -> Response:
        if camera_hub is None:
            return JSONResponse({"detail": "no camera stream in sim mode"}, status_code=404)
        worker = camera_hub.ensure_worker()

        async def frames() -> AsyncIterator[bytes]:
            # Re-send the newest annotated JPEG at ~MJPEG_FPS, decoupled from
            # cam_fps; before the first frame lands there is nothing to send.
            while True:
                jpeg = worker.latest_jpeg
                if jpeg is not None:
                    yield (
                        b"--" + _MJPEG_BOUNDARY + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                await asyncio.sleep(1.0 / MJPEG_FPS)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode('ascii')}",
        )

    @app.websocket("/ws")
    async def ws_stream(ws: WebSocket) -> None:
        if not await hub.attach(ws):
            return
        try:
            while True:
                await ws.receive_text()  # inbound ignored; raises on disconnect
        except WebSocketDisconnect:
            pass
        finally:
            hub.detach(ws)

    # check_dir=False: tolerate web/ not existing yet at app-construction time.
    app.mount("/static", StaticFiles(directory=web_dir, check_dir=False), name="static")

    return app


def main() -> None:
    """Console script ``mtmc-server`` (contract flags: --port/--seed/--people
    plus M1's --source/--cam-fps)."""
    parser = argparse.ArgumentParser(prog="mtmc-server", description="urban-mtmc server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--people", type=int, default=DEFAULT_PEOPLE)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help='"sim" (default), a webcam index like "0", or an rtsp:// URL',
    )
    parser.add_argument("--cam-fps", type=float, default=DEFAULT_CAM_FPS)
    args = parser.parse_args()

    import uvicorn  # runtime dependency of the script, not of importing this module

    if args.source != "sim":
        # macOS TCC can only show the camera-permission prompt from the MAIN
        # thread; the worker thread's open would fail with "not authorized".
        # Open once here (blocks until the user answers), release immediately.
        import cv2

        src: int | str = int(args.source) if args.source.isdigit() else args.source
        probe = cv2.VideoCapture(src)
        opened = probe.isOpened()
        probe.release()
        if not opened:
            logger.warning(
                "camera source %r did not open on preflight — if a permission "
                "prompt appeared, answer it and restart; the worker will also "
                "keep retrying every 2s.", args.source,
            )

    uvicorn.run(
        create_app(seed=args.seed, people=args.people, source=args.source, cam_fps=args.cam_fps),
        host="127.0.0.1",
        port=args.port,
        # An open /video.mjpg response never ends on its own; without a bound
        # uvicorn waits on it forever and SIGINT cannot shut the server down.
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
