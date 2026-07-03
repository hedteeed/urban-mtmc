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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mtmc.events import Observation, tick_message

logger = logging.getLogger("mtmc.server")

TICK_DT_S = 0.1  # contract: step the sim at 10 Hz
SEND_TIMEOUT_S = 1.0  # a client this far behind is dropped, not waited on

DEFAULT_PORT = 8100
DEFAULT_SEED = 42
DEFAULT_PEOPLE = 12


class SimulationLike(Protocol):
    """Structural stand-in for ``mtmc.simulator.Simulation`` (built in parallel)."""

    @property
    def clock_s(self) -> float: ...

    def step(self, dt_s: float) -> list[Observation]: ...


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


class _SimHub:
    """Owns the process-wide sim, its 10 Hz ticker task, and the client set.

    The ticker keeps running with zero clients so that connect/disconnect
    never disturbs the world (contract: disconnect must not kill the sim).
    """

    def __init__(self, sim_factory: Callable[[], SimulationLike]) -> None:
        self._sim_factory = sim_factory
        self._sim: SimulationLike | None = None
        self._task: asyncio.Task[None] | None = None
        self._clients: set[WebSocket] = set()

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

    def _ensure_sim(self) -> SimulationLike:
        if self._sim is None:
            self._sim = self._sim_factory()
        return self._sim

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="mtmc-sim-ticker")

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


def create_app(
    *,
    root: Path | None = None,
    seed: int = DEFAULT_SEED,
    people: int = DEFAULT_PEOPLE,
    sim_factory: Callable[[], SimulationLike] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``root`` and ``sim_factory`` exist for tests: the default factory imports
    ``mtmc.simulator`` at call time, never at module import.
    """
    repo_root = (root or _find_repo_root()).resolve()
    web_dir = repo_root / "web"
    plan_path = repo_root / "floorplan" / "plan.json"

    if sim_factory is None:
        sim_factory = _default_sim_factory(plan_path, seed, people)
    hub = _SimHub(sim_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
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
    """Console script ``mtmc-server`` (contract flags: --port/--seed/--people)."""
    parser = argparse.ArgumentParser(prog="mtmc-server", description="urban-mtmc M0 server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--people", type=int, default=DEFAULT_PEOPLE)
    args = parser.parse_args()

    import uvicorn  # runtime dependency of the script, not of importing this module

    uvicorn.run(create_app(seed=args.seed, people=args.people), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
