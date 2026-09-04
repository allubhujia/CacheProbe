"""Two-pane demo: a user's chat beside the adversary's timing readout.

Left pane is an ordinary chatbot. Right pane is the attacker, probing the same
cache with held-out phrasings and reporting what the clock tells it. A toggle
switches defences, and the leak visibly closes.

The demo makes an abstract vulnerability legible: nothing is hacked, no
credentials are used, and the adversary's pane is populated entirely from
response times measured over the same HTTP path any user would take.

The oracle is injected. With ``MockOracle`` the demo runs offline and still
shows the leak; with the RAG oracle it produces grounded medical answers with
citations. Everything in this module works either way - the only AI-dependent
piece is the oracle itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..attack.prober import Prober
from ..defence.base import LatencyShaper, NoDefence
from ..defence.jitter import RandomJitter
from ..defence.quantised import QuantisedBuckets
from ..defence.regions import RegionPartition
from ..defence.selective import SelectivePadding
from ..defence.uniform import UniformPadding
from ..server import CacheService

STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class AdversaryView:
    """One row of the attacker's pane."""

    concept: str
    probe_query: str
    latency_ms: float
    verdict: str  # "CACHED" | "not cached"
    #: Server-side truth, shown only so a viewer can see the attack is right.
    truth: bool | None = None


@dataclass
class DemoState:
    chat_log: list[dict[str, Any]] = field(default_factory=list)
    adversary_log: list[AdversaryView] = field(default_factory=list)
    defence_name: str = NoDefence.name


def build_shaper(
    name: str,
    partition: RegionPartition | None = None,
    pad_s: float = 2.0,
) -> LatencyShaper:
    """Map a toggle label to a shaper.

    D4 needs the region partition, and refuses to run without it rather than
    silently degrading to no protection.
    """
    if name in ("D0", "none"):
        return NoDefence()
    if name == "D1":
        return UniformPadding(pad_s=pad_s)
    if name == "D2":
        return RandomJitter()
    if name == "D3":
        return QuantisedBuckets()
    if name == "D4":
        if partition is None:
            raise ValueError("D4 requires a region partition")
        return SelectivePadding(partition)
    raise ValueError(f"unknown defence {name!r}")


class DemoController:
    """Drives both panes against one shared cache."""

    def __init__(
        self,
        service: CacheService,
        prober: Prober,
        adversary_probes: dict[str, str],
        partition: RegionPartition | None = None,
    ) -> None:
        self.service = service
        self.prober = prober
        self.adversary_probes = adversary_probes
        self.partition = partition
        self.state = DemoState()

    # ------------------------------------------------------------- left pane

    def ask(self, query: str) -> dict[str, Any]:
        """A user asks a question."""
        served = self.service.handle(query)
        entry = {
            "query": query,
            "answer": served.text,
            "sources": list(served.sources),
            "latency_ms": round(served.total_s * 1000, 1),
            "served_from_cache": served.hit,
        }
        self.state.chat_log.append(entry)
        return entry

    # ------------------------------------------------------------ right pane

    def probe_all(self, reveal_truth: bool = True) -> list[AdversaryView]:
        """Run one enumeration sweep and refresh the attacker's pane."""
        if self.prober.threshold_s is None:
            self.prober.calibrate()

        rows: list[AdversaryView] = []
        for concept, phrasing in self.adversary_probes.items():
            probe = self.prober.probe(phrasing)
            looks_cached = self.prober.looks_cached(probe)
            rows.append(
                AdversaryView(
                    concept=concept,
                    probe_query=phrasing,
                    latency_ms=round(probe.latency_s * 1000, 1),
                    verdict="CACHED" if looks_cached else "not cached",
                    truth=self._truth(phrasing) if reveal_truth else None,
                )
            )

        self.state.adversary_log = rows
        return rows

    def _truth(self, phrasing: str) -> bool:
        """Whether this concept really is resident, read from the cache directly.

        Used only to render a "correct?" column in the demo. The prober never
        calls this - it classifies from latency alone.
        """
        vector = self.service.cache._as_vector(phrasing)
        entry, similarity, _ = self.service.cache._best_match(vector)
        return entry is not None and similarity >= self.service.cache.tau

    # --------------------------------------------------------------- toggle

    def set_defence(self, name: str, pad_s: float = 2.0) -> str:
        shaper = build_shaper(name, partition=self.partition, pad_s=pad_s)
        self.service.set_defence(shaper)
        self.state.defence_name = shaper.name
        # A new defence changes the latency distribution, so the adversary's
        # old decision boundary is meaningless. Forcing recalibration models a
        # competent attacker rather than handing the defence a free win.
        self.prober.threshold_s = None
        return shaper.name


def create_demo_app(controller: DemoController) -> Any:
    """FastAPI app serving the two-pane UI and its endpoints."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    class AskRequest(BaseModel):
        query: str

    class DefenceRequest(BaseModel):
        defence: str
        pad_s: float = 2.0

    app = FastAPI(title="CacheProbe demo")

    @app.get("/")
    def index() -> HTMLResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>index.html missing</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.post("/api/ask")
    def ask(request: AskRequest) -> JSONResponse:
        return JSONResponse(controller.ask(request.query))

    @app.post("/api/probe")
    def probe() -> JSONResponse:
        rows = controller.probe_all()
        return JSONResponse(
            {
                "defence": controller.state.defence_name,
                "rows": [
                    {
                        "concept": r.concept,
                        "probe": r.probe_query,
                        "latency_ms": r.latency_ms,
                        "verdict": r.verdict,
                        "truth": r.truth,
                        "correct": (r.verdict == "CACHED") == r.truth
                        if r.truth is not None
                        else None,
                    }
                    for r in rows
                ],
            }
        )

    @app.post("/api/defence")
    def set_defence(request: DefenceRequest) -> JSONResponse:
        try:
            name = controller.set_defence(request.defence, pad_s=request.pad_s)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"defence": name})

    @app.get("/api/state")
    def state() -> JSONResponse:
        stats = controller.service.cache.stats
        return JSONResponse(
            {
                "defence": controller.state.defence_name,
                "chat": controller.state.chat_log[-20:],
                "cache_size": len(controller.service.cache),
                "hit_rate": stats.hit_rate,
            }
        )

    return app
