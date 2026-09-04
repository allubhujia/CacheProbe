"""HTTP endpoint wrapping the cache, so attack timing crosses a real network path.

The server exists because of section 12.3: measuring latency inside the process
would miss exactly the leaks that matter - queueing contention, connection
reuse, garbage collection. The adversary must be a client.

Two ordering rules make or break the defence, both from section 7.2:

  1. ``arrival`` is stamped before any cache work begins, and every deadline is
     computed from it. A deadline of ``now + delay``, computed after the lookup,
     would let a fast hit release earlier than a slow miss and shape nothing.

  2. The region is resolved from the query's embedding *before* the lookup runs,
     so the padding decision cannot depend on whether the query hit.

The shaper is never handed the hit flag; see ``defence/base.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .cache import SemanticCache
from .defence.base import LatencyShaper, NoDefence
from .defence.regions import RegionPartition


@dataclass
class ServedResponse:
    text: str
    sources: tuple[str, ...]
    #: Server-side truth, returned for the demo's adversary pane and for
    #: scoring. A real deployment would never expose this; the attack harness
    #: must not read it when classifying.
    hit: bool
    region: int | None
    added_delay_s: float
    total_s: float


class CacheService:
    """Framework-free core: one method, correct ordering, testable directly."""

    def __init__(
        self,
        cache: SemanticCache,
        shaper: LatencyShaper | None = None,
        partition: RegionPartition | None = None,
    ) -> None:
        self.cache = cache
        self.shaper = shaper or NoDefence()
        self.partition = partition

    def set_defence(self, shaper: LatencyShaper) -> None:
        """Swap defences live - what the demo's toggle calls."""
        self.shaper = shaper

    def handle(self, query: str, cluster_id: int | None = None) -> ServedResponse:
        arrival = time.perf_counter()

        region = None
        if self.partition is not None:
            # Resolved from the query alone, before the lookup, so it carries
            # no information about cache state.
            region = self.partition.region_of(self.cache._as_vector(query))

        result = self.cache.get(query, cluster_id=cluster_id)
        added = self.shaper.wait(arrival, region)

        return ServedResponse(
            text=result.response.text,
            sources=result.response.sources,
            hit=result.hit,
            region=region,
            added_delay_s=added,
            total_s=time.perf_counter() - arrival,
        )


def create_app(service: CacheService) -> Any:
    """Build the FastAPI app. Imported lazily so the core stays dependency-free."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        query: str
        cluster_id: int | None = None

    class DefenceRequest(BaseModel):
        defence: str
        pad_s: float = 2.0

    app = FastAPI(title="CacheProbe")

    @app.post("/query")
    def query(request: QueryRequest) -> JSONResponse:
        served = service.handle(request.query, cluster_id=request.cluster_id)
        # The adversary only ever times this call; the body deliberately does
        # not reveal `hit`, so the attack has to infer it from latency alone.
        return JSONResponse({"answer": served.text, "sources": list(served.sources)})

    @app.post("/admin/defence")
    def set_defence(request: DefenceRequest) -> JSONResponse:
        from .defence.jitter import RandomJitter
        from .defence.quantised import QuantisedBuckets
        from .defence.uniform import UniformPadding

        shapers = {
            "D0": NoDefence,
            "D1": lambda: UniformPadding(pad_s=request.pad_s),
            "D2": RandomJitter,
            "D3": QuantisedBuckets,
        }
        if request.defence == "D4":
            if service.partition is None:
                return JSONResponse({"error": "D4 needs a region partition"}, status_code=400)
            from .defence.selective import SelectivePadding

            service.set_defence(SelectivePadding(service.partition))
        elif request.defence in shapers:
            service.set_defence(shapers[request.defence]())
        else:
            return JSONResponse({"error": f"unknown defence {request.defence}"}, status_code=400)

        return JSONResponse({"defence": service.shaper.name})

    @app.get("/admin/stats")
    def stats() -> JSONResponse:
        s = service.cache.stats
        return JSONResponse(
            {
                "size": len(service.cache),
                "hits": s.hits,
                "misses": s.misses,
                "hit_rate": s.hit_rate,
                "false_hit_rate": s.false_hit_rate,
                "evictions": s.evictions,
                "defence": service.shaper.name,
            }
        )

    return app
