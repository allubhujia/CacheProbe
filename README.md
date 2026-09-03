# StaleGuard (CacheProbe)

Coverage-aware eviction and timing-channel defence for semantic LLM caches.

See `cacheprobe/` for the implementation, organised per module:

- `lsh_index.py`, `heap.py`, `cache.py` — core semantic cache
- `policies/` — eviction policies (LRU, LFU, FIFO, Random, coverage-aware)
- `defence/` — latency-shaping defences against the timing side channel
- `attack/` — membership-inference and reconstruction attack harness
- `oracle/` — the backend behind the cache: `mock.py` (fixed sleep) for all
  experiments, `rag.py` (retrieve + real LLM) for the demo only
- `server.py` — HTTP endpoint
- `traces.py`, `experiments.py`, `plots.py` — experiment harness

## Two data pipelines

**Experimental (Quora Question Pairs).** `traces.py` builds paraphrase clusters
from labelled duplicate pairs and samples a Zipf-skewed query stream. Labelled
duplicates are what make false-hit rate measurable, so this pipeline is the one
every reported result comes from, always against `oracle/mock.py`.

**Demo (medical).** `corpus.py` loads and chunks a fixed medical corpus for RAG
retrieval; `demo_traces.py` groups medical questions by topic and splits each
topic's phrasings into victim and adversary sets. `demo/` serves the two-pane
interface — user chat on the left, adversary timing readout on the right, with a
toggle across the four defences.

The victim/adversary phrasing split is not optional: if the adversary probes with
a string the victim used, the demo tests exact matching and demonstrates nothing
about the semantic claim.

Retrieval quality is not evaluated — no recall@k, no chunking ablation. The
generation pipeline is infrastructure, not an object of study.

**Known limitation.** Cached answers go stale if a source document changes: the
cache holds no dependency link to its inputs. Document-dependency invalidation is
future work.
# CacheProbe
