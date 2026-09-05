"""RAG-grounded medical answers from Groq. Demo only, never experiments.

On a cache miss the demo retrieves passages from the fixed medical corpus,
assembles them into a prompt, and asks the model to answer from those passages
only. Two reasons the demo does this rather than quering a bare model:

  Hallucination. A bare LLM answering health questions live will invent
  specifics in front of an examiner. Grounding the answer in retrieved text and
  showing the citation makes every claim checkable.

  Attack visibility. Retrieval plus generation lengthens the miss path, which 
  widens the hit/miss latency gap the adversary is measuring and makes the leak
  more obvious on screen.

This is the only module in the project that calls a real model API. Every reported
experimental number comes from ``mocks.py`` instead - a real API's own latency
variance would land directly on top of the timing signal Contribution B exists to measure
to measure, and a hosted model's rate limits would make sweeps non-reproducible.

Groq's free tier has per-minute request and token limits. A 429 is therefore a normal operating
condition here, not an exceptional one, and is handled as a displayable answer rather than an 
rather than an exception - a demo that crashes when the quota ticks over is worse than one that say
"rate limited, try again".

Retrieval quality is explicitly out of scope: no recall@k, no chunking ablation. The generation 
pipeline is infrastructure, not an object of study.
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from ..demo.retriever import ChunkRetriever
from .base import Oracle, OracleResponse


#: Groq production model. 131K context, 32,768 max output - far more than a
#: grounded three-sentence answer needs, so the corpus passages are never the
#: constraint. `llama-3.1-8b-instant` is the faster, weaker alternative;
#: `openai/gpt-oss-120b` the larger one. Preview models are explicitly not for
#: production and can be withdrawn at short notice, so none is used here.

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a medical information assistant for a hospital's public \
information service.

Answer using ONLY the reference passages provided in the user message. Rules:

- If the passages do not contain the answer, say no plainly. Do not fill the gap \
from your own knowledge.
- Attribute claims to the passage they came from, using the bracketed source name \
show above each passage
- Give general health information, not diagnosis, prognosis, or personalised \
treatment advice. Do not tell the user what their condition is or what they should\
take.

- Recommend contacting a clinican for anything specific to the reader's own case.
- Two to four sentences unless the question genuinely needs more.
"""

_NO_CONTEXT_REPLY = (
    "I don't have any reference material covering that question, so I can't answer "
    "it from our documents. Please contact a clinician or our information desk."
)


class RAGOracle(Oracle):
    """Retrieves passages, then asks a Groq-hosted model to answer from them.

    ``temperature`` defaults low. The job is to restate what the passages say,
    not to write creatively, and sampling variety here shows up as drift away
    from the sources - which is the one failure this design exists to prevent.
    """

    