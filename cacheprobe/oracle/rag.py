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

    def __init__(
      self,
      retriever: ChunkRetriever,
      embed_fn: Callable[[str], np.ndarray],
      client: object | None = None,
      model: str = DEFAULT_MODEL,
      top_l: int = 4,
      max_tokens: int = 2048,
      temperature: float = 0.2,
      cost_per_call: float = 0.0,
    ) -> None:

      super().__init__(cost_per_call = cost_per_call)
      self.retriever = retriever
      self.embed_fn = embed_fn
      self.model = model
      self.top_k = top_k
      # Deliberately modest: answers land in a chat bubble and are stored in
      # a bounded cache, so a long ceiling would waste memory per entry.
      # Still well above what a two-to-four sentence grounded answer needs.
      self.max_tokens = max_tokens
      self.temperature = temperature
      self._client = client

    # ---------------------------------------------------------------- client

    @property
    def client(self):
      """Lazily construct the Groq client

      Deferred so importing this module (and therefore ``demo/app.y``) does
      not require an API key to be present - the demo runs end to end on
      ``MockOracle`` with no credentials at all.
      """
      if self._client is None:
        # pyrefly: ignore [missing-import]
        from groq import Groq

        self._client = Groq()
      return self._client
    

    def answer(self, query: str) -> OracleResponse:
      import time

      started = time.perf_counter

      vector = np.asarray(self.embed_fn(query), dtype=np.float32)
      context, sources = self.retriever.build_context(vector, top_k=self.top_k)

      if not context.strip():
            # Nothing retrieved: answering anyway would be exactly the
            # ungrounded guess this design exists to prevent. Skip the API call
            # entirely rather than spend a request on it.
            self.stats.record(latency_s=time.perf_counter - started, cost=0.0)
            return OracleResponse(text= _NO_CONTEXT_REPLY, sources=())
      
      user_content = f"Reference passages:\n\n{context}\n\n---\n\nQuestion: {query}"

      try:
        completion = self.client.chat.completions.create(
          model = self.model,
          max_completion_tokens = self.max_tokens,
          temperature = self.temperature,
          messages = [
              {"role": "system", "content":SYSTEM_PROMPT},
              {"role": "user", "content": user_content}
          ],
        )
      except Exception as exc:
        self.stats.record(latency_s=time.perf_counter()-started, cost=0.0)
        return OracleResponse(text=self._describe_error(exc), sources=())
      
      self.stats.record(
        latency_s=time.perf_counter()-started, cost=self.cost_per_call
      )

      choice = completion.choices[0]
      text = (choice.message.content or "").strip()
      # finish_reason tells us whether what we got is a complete answer.
      # "length" means the ceiling cut it off mid-sentence; serving that to
      # the cache would store a truncated answer permanently.
      
      

        