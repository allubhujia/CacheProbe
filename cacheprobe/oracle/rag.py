"""RAG-grounded medical answers from Groq. Demo only, never experiments.

On a cache miss the demo retrieves passages from the fixed medical corpus,
assembles them into a prompt, and asks the model to answer *from those passages
only*. Two reasons the demo does this rather than querying a bare model:

  Hallucination. A bare LLM answering health questions live will invent
  specifics in front of an examiner. Grounding the answer in retrieved text and
  showing the citation makes every claim checkable.

  Attack visibility. Retrieval plus generation lengthens the miss path, which
  widens the hit/miss latency gap the adversary is measuring and makes the leak
  more obvious on screen.

This is the only module in the project that calls a real model API. Every
reported experimental number comes from ``mock.py`` instead - a real API's own
latency variance would land directly on top of the timing signal Contribution
B exists to measure, and a hosted model's rate limits would make sweeps
non-reproducible.

Groq's free tier has per-minute request and token limits. A 429 is therefore a
normal operating condition here, not an exceptional one, and is handled as a
displayable answer rather than an exception - a demo that crashes when the
quota ticks over is worse than one that says "rate limited, try again".

Retrieval quality is explicitly out of scope: no recall@k, no chunking
ablation. The generation pipeline is infrastructure, not an object of study.
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

- If the passages do not contain the answer, say so plainly. Do not fill the gap \
from your own knowledge.
- Attribute claims to the passage they came from, using the bracketed source name \
shown above each passage.
- Give general health information, not diagnosis, prognosis, or personalised \
treatment advice. Do not tell the user what their condition is or what they should \
take.
- Recommend contacting a clinician for anything specific to the reader's own case.
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
        top_k: int = 4,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        cost_per_call: float = 0.0,
    ) -> None:
        super().__init__(cost_per_call=cost_per_call)
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
        """Lazily construct the Groq client.

        Deferred so importing this module (and therefore ``demo/app.py``) does
        not require an API key to be present - the demo runs end to end on
        ``MockOracle`` with no credentials at all.
        """
        if self._client is None:
            # pyrefly: ignore [missing-import]
            from groq import Groq

            self._client = Groq()  # reads GROQ_API_KEY from the environment
        return self._client

    # ---------------------------------------------------------------- answer

    def answer(self, query: str) -> OracleResponse:
        import time

        started = time.perf_counter()

        vector = np.asarray(self.embed_fn(query), dtype=np.float32)
        context, sources = self.retriever.build_context(vector, top_k=self.top_k)

        if not context.strip():
            # Nothing retrieved: answering anyway would be exactly the
            # ungrounded guess this design exists to prevent. Skip the API call
            # entirely rather than spend a request on it.
            self.stats.record(latency_s=time.perf_counter() - started, cost=0.0)
            return OracleResponse(text=_NO_CONTEXT_REPLY, sources=())

        user_content = f"Reference passages:\n\n{context}\n\n---\n\nQuestion: {query}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as exc:
            self.stats.record(latency_s=time.perf_counter() - started, cost=0.0)
            return OracleResponse(text=self._describe_error(exc), sources=())

        self.stats.record(
            latency_s=time.perf_counter() - started, cost=self.cost_per_call
        )

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()

        # finish_reason tells us whether what we got is a complete answer.
        # "length" means the ceiling cut it off mid-sentence; serving that to
        # the cache would store a truncated answer permanently.
        reason = getattr(choice, "finish_reason", None)
        if reason == "length":
            text += "\n\n[Answer truncated - raise max_tokens.]"
        elif reason == "content_filter":
            return OracleResponse(
                text=(
                    "I can't answer that one. Please contact a clinician or our "
                    "information desk."
                ),
                sources=(),
            )

        if not text:
            text = _NO_CONTEXT_REPLY

        return OracleResponse(text=text, sources=sources)

    # ----------------------------------------------------------------- errors

    @staticmethod
    def _describe_error(exc: Exception) -> str:
        """Turn an API failure into something the demo can display.

        The demo must not crash mid-presentation, so failures become visible
        answers rather than tracebacks. Ordered most specific first, because a
        single broad handler would lose the distinction between a missing key
        (fix your environment) and a rate limit (wait a moment) - and on a free
        tier those need very different responses from whoever is presenting.
        """
        try:
            # pyrefly: ignore [missing-import]
            import groq
        except ImportError:
            return f"The groq package is not installed ({exc})."

        if isinstance(exc, groq.AuthenticationError):
            return "No valid API key - set GROQ_API_KEY in the environment."
        if isinstance(exc, groq.PermissionDeniedError):
            return "This API key lacks permission for that model."
        if isinstance(exc, groq.NotFoundError):
            return (
                "Model not found. Groq retires models regularly - check "
                "console.groq.com/docs/models for the current production list."
            )
        if isinstance(exc, groq.RateLimitError):
            retry_after = "a moment"
            response = getattr(exc, "response", None)
            if response is not None:
                retry_after = response.headers.get("retry-after", "a moment") + "s"
            return f"Groq rate limit reached (free tier). Retry in {retry_after}."
        if isinstance(exc, groq.BadRequestError):
            return f"The API rejected the request: {exc}"
        if isinstance(exc, groq.APIStatusError):
            if exc.status_code >= 500:
                return f"Groq had a server error ({exc.status_code}). Try again shortly."
            return f"API error: {exc}"
        if isinstance(exc, groq.APIConnectionError):
            return "Could not reach Groq - check the network connection."
        return f"Unexpected failure calling the model: {exc}"


def build_rag_oracle(
    chunks: list,
    embed_fn: Callable[[list[str]], np.ndarray],
    embed_one: Callable[[str], np.ndarray],
    **kwargs: object,
) -> RAGOracle:
    """Convenience wiring: embed the corpus, build the retriever, wrap it.

    ``embed_fn`` batches a list (used once, for the corpus); ``embed_one``
    handles a single query at serve time.
    """
    from ..demo.retriever import build_retriever

    retriever = build_retriever(chunks, embed_fn)
    return RAGOracle(retriever=retriever, embed_fn=embed_one, **kwargs)  # type: ignore[arg-type]
