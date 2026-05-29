"""Gemini async adapter.

Wraps google-genai 2.x. GeminiEmbeddingClient calls embed_content for
batch vectors; GeminiLLMClient calls generate_content for one-shot
text or structured JSON (response_schema). Function-calling lives in
the OrchestratorAgent (F5), not here.
"""

from typing import Protocol

from pydantic import BaseModel, SecretStr

from papers_agent.core.logging import get_logger

# google.genai is imported lazily inside __init__/generate to keep
# Protocol-only consumers (tools, tests with AsyncMock) free of the
# heavy module-load side effects (gRPC/httpx/OpenSSL bootstrap).

log = get_logger(__name__)


class EmbeddingClient(Protocol):
    """Async embedding contract."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMClient(Protocol):
    """Async LLM generation contract."""

    async def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str: ...


class GeminiEmbeddingClient:
    """EmbeddingClient implementation backed by google-genai embed_content."""

    def __init__(self, api_key: SecretStr, model: str) -> None:
        from google import genai

        # Materialize the secret only here, pass to the SDK, and drop the ref.
        self._client = genai.Client(api_key=api_key.get_secret_value())
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batch; returns one vector per input, same order."""
        result = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
        )
        embeddings = result.embeddings or []
        # e.values may surface as a proto repeated field on some SDK paths;
        # list(...) normalizes to a plain list[float].
        out = [list(e.values or []) for e in embeddings]
        log.info("gemini.embed.done", n_texts=len(texts), model=self._model)
        return out


class GeminiLLMClient:
    """LLMClient implementation backed by google-genai generate_content."""

    def __init__(self, api_key: SecretStr, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key.get_secret_value())
        self._model = model

    async def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """Single-shot LLM call. response_schema triggers JSON structured mode."""
        from google.genai import types

        config = types.GenerateContentConfig()
        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema
        result = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = result.text
        if text is None:
            raise RuntimeError("Gemini returned empty text")
        log.info(
            "gemini.generate.done",
            chars=len(text),
            has_schema=response_schema is not None,
        )
        return text
