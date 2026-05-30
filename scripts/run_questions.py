"""Run the 5 evaluation questions against the live API and print answers.

Invoked by `make run`. Hits the /threads and /threads/{id}/messages routes
over HTTP (no Python imports of the agent graph); one fresh thread per
question so each answer stands alone for the evaluator.
"""

import asyncio
import os

import httpx

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8080")

QUESTIONS: list[str] = [
    "Qual é o mecanismo central proposto no paper Attention Is All You Need"
    " e como ele se diferencia de RNNs?",
    "Como o RAG combina recuperação e geração? Quais são suas limitações apontadas pelos autores?",
    "Compare a abordagem do ReAct com a do Toolformer para uso de ferramentas em LLMs.",
    "Qual paper você considera mais relevante para construir um agente com"
    " uso de ferramentas externas? Justifique com base nos textos.",
    "Faça um resumo executivo dos 5 papers em no máximo 5 bullet points cada.",
]


async def _wait_for_health(client: httpx.AsyncClient) -> None:
    """Poll /health until the app reports ready (lifespan finished)."""
    for _ in range(60):
        try:
            response = await client.get(f"{API_BASE}/health", timeout=3)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(2)
    raise RuntimeError("API did not respond on /health in time")


async def _ask(client: httpx.AsyncClient, question: str) -> str:
    """Create a fresh thread, post the question, return the answer text.

    Catches httpx errors so one failed question does not abort the batch.
    """
    thread_response = await client.post(f"{API_BASE}/threads")
    thread_response.raise_for_status()
    thread_id = thread_response.json()["thread_id"]
    try:
        message_response = await client.post(
            f"{API_BASE}/threads/{thread_id}/messages",
            json={"content": question},
            timeout=600,
        )
        message_response.raise_for_status()
        return str(message_response.json()["response"])
    except httpx.HTTPError as exc:
        return f"[falha ao responder: {exc} -- provavel limite de taxa do free tier]"


async def main() -> None:
    async with httpx.AsyncClient() as client:
        await _wait_for_health(client)
        for i, question in enumerate(QUESTIONS, start=1):
            print(f"\n{'=' * 78}\nPERGUNTA {i}: {question}\n{'=' * 78}")
            answer = await _ask(client, question)
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
