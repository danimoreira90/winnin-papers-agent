"""OrchestratorAgent: function-calling driver over RAG + Analyst agents.

Owns the LLM session and the automatic-function-calling loop. Receives a
question + thread history, lets google-genai resolve which of the 5
exposed methods to call (rag_*, analyst_*), and returns the final text.
Stateless: the API handler persists the exchange after this returns.
"""

from typing import cast

from pydantic import SecretStr

from papers_agent.agents.analyst_agent import AnalystAgent
from papers_agent.agents.orchestrator_prompts import SYSTEM_PROMPT
from papers_agent.agents.rag_agent import RAGAgent
from papers_agent.core.logging import get_logger
from papers_agent.core.models import Message, Role

log = get_logger(__name__)


class OrchestratorAgent:
    """Recebe pergunta + historico, decide quais agentes acionar via function
    calling, consolida a resposta. Stateless (nao persiste; o handler faz).
    """

    def __init__(
        self,
        rag_agent: RAGAgent,
        analyst_agent: AnalystAgent,
        api_key: SecretStr,
        model: str,
    ) -> None:
        self._rag = rag_agent
        self._analyst = analyst_agent
        self._api_key = api_key
        self._model = model

    async def handle(self, question: str, history: list[Message]) -> str:
        """Run one orchestration turn. Returns the LLM's final answer text."""
        # Lazy imports preserve the T2.6 fix that keeps google.genai's heavy
        # bootstrap (gRPC/httpx/OpenSSL) out of any module-load path on Windows.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key.get_secret_value())
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[
                self._rag.rag_retrieve_context,
                self._rag.rag_extract_section,
                self._analyst.analyst_compare_papers,
                self._analyst.analyst_summarize_paper,
                self._analyst.analyst_rank_papers,
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=21,
            ),
        )
        genai_history = [
            types.Content(
                role=("user" if m.role == Role.USER else "model"),
                parts=[types.Part(text=m.content)],
            )
            for m in history
        ]
        log.info(
            "orchestrator.handle.start",
            question_len=len(question),
            history_len=len(history),
        )
        # cast: list[Content] is structurally compatible with the SDK's
        # list[Content | ContentDict] union, but list invariance blocks it.
        chat = client.aio.chats.create(
            model=self._model,
            config=config,
            history=cast(list[types.Content | types.ContentDict], genai_history),
        )
        response = await chat.send_message(question)
        n_calls = len(getattr(response, "automatic_function_calling_history", []) or [])
        log.info("orchestrator.handle.done", afc_calls=n_calls)
        text = response.text
        if not text:
            return "Nao consegui gerar uma resposta a partir dos papers."
        return text
