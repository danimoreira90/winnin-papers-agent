# Winnin Papers Agent

Sistema multi-agente que responde perguntas sobre cinco papers de Machine
Learning (Attention Is All You Need, BERT, RAG, ReAct, Toolformer) via
Retrieval-Augmented Generation. API REST em FastAPI, memoria de conversa por
thread em SQLite, 100% local em Docker. Sobe e roda com um comando:
`make setup && make run`.

---

## 1. Visao geral da arquitetura

Fluxo de uma pergunta:

```
Usuario
  |
  v
POST /threads/{id}/messages
  |
  v
+-----------------------------+
| OrchestratorAgent           |   recebe pergunta + historico da thread
|   .handle(question, hist)   |
+--------------+--------------+
               |
               | Gemini Automatic Function Calling (AFC)
               v
   +-----------+------------+        +------------------------+
   | RAGAgent               |        | AnalystAgent           |
   | - rag_retrieve_context |<------ | (injeta o RAGAgent     |
   | - rag_extract_section  |  usa   |  pra recuperar contexto)|
   +-----------+------------+        | - analyst_compare      |
               |                     | - analyst_summarize    |
               |                     | - analyst_rank         |
               |                     +-----------+------------+
               |                                 |
               v                                 v
   +-----------------------------------------------------+
   |  Tools (5 atomicas, sem memoria, IO tipado)         |
   |  search_documents | extract_section                 |  -> RAG
   |  compare_papers   | summarize       | rank_papers   |  -> Analyst
   +-----------------------------------------------------+
                          |
                          v
   +-----------------------------------------------------+
   |  Portas (core/ports.py, Protocols)                  |
   |  EmbeddingClient | LLMClient | VectorStoreClient    |
   +-----------------------------------------------------+
                          ^
                          | implementam
   +-----------------------------------------------------+
   |  Adapters (infra/)                                  |
   |  GeminiEmbeddingClient | GeminiLLMClient            |
   |  ChromaVectorStore                                  |
   +-----------------------------------------------------+
                          |
                          v
   ThreadRepository.add_message(USER + ASSISTANT) -> SQLite -> HTTP 200
```

Clean Architecture, dependencias apontando para dentro:

```
api  ->  agents  ->  tools  ->  core.ports  <-  infra (adapters)
```

- `core/` nao depende de nada do projeto (so stdlib + pydantic). `core/ports.py`
  tem as interfaces (`EmbeddingClient`, `LLMClient`, `VectorStoreClient`) como
  `typing.Protocol`, mais os tipos de resultado (`QueryResult`, `GetResult`).
- `tools/` e `agents/` dependem das portas, nunca dos adapters concretos.
- `infra/` implementa as portas (unico lugar que conhece `chromadb`,
  `google.genai`, `sqlalchemy`).
- `api/dependencies.py` e o composition root: monta o grafo concreto.

Um fitness test (`tests/test_architecture.py`) parseia o AST de todos os modulos
e falha se algum import violar essa direcao. O mesmo teste enforce o limite de
200 linhas por arquivo.

---

## 2. Distincao entre tools e agentes

Tres papeis com responsabilidades disjuntas:

**Tool** -- capacidade atomica e reutilizavel. Sem memoria, sem decisao, sem
encadeamento. Uma operacao, schema Pydantic tipado, `async`, retorno padronizado
via `ToolResult` (`success`, `data`, `error`, `metadata`). As cinco:

- `search_documents` -- busca semantica com filtro por `paper_id` (RAG).
- `extract_section` -- recupera trechos de uma secao especifica (RAG).
- `compare_papers` -- comparacao estruturada entre N papers (Analyst).
- `summarize` -- resume um paper em bullets (Analyst).
- `rank_papers` -- ranqueia papers por um criterio, com justificativa (Analyst).

**Agente** -- responsabilidade propria, conjunto exclusivo de tools, sabe quando
e como compor essas tools, reporta ao orquestrador.

- `RAGAgent` -- recupera contexto. Tools: `search_documents`, `extract_section`.
- `AnalystAgent` -- produz analises. Tools: `compare_papers`, `summarize`,
  `rank_papers`. Usa o `RAGAgent` injetado pra pre-buscar contexto.

**Orquestrador** -- sem tools proprias. Recebe pergunta + historico, expoe os
cinco metodos dos agentes ao Gemini, e deixa o Automatic Function Calling decidir
o que chamar e em que ordem. Consolida a resposta final.

As cinco funcoes expostas ao Gemini:

```
rag_retrieve_context(query, paper_ids, top_k)
rag_extract_section(paper_id, section)
analyst_compare_papers(paper_ids, aspect)
analyst_summarize_paper(paper_id, max_bullets)
analyst_rank_papers(criterion, paper_ids)
```

Por que assim: a tool e testavel isolada (com mocks, sem rede) e reutilizavel; o
agente carrega o saber de orquestrar suas tools; o orquestrador so resolve o
roteamento entre agentes. Cada camada tem seu teste: unitario pra tool,
mocked-LLM pro agente, integracao pro fluxo agente -> tool -> adapter.

---

## 3. Instrucoes de setup

**Pre-requisitos:** Docker Engine 25+ com `docker compose` v2; GNU Make; chave da
Google AI Studio (free tier, em https://aistudio.google.com/app/apikey).

```bash
git clone https://github.com/danimoreira90/winnin-papers-agent.git
cd winnin-papers-agent
cp .env.example .env     # preencha GEMINI_API_KEY; o resto ja tem defaults
make setup               # compila, sobe chroma+api, baixa os 5 PDFs e ingere no Chroma
make run                 # roda as 5 perguntas (uma thread por pergunta)
```

> **Nota sobre rate limit -- importante para a avaliacao**
>
> As perguntas usam o free tier do Gemini, que apos os cortes de dez/2025 ficou
> apertado (5-15 RPM, RPD baixo). Em um unico `make run`, uma ou duas perguntas
> podem tomar rate limit (429) -- em geral a Q3 (compare) ou a Q5 (resumo dos
> cinco), as mais pesadas. E limitacao externa de quota, nao do sistema: o
> pipeline trata com retry em dois niveis, espacamento entre perguntas
> (`_QUESTION_DELAY_SECONDS`) e degradacao graciosa.
>
> Para ver as cinco respostas limpas em um run: habilite billing na chave
> (Tier 1 = 150-300 RPM, upgrade instantaneo, custo de centavos); ou use uma
> chave free com quota diaria descansada (RPD reseta a meia-noite, horario do
> Pacifico).
>

`make setup` e idempotente (pula PDF/chunks que ja existem). Docs OpenAPI
interativas em http://localhost:8080/docs.

Outros alvos: `make build` (so compila), `make up` (sobe sem ingerir),
`make ingest` (re-ingere), `make test` (pytest no container), `make logs`,
`make down` (para, mantem volumes), `make down-volumes` (para e apaga volumes),
`make help`.

**Endpoints:**

| Metodo | Caminho                          | Descricao                          |
|--------|----------------------------------|------------------------------------|
| POST   | `/threads`                       | Cria thread, retorna `thread_id`.  |
| GET    | `/threads`                       | Lista threads.                     |
| POST   | `/threads/{thread_id}/messages`  | Posta pergunta, recebe resposta.   |
| GET    | `/threads/{thread_id}/messages`  | Historico da thread.               |
| GET    | `/health`                        | Liveness probe.                    |

**As cinco perguntas de avaliacao** (em `scripts/run_questions.py`): mecanismo do
Attention vs. RNNs; como o RAG combina recuperacao e geracao + limitacoes; ReAct
vs. Toolformer; qual paper e mais relevante pra um agente com ferramentas; resumo
executivo dos cinco papers.

---

## 4. Decisoes tecnicas

**Framework -- `google-genai` puro com Automatic Function Calling.** Dependencia
minima, `async` nativo (`client.aio`), function calling nativo
(`automatic_function_calling=True`): a SDK extrai o schema dos type hints e
docstrings e resolve o loop de tool-call sozinha. Pra dois agentes + um
orquestrador, LangGraph/CrewAI/AutoGen so somariam peso -- o AFC ja faz o
roteamento de graca.

**Vector store -- ChromaDB em HTTP server (container separado).** Persistencia
via volume, client async-friendly, metadata filters nativos
(`where={"paper_id": ...}`). Container `chromadb/chroma:1.5.9` na porta 8000,
colecao `papers`. Versus FAISS, ganhamos persistencia e filtros sem manter um
index serializado a parte.

**Embedding -- `gemini-embedding-001` (3072 dims, normalizado).** Mesmo provedor
do LLM, entao uma unica chave resolve tudo. O `text-embedding-004` foi
descontinuado pelo Google em 14/01/2026; este e o sucessor recomendado.

**Chunking -- ~800 tokens, 100 de overlap, extracao com PyMuPDF.** Parametrizado
no `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`). 800 equilibra contexto e precisao; o
overlap mantem inteira, em um dos chunks, uma frase relevante que cai na
fronteira. Cada chunk leva `paper_id`, `section`, `page` em metadata.

**LLM -- `gemini-2.5-flash-lite` (desvio documentado).** O enunciado pedia
"Gemini 2.0 Flash", mas em 2026 o Google tirou o `gemini-2.0-flash` do free tier
(0 RPD pra chaves novas) e cortou o RPD do `2.5-flash` (~20/dia). O
`2.5-flash-lite` tem o free tier mais generoso (~1000 RPD), mantem function
calling + API `aio`, e responde bem as cinco perguntas. Configuravel via
`GEMINI_MODEL` -- troca de modelo sem mexer no codigo.

**Persistencia -- SQLite via SQLAlchemy 2.0 async + `aiosqlite`.** Arquivo unico
em volume. Thread por UUID; mensagem com `role` (`USER`/`ASSISTANT`), `content`,
`created_at`. O orquestrador recebe o historico a cada turno -- viabiliza
perguntas de acompanhamento ("detalhe o ponto 2").

**Async em todo lugar.** Gemini `aio`, Chroma HTTP, sessoes SQLAlchemy async,
handlers e lifespan async. As tools do AFC sao corrotinas (`chats.create` vem de
`client.aio`), entao o loop nao bloqueia. Pydantic v2 (com `pydantic-settings`)
em todos os modelos.

**Qualidade.** `mypy --strict`, `ruff` (`E,F,I,B,UP,N,ASYNC,SIM,RUF`, linha 100),
`structlog` (zero `print` em producao), excecoes tratadas nas fronteiras (cada
tool captura e retorna `success=False`). Dois fitness tests: <=200 linhas por
arquivo e direcao de dependencia via AST.

**Rate limit.** Retry em dois niveis: o `gemini_client` faz backoff em 429
(embedding e generate); o orquestrador adiciona `HttpRetryOptions(attempts=6,
initial_delay=5s, max_delay=64s)` pros turnos internos do AFC. O
`run_questions.py` espera 30s entre perguntas e captura erro por pergunta, pra
uma falha nao abortar o lote.

**Testes -- 31, em tres grupos.** 27 unitarios (um por tool + repositorio + dois
agentes + rotas), 2 de integracao (`test_orchestration_flow.py`: agente -> tool
-> infra mockada nos limites dos adapters), 2 de arquitetura. Rodam em ~3-7s. As
rotas sao testadas chamando os handlers direto com `AsyncMock` (motivo na secao
de limitacoes).

---

## 5. Limitacoes conhecidas

**Rate limit do free tier.** O `gemini-2.5-flash-lite` free tem RPM/RPD
apertados; a Q5 (resumo dos cinco papers) e a mais pesada, com cinco `summarize`
sequenciais. Mitigado com retry em dois niveis, 30s entre perguntas e degradacao
graciosa (pergunta que estoura cota vira mensagem amigavel e o lote segue). Um
5/5 limpo sai com chave descansada ou paga; runs repetidos em janela curta podem
bater no teto.

**Eco do wrapper AFC.** A SDK do `google-genai` embrulha o retorno das tools em
`{'result': ...}` antes de devolver pro modelo. Numa execucao o modelo ecoou esse
dict cru no inicio da Q3. Mitigado pela regra 8 do `SYSTEM_PROMPT` (desembrulhar
e nunca imprimir estrutura crua) -- best-effort, o modelo nao e deterministico.

**Rotulos de citacao na Q5.** No resumo executivo o rotulo por paper varia
(`paper_id`, titulo ou abreviacao) porque o modelo compoe a formatacao. Ids
canonicos: `attention`, `bert`, `rag`, `react`, `toolformer`. Conteudo correto;
inconsistencia cosmetica.

**Detector de secao do `extract_section`.** A heuristica por regex super-casa
linhas do corpo que comecam com palavras de cabecalho, inflando a contagem de
secoes. NAO afeta a busca semantica (`search_documents` opera sobre o texto
chunkado) -- so a precisao do `extract_section`.

**Cobertura 69%.** O nao-coberto e o fio externo: `orchestrator.handle` (entra na
SDK `google.genai`), o `lifespan`, `pdf_parser.py`, e os limites de adapter dos
clients -- validados por smoke e `make run`, nao por unit. A logica de negocio
(tools, agentes, repositorio, rotas, ports) fica em 90-100%.

**Sem `TestClient` da FastAPI.** As rotas sao testadas com chamada direta + mocks
porque o `TestClient` dispara o `lifespan` -> `build_orchestrator` ->
`import google.genai`, que trava o `pytest` no Windows (conflito do bootstrap do
OpenSSL; em producao e contornado pela carga lazy no `handle`). A stack ASGI
completa e validada pelo `make run`.

**Nao implementado.** Streaming (o `run_questions` precisa do JSON inteiro);
auth/multi-tenancy (escopo e local); reranking (a `gemini-embedding-001` sozinha
bastou); verificacao de citacao (confiada a regra 2 do `SYSTEM_PROMPT`); Q5
paralela (mantida sequencial porque `summarize` e atomica por paper e o ganho de
wall-time nao compensa o risco de rate limit).
