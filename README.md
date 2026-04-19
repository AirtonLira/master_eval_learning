# master_eval_learning — Laboratório de Avaliação de LLMs e Agentes de IA

> Repositório de aprendizado prático com **todos os tipos de avaliação de sistemas de IA** — métricas determinísticas, avaliadores semânticos com LLM-as-judge, RAG evaluation, avaliação de segurança, qualidade de geração de código e muito mais.

---

## Projetos

| Projeto | Tipo de avaliação | Stack principal | Status |
|---------|-------------------|-----------------|--------|
| [`deepEval_1`](#deepeval_1) | Pipeline offline · métricas determinísticas + LLM-judge opcional | Python stdlib + OpenAI (OpenRouter) | Concluído |
| [`llm_judge_rag`](#llm_judge_rag) | RAG evaluation · métricas semânticas via DeepEval + LLM-judge | DeepEval · ChromaDB · SentenceTransformers · OpenRouter | Concluído |
| [`ragas`](#ragas) | RAG evaluation · framework RAGAS nativo (Faithfulness + AnswerRelevancy) | RAGAS 0.4 · OpenRouter · SentenceTransformers · AsyncOpenAI | Concluído |

---

## `deepEval_1`

Pipeline de **avaliação offline** de respostas de LLMs projetada como barreira de qualidade pré-deploy (quality gate). Antes de colocar um modelo em produção, você executa a pipeline contra um conjunto de casos de teste — se os casos não passarem, o deploy é bloqueado.

A pipeline é **agnóstica de modelo**: avalia qualquer saída de LLM com métricas determinísticas (gratuitas, sem API) e opcionalmente com um juiz LLM via OpenRouter.

### Métricas implementadas

| Métrica | Tipo | O que avalia |
|---------|------|--------------|
| `exact_match` | Exata | Resposta idêntica à esperada (normalizada) |
| `contains_keywords` | Fuzzy | Cobertura temática por palavras-chave |
| `valid_json_schema` | Estrutural | JSON válido com campos obrigatórios presentes |
| `groundedness` | Heurística | Sobreposição de bigramas como proxy de alucinação |
| `no_harmful_content` | Segurança | Blocklist contra prompt injection |
| `llm_judge_eval` | Semântico | Avaliação holística por LLM via OpenRouter |

### Casos de uso cobertos

| Categoria | O que avalia |
|-----------|-------------|
| `suporte` | Respostas de atendimento ao cliente (exatidão + aderência ao contexto) |
| `resumo` | Resumos de documentos (fundamentação vs. alucinação) |
| `extracao` | Extração de dados estruturados (JSON válido com campos corretos) |
| `codigo` | Geração de código (segurança + métricas personalizáveis) |

### Como rodar

```bash
# Somente métricas determinísticas
python deepEval_1/pipeline.py

# Com LLM-judge (requer OPENROUTER_API_KEY)
python deepEval_1/pipeline.py --llm-judge

# Filtrar por categoria e fail-fast
python deepEval_1/pipeline.py --category suporte --fail-fast
```

Integração com CI/CD via exit codes: `exit(0)` = deploy liberado, `exit(1)` = deploy bloqueado.

**[README completo →](deepEval_1/README.md)**

---

## `llm_judge_rag`

Sistema completo de **avaliação de RAG** usando o framework **DeepEval** com um **LLM-judge customizado via OpenRouter**. Simula um chatbot de RH corporativo e avalia a qualidade das respostas em cinco dimensões semânticas.

Enquanto o `deepEval_1` usa métricas determinísticas e heurísticas, o `llm_judge_rag` usa **métricas semânticas baseadas em LLM-judge** — permitindo avaliar aspectos subjetivos como fidelidade ao contexto, relevância e qualidade de comunicação que heurísticas não capturam.

### Arquitetura

```
hr_documents.py (7 políticas de RH)
        ↓
rag_engine.py: build_index (ChromaDB) → retrieve (semântico) → generate_answer (OpenRouter)
        ↓ respostas avaliadas por
openrouter_judge.py (OpenRouterJudge herda DeepEvalBaseLLM)
        ↓
test_hr_chatbot.py: 5 métricas × 7 test cases
```

### Métricas de avaliação

| Métrica | Threshold | O que avalia |
|---------|-----------|--------------|
| `FaithfulnessMetric` | 0.7 | O chatbot inventou algo além do contexto RAG? |
| `AnswerRelevancyMetric` | 0.7 | A resposta é relevante para a pergunta? |
| `ContextualRecallMetric` | 0.6 | O retriever trouxe contexto suficiente? |
| `GEval: Tom Profissional` | 0.7 | Linguagem adequada para comunicação de RH? |
| `GEval: Clareza` | 0.7 | Fácil de entender sem background técnico? |

### Como rodar

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Suite de eval completa
pytest llm_judge_rag/test_hr_chatbot.py -v

# Por métrica ou caso específico
pytest llm_judge_rag/test_hr_chatbot.py -v -k "faithfulness"
pytest llm_judge_rag/test_hr_chatbot.py -v -k "alucinacao"

# Batch eval (sem pytest)
python llm_judge_rag/test_hr_chatbot.py
```

**[README completo →](llm_judge_rag/README.md)**

---

## `ragas`

Avaliação de RAG usando o framework **RAGAS** (Retrieval Augmented Generation Assessment) — biblioteca dedicada a métricas de RAG, com API assíncrona e métricas implementadas nativamente sem necessidade de wrappers externos.

Enquanto o `llm_judge_rag` usa DeepEval com um judge customizado via OpenRouter, este projeto usa o **RAGAS diretamente** — uma alternativa mais focada em RAG, com vocabulário próprio e fábricas (`llm_factory` / `embedding_factory`) que abstraem o provider por trás de um cliente OpenAI-compatível apontado para o OpenRouter.

### Arquitetura

```
SingleTurnSample × N  →  EvaluationDataset  →  Faithfulness + AnswerRelevancy
       ↑                                                ↑
       │                                                │
   user_input, response,                         LLM (llm_factory) +
   retrieved_contexts,                           Embeddings (embedding_factory)
   reference                                     via OpenRouter / HuggingFace
```

### Mapeamento de vocabulário — DeepEval vs. RAGAS

| DeepEval | RAGAS |
|----------|-------|
| `input` | `user_input` |
| `actual_output` | `response` |
| `retrieval_context` | `retrieved_contexts` (lista de strings) |
| `expected_output` | `reference` |

Mesmos conceitos, nomes diferentes — entender o mapeamento facilita a transição entre os dois frameworks e evita retrabalho ao portar datasets entre ferramentas.

### Métricas implementadas

| Métrica | O que avalia |
|---------|--------------|
| `Faithfulness` | A resposta é fundamentada no `retrieved_contexts` (detecta alucinações) |
| `AnswerRelevancy` | A resposta está semanticamente relacionada à pergunta (requer embeddings) |

### Stack

- **LLM judge**: `meta-llama/llama-3.1-8b-instruct` via OpenRouter com cliente `AsyncOpenAI`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local — sem custo de API)
- **API do RAGAS**: versão 0.4+ com `llm_factory`, `embedding_factory` e `ascore` (assíncrono)

### Como rodar

```bash
export OPENROUTER_API_KEY="sk-or-..."

python ragas/ragas_eval.py
```

### Dataset de demonstração

Duas amostras sobre política de férias:
- **Amostra 1**: resposta fiel ao contexto sobre 30 dias corridos de férias (alta Faithfulness esperada)
- **Amostra 2**: alucinação intencional — o LLM inventa "abono pecuniário" que não existe no contexto recuperado (baixa Faithfulness esperada)

O dataset pequeno é proposital: foco em exercitar a API nova do RAGAS (`SingleTurnSample` / `EvaluationDataset` / `ascore`), não em cobertura exaustiva de casos.

### Estrutura do projeto

```
ragas/
├── ragas_eval.py    # Dataset + avaliação Faithfulness + AnswerRelevancy (async)
└── ragas_judge.py   # Módulo auxiliar: fábrica de LLM judge via OpenRouter
```

---

## Conceitos cobertos até agora

- Métricas determinísticas vs. semânticas — quando usar cada uma
- LLM-as-judge: como plugar qualquer provider no DeepEval via `DeepEvalBaseLLM`
- RAG evaluation: Faithfulness, AnswerRelevancy, ContextualRecall
- G-Eval: critérios customizados em linguagem natural
- Casos de falha deliberada — por que uma boa suite precisa reprovar o que merece reprovar
- Pipeline RAG do zero: indexador (ChromaDB), retriever (semântico), gerador (OpenRouter)
- Separação rígida entre sistema sendo avaliado e código de avaliação
- Integração com CI/CD via exit codes
- RAGAS como framework dedicado a RAG — `SingleTurnSample`, `EvaluationDataset`, `llm_factory`, `embedding_factory`
- Mapeamento de vocabulário entre DeepEval e RAGAS — mesmos conceitos, nomes diferentes
- Avaliação assíncrona com `ascore` para paralelizar chamadas ao LLM judge
- Uso de embeddings HuggingFace locais para eliminar custo de API em métricas que requerem similaridade semântica

---

## Requisitos gerais

```bash
# Instalar todas as dependências via Poetry
poetry install

# Ou individualmente
pip install deepeval openai chromadb sentence-transformers ragas langchain-huggingface
```

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Roadmap

- [x] Pipeline offline com métricas determinísticas (`deepEval_1`)
- [x] RAG evaluation com LLM-judge via DeepEval + OpenRouter (`llm_judge_rag`)
- [x] RAG evaluation com framework RAGAS nativo (`ragas`)
- [ ] Avaliação com a biblioteca `deepeval` nativa — métricas LLM-based out-of-the-box
- [ ] Agent Evaluation: avaliação de trajetórias de agentes multi-step
- [ ] Benchmark com datasets públicos (MMLU, HellaSwag, TruthfulQA)
- [ ] Output em JSON/CSV para análise e dashboards
- [ ] Integração com LangSmith, Weights & Biases ou Arize para rastreamento
- [ ] Testes de regressão automatizados em GitHub Actions

---

*Este repositório é um laboratório de aprendizado — cada projeto é independente e focado em um conjunto específico de técnicas de avaliação.*
