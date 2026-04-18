# master_eval_learning — Laboratório de Avaliação de LLMs e Agentes de IA

> Repositório de aprendizado prático com **todos os tipos de avaliação de sistemas de IA** — métricas determinísticas, avaliadores semânticos com LLM-as-judge, RAG evaluation, avaliação de segurança, qualidade de geração de código e muito mais.

---

## Projetos

| Projeto | Tipo de avaliação | Stack principal | Status |
|---------|-------------------|-----------------|--------|
| [`deepEval_1`](#deepeval_1) | Pipeline offline · métricas determinísticas + LLM-judge opcional | Python stdlib + OpenAI (OpenRouter) | Concluído |
| [`llm_judge_rag`](#llm_judge_rag) | RAG evaluation · métricas semânticas via DeepEval + LLM-judge | DeepEval · ChromaDB · SentenceTransformers · OpenRouter | Concluído |

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

## Conceitos cobertos até agora

- Métricas determinísticas vs. semânticas — quando usar cada uma
- LLM-as-judge: como plugar qualquer provider no DeepEval via `DeepEvalBaseLLM`
- RAG evaluation: Faithfulness, AnswerRelevancy, ContextualRecall
- G-Eval: critérios customizados em linguagem natural
- Casos de falha deliberada — por que uma boa suite precisa reprovar o que merece reprovar
- Pipeline RAG do zero: indexador (ChromaDB), retriever (semântico), gerador (OpenRouter)
- Separação rígida entre sistema sendo avaliado e código de avaliação
- Integração com CI/CD via exit codes

---

## Requisitos gerais

```bash
# Instalar todas as dependências via Poetry
poetry install

# Ou individualmente
pip install deepeval openai chromadb sentence-transformers
```

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Roadmap

- [x] Pipeline offline com métricas determinísticas (`deepEval_1`)
- [x] RAG evaluation com LLM-judge via DeepEval + OpenRouter (`llm_judge_rag`)
- [ ] Avaliação com a biblioteca `deepeval` nativa — métricas LLM-based out-of-the-box
- [ ] Agent Evaluation: avaliação de trajetórias de agentes multi-step
- [ ] Benchmark com datasets públicos (MMLU, HellaSwag, TruthfulQA)
- [ ] Output em JSON/CSV para análise e dashboards
- [ ] Integração com LangSmith, Weights & Biases ou Arize para rastreamento
- [ ] Testes de regressão automatizados em GitHub Actions

---

*Este repositório é um laboratório de aprendizado — cada projeto é independente e focado em um conjunto específico de técnicas de avaliação.*

---

## Do que se trata

`deepEval_1` é uma **pipeline de avaliação offline** de respostas de LLMs, projetada para funcionar como uma **barreira de qualidade pré-deploy** (quality gate). Antes de colocar um modelo em produção, você executa a pipeline contra um conjunto de casos de teste — se os casos não passarem, o deploy é bloqueado.

A pipeline é **agnóstica de modelo**: ela recebe como entrada qualquer saída de LLM e a avalia com métricas determinísticas (gratuitas, sem API) e opcionalmente com um juiz LLM via OpenRouter.

### Casos de uso cobertos

| Categoria | O que avalia |
|-----------|-------------|
| `suporte` | Respostas de atendimento ao cliente (exatidão + aderência ao contexto) |
| `resumo` | Resumos de documentos (fundamentação vs. alucinação) |
| `extracao` | Extração de dados estruturados (JSON válido com campos corretos) |
| `codigo` | Geração de código (segurança + métricas personalizáveis) |

---

## Aprendizados

### 1. Avaliação determinística primeiro, LLM-judge depois
A pipeline implementa a filosofia **"cheap filters before expensive judges"**: métricas determinísticas (bigram overlap, keyword matching, JSON schema) rodam localmente sem custo. O LLM-judge só é chamado quando necessário e de forma opcional.

### 2. Métricas implementadas e o que ensinam

| Métrica | Tipo | Aprendizado-chave |
|---------|------|-------------------|
| `exact_match` | Exata | Útil para respostas fechadas; normalização (case, espaços, pontuação) é essencial |
| `contains_keywords` | Fuzzy | Mede cobertura temática sem exigir resposta idêntica; threshold proporcional |
| `valid_json_schema` | Estrutural | Valida formato E presença de campos; lida com JSON dentro de markdown (` ```json ``` `) |
| `groundedness` | Heurística | Detecta alucinações via sobreposição de bigramas — proxy barato para faithfulness |
| `no_harmful_content` | Segurança | Blocklist de palavras proibidas como primeira linha de defesa contra prompt injection |
| `llm_judge_eval` | Semântico | Avaliação holística por LLM (Claude via OpenRouter) com latência monitorada |

### 3. Avaliação por categoria (category-aware routing)
Métricas diferentes para tarefas diferentes. Uma resposta de suporte ao cliente exige exatidão; um resumo exige fundamentação; uma extração exige estrutura válida. A pipeline roteia cada caso para o conjunto correto de métricas.

### 4. Casos de teste intencionalmente falhos
O dataset inclui 5 casos que **devem falhar** (1 por categoria com erro deliberado). Isso ensina que uma boa suite de avaliação precisa testar não só o caminho feliz, mas também a capacidade de **detectar saídas ruins** — a pipeline só tem valor se consegue reprovar o que merece ser reprovado.

### 5. Integração com CI/CD via exit codes
A pipeline retorna `exit(0)` em sucesso e `exit(1)` em falha, permitindo uso direto em pipelines de CI:
```bash
python pipeline.py && echo "DEPLOY LIBERADO" || echo "DEPLOY BLOQUEADO"
```

### 6. Groundedness como proxy de alucinação
A métrica `groundedness` usa sobreposição de bigramas (pares de palavras consecutivas) entre a resposta do LLM e o contexto recuperado (RAG context). É uma aproximação heurística — não tão precisa quanto um LLM-judge, mas **gratuita e instantânea**. O aprendizado é entender quando um proxy barato é suficiente e quando vale pagar pela avaliação semântica.

---

## Requisitos

### Dependências obrigatórias
Apenas a **biblioteca padrão do Python 3.8+**. Nenhuma instalação necessária para rodar as métricas determinísticas:

- `argparse`, `dataclasses`, `datetime`, `json`, `os`, `re`, `sys`, `time`, `pathlib`

### Dependência opcional (LLM-judge)
Para usar o avaliador LLM via OpenRouter:

```bash
pip install openai
```

E configurar a variável de ambiente:

```bash
export OPENROUTER_API_KEY="sua-chave-aqui"
```

> Sem a chave ou a biblioteca, a pipeline continua funcionando normalmente — o LLM-judge é silenciosamente ignorado.

---

## Como rodar a pipeline

### Execução básica (somente métricas determinísticas)
```bash
python pipeline.py
```

### Com LLM-as-judge (requer `OPENROUTER_API_KEY`)
```bash
python pipeline.py --llm-judge
```

### Filtrar por categoria específica
```bash
python pipeline.py --category suporte
python pipeline.py --category resumo
python pipeline.py --category extracao
python pipeline.py --category codigo
```

### Parar na primeira falha (fail-fast)
```bash
python pipeline.py --fail-fast
```

### Definir taxa mínima de aprovação
```bash
# Aceitar até 20% de falha (útil durante desenvolvimento)
python pipeline.py --min-pass-rate 0.8
```

### Combinações
```bash
# Só suporte, com LLM-judge, fail-fast
python pipeline.py --category suporte --llm-judge --fail-fast

# Toda a pipeline, aceitando 80% de aprovação
python pipeline.py --min-pass-rate 0.8
```

---

## O que esperar de output

### Formato por caso de teste
```
══════════════════════════════════════════════════════════════
Pipeline de Avaliação Offline | 10 casos | sem LLM-judge
Iniciado em: 2026-04-18T14:32:00
══════════════════════════════════════════════════════════════

[PASS] suporte_01 (suporte) — score: 0.93
  ok  no_harmful_content  1.00  |  Nenhum conteúdo perigoso detectado
  ok  exact_match         1.00  |  Saída normalizada bate exatamente com esperado
  ok  contains_keywords   0.80  |  4/5 palavras encontradas
  ok  groundedness        0.82  |  82% dos bigramas presentes no contexto

[FAIL] suporte_02 (suporte) — score: 0.31
  ok  no_harmful_content  1.00  |  Nenhum conteúdo perigoso detectado
  !!  exact_match         0.00  |  Esperado: '90 dias' | Obtido: '30 dias'
  !!  contains_keywords   0.25  |  1/4 palavras encontradas
  !!  groundedness        0.20  |  Possível alucinação: baixa cobertura de bigramas

  Entrada : "Qual é o prazo para troca de produto?"
  Saída   : "O prazo para troca é de 30 dias."
```

### Resumo final
```
══════════════════════════════════════════════════════════════
RESULTADO FINAL
  Casos avaliados  : 10
  Aprovados        : 5
  Reprovados       : 5
  Taxa de aprovação: 50.0%  (mínimo exigido: 100.0%)

  PIPELINE REPROVADA — deploy bloqueado
══════════════════════════════════════════════════════════════
```

### Exit codes para CI/CD
| Código | Significado |
|--------|-------------|
| `0` | Pipeline aprovada — deploy liberado |
| `1` | Pipeline reprovada — deploy bloqueado |

---

## Estrutura do projeto

```
deepEval_1/
├── dataset.py      # 10 casos de teste sintéticos (5 PASS + 5 FAIL intencionais)
├── metrics.py      # 6 métricas de avaliação implementadas do zero
├── pipeline.py     # Orquestrador principal + CLI com exit codes para CI/CD
└── README.md       # Este arquivo
```

---

## Próximos passos sugeridos (para o projeto maior)

- [ ] Avaliação com a biblioteca `deepeval` — métricas LLM-based out-of-the-box
- [ ] RAG Evaluation completo: `ContextualPrecision`, `ContextualRecall`, `Faithfulness`
- [ ] Agent Evaluation: avaliação de trajetórias de agentes multi-step
- [ ] Benchmark com datasets públicos (MMLU, HellaSwag, TruthfulQA)
- [ ] Output em JSON/CSV para análise e dashboards
- [ ] Integração com LangSmith, Weights & Biases ou Arize para rastreamento
- [ ] Testes de regressão automatizados em GitHub Actions
