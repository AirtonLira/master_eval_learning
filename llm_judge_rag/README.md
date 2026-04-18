# llm_judge_rag — RAG Evaluation com LLM-Judge via OpenRouter

> **Contexto do Projeto Maior:** Este projeto faz parte do repositório `master_eval_learning`, cujo objetivo central é aprender e experimentar todos os tipos de avaliação de agentes de IA, LLMs e sistemas de inteligência artificial — desde métricas determinísticas simples até avaliadores semânticos com LLM-as-judge, passando por RAG evaluation, avaliação de segurança e muito mais.

---

## Do que se trata

`llm_judge_rag` é um sistema completo de **avaliação de RAG (Retrieval-Augmented Generation)** usando o framework **DeepEval** com um **LLM-judge customizado via OpenRouter**. O projeto simula um chatbot de RH corporativo e avalia a qualidade das respostas em cinco dimensões semânticas.

O diferencial em relação ao `deepEval_1`: aqui as métricas são **semânticas e baseadas em LLM-judge**, não determinísticas. Isso permite avaliar aspectos subjetivos como fidelidade ao contexto, relevância da resposta e qualidade de comunicação — que métricas heurísticas não conseguem capturar.

### Domínio escolhido: Chatbot de RH

O chatbot responde perguntas de colaboradores sobre políticas internas da empresa (férias, plano de saúde, home office, rescisão, benefícios, licenças). Este domínio foi escolhido porque:

- Respostas erradas têm impacto real (ex: informar prazo de aviso prévio incorreto)
- Alucinações são facilmente detectáveis (o contexto é bem delimitado)
- Permite testar múltiplas categorias de avaliação em um domínio coeso

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         SISTEMA (RAG)                           │
│                                                                 │
│  hr_documents.py   →   rag_engine.py                           │
│  (7 documentos de RH)   ├── build_index()   → ChromaDB         │
│                         ├── retrieve()      → busca semântica   │
│                         └── generate_answer() → LLM OpenRouter  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ respostas avaliadas por
┌─────────────────────────────────────────────────────────────────┐
│                        EVAL (DeepEval)                          │
│                                                                 │
│  openrouter_judge.py  →   test_hr_chatbot.py                   │
│  (OpenRouterJudge)         ├── FaithfulnessMetric              │
│  herda DeepEvalBaseLLM     ├── AnswerRelevancyMetric           │
│                            ├── ContextualRecallMetric          │
│                            ├── GEval (Tom Profissional)        │
│                            └── GEval (Clareza)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes

### `hr_documents.py` — Base de conhecimento

7 documentos de política de RH usados como fonte de verdade pelo RAG:

| ID | Título | Conteúdo principal |
|----|--------|--------------------|
| `ferias-001` | Política de Férias | 30 dias corridos, parcelamento, 1/3 constitucional |
| `plano-saude-001` | Plano de Saúde e Odontológico | Bradesco Saúde, 80%/60% titular/dependente, coparticipação |
| `home-office-001` | Política de Home Office | R$ 150/mês híbrido, R$ 300/mês full remote |
| `rescisao-001` | Rescisão e Aviso Prévio | 30 dias base + 3 dias por ano, máximo 90 dias |
| `beneficios-001` | Benefícios Gerais | VR R$ 35/dia, VA R$ 600/mês, VT, seguro de vida, Gympass |
| `desenvolvimento-001` | Desenvolvimento e Capacitação | Bolsa educação R$ 800/mês, budget R$ 3.000/ano |
| `licencas-001` | Licenças e Afastamentos | Maternidade 180 dias, paternidade 20 dias, luto, day off |

### `rag_engine.py` — Motor RAG

Pipeline RAG com 3 etapas encadeadas:

1. **Indexador** (`build_index`) — divide os documentos em chunks e cria um índice vetorial no ChromaDB usando embeddings locais do `all-MiniLM-L6-v2` via SentenceTransformers. Sem custo de API nesta etapa.
2. **Retriever** (`retrieve`) — busca os `top_k=3` chunks mais similares por similaridade cosseno para a query do usuário.
3. **Generator** (`generate_answer`) — monta o prompt com sistema + contexto recuperado e chama o LLM via OpenRouter com `temperature=0.1` para respostas factuais consistentes.

A função `ask_hr_chatbot` orquestra o pipeline completo e retorna um dict com `input`, `actual_output` e `retrieval_context` — exatamente os campos que o DeepEval precisa para calcular as métricas.

### `openrouter_judge.py` — LLM-judge customizado

Wrapper que integra o OpenRouter ao DeepEval como judge semântico:

```python
class OpenRouterJudge(DeepEvalBaseLLM):
    def generate(self, prompt: str) -> str: ...          # síncrono
    async def a_generate(self, prompt: str) -> str: ...  # assíncrono (paralelo)
```

O DeepEval espera essa interface para calcular métricas LLM-based. Herdar de `DeepEvalBaseLLM` desacopla o framework do provedor — qualquer API compatível com OpenAI (Anthropic, Ollama, OpenRouter) pode ser usada como judge sem alterar o código de avaliação.

O judge usa `temperature=0` para resultados determinísticos e reproduzíveis.

### `test_hr_chatbot.py` — Suite de avaliação

7 test cases cobrindo cenários reais e deliberadamente falhos:

| ID | Categoria | Cenário | Comportamento esperado |
|----|-----------|---------|------------------------|
| `tc-ferias-correto` | ferias | Resposta correta sobre férias anuais | Aprovado em todas as métricas |
| `tc-ferias-alucinacao` | ferias | LLM inventou abono pecuniário não documentado | Reprovado em Faithfulness |
| `tc-saude-correto` | beneficios | Cobertura e coparticipação do plano de saúde | Aprovado em todas as métricas |
| `tc-homeoffice-correto` | home_office | Valores corretos do auxílio home office | Aprovado em todas as métricas |
| `tc-fora-escopo` | fora_escopo | Pergunta sobre celular (não coberta pelas políticas) | Chatbot deve admitir que não sabe |
| `tc-licenca-paternidade` | licencas | Duração correta da licença paternidade | Aprovado em todas as métricas |
| `tc-irrelevante` | beneficios | Confundiu vale alimentação com vale transporte | Reprovado em AnswerRelevancy |

---

## Métricas de avaliação

| Métrica | Threshold | Tipo | O que avalia |
|---------|-----------|------|--------------|
| `FaithfulnessMetric` | 0.7 | Referenceless | O chatbot inventou algo além do contexto RAG? |
| `AnswerRelevancyMetric` | 0.7 | Referenceless | A resposta é relevante para a pergunta feita? |
| `ContextualRecallMetric` | 0.6 | Reference-based | O retriever trouxe contexto suficiente para a resposta esperada? |
| `GEval: Tom Profissional` | 0.7 | Referenceless | Linguagem adequada para comunicação corporativa de RH? |
| `GEval: Clareza` | 0.7 | Referenceless | Fácil de entender sem background jurídico ou de RH? |

**Referenceless**: avalia sem `expected_output` — usa só `input`, `actual_output` e `retrieval_context`.

**Reference-based**: exige `expected_output` — avalia se o retriever trouxe o que era necessário para cobrir a resposta esperada. Mede a qualidade do *retriever*, não do gerador.

---

## Aprendizados

### 1. LLM-judge como interface, não como acoplamento
Implementar `generate()` e `a_generate()` herdando de `DeepEvalBaseLLM` é o contrato mínimo para plugar qualquer LLM como judge. Você pode trocar OpenRouter por Anthropic, Ollama ou qualquer provider sem tocar no código de avaliação — só troca o judge.

### 2. Casos de falha deliberada são essenciais
Dois test cases foram projetados especificamente para reprovar:
- `tc-ferias-alucinacao` → deve reprovar em `FaithfulnessMetric` (inventou abono pecuniário)
- `tc-irrelevante` → deve reprovar em `AnswerRelevancyMetric` (respondeu sobre VT ao invés de VA)

Uma suite que só tem casos felizes não mede nada relevante — ela só confirma que o sistema funciona quando está certo, nunca quando está errado.

### 3. GEval: critérios customizados em linguagem natural
O framework G-Eval (Wei et al., 2023) permite definir critérios de qualidade em português simples. O LLM-judge lê o critério, avalia o output e retorna um score de 0 a 1. Isso permite medir dimensões subjetivas — tom, clareza, empatia, profissionalismo — que nenhuma métrica determinística consegue capturar.

### 4. Separação rígida entre sistema e eval
`rag_engine.py` (o sistema sendo avaliado) fica completamente separado de `test_hr_chatbot.py` (o código de avaliação). O sistema não importa nada do DeepEval; o eval importa o sistema mas não interfere nele. Esta separação garante que a avaliação não contamine o comportamento do que está sendo avaliado.

### 5. Embeddings locais no retriever, LLM remoto no judge
A indexação e recuperação usam `all-MiniLM-L6-v2` localmente via SentenceTransformers, sem custo de API. Só a geração de respostas e a avaliação semântica consomem créditos no OpenRouter. Isso otimiza custo: o trabalho pesado de similaridade fica local; o julgamento semântico vai para o LLM.

### 6. Modo degradado sem API key
Os testes pulam automaticamente com `pytest.skip()` se `OPENROUTER_API_KEY` não estiver definida, em vez de falhar com erro obscuro. Isso facilita explorar o código, entender a estrutura e estudar os test cases sem precisar configurar credenciais.

### 7. Avaliação síncrona vs. assíncrona
O `a_generate()` assíncrono permite que o DeepEval avalie múltiplos test cases em paralelo com `run_async=True`. Para 7 casos × 5 métricas = 35 chamadas ao judge, a execução paralela reduz o tempo de 35× latência para ~5× latência.

---

## Requisitos

### Dependências

```bash
pip install deepeval openai chromadb sentence-transformers
```

Ou via Poetry (usa o `pyproject.toml` na raiz do projeto):

```bash
cd ..  # raiz do master_eval_learning
poetry install
```

### Variável de ambiente

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Como rodar

### Testar o motor RAG isoladamente (sem eval)

```bash
python rag_engine.py
```

Executa 3 perguntas de demonstração, imprime as respostas do chatbot e o número de chunks recuperados por pergunta.

### Testar o wrapper do judge

```bash
python openrouter_judge.py
```

Faz uma pergunta simples ao judge e imprime a resposta — confirma que a API key e o modelo estão funcionando.

### Rodar a suite de eval com pytest

```bash
# Todas as métricas, todos os casos
pytest test_hr_chatbot.py -v

# Filtrar por métrica específica
pytest test_hr_chatbot.py -v -k "faithfulness"
pytest test_hr_chatbot.py -v -k "answer_relevancy"
pytest test_hr_chatbot.py -v -k "contextual_recall"
pytest test_hr_chatbot.py -v -k "tom_profissional"

# Filtrar por caso específico
pytest test_hr_chatbot.py -v -k "alucinacao"
pytest test_hr_chatbot.py -v -k "irrelevante"
```

### Rodar avaliação em batch (sem pytest)

```bash
python test_hr_chatbot.py
```

Avalia todos os casos com `deepeval.evaluate()` e `run_async=True`, imprime relatório consolidado e (com login no DeepEval Cloud) envia resultados para a plataforma de rastreamento.

---

## O que esperar de output (pytest)

```
test_hr_chatbot.py::test_faithfulness[tc-ferias-correto] PASSED
  Faithfulness score: 0.950
  Reason: All claims in the output can be attributed to the retrieval context.

test_hr_chatbot.py::test_faithfulness[tc-ferias-alucinacao] FAILED
  Faithfulness score: 0.120
  Reason: The claim about 'abono pecuniário' cannot be found in the retrieval context.
  [FALHA] tc-ferias-alucinacao
  Score: 0.120 (threshold: 0.700)
```

---

## Estrutura do projeto

```
llm_judge_rag/
├── hr_documents.py       # Base de conhecimento: 7 políticas de RH
├── rag_engine.py         # Motor RAG: indexador (ChromaDB), retriever e gerador
├── openrouter_judge.py   # LLM-judge: wrapper OpenRouter ↔ DeepEvalBaseLLM
├── test_hr_chatbot.py    # Suite de eval: 5 métricas × 7 test cases com pytest
└── README.md             # Este arquivo
```

---

## Próximos passos sugeridos

- [ ] Substituir respostas mockadas por chamadas reais ao `rag_engine` nos testes (modo `--live`)
- [ ] Adicionar `ContextualPrecisionMetric` — avalia se os chunks recuperados são todos relevantes (qualidade do ranking)
- [ ] Testar com modelos diferentes no OpenRouter (`google/gemini-flash`, `meta-llama/llama-3`)
- [ ] Output dos resultados em JSON/CSV para análise histórica e dashboards
- [ ] Integração com DeepEval Cloud para rastreamento de regressões entre versões
- [ ] Chunking por parágrafo ou sliding window no indexador (atualmente 1 chunk por documento)
- [ ] Testes de regressão automatizados em GitHub Actions
