"""
test_hr_chatbot.py
------------------
Suite de eval com DeepEval — Módulo 2 do curso.
 
Este arquivo é o coração do projeto: aqui você vê na prática
como o DeepEval estrutura avaliações de sistemas RAG.
 
Métricas utilizadas:
  1. FaithfulnessMetric   — o chatbot inventou algo além do contexto?
  2. AnswerRelevancyMetric — a resposta é relevante para a pergunta?
  3. ContextualRecallMetric — o retriever trouxe o que era necessário?
  4. GEval (tom profissional)  — critério customizado via LLM-judge
  5. GEval (clareza)           — critério customizado via LLM-judge
 
Anatomia de um LLMTestCase no DeepEval:
  ┌─────────────────────────────────────────────────────┐
  │  LLMTestCase(                                       │
  │    input            = pergunta do usuário           │
  │    actual_output    = resposta gerada pelo LLM      │
  │    expected_output  = resposta esperada (opcional)  │
  │    retrieval_context= chunks do RAG ← CHAVE para    │
  │  )                    FaithfulnessMetric            │
  └─────────────────────────────────────────────────────┘
 
Como rodar:
  # Sem API (usa respostas mockadas — modo de aprendizado)
  pytest test_hr_chatbot.py -v
 
  # Com API real (chama OpenRouter de verdade)
  arquivo .env com OPENROUTER_API_KEY=sk-or-...
  pytest test_hr_chatbot.py -v --live
"""

import os
import pytest
 
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
 
from openrouter_judge import OpenRouterJudge


# 1. configuração do judge com openrouter
def get_judge() -> OpenRouterJudge:
    """
    Retorna o LLM-judge configurado.
 
    Em modo mock (sem API key), retorna None e as métricas
    usarão o judge padrão (que falhará graciosamente nos testes).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    return OpenRouterJudge(
        model="openrouter/free",
        api_key=api_key,
    )
    

# 2. MÉTRICAS
#
# Cada métrica tem um `threshold` — score mínimo para aprovação.
# O DeepEval calcula internamente usando o LLM-judge e retorna
# um float de 0.0 a 1.0.
#
# IMPORTANTE: métricas são instanciadas com o judge customizado.
# Se judge=None, o DeepEval tentará usar OpenAI (vai falhar sem key).

def build_metrics(judge: OpenRouterJudge) -> dict:
    """
    Constrói todas as métricas do pipeline de eval.
 
    Retorna um dict nomeado para facilitar seleção por teste.
    """
    kwargs = {"model": judge} if judge else {}
 
    # ── Métricas nativas do DeepEval ────────────────────────────────────────
 
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        # Verifica se cada claim do output pode ser
        # inferido a partir do retrieval_context.
        # Referenceless: não precisa de expected_output.
        **kwargs,
    )
 
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        # Verifica se a resposta é relevante para o input.
        # Referenceless: avalia input ↔ actual_output.
        **kwargs,
    )
 
    contextual_recall = ContextualRecallMetric(
        threshold=0.6,
        # Verifica se o expected_output pode ser atribuído
        # ao retrieval_context. Reference-based: precisa de
        # expected_output. Mede se o retriever trouxe o suficiente.
        **kwargs,
    )
 
    # ── Métricas customizadas com GEval ─────────────────────────────────────
    #
    # GEval = G-Eval framework (Wei et al., 2023)
    # Você define o critério em linguagem natural.
    # O judge avalia e retorna um score de 0 a 1.
    #
    # evaluation_params define quais campos do LLMTestCase
    # o judge vai receber para fazer a avaliação.
 
    tom_profissional = GEval(
        name="Tom Profissional de RH",
        criteria=(
            "A resposta usa linguagem profissional e respeitosa, "
            "adequada para comunicação corporativa de RH. "
            "Evita gírias, jargões técnicos desnecessários e tom informal. "
            "É empática e prestativa."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        **kwargs,
    )
 
    clareza = GEval(
        name="Clareza para Colaboradores",
        criteria=(
            "A resposta é clara e fácil de entender para um colaborador "
            "sem background jurídico ou de RH. Valores e prazos são "
            "apresentados de forma direta. Se houver múltiplas informações, "
            "estão bem organizadas."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        **kwargs,
    )
 
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "contextual_recall": contextual_recall,
        "tom_profissional": tom_profissional,
        "clareza": clareza,
    }
    
    # 3. DATASET DE TESTE
    # Cada entrada representa uma interação real do chatbot.
    # Para cada caso, definimos:
    #   - input: o que o colaborador perguntou
    #   - actual_output: o que o chatbot respondeu (mockado aqui,
    #     substituído pela resposta real quando --live)
    #   - expected_output: resposta esperada (para contextual_recall)
    #   - retrieval_context: chunks que o RAG recuperou
    #   - category: para análise segmentada nos relatórios
    
TEST_CASES_DATA = [

    # ── Caso 1: Férias ───────────────────────────────────────────────────────
    {
        "id": "tc-ferias-correto",
        "category": "ferias",
        "input": "Quantos dias de férias tenho direito por ano?",
        "actual_output": (
            "Você tem direito a 30 dias corridos de férias após completar "
            "12 meses de trabalho. As férias podem ser divididas em até 3 "
            "períodos: o maior deve ter no mínimo 14 dias e os outros no "
            "mínimo 5 dias cada. O pagamento é feito até 2 dias antes do "
            "início, com acréscimo de 1/3 constitucional."
        ),
        "expected_output": (
            "30 dias corridos após 12 meses de trabalho, com possibilidade "
            "de parcelamento em até 3 períodos."
        ),
        "retrieval_context": [
            "[Política de Férias]\nTodo colaborador tem direito a 30 dias corridos "
            "de férias após completar 12 meses de trabalho (período aquisitivo). "
            "As férias podem ser parceladas em até 3 períodos, sendo que um deles "
            "não pode ser inferior a 14 dias corridos e os demais não podem ser "
            "inferiores a 5 dias corridos cada. O pagamento das férias deve ser "
            "realizado até 2 dias antes do início do período de descanso, com "
            "acréscimo de 1/3 constitucional."
        ],
    },

    # ── Caso 2: Alucinação de férias ─────────────────────────────────────────
    # Este caso simula um LLM que INVENTOU informação além do contexto.
    # A faithfulness deve reprovar.
    {
        "id": "tc-ferias-alucinacao",
        "category": "ferias",
        "input": "Posso vender meus dias de férias?",
        "actual_output": (
            # ALUCINAÇÃO: venda de férias não está nos documentos
            "Sim, é possível vender até 10 dias de férias (abono pecuniário). "
            "O valor é calculado proporcionalmente ao seu salário. "
            "Basta solicitar ao RH até 15 dias antes do início das férias."
        ),
        "expected_output": (
            "A política não menciona venda ou abono de férias."
        ),
        "retrieval_context": [
            "[Política de Férias]\nTodo colaborador tem direito a 30 dias corridos "
            "de férias após completar 12 meses de trabalho. As férias podem ser "
            "parceladas em até 3 períodos."
        ],
    },

    # ── Caso 3: Plano de saúde ───────────────────────────────────────────────
    {
        "id": "tc-saude-correto",
        "category": "beneficios",
        "input": "A empresa paga todo o plano de saúde ou tenho desconto em folha?",
        "actual_output": (
            "O custo do plano é compartilhado. A empresa cobre 80% da mensalidade "
            "do titular e 60% dos dependentes diretos (cônjuge e filhos até 21 anos, "
            "ou 24 anos se universitários). Além disso, há coparticipação de 20% "
            "em consultas e 30% em exames. O restante é descontado em folha."
        ),
        "expected_output": (
            "A empresa cobre 80% do titular e 60% dos dependentes. "
            "Há coparticipação de 20% em consultas e 30% em exames."
        ),
        "retrieval_context": [
            "[Plano de Saúde e Odontológico]\nA empresa oferece plano de saúde "
            "Bradesco Saúde (plano Flex Nacional) para todos os colaboradores "
            "a partir do primeiro dia de trabalho. O custo é compartilhado: "
            "a empresa cobre 80% da mensalidade do titular e 60% dos dependentes "
            "diretos (cônjuge e filhos até 21 anos ou 24 anos se universitários). "
            "Coparticipação de 20% em consultas e 30% em exames."
        ],
    },

    # ── Caso 4: Home office ──────────────────────────────────────────────────
    {
        "id": "tc-homeoffice-correto",
        "category": "home_office",
        "input": "Quanto recebo de auxílio home office por mês?",
        "actual_output": (
            "Em regime híbrido, você recebe R$ 150 por mês para cobrir "
            "custos de internet e energia. Se for full remote (aprovado pelo "
            "gestor e RH), o auxílio é de R$ 300 por mês. Ambos são pagos "
            "junto com o salário."
        ),
        "expected_output": (
            "R$ 150/mês para híbrido, R$ 300/mês para full remote."
        ),
        "retrieval_context": [
            "[Política de Home Office]\nA empresa fornece auxílio home office de "
            "R$ 150 por mês para cobrir custos de internet e energia elétrica, "
            "pago junto com o salário. Colaboradores full remote (aprovados pelo "
            "gestor e RH) recebem R$ 300 por mês de auxílio."
        ],
    },

    # ── Caso 5: Pergunta fora do escopo ─────────────────────────────────────
    # O chatbot deve dizer que não sabe — não inventar resposta.
    {
        "id": "tc-fora-escopo",
        "category": "fora_escopo",
        "input": "Qual é a política de uso de celular pessoal no trabalho?",
        "actual_output": (
            "Não encontrei informações sobre política de uso de celular pessoal "
            "em nossas políticas de RH. Recomendo consultar diretamente o seu "
            "gestor ou entrar em contato com o time de RH para obter essa "
            "orientação."
        ),
        "expected_output": (
            "Não há informação sobre isso na política."
        ),
        "retrieval_context": [
            "[Benefícios Gerais]\nVale Refeição: R$ 35 por dia útil trabalhado. "
            "Vale Alimentação: R$ 600 por mês.",
            "[Política de Home Office]\nColaboradores em regime híbrido trabalham "
            "presencialmente 3 dias por semana.",
        ],
    },

    # ── Caso 6: Licença paternidade ──────────────────────────────────────────
    {
        "id": "tc-licenca-paternidade",
        "category": "licencas",
        "input": "Quantos dias de licença paternidade a empresa oferece?",
        "actual_output": (
            "A empresa oferece 20 dias de licença paternidade remunerada, "
            "o que é 15 dias a mais do que o mínimo exigido pela legislação."
        ),
        "expected_output": "20 dias de licença paternidade remunerada.",
        "retrieval_context": [
            "[Licenças e Afastamentos]\nLicença Paternidade: 20 dias de licença "
            "paternidade remunerada (15 dias a mais que o mínimo legal)."
        ],
    },

    # ── Caso 7: Resposta irrelevante ─────────────────────────────────────────
    # Simula um LLM que deu uma resposta não relacionada à pergunta.
    # A answer_relevancy deve reprovar.
    {
        "id": "tc-irrelevante",
        "category": "beneficios",
        "input": "Qual é o valor do vale alimentação?",
        "actual_output": (
            # Responde sobre vale transporte ao invés de vale alimentação
            "O vale transporte é fornecido conforme sua rota declarada, "
            "com desconto de 6% do salário bruto em folha."
        ),
        "expected_output": "R$ 600 por mês, creditado todo dia 1º.",
        "retrieval_context": [
            "[Benefícios Gerais]\nVale Alimentação: R$ 600 por mês, creditado "
            "todo dia 1º. Vale Transporte: fornecido conforme necessidade, "
            "desconto de 6% do salário bruto."
        ],
    },
]

# 4. Converte dados em LLMTestCase

def make_test_case(data: dict) -> LLMTestCase:
    """
    Converte um dict de dados em um LLMTestCase do DeepEval.
 
    LLMTestCase é a unidade básica — guarda todos os campos
    que as métricas precisam para avaliar.
    """
    return LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"],
        expected_output=data.get("expected_output"),
        retrieval_context=data.get("retrieval_context", []),
        # O campo `name` aparece nos relatórios
        name=data["id"],
    )
    
# 5. Testes com pytest
# Métricas são instanciadas DENTRO de cada teste para evitar
# falha no import quando OPENROUTER_API_KEY não está definida.
# Isso também garante isolamento entre testes.

def _skip_if_no_key():
    """Pula o teste se não houver API key configurada."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY não definida — configure e rode novamente")
        
@pytest.mark.parametrize(
    "data",
    TEST_CASES_DATA,
    ids=[d["id"] for d in TEST_CASES_DATA],
)
def test_faithfulness(data: dict):
    """
    Faithfulness (fidelidade ao contexto).
 
    Pergunta: o chatbot inventou algo que não estava no contexto?
    Tipo: REFERENCELESS — não precisa de expected_output.
    Quando reprovar: caso tc-ferias-alucinacao deve reprovar aqui.
    """
    _skip_if_no_key()
    judge = get_judge()
    metric = build_metrics(judge)["faithfulness"]
 
    tc = make_test_case(data)
    metric.measure(tc)
 
    print(f"\n  Faithfulness score: {metric.score:.3f}")
    print(f"  Reason: {metric.reason}")
 
    assert metric.score >= metric.threshold, (
        f"\n[FALHA] {data['id']}"
        f"\nInput: {data['input']}"
        f"\nOutput: {data['actual_output'][:150]}"
        f"\nScore: {metric.score:.3f} (threshold: {metric.threshold})"
        f"\nMotivo: {metric.reason}"
    )
 
 
@pytest.mark.parametrize(
    "data",
    TEST_CASES_DATA,
    ids=[d["id"] for d in TEST_CASES_DATA],
)
def test_answer_relevancy(data: dict):
    """
    Answer Relevancy (relevância da resposta).
 
    Pergunta: a resposta responde o que foi perguntado?
    Tipo: REFERENCELESS — avalia só input ↔ actual_output.
    Quando reprovar: tc-irrelevante deve reprovar aqui.
    """
    _skip_if_no_key()
    judge = get_judge()
    metric = build_metrics(judge)["answer_relevancy"]
 
    tc = make_test_case(data)
    metric.measure(tc)
 
    print(f"\n  Answer Relevancy score: {metric.score:.3f}")
    print(f"  Reason: {metric.reason}")
 
    assert metric.score >= metric.threshold, (
        f"\n[FALHA] {data['id']}"
        f"\nInput: {data['input']}"
        f"\nOutput: {data['actual_output'][:150]}"
        f"\nScore: {metric.score:.3f}"
        f"\nMotivo: {metric.reason}"
    )
 
 
@pytest.mark.parametrize(
    "data",
    [d for d in TEST_CASES_DATA if d.get("expected_output")],
    ids=[d["id"] for d in TEST_CASES_DATA if d.get("expected_output")],
)
def test_contextual_recall(data: dict):
    """
    Contextual Recall (cobertura do retriever).
 
    Pergunta: o retriever trouxe contexto suficiente para
    responder o que estava em expected_output?
    Tipo: REFERENCE-BASED — precisa de expected_output.
    Avalia o RETRIEVER, não o LLM gerador.
    """
    _skip_if_no_key()
    judge = get_judge()
    metric = build_metrics(judge)["contextual_recall"]
 
    tc = make_test_case(data)
    metric.measure(tc)
 
    print(f"\n  Contextual Recall score: {metric.score:.3f}")
    print(f"  Reason: {metric.reason}")
 
    assert metric.score >= metric.threshold, (
        f"\n[FALHA RETRIEVER] {data['id']}"
        f"\nO retriever não trouxe contexto suficiente."
        f"\nScore: {metric.score:.3f}"
        f"\nMotivo: {metric.reason}"
    )
 
 
@pytest.mark.parametrize(
    "data",
    TEST_CASES_DATA,
    ids=[d["id"] for d in TEST_CASES_DATA],
)
def test_tom_profissional(data: dict):
    """
    Tom Profissional (GEval customizado).
 
    Avalia se o chatbot usa linguagem adequada para comunicação
    corporativa de RH. Critério definido em linguagem natural.
    Tipo: REFERENCELESS.
    """
    _skip_if_no_key()
    judge = get_judge()
    metric = build_metrics(judge)["tom_profissional"]
 
    tc = make_test_case(data)
    metric.measure(tc)
 
    print(f"\n  Tom Profissional score: {metric.score:.3f}")
    print(f"  Reason: {metric.reason}")
 
    assert metric.score >= metric.threshold, (
        f"\n[FALHA] {data['id']}"
        f"\nTom inadequado para comunicação de RH."
        f"\nScore: {metric.score:.3f}"
        f"\nMotivo: {metric.reason}"
    )
 
 
 # 6. Avaliação
 # O `evaluate()` do DeepEval roda todos os test cases de uma vez,
# gera relatório consolidado e (com login) envia para a plataforma.


def run_batch_eval():
    """
    Alternativa ao pytest: avalia todos os casos em batch.
    Útil para relatórios executivos e pipelines de CI/CD.
 
    Uso: python test_hr_chatbot.py
    """
    
    judge = get_judge()

    
    if not judge:
        print("OPENROUTER_API_KEY não definida. Configure e tente novamente.")
        return 
 
    print("\nPreparando test cases...")
    test_cases = [make_test_case(d) for d in TEST_CASES_DATA]
    
    metrics = build_metrics(judge)
 
    active_metrics = [
        metrics["faithfulness"],
        metrics["answer_relevancy"],
        metrics["tom_profissional"],
        metrics["clareza"],
    ]
 
    print(f"Rodando eval em batch: {len(test_cases)} casos * {len(active_metrics)} métricas\n")
 
    results = evaluate(
        test_cases=test_cases,
        metrics=active_metrics,
        run_async=True,          # avalia em paralelo
        show_indicator=True,     # progress bar
        print_results=True,      # imprime cada resultado
    )
 
    return results
 
 
if __name__ == "__main__":
    run_batch_eval()