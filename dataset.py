"""
dataset.py
----------
Dataset de teste para o pipeline de offline eval.
 
Cada entrada simula uma chamada real ao seu LLM em produção.
Substitua `llm_output` pelo output real do seu modelo.

"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvalCase:
    id: str
    input: str                       # Pergunta / prompt do usuário
    llm_output: str                  # Resposta gerada pelo LLM (o que você está avaliando)
    expected_output: Optional[str]   # Resposta correta (para métricas determinísticas)
    retrieval_context: list[str]     # Chunks recuperados pelo RAG (se houver)
    category: str                    # Categoria do caso (para análise segmentada)
    min_score: float = 0.7           # Score mínimo aceitável para este caso
 
 
 #Dataset de demonstração:
 
DATASET: list[EvalCase] = [
         # --- Casos: Suporte ao cliente ---
    EvalCase(
        id="suporte-001",
        category="suporte",
        input="Qual é o prazo para troca de produto com defeito?",
        llm_output=(
            "Produtos com defeito podem ser trocados em até 90 dias após a compra, "
            "conforme o Código de Defesa do Consumidor. Para iniciar a troca, "
            "entre em contato pelo nosso chat ou ligue para 0800-123-4567."
        ),
        expected_output="90 dias",
        retrieval_context=[
            "O prazo legal para reclamação de vícios ocultos é de 90 dias para produtos duráveis.",
            "Para acionar a garantia, o cliente deve contatar o SAC em até 90 dias da compra.",
        ],
        min_score=0.8,
    ),
    EvalCase(
        id="suporte-002",
        category="suporte",
        input="Qual é o prazo para troca de produto com defeito?",
        llm_output=(
            # Resposta ERRADA intencionalmente — para o eval detectar
            "O prazo para troca é de 30 dias para qualquer produto."
        ),
        expected_output="90 dias",
        retrieval_context=[
            "O prazo legal para reclamação de vícios ocultos é de 90 dias para produtos duráveis.",
        ],
        min_score=0.8,
    ),
 
    # --- Casos: Resumo de documentos ---
    EvalCase(
        id="resumo-001",
        category="resumo",
        input="Resuma os principais pontos do relatório trimestral.",
        llm_output=(
            "O relatório do Q3 mostra crescimento de 12% na receita (R$ 45M), "
            "margem EBITDA de 23%, lançamento de 3 novos produtos e expansão "
            "para 2 novos estados. O principal risco identificado é a pressão "
            "cambial no segmento de importação."
        ),
        expected_output=None,
        retrieval_context=[
            "Receita Q3: R$ 45 milhões, crescimento de 12% YoY.",
            "EBITDA margin: 23%. Foram lançados 3 produtos no período.",
            "Expansão: entrada nos mercados do Pará e Tocantins.",
            "Risco principal: variação cambial impacta 18% do custo de produtos.",
        ],
        min_score=0.75,
    ),
    EvalCase(
        id="resumo-002",
        category="resumo",
        input="Resuma os principais pontos do relatório trimestral.",
        llm_output=(
            # Alucinação: inventa números que não estão no contexto
            "O relatório mostra crescimento de 25% com receita de R$ 80M "
            "e abertura de 10 lojas físicas."
        ),
        expected_output=None,
        retrieval_context=[
            "Receita Q3: R$ 45 milhões, crescimento de 12% YoY.",
            "EBITDA margin: 23%. Foram lançados 3 produtos no período.",
        ],
        min_score=0.75,
    ),
 
    # --- Casos: Extração estruturada ---
    EvalCase(
        id="extracao-001",
        category="extracao",
        input='Extraia: {"empresa": ..., "cnpj": ..., "valor": ...} do contrato.',
        llm_output='{"empresa": "Acme Ltda", "cnpj": "12.345.678/0001-90", "valor": "R$ 250.000"}',
        expected_output='{"empresa": "Acme Ltda", "cnpj": "12.345.678/0001-90", "valor": "R$ 250.000"}',
        retrieval_context=[
            "Contratante: Acme Ltda, CNPJ 12.345.678/0001-90.",
            "Valor total do contrato: R$ 250.000 (duzentos e cinquenta mil reais).",
        ],
        min_score=0.9,
    ),
    EvalCase(
        id="extracao-002",
        category="extracao",
        input='Extraia: {"empresa": ..., "cnpj": ..., "valor": ...} do contrato.',
        llm_output=(
            # Resposta em texto livre — não seguiu o formato
            "A empresa é Acme Ltda com CNPJ 12.345.678/0001-90 e o valor é 250 mil reais."
        ),
        expected_output='{"empresa": "Acme Ltda", "cnpj": "12.345.678/0001-90", "valor": "R$ 250.000"}',
        retrieval_context=[
            "Contratante: Acme Ltda, CNPJ 12.345.678/0001-90.",
        ],
        min_score=0.9,
    ),
 
    # --- Casos: Geração de código ---
    EvalCase(
        id="codigo-001",
        category="codigo",
        input="Escreva uma função Python que calcula o IMC dado peso e altura.",
        llm_output=(
            "def calcular_imc(peso_kg: float, altura_m: float) -> float:\n"
            "    if altura_m <= 0:\n"
            "        raise ValueError('Altura deve ser maior que zero')\n"
            "    return round(peso_kg / (altura_m ** 2), 2)\n"
        ),
        expected_output=None,
        retrieval_context=[],
        min_score=0.8,
    ),
    EvalCase(
        id="codigo-002",
        category="codigo",
        input="Escreva uma função Python que calcula o IMC dado peso e altura.",
        llm_output=(
            # Código com bug: divide peso por altura ao invés de altura²
            "def calcular_imc(peso, altura):\n"
            "    return peso / altura\n"
        ),
        expected_output=None,
        retrieval_context=[],
        min_score=0.8,
    ),
     
 ]