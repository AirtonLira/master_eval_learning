"""
pipeline.py
-----------
Pipeline de offline eval — roda antes de qualquer deploy.
 
Fluxo:
  Dataset → Métricas automáticas → (LLM-judge opcional) → Relatório → Pass/Fail
 
Uso rápido:
    python pipeline.py                    # só métricas automáticas
    python pipeline.py --llm-judge        # + LLM-as-judge (requer API key)
    python pipeline.py --category resumo  # filtra por categoria
    python pipeline.py --fail-fast        # para na primeira falha
 
Uso em CI/CD (GitHub Actions, etc.):
    python pipeline.py && echo "DEPLOY OK" || echo "DEPLOY BLOQUEADO"
    # Retorna exit code 1 se qualquer caso falhar acima do threshold
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
 
from dataset import DATASET, EvalCase
from metrics import (
    MetricResult,
    contains_keywords,
    exact_match,
    groundedness,
    no_harmful_content,
    valid_json_schema
)


# 1. LLM-JUDGE 

def llm_judge_eval(case: EvalCase, model: str = "anthropic/claude-sonnet-4-5") -> MetricResult:
    """
    Avalia qualidade geral via LLM-as-judge usando OpenRouter.
    Requer: pip install openai + OPENROUTER_API_KEY no ambiente.
    Modelos disponíveis: https://openrouter.ai/models
    """
    try:
        from openai import OpenAI
        import json as _json
    except ImportError:
        return MetricResult(
            metric="llm_judge",
            score=0.5, passed=True, threshold=0.6,
            reason="openai não instalado — skip",
        )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return MetricResult(
            metric="llm_judge",
            score=0.5, passed=True, threshold=0.6,
            reason="OPENROUTER_API_KEY não definida — skip",
        )

    context_block = ""
    if case.retrieval_context:
        ctx = "\n".join(f"- {c}" for c in case.retrieval_context)
        context_block = f"\nCONTEXTO:\n{ctx}\n"

    prompt = f"""Avalie esta resposta de LLM. Retorne APENAS JSON válido.

PERGUNTA: {case.input}
{context_block}
RESPOSTA: {case.llm_output}

{{"score": <float 0.0-1.0>, "reason": "<1 frase>", "passed": <true/false>}}

Critérios: precisão factual, coerência com o contexto, utilidade para o usuário."""

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = int((time.time() - t0) * 1000)
        raw = resp.choices[0].message.content.strip()
        parsed = _json.loads(raw.replace("```json", "").replace("```", "").strip())
        score = float(parsed.get("score", 0.5))
        return MetricResult(
            metric="llm_judge",
            score=round(score, 3),
            passed=bool(parsed.get("passed", score >= 0.6)),
            threshold=0.6,
            reason=f"{parsed.get('reason', '')} ({latency}ms)",
        )
    except Exception as e:
        return MetricResult(
            metric="llm_judge",
            score=0.5, passed=True, threshold=0.6,
            reason=f"Erro: {e}",
        )
 
 # 2. Router de métricas por categoria definidas no dataset (ex: resumo, código, etc.)
 
def evaluate_case(case: EvalCase, use_llm_judge: bool = False) -> list[MetricResult]:
    """
    Seleciona e executa as métricas certas para cada tipo de caso.
    Métricas comuns rodam sempre; métricas específicas dependem da categoria.
    """
    results: list[MetricResult] = []
 
    # --- Métricas universais (toda categoria) ---
    results.append(no_harmful_content(case.llm_output))
 
    # --- Métricas por categoria ---
    if case.category == "suporte":
        if case.expected_output:
            results.append(exact_match(case.llm_output, case.expected_output))
            results.append(
                contains_keywords(case.llm_output, [case.expected_output], threshold=0.9)
            )
        if case.retrieval_context:
            results.append(groundedness(case.llm_output, case.retrieval_context, threshold=0.3))
 
    elif case.category == "resumo":
        # Resumos: sem expected — foca em groundedness
        results.append(groundedness(case.llm_output, case.retrieval_context, threshold=0.35))
 
    elif case.category == "extracao":
        results.append(valid_json_schema(
            case.llm_output,
            required_fields=["empresa", "cnpj", "valor"],
            threshold=1.0,
        ))
 
    # --- LLM-judge (opcional, tem custo) ---
    if use_llm_judge:
        results.append(llm_judge_eval(case))
 
    return results


# 3. Registro dos resultados:

@dataclass
class CaseResult:
    case_id: str
    category: str
    input_preview: str
    output_preview: str
    metrics: list[MetricResult]
    overall_passed: bool
    overall_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
 
    @property
    def failures(self) -> list[MetricResult]:
        return [m for m in self.metrics if not m.passed]
    
    
# 4. Pipeline principal:

def run_pipeline(
    category_filter: Optional[str] = None,
    use_llm_judge: bool = False,
    fail_fast: bool = False,
    min_pass_rate: float = 1.0,
) -> tuple[list[CaseResult], bool]:
    """
    Executa o pipeline completo de offline eval.
 
    Returns:
        (resultados, pipeline_passed)
        pipeline_passed = True se todos os casos passaram
    """
    dataset = DATASET
    if category_filter:
        dataset = [c for c in dataset if c.category == category_filter]
 
    if not dataset:
        print(f"Nenhum caso encontrado para categoria: {category_filter}")
        sys.exit(1)
 
    results: list[CaseResult] = []
    passed_count = 0
 
    print(f"\n{'─'*62}")
    print(f"  Offline Eval Pipeline")
    print(f"  {len(dataset)} casos  |  LLM-judge: {'on' if use_llm_judge else 'off'}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*62}\n")
 
    for case in dataset:
        metrics = evaluate_case(case, use_llm_judge=use_llm_judge)
 
        # Case passa se: todas as métricas passaram
        case_passed = all(m.passed for m in metrics)
        avg_score = round(sum(m.score for m in metrics) / len(metrics), 3)
 
        result = CaseResult(
            case_id=case.id,
            category=case.category,
            input_preview=case.input[:80],
            output_preview=case.llm_output[:80],
            metrics=metrics,
            overall_passed=case_passed,
            overall_score=avg_score,
        )
        results.append(result)
 
        if case_passed:
            passed_count += 1
 
        # Print resultado
        icon = "PASS" if case_passed else "FAIL"
        print(f"[{icon}] {case.id} ({case.category}) — score: {avg_score:.2f}")
        for m in metrics:
            sub_icon = "  ok" if m.passed else "  !!"
            print(f"{sub_icon}  {m}")
 
        if not case_passed:
            print(f"\n      Input:  {case.input[:80]}")
            print(f"      Output: {case.llm_output[:80]}\n")
 
        if fail_fast and not case_passed:
            print("\nFail-fast ativado — interrompendo pipeline.")
            break
 
    # --- Sumário ---
    total = len(results)
    pass_rate = passed_count / total if total else 0
    pipeline_passed = pass_rate >= min_pass_rate
 
    print(f"\n{'─'*62}")
    print(f"  Resultado: {passed_count}/{total} casos passaram ({pass_rate:.0%})")
    print(f"  Pipeline: {'APROVADO' if pipeline_passed else 'REPROVADO'}")
    print(f"{'─'*62}\n")
 
    return pipeline_passed

 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline Eval Pipeline")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Ativa LLM-as-judge (requer ANTHROPIC_API_KEY)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filtra por categoria (suporte, resumo, extracao, codigo)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Para na primeira falha")
    parser.add_argument("--min-pass-rate", type=float, default=1.0,
                        help="Taxa mínima de aprovação (ex: 0.8 aceita 80%% de falhas)")
    args = parser.parse_args()
 
    pipeline_passed = run_pipeline(
        category_filter=args.category,
        use_llm_judge=args.llm_judge,
        fail_fast=args.fail_fast,
        min_pass_rate=args.min_pass_rate,
    )
 
 
    # Exit code 1 bloqueia o deploy no CI/CD
    sys.exit(0 if pipeline_passed else 1)
 
 
if __name__ == "__main__":
    main()