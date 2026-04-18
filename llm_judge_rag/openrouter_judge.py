"""
openrouter_judge.py
-------------------
Wrapper que conecta o OpenRouter ao DeepEval como LLM-judge.

Por que isso é necessário?
  O DeepEval usa seu próprio protocolo interno para chamar o LLM-juiz.
  Ele espera uma classe que herda de `DeepEvalBaseLLM` e implementa
  dois métodos: `generate()` (síncrono) e `a_generate()` (assíncrono).

  Isso desacopla o framework do provedor — você pode usar
  OpenAI, Anthropic, Ollama, ou qualquer provedor compatível
  com a API OpenAI, como é o caso do OpenRouter.
"""

import os
from typing import Optional

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

# Modelo padrão para o judge (diferente do generator para independência)
DEFAULT_JUDGE_MODEL = "openrouter/free"


class OpenRouterJudge(DeepEvalBaseLLM):
    """
    LLM-judge customizado usando OpenRouter.

    Herda de DeepEvalBaseLLM e implementa a interface que o
    DeepEval espera para calcular todas as métricas (Faithfulness,
    AnswerRelevancy, GEval, etc.).

    Uso:
        judge = OpenRouterJudge()
        metric = FaithfulnessMetric(model=judge)
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY não definida.\n"
                "Execute: export OPENROUTER_API_KEY=sk-or-..."
            )

        # Cliente síncrono (usado em generate)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Cliente assíncrono (usado em a_generate)
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_model_name(self) -> str:
        """Nome do modelo — exibido nos relatórios do DeepEval."""
        return self.model

    def load_model(self):
        """
        DeepEval chama este método antes de usar o judge.
        Retornamos o cliente síncrono.
        """
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Geração síncrona — usada pelo DeepEval durante o eval.

        O DeepEval monta o prompt internamente (com o critério,
        o output a avaliar, o contexto, etc.) e passa aqui.
        Nossa responsabilidade é só chamar o modelo e retornar o texto.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,      # determinístico para avaliação
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                f"O modelo '{self.model}' retornou content=None. "
                "Possíveis causas: rate limit, content filter ou falha do provider."
            )
        return content

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """
        Geração assíncrona — usada quando o DeepEval roda em paralelo.
        Permite avaliar múltiplos test cases simultaneamente.
        """
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                f"O modelo '{self.model}' retornou content=None. "
                "Possíveis causas: rate limit, content filter ou falha do provider."
            )
        return content


# ---------------------------------------------------------------------------
# Teste rápido do wrapper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    judge = OpenRouterJudge()
    print(f"Judge configurado: {judge.get_model_name()}")

    response = judge.generate("Responda em 1 frase: o que é faithfulness em RAG?")
    print(f"Resposta de teste: {response}")