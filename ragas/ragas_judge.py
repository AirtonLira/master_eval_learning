import os
from dotenv import load_dotenv
from openai import OpenAI
from ragas.llms import llm_factory

load_dotenv()


DEFAULT_MODEL = "openrouter/free"

def get_ragas_llm():
    """
    Configura o LLM judge do RAGAS usando a API nova (llm_factory).
    O client OpenAI aponta para o OpenRouter via base_url.
    """
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    return llm_factory(
        model=DEFAULT_MODEL,
        client=client,
    )


if __name__ == "__main__":
    llm = get_ragas_llm()
    print(f"RAGAS LLM configurado: {llm}")