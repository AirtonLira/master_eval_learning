from openai import AsyncOpenAI  
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

def get_ragas_llm():
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    return llm_factory(
        model="meta-llama/llama-3.1-8b-instruct",
        client=client,
    )
    
# Dataset 
# SingleTurnSample é a unidade básica do RAGAS.
# Repara nos campos — são os mesmos conceitos do DeepEval do outro projeto,
# mas com nomes diferentes:
#
#   DeepEval              RAGAS
#   ─────────────────     ─────────────────
#   input              -  user_input
#   actual_output      -  response
#   retrieval_context  -  retrieved_contexts  (lista de strings)
#   expected_output    -  reference

samples = [
    SingleTurnSample(
        user_input="Quantos dias de férias tenho direito por ano?",
        response=(
            "Você tem direito a 30 dias corridos de férias após completar "
            "12 meses de trabalho. As férias podem ser divididas em até 3 "
            "períodos: o maior deve ter no mínimo 14 dias e os outros no "
            "mínimo 5 dias cada."
        ),
        retrieved_contexts=[
            "Todo colaborador tem direito a 30 dias corridos de férias após "
            "completar 12 meses de trabalho. As férias podem ser parceladas "
            "em até 3 períodos, sendo que um deles não pode ser inferior a "
            "14 dias corridos e os demais não podem ser inferiores a 5 dias."
        ],
        reference="30 dias corridos após 12 meses de trabalho.",
    ),
    SingleTurnSample(
        user_input="Posso vender meus dias de férias?",
        response=(
            # ALUCINAÇÃO intencional — o LLM inventou isso
            "Sim, você pode vender até 10 dias de férias (abono pecuniário). "
            "Basta solicitar ao RH até 15 dias antes do início das férias."
        ),
        retrieved_contexts=[
            "Todo colaborador tem direito a 30 dias corridos de férias após "
            "completar 12 meses de trabalho. As férias podem ser parceladas "
            "em até 3 períodos."
        ],
        reference="A política não menciona venda ou abono de férias.",
    ),
]

dataset = EvaluationDataset(samples=samples)
print(f"Dataset criado: {len(dataset.samples)} amostras")
print(f"Campos do primeiro sample: {list(samples[0].__dict__.keys())}")

print("-----------------------Rodando avaliação----------------------- \n")

def get_ragas_embeddings():
    embedding_client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    return embedding_factory(
        "huggingface",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

llm = get_ragas_llm()
embeddings = get_ragas_embeddings()
faith_scorer = Faithfulness(llm=llm)
relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)


async def rodar_eval():
    print("\nRodando Faithfulness + AnswerRelevancy...\n")
    for sample in samples:

        faith = await faith_scorer.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )

        relevancy = await relevancy_scorer.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )

        print(f"Pergunta:         {sample.user_input}")
        print(f"Faithfulness:     {faith.value:.3f}")
        print(f"AnswerRelevancy:  {relevancy.value:.3f}")
        print()

asyncio.run(rodar_eval())