"""
rag_engine.py
-------------
Motor RAG do chatbot de RH.
 
Componentes:
  1. Indexador   — divide docs em chunks e indexa no ChromaDB
  2. Retriever   — busca chunks relevantes por similaridade semântica
  3. Generator   — monta o prompt e chama o LLM via OpenRouter
 
Este módulo é o SISTEMA sendo avaliado.
O eval (DeepEval) fica separado, em test_hr_chatbot.py.
"""

import os
from typing import Optional
 
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
 
from hr_documents import HR_DOCUMENTS

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
 
# Modelo para GERAÇÃO de resposta (gratuito no OpenRouter)
GENERATOR_MODEL = "openrouter/free"
 
# Número de chunks recuperados por query
TOP_K = 3
 
 
# 1. Indexador: Ingestão e indexação dos documentos de RH
def build_index() -> chromadb.Collection:
   """
   Cria e popula o índice vetorial com os documentos de RH.

   Usa sentence-transformers localmente para embeddings —
   sem custo de API na fase de indexação.

   Returns:
       collection: ChromaDB collection pronta para busca
   """
   client = chromadb.Client()

   # Embedding function local 
   ef = embedding_functions.SentenceTransformerEmbeddingFunction(
       model_name="all-MiniLM-L6-v2"
   )

   # Cria (ou recria) a collection
   try:
       client.delete_collection("hr_policies")
   except Exception:
       pass

   collection = client.create_collection(
       name="hr_policies",
       embedding_function=ef,
       metadata={"hnsw:space": "cosine"},
   )

   # Chunking simples: um chunk por documento
   # Em produção: chunk por parágrafo ou sliding window
   documents, ids, metadatas = [], [], []
   for doc in HR_DOCUMENTS:
       content = doc["content"].strip()
       documents.append(content)
       ids.append(doc["id"])
       metadatas.append({"title": doc["title"]})

   collection.add(documents=documents, ids=ids, metadatas=metadatas)
   print(f"[RAG] Índice criado com {len(documents)} documentos.")
   return collection


# 2. Retriever: Busca por similaridade semântica

def retrieve(query: str, collection: chromadb.Collection, top_k: int = TOP_K) -> list[str]:
    """
    Recupera os chunks mais relevantes para a query.
 
    Args:
        query: pergunta do usuário
        collection: índice ChromaDB
        top_k: número de chunks a retornar
 
    Returns:
        lista de strings com o conteúdo dos chunks recuperados
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
 
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        title = results["metadatas"][0][i].get("title", "")
        chunks.append(f"[{title}]\n{doc.strip()}")
 
    return chunks


# 3. Generator: Monta o prompt e chama o LLM para gerar a resposta
SYSTEM_PROMPT = """Você é o assistente de RH da empresa. Responda perguntas dos
colaboradores com base EXCLUSIVAMENTE nas políticas fornecidas no contexto.
 
Regras:
- Use apenas informações do contexto. Nunca invente dados.
- Se a informação não estiver no contexto, diga claramente que não sabe.
- Seja direto, claro e profissional.
- Use linguagem acessível, sem jargão jurídico desnecessário.
- Quando citar valores ou prazos, seja preciso."""
 
 
def generate_answer(
    question: str,
    context_chunks: list[str],
    model: str = GENERATOR_MODEL,
) -> str:
    """
    Gera a resposta do chatbot usando o LLM via OpenRouter.
 
    Args:
        question: pergunta do usuário
        context_chunks: chunks recuperados pelo retriever
        model: modelo do OpenRouter a usar
 
    Returns:
        resposta gerada pelo LLM
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY não definida.\n"
            "Execute: export OPENROUTER_API_KEY=sk-or-..."
        )
 
    context = "\n\n---\n\n".join(context_chunks)
 
    user_message = f"""Contexto das políticas de RH:
{context}
 
Pergunta do colaborador:
{question}
 
Responda com base apenas no contexto acima."""
 
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
 
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # baixo para respostas factuais e consistentes
        max_tokens=512,
    )
 
    return response.choices[0].message.content.strip()
 
 
# ---------------------------------------------------------------------------
# 4. PIPELINE COMPLETO (retriever + generator)
# ---------------------------------------------------------------------------
 
def ask_hr_chatbot(
    question: str,
    collection: chromadb.Collection,
    model: str = GENERATOR_MODEL,
) -> dict:
    """
    Pipeline RAG completo: query → retrieve → generate.
 
    Retorna um dict com tudo que o DeepEval precisa para avaliar:
      - input: a pergunta
      - actual_output: a resposta gerada
      - retrieval_context: os chunks usados (para faithfulness)
 
    Args:
        question: pergunta do colaborador
        collection: índice ChromaDB
        model: modelo do OpenRouter
 
    Returns:
        dict com input, actual_output e retrieval_context
    """
    # Passo 1 — Recupera contexto relevante
    chunks = retrieve(question, collection)
 
    # Passo 2 — Gera resposta com base no contexto
    answer = generate_answer(question, chunks, model=model)
 
    return {
        "input": question,
        "actual_output": answer,
        "retrieval_context": chunks,  # ← DeepEval usa isso para FaithfulnessMetric
    }
 
 
# ---------------------------------------------------------------------------
# Demo rápida (sem eval)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    print("Construindo índice...")
    col = build_index()
 
    perguntas = [
        "Quantos dias de férias tenho direito por ano?",
        "A empresa cobre o plano de saúde dos meus filhos?",
        "Quanto recebo de auxílio home office?",
    ]
 
    for q in perguntas:
        print(f"\n{'─'*60}")
        print(f"Pergunta: {q}")
        result = ask_hr_chatbot(q, col)
        print(f"Resposta: {result['actual_output']}")
        print(f"Chunks usados: {len(result['retrieval_context'])}")
 