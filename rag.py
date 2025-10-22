# rag.py
"""
RAG helper utilities that integrate datapizza ingestion/retrieval with the ChatBotAgent
defined in app.py (initialize_agent).
- Ingesta PDFs in a Qdrant collection
- Provide a retrieval function that supplies retrieved chunks to the ChatBotAgent
- Optionally persist chat messages as 'memory' documents in the same collection
"""

import logging
import os
from typing import List, Optional

from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import DagPipeline, IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

# Importa l'initialize_agent dal tuo app.py (attento al path)
from app import initialize_agent

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)


def create_qdrant_vectorstore(
    location: str = ":memory:", collection_name: str = "my_documents", dims: int = 1536
):
    """
    Creates a QdrantVectorstore and ensures collection exists.
    Use location=":memory:" for in-memory, or host/port based Qdrant config as needed.
    """
    vs = QdrantVectorstore(location=location)
    # Create if not exists (docpizza API create_collection)
    try:
        vs.create_collection(
            collection_name,
            vector_config=[VectorConfig(name="embedding", dimensions=dims)],
        )
    except Exception as e:
        logger.warning("create_collection may have failed or collection exists: %s", e)
    return vs


def ingest_pdfs_to_qdrant(
    pdf_paths: List[str],
    qdrant_location=":memory:",
    collection_name="my_documents",
    openai_api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    chunk_max_char: int = 1000,
):
    """
    Ingest a list of pdf file paths into Qdrant using datapizza IngestionPipeline.
    """
    if openai_api_key is None:
        raise ValueError("openai_api_key is required for embeddings in this example")

    logger.info(
        "Creating Qdrant vectorstore at %s, collection %s",
        qdrant_location,
        collection_name,
    )
    vectorstore = create_qdrant_vectorstore(
        location=qdrant_location, collection_name=collection_name
    )

    logger.info("Configuring embedder with model %s", embedding_model)
    embedder_client = OpenAIEmbedder(api_key=openai_api_key, model_name=embedding_model)

    # Build pipeline: parser -> splitter -> chunk embedder
    ingestion_pipeline = IngestionPipeline(
        modules=[
            DoclingParser(),  # parses PDFs, images, etc.
            NodeSplitter(max_char=chunk_max_char),
            ChunkEmbedder(client=embedder_client),
        ],
        vector_store=vectorstore,
        collection_name=collection_name,
    )

    for p in pdf_paths:
        logger.info("Ingesting %s", p)
        ingestion_pipeline.run(p, metadata={"source": os.path.basename(p)})
    logger.info("Ingestion complete")
    return vectorstore


def build_dag_pipeline(
    openai_api_key: str,
    qdrant_location=":memory:",
    collection_name="my_documents",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
):
    """
    Build a DagPipeline for retrieval and generation, following Datapizza example.
    We'll return the dag_pipeline object (datapizza.pipeline.DagPipeline)
    """
    openai_client = OpenAIClient(model=llm_model, api_key=openai_api_key)
    query_rewriter = ToolRewriter(
        client=openai_client,
        system_prompt="Rewrite user queries to improve retrieval accuracy.",
    )
    embedder = OpenAIEmbedder(api_key=openai_api_key, model_name=embedding_model)

    # Reuse/create qdrant retriever
    retriever = QdrantVectorstore(location=qdrant_location)
    # ensure collection exists
    try:
        retriever.create_collection(
            collection_name,
            vector_config=[VectorConfig(name="embedding", dimensions=1536)],
        )
    except Exception:
        pass

    prompt_template = ChatPromptTemplate(
        user_prompt_template="User question: {{user_prompt}}\n",
        retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}",
    )

    dag_pipeline = DagPipeline()
    dag_pipeline.add_module("rewriter", query_rewriter)
    dag_pipeline.add_module("embedder", embedder)
    dag_pipeline.add_module("retriever", retriever)
    dag_pipeline.add_module("prompt", prompt_template)
    dag_pipeline.add_module("generator", openai_client)

    # Connect modules (keys as in docs)
    dag_pipeline.connect("rewriter", "embedder", target_key="text")
    dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
    dag_pipeline.connect("retriever", "prompt", target_key="chunks")
    dag_pipeline.connect("prompt", "generator", target_key="memory")

    return dag_pipeline


def rag_query(
    agent_provider: str,
    dag_pipeline,
    query: str,
    collection_name: str = "my_documents",
    k: int = 3,
    session_id: Optional[str] = None,
    openai_api_key: Optional[str] = None,
):
    """
    Run a full RAG query:
    - Use DagPipeline to retrieve chunks (k)
    - Build a prompt combining retrieved chunks and optionally session memory
    - Call ChatBotAgent.chat(...) with that enriched prompt
    """

    # Initialize the chatbot agent (from app.py)
    agent = initialize_agent(agent_provider)

    # Prepare dag input (following Datapizza example)
    dag_input = {
        "rewriter": {"user_prompt": query},
        "prompt": {"user_prompt": query},
        "retriever": {"collection_name": collection_name, "k": k},
        "generator": {"input": query},
    }
    logger.info("Running DagPipeline retrieval for query: %s", query)
    result = dag_pipeline.run(dag_input)

    # 'result' will have generator output under key 'generator' (string)
    # but we want to explicitly compose a context for agent.chat: retrieved chunks + user prompt + optionally session memory
    # The dag pipeline's prompt module produced 'memory' (depending on config). We'll extract retrieved chunks:
    retrieved_chunks = []
    if "retriever" in result and isinstance(result["retriever"], dict):
        # Some implementations return results inside result['retriever']; fallback to 'prompt' results
        pass

    # Try to get prompt memory / retrieved chunks
    retrieved_texts = []
    # The DagPipeline in datapizza returns each module output in the result dict
    # check keys for 'prompt' or 'retriever'
    if (
        "prompt" in result
        and isinstance(result["prompt"], dict)
        and "memory" in result["prompt"]
    ):
        # memory may be a string
        retrieved_texts.append(result["prompt"]["memory"])
    # If retriever module returned 'chunks' as a list
    if (
        "retriever" in result
        and isinstance(result["retriever"], dict)
        and "chunks" in result["retriever"]
    ):
        chunks = result["retriever"]["chunks"]
        for c in chunks:
            # chunk.text or chunk['text']
            t = (
                getattr(c, "text", None) or c.get("text", None)
                if isinstance(c, dict)
                else None
            )
            if t:
                retrieved_texts.append(t)

    # Compose final prompt for ChatBotAgent
    context = (
        "\n--- Retrieved documents ---\n"
        + ("\n\n".join(retrieved_texts))
        + "\n--- End retrieved ---\n\n"
    )
    final_prompt = f"{context}\nUser question: {query}\n\nAnswer as a helpful financial assistant concisely:"

    # Optionally, append chat memory from the session, if present in agent or provided externally.
    # If you stored messages in vectorstore as 'memory' documents, you could also query them here similarly.

    logger.info("Passing enriched prompt to ChatBotAgent")
    response_text = agent.chat(final_prompt)

    # Optionally: persist the user question + response as 'memory' documents in same vectorstore
    # (this part is left as optional helper below)

    return {
        "response": response_text,
        "retrieved": retrieved_texts,
    }


def persist_chat_as_memory(
    vectorstore: QdrantVectorstore,
    collection_name: str,
    messages: List[dict],
    embedder_client,
    session_id: str,
):
    """
    (Optional) Persist a list of chat messages into the same vectorstore as 'memory' docs.
    messages: list of {"role": "user"|"assistant", "content": "..."}
    embedder_client: embedder (OpenAIEmbedder) used to create embeddings and add to vectorstore
    """
    # This function demonstrates approach: create short documents from messages and add embeddings + metadata.
    from datapizza.core.node import Node  # depends on datapizza internals

    insert_items = []
    for i, m in enumerate(messages):
        text = f"{m['role']}: {m['content']}"
        metadata = {"type": "memory", "session_id": session_id, "index": i}
        # create Node/Chunk object expected by vectorstore.add (datapizza API specifics vary)
        # Here we show conceptual approach; adapt to your datapizza version.
        insert_items.append({"text": text, "metadata": metadata})
    # Many vectorstores allow a simple 'upsert' or 'add' API. Consult your datapizza QdrantVectorstore wrapper.
    try:
        vectorstore.upsert(collection_name=collection_name, items=insert_items)
    except Exception as e:
        logger.warning(
            "Could not persist chat memory to vectorstore automatically: %s", e
        )


# Example helper usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-dir", required=True, help="Directory with PDFs to ingest"
    )
    parser.add_argument(
        "--openai-key", required=True, help="OpenAI API key for embeddings & LLM"
    )
    parser.add_argument(
        "--qdrant-location",
        default=":memory:",
        help="Qdrant location string (':memory:' or host configuration)",
    )
    parser.add_argument("--collection", default="my_documents")
    parser.add_argument(
        "--provider",
        default="openai",
        help="Provider to pass to initialize_agent (depends on app.py)",
    )
    args = parser.parse_args()

    pdfs = [
        os.path.join(args.pdf_dir, f)
        for f in os.listdir(args.pdf_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise SystemExit("No PDFs found in the given directory")

    vs = ingest_pdfs_to_qdrant(
        pdfs,
        qdrant_location=args.qdrant_location,
        collection_name=args.collection,
        openai_api_key=args.openai_key,
    )
    dag = build_dag_pipeline(
        openai_api_key=args.openai_key,
        qdrant_location=args.qdrant_location,
        collection_name=args.collection,
    )
    # Example interactive loop
    print("Ready. Ask a question (type 'exit' to stop).")
    agent_provider = args.provider
    while True:
        q = input("Q: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        out = rag_query(
            agent_provider,
            dag,
            q,
            collection_name=args.collection,
            k=3,
            openai_api_key=args.openai_key,
        )
        print("A:", out["response"])
        print("---Retrieved chunks---")
        for r in out["retrieved"]:
            print(r[:400].replace("\n", " "))
            print("----")
