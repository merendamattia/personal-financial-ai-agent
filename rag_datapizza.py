#!/usr/bin/env python3
# -*- c_oding: utf-8 -*-

"""
RAG su PDF con Datapizza AI (Ollama Locale) + Qdrant locale su directory:
...
"""

import argparse
import glob
import os
import shutil
import sys
from typing import List

# --- Import corretti ---
from datapizza.clients.openai_like import OpenAILikeClient

# Datapizza AI
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import DagPipeline, IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv

# --- MODIFICA: Aggiungi import per il client Qdrant base ---
from qdrant_client import QdrantClient

DATASET_PATH = "./dataset/ETFs_2"

# --- Carica configurazione da .env ---
load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")

if not OLLAMA_API_URL.endswith("/v1"):
    OLLAMA_API_URL = OLLAMA_API_URL.rstrip("/") + "/v1"


# Modello LLM (Generazione) dal .env
LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
# LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")

# Modello Embedding (Retrieval) - Locale via Ollama
# EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_MODEL_NAME = "mxbai-embed-large"
# EMBEDDING_DIMS = 768
EMBEDDING_DIMS = 1024

# Qdrant locale
DEFAULT_COLLECTION = "my_documents"
DEFAULT_QDRANT_LOCATION = "./qdrant_local"


def _collect_pdf_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True))
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        return [path]
    return sorted(glob.glob(path, recursive=True))


# --- MODIFICA DEFINITIVA: Inizializzazione Qdrant (Hack per bug Datapizza) ---
def _get_vectorstore(location: str) -> QdrantVectorstore:
    """
    Crea un'istanza del vector store Qdrant in modalità locale (on-disk).
    """
    # 1. Creiamo manualmente il client Qdrant specificando 'path'
    #    per forzare la modalità "on-disk".
    client = QdrantClient(path=location)

    # 2. Passiamo il client E un location fittizio (es. ":memory:")
    #    per superare il bug di validazione del wrapper Datapizza.
    #    Il wrapper darà priorità al 'client' che abbiamo passato.
    return QdrantVectorstore(client=client, location=":memory:")


def ensure_collection(
    vectorstore: QdrantVectorstore, collection_name: str = DEFAULT_COLLECTION
):
    try:
        vectorstore.create_collection(
            collection_name,
            vector_config=[VectorConfig(name="embedding", dimensions=EMBEDDING_DIMS)],
        )
    except Exception:
        pass


def build_ingestion_pipeline(
    vectorstore: QdrantVectorstore, collection_name: str = DEFAULT_COLLECTION
) -> IngestionPipeline:
    # 1. Definiamo il client di embedding di base (che riceve solo stringhe)
    base_embedder_client = OpenAIEmbedder(
        model_name=EMBEDDING_MODEL_NAME, api_key="", base_url=OLLAMA_API_URL
    )

    # 2. Usiamo il wrapper ChunkEmbedder (che riceve 'Chunk' e usa il client)
    embedder_component = ChunkEmbedder(client=base_embedder_client)

    ingestion_pipeline = IngestionPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=1000),
            embedder_component,  # <-- Usiamo il wrapper corretto
        ],
        vector_store=vectorstore,
        collection_name=collection_name,
    )
    return ingestion_pipeline


def ingest_dataset(collection_name: str, db_dir: str):
    pdfs = _collect_pdf_paths(DATASET_PATH)
    if not pdfs:
        print(f"Nessun PDF trovato nel percorso: {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(db_dir, exist_ok=True)
    vs = _get_vectorstore(location=db_dir)
    ensure_collection(vs, collection_name)
    pipe = build_ingestion_pipeline(vs, collection_name)

    print(f"Avvio ingestione su Ollama (URL: {OLLAMA_API_URL})...")
    print(f"Modello Embedding: {EMBEDDING_MODEL_NAME}")
    for pdf in pdfs:
        print(f"Ingestione: {pdf}")
        pipe.run(pdf, metadata={"source": os.path.basename(pdf)})

    print(f"Ingestione completata su collection: {collection_name} (db: {db_dir})")


def build_rag_pipeline(collection_name: str, db_dir: str) -> DagPipeline:
    # --- RAG ---
    llm_client = OpenAILikeClient(
        model=LLM_MODEL_NAME, api_key="", base_url=OLLAMA_API_URL
    )
    print(f"Pipeline RAG avviata con LLM: {LLM_MODEL_NAME} (su {OLLAMA_API_URL})")

    rewriter = ToolRewriter(
        client=llm_client,
        system_prompt="Rewrite user queries to improve retrieval accuracy.",
    )

    query_embedder = OpenAIEmbedder(
        model_name=EMBEDDING_MODEL_NAME, api_key="", base_url=OLLAMA_API_URL
    )

    os.makedirs(db_dir, exist_ok=True)
    vs = _get_vectorstore(location=db_dir)
    ensure_collection(vs, collection_name)
    retriever = vs.as_retriever(collection_name=collection_name, k=5)

    prompt_template = ChatPromptTemplate(
        user_prompt_template="User question: {{user_prompt}}\n",
        retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}",
    )

    dag = DagPipeline()
    dag.add_module("rewriter", rewriter)
    dag.add_module("embedder", query_embedder)
    dag.add_module("retriever", retriever)
    dag.add_module("prompt", prompt_template)
    dag.add_module("generator", llm_client)

    dag.connect("rewriter", "embedder", target_key="text")
    dag.connect("embedder", "retriever", target_key="query_vector")
    dag.connect("retriever", "prompt", target_key="chunks")
    dag.connect("prompt", "generator", target_key="memory")

    return dag


def ask(question: str, k: int, collection_name: str, db_dir: str) -> str:
    dag = build_rag_pipeline(collection_name=collection_name, db_dir=db_dir)
    result = dag.run(
        {
            "rewriter": {"user_prompt": question},
            "prompt": {"user_prompt": question},
            "retriever": {"collection_name": collection_name, "k": int(k)},
            "generator": {"input": question},
        }
    )
    return str(result["generator"])


def main():
    parser = argparse.ArgumentParser(
        description="RAG su PDF con Datapizza AI (Ollama) + Qdrant locale + dataset fisso"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser(
        "ingest", help="Indicizza i PDF dal dataset fisso ./dataset/ETFs_2"
    )
    p_ing.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Nome collection Qdrant",
    )
    p_ing.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_QDRANT_LOCATION,
        help="Directory locale per Qdrant (persistente)",
    )

    p_ask = sub.add_parser("ask", help="Poni una domanda sui PDF indicizzati")
    p_ask.add_argument("question", type=str, help="Domanda")
    p_ask.add_argument("--k", type=int, default=4, help="Numero di chunk da recuperare")
    p_ask.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Nome collection Qdrant",
    )
    p_ask.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_QDRANT_LOCATION,
        help="Directory locale per Qdrant (persistente)",
    )

    args = parser.parse_args()

    try:
        import httpx

        check_url = OLLAMA_API_URL.replace("/v1", "")
        httpx.get(check_url)
    except Exception as e:
        print(
            f"Errore: Impossibile connettersi a Ollama a '{check_url}'", file=sys.stderr
        )
        print(
            "Assicurati che Ollama sia in esecuzione ('ollama serve').", file=sys.stderr
        )
        sys.exit(1)

    if args.cmd == "ingest":
        ingest_dataset(collection_name=args.collection, db_dir=args.db_dir)
    elif args.cmd == "ask":
        answer = ask(
            args.question, k=args.k, collection_name=args.collection, db_dir=args.db_dir
        )
        print(answer)


def default_run():
    db_dir = DEFAULT_QDRANT_LOCATION
    collection = DEFAULT_COLLECTION

    try:
        import httpx

        check_url = OLLAMA_API_URL.replace("/v1", "")
        httpx.get(check_url)
        print(f"Connesso a Ollama a {check_url} (endpoint API: {OLLAMA_API_URL})")
    except Exception as e:
        print(
            f"Errore: Impossibile connettersi a Ollama a '{check_url}'", file=sys.stderr
        )
        print(f"Dettagli: {e}", file=sys.stderr)
        print(
            "Assicurati che Ollama sia in esecuzione (es. 'ollama serve').",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- MODIFICA: Ripristinata la pulizia del DB ---
    # È fondamentale per assicurare che non si lavori su un DB corrotto
    if os.path.exists(db_dir):
        print(f"Rilevato vecchio DB in {db_dir}. Rimuovo per evitare conflitti.")
        try:
            shutil.rmtree(db_dir)
            print(f"Vecchia directory DB {db_dir} rimossa.")
        except Exception as e:
            print(
                f"Impossibile rimuovere {db_dir}: {e}. Rimuovila manually ed esegui di nuovo."
            )
            sys.exit(1)

    # 1) Esegui Ingest
    print(f"Eseguo ingest su {DATASET_PATH} -> {db_dir}")
    ingest_dataset(collection_name=collection, db_dir=db_dir)

    # 2) Modalità Q&A interattiva
    print("\n--- Ingestione completata ---")
    print("Modalità Q&A (digita 'exit' per uscire)")

    dag = build_rag_pipeline(collection_name=collection, db_dir=db_dir)

    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        try:
            result = dag.run(
                {
                    "rewriter": {"user_prompt": q},
                    "prompt": {"user_prompt": q},
                    "retriever": {"collection_name": collection, "k": 4},
                    "generator": {"input": q},
                }
            )
            ans = str(result["generator"])
            print(f"\nRisposta: {ans}\n")
        except Exception as e:
            print(f"Errore: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_run()
    else:
        main()
