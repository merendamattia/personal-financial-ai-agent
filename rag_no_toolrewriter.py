#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG su PDF con Datapizza AI (Ollama Locale) + Qdrant locale su directory.
Questo script usa un parser PDF personalizzato e un embedder locale
basato su sentence-transformers per la massima compatibilità e performance.
"""

import argparse
import glob
import os
import re
import shutil
import sys
from typing import List

# --- Dipendenze ---
import pypdf
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.core.embedder.base import BaseEmbedder
from datapizza.core.models import PipelineComponent
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.splitters import RecursiveSplitter
from datapizza.pipeline import DagPipeline, IngestionPipeline
from datapizza.type.type import Node
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Logica di Parsing e Embedding Personalizzata ---

# def read_pdf_text(pdf_path: str) -> str:
#     """Estrae tutto il testo da un file PDF."""
#     try:
#         reader = pypdf.PdfReader(pdf_path)
#         texts = [page.extract_text() or "" for page in reader.pages]
#         return "\n".join(texts)
#     except Exception as e:
#         print(f"Errore pypdf nel parsing {pdf_path}: {e}")
#         return ""

# In rag_no_toolrewriter.py


def read_pdf_text(pdf_path: str) -> str:
    """Estrae e pulisce il testo da un file PDF."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        texts = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(texts)

        # --- INIZIO BLOCCO DI PULIZIA DEL TESTO ---
        # 1. Sostituisce le interruzioni di riga multiple con un singolo spazio.
        # Questo unisce le frasi spezzate su più righe.
        cleaned_text = re.sub(r"\n+", " ", full_text)

        # 2. Sostituisce gli spazi multipli con un singolo spazio.
        cleaned_text = re.sub(r" +", " ", cleaned_text)
        # --- FINE BLOCCO DI PULIZIA DEL TESTO ---

        return cleaned_text.strip()

    except Exception as e:
        print(f"Errore pypdf nel parsing {pdf_path}: {e}")
        return ""


class PyPdfParser(PipelineComponent):
    """Parser personalizzato che usa pypdf per estrarre testo."""

    def _run(self, file_path: str, metadata: dict = None) -> Node:
        full_text = read_pdf_text(file_path)
        return Node(content=full_text, metadata=metadata or {})


class CustomSentenceTransformerEmbedder(BaseEmbedder):
    """Embedder personalizzato che usa sentence-transformers."""

    _model_instance = None

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        if CustomSentenceTransformerEmbedder._model_instance is None:
            print(
                f"--- Carico il modello SentenceTransformer: {model_name} (potrebbe richiedere tempo)... ---"
            )
            CustomSentenceTransformerEmbedder._model_instance = SentenceTransformer(
                model_name
            )
            print("--- Modello caricato. ---")
        self.model = CustomSentenceTransformerEmbedder._model_instance

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        is_list = isinstance(text, list)
        if not is_list:
            text = [text]

        embeddings_np = self.model.encode(
            text, show_progress_bar=False, convert_to_numpy=True
        )
        embeddings_list = embeddings_np.tolist()
        return embeddings_list if is_list else embeddings_list[0]

    def _run(self, text: str | List[str]) -> List[float] | List[List[float]]:
        return self.embed(text)

    def _set_client(self):
        self.client = self.model

    def _set_a_client(self):
        pass


# ---------------------------------------------------------
# --- Configurazione Globale ---
# ---------------------------------------------------------

DATASET_PATH = "./dataset/ETFs_2"

load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")
if not OLLAMA_API_URL.endswith("/v1"):
    OLLAMA_API_URL = OLLAMA_API_URL.rstrip("/") + "/v1"

LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMS = 1024

DEFAULT_COLLECTION = "my_documents"
DEFAULT_QDRANT_LOCATION = "./qdrant_local"
DEFAULT_K = 4

# ---------------------------------------------------------
# --- Funzioni Principali ---
# ---------------------------------------------------------


def _collect_pdf_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True))
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        return [path]
    return sorted(glob.glob(path, recursive=True))


def _get_vectorstore(location: str) -> QdrantVectorstore:
    client = QdrantClient(path=location)
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
    base_embedder_client = CustomSentenceTransformerEmbedder(
        model_name=EMBEDDING_MODEL_NAME
    )
    embedder_component = ChunkEmbedder(client=base_embedder_client)

    return IngestionPipeline(
        modules=[
            PyPdfParser(),
            RecursiveSplitter(max_char=1500, overlap=300),
            embedder_component,
        ],
        vector_store=vectorstore,
        collection_name=collection_name,
    )


def ingest_dataset(vectorstore: QdrantVectorstore, collection_name: str, db_dir: str):
    pdfs = _collect_pdf_paths(DATASET_PATH)
    if not pdfs:
        print(f"Nessun PDF trovato nel percorso: {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    ensure_collection(vectorstore, collection_name)
    pipe = build_ingestion_pipeline(vectorstore, collection_name)

    print(f"Avvio ingestione (Embedding model: {EMBEDDING_MODEL_NAME})...")
    for pdf in pdfs:
        #     print(f"Ingestione: {pdf}")
        #     pipe.run(pdf, metadata={"source": os.path.basename(pdf)})

        # print(f"Ingestione completata su collection: {collection_name} (db: {db_dir})")

        # TODO: eliminare debug e scommentare le righe sopra
        # --- INIZIO CODICE DI DEBUG TEMPORANEO ---
        if "CRPA.pdf" in os.path.basename(pdf):
            print("\n\n--- DEBUG: CONTENUTO RAW DI CRPA.PDF ---\n")
            raw_text = read_pdf_text(pdf)
            print(raw_text)
            print("\n--- FINE DEBUG RAW ---\n\n")
        # --- FINE CODICE DI DEBUG TEMPORANEO ---

        print(f"Ingestione: {pdf}")
        pipe.run(pdf, metadata={"source": os.path.basename(pdf)})

    print(f"Ingestione completata su collection: {collection_name} (db: {db_dir})")


def build_rag_pipeline(
    vectorstore: QdrantVectorstore, collection_name: str
) -> DagPipeline:
    llm_client = OpenAILikeClient(
        model=LLM_MODEL_NAME, api_key="", base_url=OLLAMA_API_URL, temperature=0.01
    )
    query_embedder = CustomSentenceTransformerEmbedder(model_name=EMBEDDING_MODEL_NAME)
    retriever = vectorstore.as_retriever(collection_name=collection_name, k=DEFAULT_K)

    # --- CORREZIONE: Prompt diviso in due parti ---
    CONTEXT_TEMPLATE = """--- CONTESTO FORNITO ---
{% for chunk in chunks %}
Fonte: {{ chunk.metadata.get('source', 'N/A') }}
Contenuto: {{ chunk.text }}
---
{% endfor %}
"""
    USER_TEMPLATE = """Sei un assistente per l'analisi di documenti. Il tuo unico compito è rispondere alle domande basandoti ESCLUSIVAMENTE sui testi forniti nel CONTESTO.

Regole Obbligatorie:
1.  **NON usare alcuna conoscenza esterna**. La tua memoria è limitata solo al contesto qui sotto.
2.  Rispondi alla "Domanda" usando solo le informazioni trovate nel "CONTESTO".
3.  Se la risposta non si trova nel CONTESTO, rispondi **esattamente e solo**: "L'informazione non è presente nei documenti forniti." Non aggiungere altre parole.
4.  Quando rispondi, cita la fonte esatta da cui hai preso l'informazione, usando il nome del file indicato nel campo "Fonte".

--- CONTESTO ---
{{context}}
--- FINE CONTESTO ---

Domanda: {{user_prompt}}
"""

    prompt_template = ChatPromptTemplate(
        user_prompt_template=USER_TEMPLATE, retrieval_prompt_template=CONTEXT_TEMPLATE
    )

    dag = DagPipeline()
    dag.add_module("embedder", query_embedder)
    dag.add_module("retriever", retriever)
    dag.add_module("prompt", prompt_template)
    dag.add_module("generator", llm_client)
    dag.connect("embedder", "retriever", target_key="query_vector")
    dag.connect("retriever", "prompt", target_key="chunks")
    dag.connect("prompt", "generator", target_key="memory")
    return dag


def ask(
    question: str, k: int, collection_name: str, vectorstore: QdrantVectorstore
) -> str:
    dag = build_rag_pipeline(vectorstore=vectorstore, collection_name=collection_name)
    result = dag.run(
        {
            "embedder": {"text": question},
            "prompt": {"user_prompt": question},
            "retriever": {"collection_name": collection_name, "k": int(k)},
            "generator": {"input": question},
        }
    )
    return str(result["generator"])


def default_run():
    db_dir = DEFAULT_QDRANT_LOCATION
    collection = DEFAULT_COLLECTION

    try:
        import httpx

        check_url = OLLAMA_API_URL.replace("/v1", "")
        httpx.get(check_url, timeout=5)
        print(f"Connesso a Ollama a {check_url} (endpoint API: {OLLAMA_API_URL})")
    except Exception:
        print(
            f"Errore: Impossibile connettersi a Ollama a '{check_url}'. Assicurati che sia in esecuzione.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.exists(db_dir):
        print(f"Rilevato vecchio DB in {db_dir}. Rimuovo per evitare conflitti.")
        try:
            shutil.rmtree(db_dir)
            print(f"Vecchia directory DB {db_dir} rimossa.")
        except Exception:
            sys.exit(1)

    print("--- Creazione Vector Store persistente... ---")
    os.makedirs(db_dir, exist_ok=True)
    vs = _get_vectorstore(location=db_dir)

    print(f"Eseguo ingest su {DATASET_PATH} -> {db_dir}")
    ingest_dataset(vectorstore=vs, collection_name=collection, db_dir=db_dir)

    print("\n--- Ingestione completata ---")
    print("Modalità Q&A (digita 'exit' per uscire)")

    dbg_embedder = CustomSentenceTransformerEmbedder(model_name=EMBEDDING_MODEL_NAME)
    dbg_retriever = vs.as_retriever(collection_name=collection, k=DEFAULT_K)

    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        try:
            query_vector = dbg_embedder.run(text=q)

            retrieved_chunks = dbg_retriever.run(
                query_vector=query_vector, collection_name=collection
            )

            print("\n--- DEBUG: 3. CHUNK RECUPERATI ---")
            if not retrieved_chunks:
                print("!!! NESSUN CHUNK RECUPERATO !!!")
            else:
                for i, chunk in enumerate(retrieved_chunks):
                    metadata = chunk.metadata or {}
                    text = chunk.text or ""
                    print(f"\n[CHUNK {i+1} | Source: {metadata.get('source', 'N/A')}]")
                    clean_text = text.replace("\n", " ")
                    print(f"Text: {clean_text[0:300]}...")
                    print("-----------------------------------")

            ans = ask(
                question=q, k=DEFAULT_K, collection_name=collection, vectorstore=vs
            )

            if "TextBlock(content=" in ans:
                try:
                    ans_raw = ans.split("TextBlock(content=", 1)[1].rsplit(")]", 1)[0]
                    if ans_raw.startswith("'") and ans_raw.endswith("'"):
                        ans_raw = ans_raw[1:-1]
                    ans = bytes(ans_raw, "utf-8").decode("unicode_escape")
                except Exception:
                    pass

            print(f"\nRisposta: {ans}\n")

        except Exception as e:
            print(f"Errore durante la generazione della risposta: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_run()
    else:
        print("Esegui lo script senza argomenti per la modalità interattiva.")
