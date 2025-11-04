#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema RAG completo e robusto basato su LangChain, Qdrant e Ollama.
Questa versione sostituisce l'implementazione precedente per garantire
controllo, trasparenza e affidabilità.
"""

import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Componenti LangChain ---
# --- Componenti LangChain (con import corretti) ---
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser

# CORREZIONE: Importa i componenti dal pacchetto 'langchain_core'
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configurazione Globale ---
load_dotenv()
DATASET_PATH = "./dataset/ETFs_2"
QDRANT_PATH = "./qdrant_lc"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
DEFAULT_K = 3


def load_and_split_documents(path: str) -> list:
    """
    Carica i PDF da una directory, li pulisce e li divide in chunk.
    """
    print(f"--- Caricamento e splitting dei documenti da: {path} ---")

    # 1. Carica i documenti PDF dalla directory
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    if not docs:
        raise ValueError(f"Nessun documento PDF trovato in {path}")

    print(f"Trovati {len(docs)} documenti.")

    # 2. Inizializza lo splitter di testo robusto di LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300, length_function=len
    )

    # 3. Dividi i documenti in chunk
    chunks = text_splitter.split_documents(docs)
    print(f"Documenti divisi in {len(chunks)} chunk.")

    return chunks


def setup_vector_store(chunks: list, embedding_model: HuggingFaceEmbeddings) -> Qdrant:
    """
    Crea e popola il database vettoriale Qdrant con i chunk di testo.
    """
    print(f"--- Creazione del Vector Store in: {QDRANT_PATH} ---")

    # Pulisce la directory precedente per garantire una partenza pulita
    if os.path.exists(QDRANT_PATH):
        print("Rilevato vecchio DB, lo rimuovo...")
        shutil.rmtree(QDRANT_PATH)

    # Crea il vector store da zero utilizzando Qdrant on-disk
    # LangChain gestisce l'embedding e l'inserimento automaticamente
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        path=QDRANT_PATH,
        collection_name="etf_documents",
        force_recreate=True,
    )
    print("--- Vector Store creato e popolato con successo. ---")
    return vector_store


def create_rag_chain(vector_store: Qdrant, llm: Ollama):
    """
    Crea e assembla la pipeline RAG usando LangChain Expression Language (LCEL).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": DEFAULT_K})

    # Template del prompt, rigido e fattuale
    prompt_template = """
Sei un assistente per l'analisi di documenti finanziari. Il tuo unico compito è rispondere alle domande basandoti ESCLUSIVAMENTE sui testi forniti nel CONTESTO.

Regole Obbligatorie:
1. NON usare alcuna conoscenza pregressa. La tua memoria è limitata solo al contesto qui sotto.
2. Rispondi alla "Domanda" usando solo le informazioni trovate nel "CONTESTO".
3. Se la risposta non è nel CONTESTO, rispondi esattamente e solo: "L'informazione non è presente nei documenti forniti."
4. Quando rispondi, cita la fonte esatta da cui hai preso l'informazione, usando il nome del file indicato nel campo "source" del contesto.

--- CONTESTO ---
{context}
--- FINE CONTESTO ---

Domanda: {question}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Assemblaggio della chain con LCEL (LangChain Expression Language)
    # È una sintassi pulita per definire il flusso di dati
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":
    # 1. Inizializza i componenti principali
    print("--- Inizializzazione del sistema RAG con LangChain ---")

    # Modello di embedding (usa il wrapper di LangChain)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Modello di linguaggio (usa il wrapper di LangChain per Ollama)
    # Impostiamo una temperatura bassa per risposte fattuali
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL_NAME, temperature=0.01)

    # 2. Flusso di Ingestione
    document_chunks = load_and_split_documents(DATASET_PATH)
    vector_store = setup_vector_store(document_chunks, embeddings)

    # 3. Creazione della RAG Chain
    rag_chain = create_rag_chain(vector_store, llm)

    print("\n--- Sistema RAG pronto. ---")
    print("Modalità Q&A (digita 'exit' per uscire)")

    # 4. Loop Interattivo per le Domande
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        print("\nRecupero informazioni...")
        try:
            # 5. Invocazione della chain
            # LangChain gestisce l'intero processo di recupero e generazione
            response = rag_chain.invoke(question)
            print(f"\nRisposta:\n{response}\n")
        except Exception as e:
            print(f"\nSi è verificato un errore: {e}\n")
