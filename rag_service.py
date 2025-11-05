#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Servizio RAG basato su LangChain.
Questa versione è strutturata come una classe riusabile per essere importata
e utilizzata da altre applicazioni, come un agente.
"""

import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Componenti LangChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Caricamento configurazione
load_dotenv()


class RAGService:
    """
    Una classe che incapsula l'intera pipeline RAG, dall'ingestione alla query.
    """

    def __init__(self):
        print("--- [RAGService] Inizializzazione... ---")
        # Configurazione
        self.dataset_path = "./dataset/ETFs_2"
        self.qdrant_path = "./qdrant_lc"
        self.embedding_model_name = "BAAI/bge-m3"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model_name = os.getenv("OLLAMA_MODEL", "llama3:8b")
        self.default_k = 3

        # Inizializza componenti
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.llm = Ollama(
            base_url=self.ollama_base_url, model=self.llm_model_name, temperature=0.01
        )

        # Carica, splitta e indicizza i documenti
        self.vector_store = self._load_and_index_documents()

        # Crea la RAG chain
        self.rag_chain = self._create_rag_chain()
        print("--- [RAGService] Pronto. ---")

    def _load_and_index_documents(self) -> Qdrant:
        """Carica, processa e indicizza i documenti PDF in Qdrant."""
        print("[RAGService] Avvio caricamento e indicizzazione documenti...")

        # Pulisce la vecchia directory per garantire dati aggiornati
        if os.path.exists(self.qdrant_path):
            shutil.rmtree(self.qdrant_path)

        loader = PyPDFDirectoryLoader(self.dataset_path)
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(
                f"Nessun documento PDF trovato in {self.dataset_path}"
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300
        )
        chunks = text_splitter.split_documents(docs)

        print(
            f"[RAGService] Documenti divisi in {len(chunks)} chunk. Creazione Vector Store..."
        )
        vector_store = Qdrant.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            path=self.qdrant_path,
            collection_name="etf_documents",
            force_recreate=True,
        )
        print("[RAGService] Vector Store creato con successo.")
        return vector_store

    def _create_rag_chain(self):
        """Crea la chain LangChain per il processo di domanda e risposta."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.default_k})

        prompt_template = """
Sei un assistente esperto di finanza il cui unico compito è estrarre informazioni fattuali dai documenti forniti.

Regole Assolute:
1. Rispondi alla "Domanda" usando ESCLUSIVAMENTE le informazioni trovate nel "Contesto".
2. NON usare mai la tua conoscenza pregressa.
3. Se l'informazione non è nel Contesto, rispondi solo e unicamente: "L'informazione non è presente nei documenti forniti."
4. Cita sempre la fonte del documento quando trovi una risposta.

--- CONTESTO ---
{context}
--- FINE CONTESTO ---

Domanda: {question}
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def query(self, question: str) -> str:
        """
        Interroga il sistema RAG con una domanda e restituisce la risposta.
        Questo è il metodo che verrà esposto come tool.
        """
        print(f"[RAGService] Ricevuta query: '{question}'")
        return self.rag_chain.invoke(question)


# Blocco per testare il servizio in modo indipendente
if __name__ == "__main__":
    rag_service = RAGService()

    print("\nModalità Test RAG Service (digita 'exit' per uscire)")
    while True:
        q = input("> ")
        if q.lower() in ("exit", "quit"):
            break
        if q:
            response = rag_service.query(q)
            print(f"\nRisposta:\n{response}\n")
