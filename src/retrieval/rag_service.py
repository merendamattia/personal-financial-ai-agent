"""
Modulo del Servizio RAG basato su LangChain.
Sostituisce il retriever manuale originale con una pipeline robusta.
"""
import logging
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

############## TODO: cambiare commenti --> rebderli più completi e tradurli in inglese #############

load_dotenv()
logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, dataset_path="./dataset/ETFs_2", qdrant_path="./qdrant_db_lc"):
        logger.info("Inizializzazione RAGService con motore LangChain...")
        self.dataset_path = dataset_path
        self.qdrant_path = qdrant_path
        self.embedding_model_name = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.llm_model_name = os.getenv("OLLAMA_MODEL", "llama3:8b")
        self.ollama_base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.llm = Ollama(
            base_url=self.ollama_base_url, model=self.llm_model_name, temperature=0.0
        )

        self.vector_store = self._load_and_index_documents()
        self.rag_chain = self._create_rag_chain()
        logger.info("RAGService pronto.")

    def _load_and_index_documents(self):
        logger.info("Controllo e indicizzazione documenti da: %s", self.dataset_path)
        if os.path.exists(self.qdrant_path):
            shutil.rmtree(self.qdrant_path)
            logger.info("Vecchia directory DB Qdrant rimossa: %s", self.qdrant_path)

        loader = PyPDFDirectoryLoader(self.dataset_path)
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"Nessun PDF trovato in {self.dataset_path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)

        logger.info("Creazione Vector Store Qdrant in: %s", self.qdrant_path)
        vector_store = Qdrant.from_documents(
            chunks,
            self.embeddings,
            path=self.qdrant_path,
            collection_name="etf_documents_lc",
            force_recreate=True,
        )
        return vector_store

    def _create_rag_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        prompt_template = """Sei un analista finanziario che risponde basandosi SOLO sul Contesto fornito.
Contesto:
{context}

Domanda: {question}
Risposta Fattuale:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve_context(self, question: str, k: int = 5) -> list:
        """
        Questo è il metodo che l'agente userà. Restituisce i documenti recuperati.
        """
        logger.debug(
            "RAGService (LangChain) interrogato con la domanda: '%s' per k=%d",
            question,
            k,
        )
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # Invece di una chain complessa, facciamo solo il retrieval
        docs = retriever.get_relevant_documents(question)

        # Formattiamo i documenti in un formato simile all'originale
        results = []
        for doc in docs:
            # Punteggio di rilevanza non è sempre disponibile, usiamo un placeholder
            score = doc.metadata.get("score", 1.0)
            results.append(
                {
                    "id": doc.metadata.get(
                        "source", "unknown"
                    ),  # Usa il nome del file come ID
                    "score": score,
                    "text": doc.page_content,
                }
            )
        return results
