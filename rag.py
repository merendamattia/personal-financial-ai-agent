import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Load environment variables
logger.debug("Loading environment variables")
load_dotenv()
logger.info("Environment variables loaded")

from google import genai

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY non impostata. Esporta prima la variabile, ad es.: "
        "export GOOGLE_API_KEY='la_tua_chiave'"
    )
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro")
if not GOOGLE_MODEL:
    raise EnvironmentError(
        "GOOGLE_MODEL non impostato. Esporta prima la variabile, ad es.: "
        "export GOOGLE_MODEL='gemini-2.5-pro'"
    )
client = genai.Client(api_key=GOOGLE_API_KEY)

DATA_DIR = Path("dataset/ETFs")
CACHE_DIR = DATA_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
EMB_CACHE = CACHE_DIR / "embeddings.pkl"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# Embedding model
EMB_MODEL_NAME = "all-roberta-large-v1"
_model = None


def load_embedder():
    """
    Load and cache the sentence transformer embedding model.

    Returns:
        SentenceTransformer: The embedding model instance. Uses lazy loading and caching
                            to avoid reloading the model on multiple calls.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL_NAME)
    return _model


def read_pdf_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_path (Path): Path to the PDF file.

    Returns:
        str: Complete text content from all pages joined with newlines.
             Returns empty string if extraction fails for a page.
    """
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)


def chunk_text(
    text: str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks to preserve context.

    Args:
        text (str): The text to chunk.
        chunk_size (int): Maximum number of characters per chunk. Default: 800.
        overlap (int): Number of overlapping characters between consecutive chunks. Default: 120.

    Returns:
        List[str]: List of text chunks with specified overlap.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# Reading and chunking of pdf files
def ingest_pdfs(pdf_dir: Path) -> List[Dict]:
    """
    Read all PDF files from a directory and chunk them.

    Args:
        pdf_dir (Path): Directory path containing PDF files to ingest.

    Returns:
        List[Dict]: List of documents with keys: 'id' (unique identifier),
                   'source' (file path), and 'text' (chunk content).
                   Skips PDFs with no extractable text.

    Raises:
        Logs errors for individual PDFs but continues processing.
    """
    docs = []
    for pdf in sorted(pdf_dir.rglob("*.pdf")):
        try:
            txt = read_pdf_text(pdf)
            if not txt.strip():
                continue
            for i, ch in enumerate(chunk_text(txt)):
                docs.append(
                    {"id": f"{pdf.name}::chunk_{i}", "source": str(pdf), "text": ch}
                )
        except Exception as e:
            logging.error(f"Errore nel parsing {pdf}: {e}")
    return docs


def build_or_load_index() -> Tuple[List[Dict], np.ndarray]:
    """
    Build or load a cached index of documents and their embeddings.

    Returns:
        Tuple[List[Dict], np.ndarray]: A tuple containing:
            - List of document dictionaries with 'id', 'source', and 'text' keys
            - Numpy array of embeddings (one per document, shape: [n_docs, embedding_dim])

    Process:
        - If cached embeddings exist, loads them
        - Otherwise, ingests all PDFs from DATA_DIR, encodes them, and caches results

    Raises:
        RuntimeError: If no PDFs are found in DATA_DIR.
    """
    if EMB_CACHE.exists():
        with open(EMB_CACHE, "rb") as f:
            payload = pickle.load(f)
        return payload["docs"], payload["embeddings"]
    # Reindexing
    docs = ingest_pdfs(DATA_DIR)
    if not docs:
        raise RuntimeError(f"Nessun PDF trovato in {DATA_DIR}")
    embedder = load_embedder()
    texts = [d["text"] for d in docs]
    embs = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    with open(EMB_CACHE, "wb") as f:
        pickle.dump({"docs": docs, "embeddings": embs}, f)
    return docs, embs


def retrieve(query: str, docs: List[Dict], embs: np.ndarray, k: int = 5) -> List[Dict]:
    """
    Retrieve the k most similar documents for a given query using semantic similarity.

    Args:
        query (str): The user query string.
        docs (List[Dict]): List of documents with embeddings.
        embs (np.ndarray): Document embeddings array (shape: [n_docs, embedding_dim]).
        k (int): Number of top results to return. Default: 5.

    Returns:
        List[Dict]: List of k most similar documents sorted by similarity score (descending),
                   each with added 'score' key containing the cosine similarity [0, 1].
    """
    embedder = load_embedder()
    q = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q, embs)[0]
    idxs = np.argsort(-sims)[:k]
    results = []
    for idx in idxs:
        d = docs[idx].copy()
        d["score"] = float(sims[idx])
        results.append(d)
    return results


def call_llm(prompt: str) -> str:
    """
    Generate a response from the Gemini LLM given a prompt.

    Args:
        prompt (str): The prompt to send to the LLM.

    Returns:
        str: The generated response text from the model.

    Note:
        Uses the Gemini model specified by the GOOGLE_MODEL variable. Requires GOOGLE_API_KEY environment variable.
    """
    resp = client.models.generate_content(model=GOOGLE_MODEL, contents=prompt)
    return getattr(resp, "text", str(resp))


def make_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Create an augmented prompt for the LLM combining context and user query.

    Args:
        query (str): The user's question.
        contexts (List[Dict]): List of retrieved context documents with keys:
                              'id', 'source', 'text', and 'score'.

    Returns:
        str: Formatted prompt ready to send to the LLM, including:
            - System instructions for the assistant behavior
            - Retrieved document contexts with sources
            - User query
            - Specific guidelines for response generation

    Note:
        The prompt encourages the model to cite sources and acknowledge when
        information is not available in the provided contexts.
    """
    # Augmented prompt with better instructions for LLM
    header = (
        "Sei un assistente finanziario esperto e utile.\n"
        "Usa i contesti forniti per rispondere alle domande in modo chiaro e accurato.\n"
        "Cita sempre le fonti quando usi informazioni specifiche dai documenti.\n\n"
    )
    ctx_blocks = []
    for c in contexts:
        ctx_blocks.append(
            # [chunk | pdf]
            f"[CTX id={c['id']} | source={Path(c['source']).name}]\n{c['text']}\n"
        )
    ctx_text = "\n---\n".join(ctx_blocks)
    user = f"Domanda: {query}\n"
    instr = (
        "Istruzioni:\n"
        "- Fornisci una risposta utile basata sui documenti forniti.\n"
        "- Se l'informazione è disponibile nei contesti, usala per rispondere completamente.\n"
        "- Cita le fonti come [source: <file>, <chunk_index>] quando rilevante.\n"
        "- Se l'informazione non è nei contesti, dillo chiaramente.\n"
        "- Rispondi in modo conversazionale e amichevole, non robotico.\n"
    )
    return f"{header}{ctx_text}\n\n{user}{instr}"


def answer(query: str, k: int = 15) -> Tuple[str, List[Dict]]:
    """
    Complete RAG pipeline: retrieve relevant documents and generate an answer.

    Args:
        query (str): The user's question.
        k (int): Number of context documents to retrieve. Default: 15.

    Returns:
        Tuple[str, List[Dict]]: A tuple containing:
            - Generated response from the LLM
            - List of retrieved context documents used for the answer

    Pipeline Steps:
        1. Build or load the document index with embeddings
        2. Retrieve the k most similar documents to the query
        3. Create an augmented prompt with retrieved contexts
        4. Call the LLM to generate the answer
    """
    docs, embs = build_or_load_index()
    ctxs = retrieve(query, docs, embs, k=k)
    prompt = make_prompt(query, ctxs)
    text = call_llm(prompt)
    return text, ctxs


if __name__ == "__main__":
    print("Indice: creazione o caricamento cache…")
    _ = build_or_load_index()
    print("Pronto. Scrivi una domanda (CTRL+C per uscire).")
    while True:
        try:
            q = input("\n> ")
            if not q.strip():
                continue
            resp, ctx = answer(q, k=15)
            print("\n=== RISPOSTA ===")
            print(resp)
            print("\n=== FONTI RECUPERATE ===")
            for c in ctx:
                print(
                    f"- {Path(c['source']).name} | {c['id']} | score={c['score']:.3f}"
                )
        except KeyboardInterrupt:
            print("\nArrivederci!")
            break
