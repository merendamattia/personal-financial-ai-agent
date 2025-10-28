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
        "GOOGLE_API_KEY not set. Export the variable first, for example: "
        "export GOOGLE_API_KEY='your_key'"
    )
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro")
if not GOOGLE_MODEL:
    raise EnvironmentError(
        "GOOGLE_MODEL not set. Export the variable first, for example: "
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

    This function recursively searches for all PDF files in the specified directory,
    extracts their text content, and splits them into overlapping chunks to preserve
    semantic context. Each chunk is indexed with a unique identifier and source reference.

    Args:
        pdf_dir (Path): Directory path containing PDF files to ingest.

    Returns:
        List[Dict]: List of documents with keys:
            - 'id' (str): Unique identifier in format "{filename}::chunk_{index}"
            - 'source' (str): Full file path to the PDF
            - 'text' (str): The text chunk content

            Returns empty list if no PDFs with extractable text are found.
            Skips PDFs with empty or unparseable content.

    Raises:
        Logs warnings for individual PDFs that fail to parse, but continues processing
        other files. The function is resilient to individual PDF failures.

    Example:
        >>> docs = ingest_pdfs(Path("dataset/ETFs"))
        >>> print(f"Ingested {len(docs)} document chunks")
        >>> print(docs[0])
        {'id': 'fund.pdf::chunk_0', 'source': '/path/to/fund.pdf', 'text': '...'}
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
            logging.error(f"Error parsing {pdf}: {e}")
    return docs


def build_or_load_index() -> Tuple[List[Dict], np.ndarray]:
    """
    Build or load a cached index of documents and their embeddings.

    This function implements a caching mechanism to avoid recomputing embeddings on every run.
    It checks for a cached pickle file containing previously computed embeddings and document
    metadata. If the cache exists, it loads and returns it immediately. Otherwise, it:
    1. Ingests all PDF documents from DATA_DIR
    2. Generates embeddings using the SentenceTransformer model
    3. Saves both documents and embeddings to a cache file for future runs

    Returns:
        Tuple[List[Dict], np.ndarray]: A tuple containing:
            - List of document dictionaries with keys:
                - 'id': unique identifier (format: "{filename}::chunk_{index}")
                - 'source': full path to the source PDF
                - 'text': the document chunk content
            - Numpy array of embeddings with shape [n_docs, embedding_dim]
              where embedding_dim = 1024 (for all-roberta-large-v1 model)

    Raises:
        RuntimeError: If no PDFs are found in DATA_DIR after ingestion attempt.
        FileNotFoundError: If DATA_DIR does not exist (implicitly through ingest_pdfs).

    Performance:
        First run: O(n_docs * text_length) for embedding generation
        Subsequent runs: O(1) disk read operation
        Cache is stored at: {DATA_DIR}/.cache/embeddings.pkl

    Example:
        >>> docs, embeddings = build_or_load_index()
        >>> print(f"Loaded {len(docs)} documents with shape {embeddings.shape}")
        Loaded 150 documents with shape (150, 1024)
    """
    if EMB_CACHE.exists():
        with open(EMB_CACHE, "rb") as f:
            payload = pickle.load(f)
        return payload["docs"], payload["embeddings"]
    # Reindexing
    docs = ingest_pdfs(DATA_DIR)
    if not docs:
        raise RuntimeError(f"No PDFs found in {DATA_DIR}")
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

    This function constructs a carefully formatted prompt that:
    1. Sets the LLM's role and instructions (system message)
    2. Includes retrieved document contexts with their metadata
    3. Presents the user's question
    4. Provides specific guidelines for generating a helpful response

    The prompt engineering emphasizes source citation and honest acknowledgment
    of information limitations, encouraging the model to only use information
    from the provided contexts rather than relying on general knowledge.

    Args:
        query (str): The user's question or query string.
        contexts (List[Dict]): List of retrieved context documents. Each dict should have:
            - 'id' (str): unique document identifier
            - 'source' (str): full path to source file
            - 'text' (str): the document chunk content
            - 'score' (float): semantic similarity score [0, 1]

    Returns:
        str: Formatted prompt ready to send to the LLM, structured as:
            - Header: System instructions for assistant behavior
            - Context blocks: Retrieved documents with sources and metadata
            - Query section: The user's question formatted as "Domanda: {query}"
            - Instructions: Specific guidelines for response generation

    Key Features:
        - Clear source attribution format for traceability
        - Instructions to acknowledge information gaps
        - Conversational tone encouragement
        - Separation of context blocks with markdown separators

    Example:
        >>> contexts = [
        ...     {
        ...         'id': 'fund.pdf::chunk_0',
        ...         'source': '/path/to/fund.pdf',
        ...         'text': 'ETF information...',
        ...         'score': 0.95
        ...     }
        ... ]
        >>> prompt = make_prompt("What is an ETF?", contexts)
        >>> # prompt contains formatted instructions, context, and question
    """
    # Augmented prompt with better instructions for LLM
    header = (
        "You are a helpful and expert financial assistant.\n"
        "Use the provided contexts to answer questions clearly and accurately.\n"
        "Always cite sources when using specific information from the documents.\n\n"
    )
    ctx_blocks = []
    for c in contexts:
        ctx_blocks.append(
            # [chunk | pdf]
            f"[CTX id={c['id']} | source={Path(c['source']).name}]\n{c['text']}\n"
        )
    ctx_text = "\n---\n".join(ctx_blocks)
    user = f"Question: {query}\n"
    instr = (
        "Instructions:\n"
        "- Provide a helpful answer based on the provided documents.\n"
        "- If information is available in the contexts, use it to answer completely.\n"
        "- Cite sources as [source: <file>, <chunk_index>] when relevant.\n"
        "- If information is not in the contexts, state this clearly.\n"
        "- Answer in a conversational and friendly manner, not robotic.\n"
    )
    return f"{header}{ctx_text}\n\n{user}{instr}"


def answer(query: str, k: int = 15) -> Tuple[str, List[Dict]]:
    """
    Complete RAG pipeline: retrieve relevant documents and generate an answer.

    This is the main orchestrator function that implements the full Retrieval-Augmented
    Generation workflow. It takes a user query and returns an LLM-generated answer
    augmented with relevant context from the document corpus.

    The RAG pipeline follows these steps:
    1. Build or load the pre-computed document index with embeddings
    2. Retrieve the k most semantically similar documents to the query
    3. Construct a prompt with retrieved contexts and user query
    4. Call the LLM to generate a context-aware answer
    5. Return both the answer and source documents for transparency

    Args:
        query (str): The user's question or query string.
        k (int): Number of context documents to retrieve for augmentation. Default: 15.
                 Higher values provide more context but increase prompt size and cost.

    Returns:
        Tuple[str, List[Dict]]: A tuple containing:
            - Generated response text from the LLM
            - List of retrieved context documents used for the answer, each with:
                - 'id': unique identifier
                - 'source': source file path
                - 'text': document chunk
                - 'score': semantic similarity score

    Performance:
        - Embedding computation: O(|query|) (shared with other queries after first run)
        - Retrieval: O(k * n_docs) for similarity computation
        - LLM call: depends on model latency and prompt length

    Example:
        >>> answer_text, sources = answer("What are bond ETFs?", k=10)
        >>> print(answer_text)
        "Bond ETFs are exchange-traded funds..."
        >>> for source in sources:
        ...     print(f"Source: {source['source']} (score: {source['score']:.2f})")
    """
    docs, embs = build_or_load_index()
    ctxs = retrieve(query, docs, embs, k=k)
    prompt = make_prompt(query, ctxs)
    text = call_llm(prompt)
    return text, ctxs


if __name__ == "__main__":
    print("Index: creating or loading cache...")
    _ = build_or_load_index()
    print("Ready. Write a question (CTRL+C to exit).")
    while True:
        try:
            q = input("\n> ")
            if not q.strip():
                continue
            resp, ctx = answer(q, k=15)
            print("\n=== ANSWER ===")
            print(resp)
            print("\n=== RETRIEVED SOURCES ===")
            for c in ctx:
                print(
                    f"- {Path(c['source']).name} | {c['id']} | score={c['score']:.3f}"
                )
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
