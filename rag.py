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

# -------- CARICAMENTO .ENV -------#
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
# ---------------------------------#

# LLM initialization
from google import genai

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY non impostata. Esporta prima la variabile, ad es.: "
        "export GEMINI_API_KEY='la_tua_chiave'"
    )
client = genai.Client(api_key=API_KEY)

DATA_DIR = Path("dataset/ETFs")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
EMB_CACHE = CACHE_DIR / "embeddings.pkl"

# Embedding model
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def load_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL_NAME)
    return _model


def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)


def chunk_text(text: str, chunk_size=800, overlap=120) -> List[str]:
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
            print(f"Errore nel parsing {pdf}: {e}")
    return docs


# Indexing for retrieval
def build_or_load_index() -> Tuple[List[Dict], np.ndarray]:
    if EMB_CACHE.exists():
        with open(EMB_CACHE, "rb") as f:
            payload = pickle.load(f)
        return payload["docs"], payload["embeddings"]
    # Reindixing
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


# Retrieval
def retrieve(query: str, docs: List[Dict], embs: np.ndarray, k: int = 5) -> List[Dict]:
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


# Calls the LLM to generate responses
def call_llm(prompt: str) -> str:
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return getattr(resp, "text", str(resp))


def make_prompt(query: str, contexts: List[Dict]) -> str:
    # Argumented prompt (force the model not to invent answers and
    # to always cite where the data comes from)
    header = (
        "Sei un assistente che risponde SOLO usando i contesti forniti.\n"
        "Cita sempre le fonti come [source: <file>, chunk_n] alla fine dei passaggi rilevanti.\n"
        "Se l'informazione non è nei contesti, dì che non è disponibile.\n\n"
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
        "- Rispondi in modo conciso e puntuale.\n"
        "- Cita le fonti con [source: <file>, <chunk_index>].\n"
        "- Non inventare contenuti.\n"
    )
    return f"{header}{ctx_text}\n\n{user}{instr}"


# Orchestration of all RAG phases
def answer(query: str, k: int = 5) -> Tuple[str, List[Dict]]:
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
            resp, ctx = answer(q, k=5)
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
