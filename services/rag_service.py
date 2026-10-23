"""
Serviço RAG (Retrieval-Augmented Generation).

Busca híbrida sobre trechos da base de conhecimento:
- embeddings (cosseno), com o MESMO modelo e prefixos gravados pelo build_index.py
- BM25 (palavras-chave), que ajuda em nomes e termos específicos
- fusão dos dois rankings (RRF) e montagem do contexto a partir do texto ORIGINAL dos trechos
  (sem pedir ao modelo para reescrever documentos, o que gerava risco e custo)

Configuração opcional no Config:
- RAG_MIN_SCORE: similaridade mínima (cosseno, 0 a 1) para um trecho valer. Sem valor = sem corte.
  Se nenhum trecho passar, o contexto fica vazio e o chat transfere para um humano.
  Use rag_check.py para escolher o valor.
- OPENAI_TIMEOUT: timeout das chamadas à OpenAI (padrão 20s)
"""

import json
import logging
import math
import re
import threading
import unicodedata
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=Config.OPENAI_API_KEY,
    timeout=getattr(Config, "OPENAI_TIMEOUT", 20),
    max_retries=1,
)

DENSE_K = 12            # candidatos por embeddings
LEXICAL_K = 12          # candidatos por BM25
FINAL_K = 5             # trechos no contexto final
RRF_K = 60
LEXICAL_WEIGHT = 0.6    # peso do BM25 na fusão (embeddings = 1.0)
MAX_CONTEXT_CHARS = 7000


# ================================
# TEXTO / BM25
# ================================
_STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre entre ate ao aos
que qual quais quem quando onde como porque pq se e ou mas ja nao sim eu voce vc voces nos ele ela eles
elas meu minha meus minhas seu sua seus suas esse essa esses essas este esta estes estas isso isto aquilo
aquele aquela ser estar ter haver fazer ficar ir vai vou tem tenho temos tinha foi sao era sera pode posso
podem preciso quero queria gostaria saber ola oi bom boa dia tarde noite obrigado obrigada favor mais muito
muita tambem so tudo todo toda
""".split())


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str):
    tokens = []
    for w in _normalize(text).split():
        if len(w) < 3 or w in _STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s"):  # plural simples: horarios -> horario
            w = w[:-1]
        tokens.append(w)
    return tokens


class _BM25:
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(_tokenize(d)) for d in docs]
        self.dl = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avgdl = max(float(self.dl.mean()), 1.0) if len(docs) else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for term in set(_tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    )
        return out


# ================================
# CARREGAMENTO DO ÍNDICE
# ================================
class _RagState:
    def __init__(self, index_path: str, metadata_path: str):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict) or "chunks" not in meta or "model" not in meta:
            raise RuntimeError(
                "Índice em formato antigo. Rode 'python build_index.py' para recriá-lo."
            )

        self.chunks = meta["chunks"]
        self.model_name = meta["model"]
        self.query_prefix = meta.get("query_prefix", "")
        if not self.chunks:
            raise RuntimeError("Índice vazio. Verifique a pasta da base de conhecimento.")

        index = faiss.read_index(index_path)
        if index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Índice ({index.ntotal}) e metadados ({len(self.chunks)}) não combinam. "
                "Rode 'python build_index.py' novamente."
            )
        # base pequena: mantemos a matriz em memória e calculamos o cosseno exato com todos os trechos
        self.embeddings = index.reconstruct_n(0, index.ntotal).astype("float32")

        configured = getattr(Config, "EMBEDDING_MODEL", None)
        if configured and configured != self.model_name:
            logger.warning(
                "Config.EMBEDDING_MODEL (%s) difere do modelo do índice (%s). Usando o do índice.",
                configured, self.model_name,
            )

        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        if dim != self.embeddings.shape[1]:
            raise RuntimeError(
                f"Dimensão do modelo ({dim}) diferente do índice ({self.embeddings.shape[1]})."
            )
        self.model.max_seq_length = min(self.model.max_seq_length or 512, 512)

        self.bm25 = _BM25([c["text"] for c in self.chunks])
        self.encode_lock = threading.Lock()
        logger.info("RAG carregado: %d trechos, modelo %s", len(self.chunks), self.model_name)


_state = None
_state_lock = threading.Lock()


def _get_state() -> _RagState:
    """Carrega o índice na primeira necessidade. Se falhar, lança erro (o chat responde 'problema técnico')."""
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = _RagState(Config.EMBEDDINGS_INDEX_PATH, Config.EMBEDDINGS_METADATA_PATH)
    return _state


try:  # carrega no start para a primeira pergunta não ficar lenta
    _get_state()
except Exception:
    logger.exception("Falha ao inicializar o serviço RAG (nova tentativa na próxima pergunta)")


# ================================
# BUSCA
# ================================
def _min_score():
    value = getattr(Config, "RAG_MIN_SCORE", None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def search(question: str, top_k: int = FINAL_K) -> list:
    """Busca híbrida. Retorna lista de dicts com id, file, heading, text, dense, lexical, fused."""
    question = (question or "").strip()
    if not question:
        return []

    st = _get_state()
    with st.encode_lock:
        q = st.model.encode(
            [st.query_prefix + question], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype("float32")

    dense = st.embeddings @ q                 # cosseno com todos os trechos
    lexical = st.bm25.scores(question)

    fused = {}
    for rank, i in enumerate(np.argsort(-dense)[:DENSE_K]):
        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
    lex_order = [int(i) for i in np.argsort(-lexical)[:LEXICAL_K] if lexical[i] > 0]
    for rank, i in enumerate(lex_order):
        fused[i] = fused.get(i, 0.0) + LEXICAL_WEIGHT / (RRF_K + rank + 1)

    min_score = _min_score()
    results = []
    for i in sorted(fused, key=lambda k: fused[k], reverse=True):
        if min_score is not None and dense[i] < min_score:
            continue
        chunk = st.chunks[i]
        results.append({
            "id": i,
            "file": chunk.get("file", ""),
            "heading": chunk.get("heading", ""),
            "text": chunk["text"],
            "dense": float(dense[i]),
            "lexical": float(lexical[i]),
            "fused": fused[i],
        })
        if len(results) >= top_k:
            break
    return results


def get_reranked_context(question: str, top_k_final: int = FINAL_K) -> tuple:
    """
    Retorna (contexto, fontes). O contexto é o texto original dos melhores trechos
    (nada é reescrito por modelo). Sem trechos relevantes: ("", []).
    """
    results = search(question, top_k_final)
    if not results:
        return "", []

    parts, sources, total = [], [], 0
    for r in results:
        if parts and total + len(r["text"]) > MAX_CONTEXT_CHARS:
            break
        parts.append(r["text"])
        sources.append(f'{r["file"]} › {r["heading"]}')
        total += len(r["text"])

    if logger.isEnabledFor(logging.DEBUG):
        for r in results:
            logger.debug("RAG: %s | cos=%.3f bm25=%.2f", r["heading"], r["dense"], r["lexical"])

    return "\n\n---\n\n".join(parts), sources


get_context = get_reranked_context  # alias


# ================================
# PERGUNTA DE BUSCA (usa o histórico)
# ================================
def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def generate_search_query(question: str, history: list) -> str:
    """Reescreve a última pergunta como uma pergunta autônoma, usando o histórico recente."""
    if not history:
        return question

    history_str = "\n".join(
        f"Usuário: {_clip(h['user'], 300)}\nAssistente: {_clip(h['bot'], 300)}" for h in history[-4:]
    )
    prompt = (
        "Reescreva a última mensagem do usuário como uma única pergunta de busca, em português, "
        "que faça sentido sozinha (sem depender do histórico). Mantenha termos específicos "
        "(séries, documentos, nomes, valores). Se a mensagem já for autônoma, apenas repita-a. "
        "Responda somente com a pergunta, sem explicações.\n\n"
        f"--- Histórico ---\n{history_str}\n\n"
        f"--- Última mensagem ---\n{question}\n"
    )
    try:
        response = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        text = text.splitlines()[0].strip() if text else ""
        text = re.sub(r"^(pergunta( de busca)?( otimizada)?\s*:\s*)", "", text, flags=re.IGNORECASE)
        text = text.strip(" \"'“”")
        return text if 3 <= len(text) <= 300 else question
    except Exception:
        logger.exception("Falha ao gerar pergunta de busca; usando a pergunta original")
        return question