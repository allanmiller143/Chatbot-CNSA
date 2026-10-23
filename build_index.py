"""
Indexador da base de conhecimento. Rode sempre que alterar os arquivos .md:

    python build_index.py

O que ele faz:
- Divide cada .md em trechos por título (#, ##, ###), mantendo o caminho do título em cada trecho
- Trechos muito longos são divididos por parágrafo; trechos minúsculos são juntados
- Gera embeddings normalizados com um modelo multilíngue (cosseno)
- Grava no metadata.json o nome do modelo e os prefixos usados, para o rag_service
  carregar exatamente o mesmo modelo (impossível divergir)
"""

import json
import os
import re
import sys
import unicodedata

import faiss
import numpy as np

# --- Configurações ---
BASE_DIR = "knowledge_base"
INDEX_DIR = "embeddings"
INDEX_FILE = "kb.index"
METADATA_FILE = "metadata.json"

# Modelo multilíngue, bom em português. Alternativa mais leve (menos memória):
# "intfloat/multilingual-e5-base". Os prefixos de cada modelo são tratados abaixo.
MODEL_NAME = "BAAI/bge-m3"

MAX_CHARS = 1400     # tamanho máximo de um trecho (~350 tokens em português)
MIN_CHARS = 250      # trechos menores que isso são juntados ao seguinte
OVERLAP_CHARS = 300  # parágrafo final repetido no início do próximo trecho (só em seções longas)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def get_prefixes(model_name: str):
    """Prefixos de consulta/passagem exigidos por cada família de modelo."""
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge-m3" in name:
        return "", ""
    if "bge" in name:  # bge-*-en-v1.5 e similares
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def doc_title_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem[:1].upper() + stem[1:] if stem else "Documento"


def parse_sections(text: str, doc_title: str):
    """Retorna [(caminho_de_títulos, corpo)], respeitando blocos de código."""
    sections = []
    stack = []  # [(nível, título)]
    buf = []
    in_code = False

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if body and re.search(r"\w", body):
            sections.append(([t for _, t in stack] or [doc_title], body))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue
        match = None if in_code else HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _common_prefix(a, b):
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def merge_small_sections(sections):
    """
    Junta seções minúsculas às seguintes, para não gerar trechos sem informação.
    O título do trecho vira o caminho comum das seções juntadas, e cada sub-título
    continua visível no texto.
    """
    groups = []  # cada grupo: {"path": [...], "parts": [(rótulo, corpo), ...]}
    for path, body in sections:
        if groups:
            g = groups[-1]
            size = sum(len(b_) for _, b_ in g["parts"])
            if size < MIN_CHARS and size + len(body) + 2 <= MAX_CHARS:
                g["parts"].append((path[-1], body))
                g["path"] = _common_prefix(g["path"], path) or g["path"][:1]
                continue
        groups.append({"path": path, "parts": [(path[-1], body)]})

    merged = []
    for g in groups:
        if len(g["parts"]) == 1:
            merged.append((g["path"], g["parts"][0][1]))
        else:
            body = "\n\n".join(f"**{label}**\n{b_}" for label, b_ in g["parts"])
            merged.append((g["path"], body))
    return merged


def split_long(body: str, max_chars: int = MAX_CHARS):
    """Divide uma seção longa por parágrafos (depois linhas, depois frases), com pequena sobreposição."""
    if len(body) <= max_chars:
        return [body]

    units = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for line in para.split("\n"):
            line = line.strip()
            while len(line) > max_chars:
                cut = line.rfind(". ", 0, max_chars)
                cut = cut + 1 if cut > max_chars // 2 else max_chars
                units.append(line[:cut].strip())
                line = line[cut:].strip()
            if line:
                units.append(line)

    chunks, cur, cur_len = [], [], 0
    for unit in units:
        if cur and cur_len + len(unit) + 1 > max_chars:
            chunks.append("\n".join(cur))
            last = cur[-1]
            cur = [last] if len(last) <= OVERLAP_CHARS else []
            cur_len = sum(len(u) + 1 for u in cur)
        cur.append(unit)
        cur_len += len(unit) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_document(text: str, doc_title: str):
    """Retorna [(heading, texto_do_trecho)] de um arquivo .md."""
    text = FRONT_MATTER_RE.sub("", text.replace("\r\n", "\n"), count=1)
    sections = merge_small_sections(parse_sections(text, doc_title))

    out = []
    for path, body in sections:
        # inclui o nome do arquivo no caminho, a menos que o título já o contenha
        first = _norm(path[0])
        title_n = _norm(doc_title)
        if title_n and title_n not in first and first not in title_n:
            path = [doc_title] + path
        heading = " > ".join(path)
        for piece in split_long(body):
            out.append((heading, f"{heading}\n\n{piece}"))
    return out


def load_chunks(base_dir: str):
    chunks = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for file in sorted(files):
            if not file.lower().endswith(".md"):
                continue
            path = os.path.join(root, file)
            rel = os.path.relpath(path, base_dir).replace(os.sep, "/")
            print(f"Processando arquivo: {rel}")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for heading, chunk_text in chunk_document(text, doc_title_from_filename(file)):
                chunks.append({"id": len(chunks), "file": rel, "heading": heading, "text": chunk_text})
    return chunks


def build(base_dir=BASE_DIR, index_dir=INDEX_DIR, model_name=MODEL_NAME):
    os.makedirs(index_dir, exist_ok=True)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Pasta '{base_dir}' criada. Adicione seus arquivos .md nela e rode de novo.")
        return 1

    chunks = load_chunks(base_dir)
    if not chunks:
        print("\nAVISO: nenhum trecho encontrado para indexar. Nada foi gravado.")
        return 1

    sizes = [len(c["text"]) for c in chunks]
    print(f"\n{len(chunks)} trechos (tamanho mín/médio/máx: {min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)} caracteres)")

    from sentence_transformers import SentenceTransformer  # import tardio: só o indexador precisa

    print(f"Carregando modelo '{model_name}'...")
    model = SentenceTransformer(model_name)
    model.max_seq_length = min(model.max_seq_length or 512, 512)

    query_prefix, passage_prefix = get_prefixes(model_name)
    print("Criando embeddings...")
    embeddings = model.encode(
        [passage_prefix + c["text"] for c in chunks],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # produto interno = cosseno (vetores normalizados)
    index.add(np.ascontiguousarray(embeddings))

    metadata = {
        "version": 2,
        "model": model_name,
        "metric": "cosine",
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "chunks": chunks,
    }

    # Grava em arquivo temporário e troca no final: o app nunca lê um índice pela metade
    index_path = os.path.join(index_dir, INDEX_FILE)
    meta_path = os.path.join(index_dir, METADATA_FILE)
    faiss.write_index(index, index_path + ".tmp")
    with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(index_path + ".tmp", index_path)
    os.replace(meta_path + ".tmp", meta_path)

    print("\nÍndice e metadados criados com sucesso! Reinicie o servidor para carregar o novo índice.")
    return 0


if __name__ == "__main__":
    sys.exit(build())