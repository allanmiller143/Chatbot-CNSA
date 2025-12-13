import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import markdown
from bs4 import BeautifulSoup
import json

BASE_DIR = "knowledge_base"
INDEX_DIR = "embeddings"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

os.makedirs(INDEX_DIR, exist_ok=True)

print("Carregando modelo...")
model = SentenceTransformer(MODEL_NAME)

documents = []
metadata = []

print("Lendo arquivos...")
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".md"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            html = markdown.markdown(text)
            soup = BeautifulSoup(html, "html.parser")
            plain_text = soup.get_text(separator='\n', strip=True)

            documents.append(plain_text)
            metadata.append({"file": file, "path": path})

if not documents:
    print("Nenhum documento encontrado.")
else:
    print(f"Criando embeddings ({len(documents)})...")

    # 🔥 NORMALIZAÇÃO OBRIGATÓRIA
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]

    # 🔥 ÍNDICE CORRETO PARA BGE
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, f"{INDEX_DIR}/kb.index")

    with open(f"{INDEX_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("Index criado com sucesso!")
