import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import markdown
from bs4 import BeautifulSoup
import json

# --- Configurações ---
BASE_DIR = "knowledge_base"
INDEX_DIR = "embeddings"
# MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


# --- Garantir que os diretórios existam ---
os.makedirs(INDEX_DIR, exist_ok=True)
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f"Pasta '{BASE_DIR}' criada. Por favor, adicione seus arquivos .md nela.")

print("Carregando modelo de sentenças...")
model = SentenceTransformer(MODEL_NAME)

documents = []
metadata = []

print(f"Lendo arquivos do diretório: {BASE_DIR}")
# --- Carrega cada arquivo .md como um único documento ---
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".md"):
            path = os.path.join(root, file)
            print(f"Processando arquivo: {file}")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # Converte Markdown para texto simples de forma limpa, preservando parágrafos
            html = markdown.markdown(text)
            soup = BeautifulSoup(html, 'html.parser')
            plain_text = soup.get_text(separator='\n', strip=True)

            documents.append(plain_text)
            # O metadado agora só precisa do caminho, pois o documento é inteiro
            metadata.append({"file": file, "path": path})

if not documents:
    print("\nAVISO: Nenhum documento encontrado para indexar. O script foi concluído sem criar um índice.")
else:
    print(f"\nCriando embeddings para {len(documents)} documentos...")
    embeddings = model.encode(documents, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # --- Salva o índice e os metadados ---
    faiss.write_index(index, f"{INDEX_DIR}/kb.index")
    with open(f"{INDEX_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("\nÍndice e metadados criados com sucesso!")
