'''Módulo de serviço RAG (Retrieval-Augmented Generation).

Contém a lógica para gerar a pergunta de busca, buscar o contexto e re-rankear os documentos.
'''

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

from config import Config

# --- Inicialização dos Modelos e Clientes ---
try:
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    index = faiss.read_index(Config.EMBEDDINGS_INDEX_PATH)
    with open(Config.EMBEDDINGS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
except FileNotFoundError as e:
    print(f"ERRO: Arquivo de índice não encontrado em {e.filename}. Execute o script de criação de índice.")
    # Não vamos sair, apenas imprimir o erro para permitir o desenvolvimento
    # Em um ambiente de produção, isso seria um erro fatal.
    pass
except Exception as e:
    print(f"Erro ao inicializar o serviço RAG: {e}")
    pass

def generate_search_query(question: str, history: list) -> str:
    """Gera uma pergunta de busca otimizada usando o histórico da conversa."""
    if not history:
        return question
    history_str = "\n".join([f"Usuário: {h['user']}\nAssistente: {h['bot']}" for h in history])
    prompt = (
        f'Com base no histórico da conversa e na última pergunta do usuário, gere uma única e autônoma pergunta de busca em português que capture a intenção completa do usuário.\n\n'
        f'--- Histórico ---\n{history_str}\n\n'
        f'--- Última Pergunta ---\n{question}\n\n'
        f'--- Pergunta de Busca Otimizada ---\n'
    )
    try:
        response = client.chat.completions.create(
            model=Config.CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question

def get_reranked_context(question: str, top_k_initial: int = 7, top_k_final: int = 5) -> tuple[str, list[str]]:
    """Busca, re-classifica e retorna o melhor contexto e os nomes dos arquivos usados."""
    
    # Verifica se os objetos de RAG foram inicializados
    if 'model' not in globals() or 'index' not in globals() or 'metadata' not in globals():
        print("Aviso: Serviço RAG não inicializado corretamente. Retornando contexto vazio.")
        return "", []

    # 1. Busca Ampla
    q_emb = model.encode([question])
    distances, indices = index.search(np.array(q_emb), top_k_initial)
    
    retrieved_paths = [metadata[i]["path"] for i in indices[0]]
    if not retrieved_paths:
        return "", []

    # 2. Re-ranking
    docs_content = ""
    for i, path in enumerate(retrieved_paths):
        # Acessa o arquivo original para obter o conteúdo
        try:
            with open(path, "r", encoding="utf-8") as f:
                docs_content += f'--- Documento {i+1} ({os.path.basename(path)}) ---\n{f.read()}\n\n'
        except FileNotFoundError:
            print(f"Aviso: Documento de contexto não encontrado: {path}")
            continue

    rerank_prompt = (
        f"Você é um assistente de busca. Sua tarefa é analisar uma lista de documentos e identificar "
        f"os {top_k_final} mais relevantes para responder à pergunta. "
        f"Cada documento possui seu nome real entre parênteses — você DEVE usar exatamente esse nome sem alterar, sem resumir e sem inventar novos nomes.\n\n"
        f"Retorne os textos completos desses documentos, separados por:\n\n=== FIM DO DOCUMENTO ===\n\n"
        f"No final, adicione uma linha começando por 'FONTES_UTILIZADAS:' seguida EXATAMENTE "
        f"dos nomes reais dos arquivos usados. Exemplo de formato (não use esses nomes):\n"
        f"FONTES_UTILIZADAS: arquivo1.md, arquivo2.md\n"
        f"NÃO invente nomes como doc1.md ou doc2.md.\n"
    )


    try:
        response = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[{"role": "user", "content": rerank_prompt + f"\n\n--- Pergunta do Usuário ---\n{question}\n\n--- Lista de Documentos ---\n{docs_content}"}],
            temperature=0.0,
        )
        full_response = response.choices[0].message.content.strip()
        
        parts = full_response.split("FONTES_UTILIZADAS:")
        best_contexts = parts[0].strip()
        source_files_str = parts[1].strip() if len(parts) > 1 else ""
        source_files = [name.strip() for name in source_files_str.split(',')]
        
        return best_contexts, source_files
    except Exception as e:
        print(f"Erro no re-ranking: {e}")
        # Fallback
        contexts = []
        fallback_paths = retrieved_paths[:top_k_final]
        for path in fallback_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    contexts.append(f.read())
            except FileNotFoundError:
                continue
        return "\n\n".join(contexts), [os.path.basename(p) for p in fallback_paths]
