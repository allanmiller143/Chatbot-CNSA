'''Módulo de configuração da aplicação.

Carrega variáveis de ambiente e define constantes utilizadas em toda a aplicação.
'''

import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

class Config:
    """Classe que armazena as configurações da aplicação."""
    
    # Chave da API da OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Chave OPENAI_API_KEY não encontrada no arquivo .env")
    
    WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
    if not WHATSAPP_VERIFY_TOKEN:
        raise ValueError("Chave WHATSAPP_VERIFY_TOKEN não encontrada no arquivo .env")
    
    WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
    if not WHATSAPP_ACCESS_TOKEN:
        raise ValueError("Chave WHATSAPP_ACCESS_TOKEN não encontrada no arquivo .env")
      
    WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
    if not WHATSAPP_PHONE_ID:
        raise ValueError("Chave WHATSAPP_PHONE_ID não encontrada no arquivo .env")
       

    # Arquivo do banco de dados
    DB_FILE = 'chatbot_log.db'

    # Caminhos para os arquivos de embeddings
    EMBEDDINGS_INDEX_PATH = "embeddings/kb.index"
    EMBEDDINGS_METADATA_PATH = "embeddings/metadata.json"

    # Modelos
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    CHAT_MODEL = "gpt-4.1"
