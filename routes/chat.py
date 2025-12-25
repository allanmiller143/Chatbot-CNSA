"""
Módulo de rotas para o endpoint /chat (Flask Blueprint) com:
- RAG (generate_search_query + get_reranked_context)
- Anti-alucinação (sem contexto -> FLAG_HUMAN_TRANSFER)
- Handoff para humano (banco human_support_db)
- Melhor engenharia de prompt (START/MIDDLE/END)
- Histórico limitado e timeout de sessão
"""

from flask import Blueprint, request, jsonify
from openai import OpenAI
import datetime
import time
import re

from config import Config
from db.db_manager import log_interaction
from db.human_support_db import (
    add_user_to_human_support,
    remove_user_from_human_support,
    get_human_support_users
)
from services.rag_service import generate_search_query, get_reranked_context


chat_bp = Blueprint("chat", __name__)

client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Histórico em memória (atenção: por processo)
conversation_history = {}  # user_id -> list[{"user":..., "bot":..., "ts":...}]
last_seen = {}  # user_id -> epoch seconds

HISTORY_LIMIT = 10
SESSION_TIMEOUT_SEC = 25 * 60  # 25 minutos


# ================================
# HUMAN SUPPORT
# ================================
def user_is_in_human_support(user_id: str) -> bool:
    active_users = get_human_support_users(status="active")
    return any(u.get("user_id") == user_id for u in active_users)


# ================================
# CONVERSATION STAGE
# ================================
_ENDING_PATTERNS = [
    r"\b(tchau|até\s+mais|até\s+logo|até|valeu|obrigad[oa]|boa\s+noite|bom\s+dia|boa\s+tarde)\b",
    r"\b(encerrar|finalizar|pode\s+encerrar|isso\s+é\s+tudo)\b"
]

def detect_end_message(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(re.search(p, t) for p in _ENDING_PATTERNS)

def compute_stage(user_id: str, question: str, history: list) -> str:
    now = time.time()
    prev = last_seen.get(user_id)

    # START se não existe histórico ou se passou o timeout
    if (not history) or (prev is None) or ((now - prev) > SESSION_TIMEOUT_SEC):
        stage = "START"
        history.clear()  # novo atendimento -> limpa contexto conversacional antigo
    else:
        stage = "MIDDLE"

    # END se mensagem de encerramento
    if detect_end_message(question):
        stage = "END"

    last_seen[user_id] = now
    return stage


# ================================
# PROMPT
# ================================
def build_system_prompt(final_context: str, stage: str) -> str:
    return f"""
Você é o Amparo, assistente virtual do Colégio Nossa Senhora do Amparo.
Você escreve como um atendente de secretaria: humano, educado e direto ao ponto.

ESTADO DA CONVERSA:
- CONVERSATION_STAGE = {stage}  (START | MIDDLE | END)

ESTILO (para soar natural):
- START: cumprimente em 1 frase curta e inclua "Paz e bem!" exatamente 1 vez; depois responda.
- MIDDLE: não cumprimente e não use "Paz e bem!".
- END: finalize com 1 frase cordial e inclua "Paz e bem!" exatamente 1 vez.
- Não use linguagem robótica (evite repetir fórmulas).
- Máximo 2–4 frases; só faça 1 pergunta se for realmente necessário para ajudar.

REGRAS IMPORTANTES:
1) Use somente informações presentes no texto abaixo.
2) Pode interpretar, relacionar e sintetizar ideias presentes no texto.
3) Não invente fatos ausentes no texto.
4) Se a pergunta exigir algum dado que não esteja no texto, responda exatamente: FLAG_HUMAN_TRANSFER.
5) Nunca mencione palavras como "documento", "contexto", "base" ou similares.
6) Se o usuário pedir atendente humano, responda: FLAG_HUMAN_TRANSFER.

TEXTO:
{final_context}
""".strip()


# ================================
# CORE CHAT
# ================================
def get_chat_response(question: str, user_id: str, user_name: str = "Usuário"):
    """Lógica principal do chatbot RAG."""
    question = (question or "").strip()

    # Validação mínima
    if not question:
        return "Por favor, envie sua dúvida na mensagem.", "ERRO_VALIDACAO"

    # Se usuário está em atendimento humano -> resposta padrão
    if user_is_in_human_support(user_id):
        final_answer = "Um momento, por favor. Vou transferir sua solicitação para um de nossos colaboradores."
        status = "MANTIDO_HUMANO"

        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "user_name": user_name,
            "original_question": question,
            "search_question": None,
            "bot_answer": final_answer,
            "status": status,
            "context_docs": []
        }
        log_interaction(log_data)
        return final_answer, status

    # Carrega histórico do usuário
    history = conversation_history.get(user_id, [])
    stage = compute_stage(user_id, question, history)

    # Adapta histórico para as funções existentes (mantém compatibilidade)
    history_simple = [{"user": h["user"], "bot": h["bot"]} for h in history]

    # Gera pergunta de busca + recupera contexto
    search_question = generate_search_query(question, history_simple)
    final_context, context_docs = get_reranked_context(search_question)

    # Anti-alucinação: sem contexto -> transferir
    if not final_context or not final_context.strip():
        raw_answer = "FLAG_HUMAN_TRANSFER"
    else:
        system_prompt = build_system_prompt(final_context=final_context, stage=stage)

        messages = [{"role": "system", "content": system_prompt}]

        # Injeta histórico (últimas interações)
        for h in history[-HISTORY_LIMIT:]:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["bot"]})

        # Mensagem atual
        messages.append({"role": "user", "content": question})

        # Chamada OpenAI
        response = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=messages,
            temperature=getattr(Config, "TEMPERATURE", 0.6),
        )
        raw_answer = response.choices[0].message.content.strip()

    # Processamento da FLAG
    status = "RESPOSTA_OK"
    final_answer = raw_answer

    if "FLAG_HUMAN_TRANSFER" in (raw_answer or ""):
        status = "TRANSFERIDO"
        final_answer = "Um momento, por favor. Vou transferir sua solicitação para um de nossos colaboradores."

    # Salva histórico limitado
    history.append({
        "user": question,
        "bot": final_answer,
        "ts": datetime.datetime.now().isoformat()
    })
    conversation_history[user_id] = history[-HISTORY_LIMIT:]

    # Log
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "user_name": user_name,
        "original_question": question,
        "search_question": search_question,
        "bot_answer": final_answer,
        "status": status,
        "context_docs": context_docs
    }
    log_interaction(log_data)

    # Se transferiu, salva no banco de atendimento humano
    if status == "TRANSFERIDO":
        add_user_to_human_support(
            user_id=user_id,
            user_name=user_name,
            started_at=datetime.datetime.now().isoformat(),
            status="active"
        )

    return final_answer, status


# ================================
# ENDPOINTS HTTP
# ================================
@chat_bp.route("/chat", methods=["POST"])
def chat():
    # Flask: request.get_json / request.json para ler payload JSON
    data = request.get_json(silent=True) or {}  # evita erro se vier payload inválido [web:33]
    question = data.get("question")
    user_id = data.get("user_id", "default_user")
    user_name = data.get("user_name", "Usuário")

    final_answer, status = get_chat_response(question, user_id, user_name)

    http_code = 200 if status != "ERRO_VALIDACAO" else 400
    return jsonify({
        "answer": final_answer,
        "status": status,
        "user_name": user_name
    }), http_code


@chat_bp.route("/chat/remove", methods=["POST"])
def human_chat_remove():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "default_user")

    remove_user_from_human_support(user_id)

    return jsonify({"status": "REMOVIDO"})


@chat_bp.route("/chat/humanChats", methods=["GET"])
def get_human_chat():
    users = get_human_support_users(status="active")
    return jsonify({"human_chats": users})
