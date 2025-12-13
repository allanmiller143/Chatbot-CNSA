'''
Módulo de rotas para o endpoint /chat.
'''

from flask import Blueprint, request, jsonify
from openai import OpenAI
import datetime

from config import Config
from db.db_manager import log_interaction
from db.human_support_db import (
    add_user_to_human_support,
    remove_user_from_human_support,
    get_human_support_users
)
from services.rag_service import generate_search_query, get_reranked_context

chat_bp = Blueprint('chat', __name__)

conversation_history = {}
client = OpenAI(api_key=Config.OPENAI_API_KEY)


# 🔍 Função utilitária para verificar se user está em atendimento humano
def user_is_in_human_support(user_id):
    active_users = get_human_support_users(status="active")
    return any(u["user_id"] == user_id for u in active_users)


def get_chat_response(question: str, user_id: str, user_name: str = "Usuário"):
    """Lógica principal do chatbot RAG."""

    # 🔹 Se o usuário está em atendimento humano → resposta padrão
    if user_is_in_human_support(user_id):
        final_answer = (
            "Um momento, por favor. Vou transferir sua solicitação para um de nossos colaboraASdores."
        )
        status = "MANTIDO_HUMANO"

        log_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'original_question': question,
            'search_question': None,
            'bot_answer': final_answer,
            'status': status,
            'context_docs': []
        }
        log_interaction(log_data)
        return final_answer, status

    # Carrega o histórico do usuário
    history = conversation_history.get(user_id, [])

    # Gera pergunta de busca
    search_question = generate_search_query(question, history)
    final_context, context_docs = get_reranked_context(search_question)

    # 🔒 Anti-alucinação: Sem contexto → transferir para humano
    if not final_context or final_context.strip() == "":
        raw_answer = "FLAG_HUMAN_TRANSFER"
    else:
        system_prompt = (
            "Você é o Amparo, assistente virtual do Colégio Nossa Senhora do Amparo. "
            "Normalmente no início/fim do atendimento você pode dar alguma saudação e em seguida usar o termo 'Paz e bem!'. "
            "Você fala como um funcionário da secretaria: educado, humano e direto ao ponto.\n\n"

            "REGRAS IMPORTANTES:\n"
            "1. Você só pode usar informações presentes no contexto abaixo.\n"
            "2. Você PODE interpretar, relacionar e sintetizar ideias presentes nos documentos.\n"
            "3. Você NÃO pode inventar fatos ausentes no contexto.\n"
            "4. Se a pergunta exigir algum dado que não esteja no contexto, responda exatamente: FLAG_HUMAN_TRANSFER.\n"
            "5. Seja direto. Máximo 3–4 frases.\n"
            "6. Nunca mencione palavras como 'documento', 'contexto' ou 'base'.\n"
            "7. Se o usuário pedir atendente humano → responda FLAG_HUMAN_TRANSFER.\n\n"

            "=== CONTEXTO ===\n"
            f"{final_context}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]

        for h in history:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["bot"]})

        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=messages
        )

        raw_answer = response.choices[0].message.content.strip()

    # PROCESSAMENTO DA FLAG
    status = "RESPOSTA_OK"
    final_answer = raw_answer

    if "FLAG_HUMAN_TRANSFER" in raw_answer:
        status = "TRANSFERIDO"
        final_answer = (
            "Um momento, por favor. Vou transferir sua solicitação para um de nossos colaboradores."
        )

    # Salva histórico limitado a 10 mensagens
    history.append({"user": question, "bot": final_answer})
    conversation_history[user_id] = history[-10:]

    # LOG
    log_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'user_id': user_id,
        'user_name': user_name,
        'original_question': question,
        'search_question': search_question,
        'bot_answer': final_answer,
        'status': status,
        'context_docs': context_docs
    }
    log_interaction(log_data)

    # Se foi transferido, salva no banco
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
    data = request.json
    question = data.get("question")
    user_id = data.get("user_id", "default_user")
    user_name = data.get("user_name", "Usuário")

    final_answer, status = get_chat_response(question, user_id, user_name)

    return jsonify({
        "answer": final_answer,
        "status": status,
        "user_name": user_name
    })


@chat_bp.route("/chat/remove", methods=["POST"])
def human_chat_remove():
    data = request.json
    user_id = data.get("user_id", "default_user")

    remove_user_from_human_support(user_id)

    return jsonify({
        "status": "REMOVIDO"
    })


@chat_bp.route("/chat/humanChats", methods=["GET"])
def get_human_chat():
    users = get_human_support_users(status="active")
    return jsonify({
        "human_chats": users
    })
