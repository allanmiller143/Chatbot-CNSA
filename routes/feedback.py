'''Módulo de rotas para o endpoint /feedback.
'''

from flask import Blueprint, request, jsonify
from db.db_manager import get_unclassified_interactions, update_interaction_feedback
from db.db_manager import delete_interaction, modify_interaction # Adicione estas importações no topo do arquivo

# Cria um Blueprint para as rotas de feedback
feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route("/feedback/unclassified", methods=["GET"])
def get_unclassified():
    """Retorna uma lista de interações que ainda não foram classificadas."""
    try:
        limit = request.args.get('limit', default=10, type=int)
        interactions = get_unclassified_interactions(limit=limit)
        
        # Converte a lista de objetos Row para uma lista de dicionários para jsonify
        return jsonify([dict(i) for i in interactions])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@feedback_bp.route("/feedback/classify", methods=["POST"])
def classify_interaction():
    """Recebe o feedback humano e atualiza a interação no banco de dados."""
    data = request.json
    interaction_id = data.get("id")
    rating = data.get("rating") # 1 (Correto) ou 0 (Incorreto)
    correct_answer = data.get("correct_answer")
    category = data.get("category")

    if not interaction_id or rating is None or rating not in [0, 1]:
        return jsonify({"error": "ID da interação é obrigatório, e rating deve ser 0 (Incorreto) ou 1 (Correto)."}), 400

    # Se o rating for 0 (Incorreto), a resposta correta é obrigatória.
    if rating == 0 and not correct_answer:
        return jsonify({"error": "Para rating 0 (Incorreto), 'correct_answer' é obrigatório."}), 400

    try:
        success = update_interaction_feedback(
            interaction_id=interaction_id, 
            rating=rating, 
            correct_answer=correct_answer,
            category=category
        )
        
        if success:
            return jsonify({"message": f"Feedback para interação {interaction_id} registrado com sucesso."})
        else:
            return jsonify({"error": f"Interação com ID {interaction_id} não encontrada ou não atualizada."}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@feedback_bp.route("/feedback/delete/<int:interaction_id>", methods=["DELETE"])
def delete_interaction_route(interaction_id):
    """Deleta uma interação pelo ID."""
    try:
        success = delete_interaction(interaction_id)
        if success:
            return jsonify({"message": f"Interação {interaction_id} deletada com sucesso."})
        else:
            return jsonify({"error": f"Interação com ID {interaction_id} não encontrada."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@feedback_bp.route("/feedback/modify/<int:interaction_id>", methods=["PUT"])
def modify_interaction_route(interaction_id):
    """Modifica campos de uma interação pelo ID."""
    data = request.json
    
    # Campos que podem ser modificados
    original_question = data.get("original_question")
    bot_answer = data.get("bot_answer")
    status = data.get("status")
    
    if not any([original_question, bot_answer, status]):
        return jsonify({"error": "Nenhum campo válido fornecido para modificação (original_question, bot_answer, status)."}), 400

    try:
        success = modify_interaction(
            interaction_id=interaction_id,
            original_question=original_question,
            bot_answer=bot_answer,
            status=status
        )
        
        if success:
            return jsonify({"message": f"Interação {interaction_id} modificada com sucesso."})
        else:
            return jsonify({"error": f"Interação com ID {interaction_id} não encontrada ou nenhum campo modificado."}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500