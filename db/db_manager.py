'''Módulo para gerenciamento do banco de dados SQLite.

Contém funções para inicialização do banco de dados e logging das interações.
'''

import sqlite3
import json
from config import Config

def init_db():
    """Cria a tabela de interações do banco de dados se ela não existir."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            user_name TEXT,
            original_question TEXT,
            search_question TEXT,
            bot_answer TEXT,
            status TEXT,
            context_docs TEXT,
            human_feedback_rating BOOLEAN,
            human_feedback_correct_answer TEXT,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_interaction(log_data):
    """Insere uma nova interação no banco de dados."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO interactions (
            timestamp, user_id, user_name, original_question, search_question, 
            bot_answer, status, context_docs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        log_data['timestamp'],
        log_data['user_id'],
        log_data['user_name'],
        log_data['original_question'],
        log_data['search_question'],
        log_data['bot_answer'],
        log_data['status'],
        json.dumps(log_data['context_docs']) # Salva a lista de docs como um texto JSON
    ))
    conn.commit()
    conn.close()

def get_unclassified_interactions(limit=10):
    """Busca interações que ainda não foram classificadas (human_feedback_rating IS NULL)."""
    conn = sqlite3.connect(Config.DB_FILE)
    conn.row_factory = sqlite3.Row # Permite acessar colunas por nome
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            *
        FROM 
            interactions 
        WHERE 
            human_feedback_rating IS NULL 
        ORDER BY 
            timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    interactions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return interactions

def update_interaction_feedback(interaction_id, rating, correct_answer=None, category=None):
    """
    Atualiza o feedback humano para uma interação específica.
    Rating: 1 (Correto) ou 0 (Incorreto).
    Se rating for 1, correct_answer é ignorado/anulado.
    """
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    # Se o rating for 1 (Correto), a resposta correta não é necessária.
    if rating == 1:
        correct_answer = None
        
    cursor.execute('''
        UPDATE interactions
        SET 
            human_feedback_rating = ?,
            human_feedback_correct_answer = ?,
            category = ?
        WHERE 
            id = ?
    ''', (rating, correct_answer, category, interaction_id))
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def delete_interaction(interaction_id):
    """Deleta uma interação específica do banco de dados."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        DELETE FROM interactions
        WHERE id = ?
    ''', (interaction_id,))
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def modify_interaction(interaction_id, original_question=None, bot_answer=None, status=None):
    """Modifica campos específicos de uma interação no banco de dados."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    # Cria a query de forma dinâmica para modificar apenas os campos fornecidos
    updates = []
    params = []
    
    if original_question is not None:
        updates.append("original_question = ?")
        params.append(original_question)
        
    if bot_answer is not None:
        updates.append("bot_answer = ?")
        params.append(bot_answer)
        
    if status is not None:
        updates.append("status = ?")
        params.append(status)
        
    if not updates:
        conn.close()
        return False # Nada para modificar

    query = f"UPDATE interactions SET {', '.join(updates)} WHERE id = ?"
    params.append(interaction_id)
    
    cursor.execute(query, tuple(params))
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

