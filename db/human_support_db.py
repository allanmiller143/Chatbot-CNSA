"""
Módulo para gerenciamento da tabela de usuários em atendimento humano.

Contém funções para inicialização, inserção, leitura, atualização
e remoção de usuários da fila de atendimento humano.
"""

import sqlite3
from config import Config


def init_human_support_table():
    """Cria a tabela de atendimento humano se ela não existir."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS human_support (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            user_name TEXT,
            started_at TEXT NOT NULL,
            status TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def add_user_to_human_support(user_id, user_name, started_at, status="active"):
    """
    Insere ou substitui um usuário na fila de atendimento humano.
    UNIQUE(user_id) evita duplicação.
    """
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO human_support (user_id, user_name, started_at, status)
        VALUES (?, ?, ?, ?)
    ''', (user_id, user_name, started_at, status))

    conn.commit()
    conn.close()


def get_human_support_users(status=None):
    """
    Retorna todos os usuários em atendimento humano.
    Se status for informado, filtra por ele ("active", "finished", etc.).
    """
    conn = sqlite3.connect(Config.DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if status:
        cursor.execute("SELECT * FROM human_support WHERE status = ?", (status,))
    else:
        cursor.execute("SELECT * FROM human_support")

    users = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return users


def update_human_support_status(user_id, new_status):
    """Atualiza o status do atendimento do usuário."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE human_support
        SET status = ?
        WHERE user_id = ?
    ''', (new_status, user_id))

    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def remove_user_from_human_support(user_id):
    """Remove um usuário da fila de atendimento humano."""
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        DELETE FROM human_support
        WHERE user_id = ?
    ''', (user_id,))

    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success
