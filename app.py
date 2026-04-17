# app.py
from flask import Flask, render_template
from db.db_manager import init_db
from db.human_support_db import init_human_support_table
from routes.chat import chat_bp
from routes.feedback import feedback_bp
from routes.dashboard_routes import dashboard_bp

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')

    # registra blueprints
    app.register_blueprint(chat_bp)

    try:
        app.register_blueprint(feedback_bp)
        app.register_blueprint(dashboard_bp)
    except Exception:
        app.logger.debug("feedback blueprint não encontrado ou já registrado")

    # rota principal
    @app.route("/")
    def index():
        return render_template("dashboard.html")

    # inicialização do banco
    init_db()
    init_human_support_table()

    return app


if __name__ == "__main__":
    app = create_app()
    # servidor Flask normal
    app.run(host="0.0.0.0", port=5000, debug=True)
