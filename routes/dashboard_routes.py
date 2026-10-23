# routes/dashboard_routes.py
from flask import Blueprint, render_template

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
def dashboard():
    """
    Serve o painel web de status do WhatsApp.
    """
    return render_template("dashboard.html")
