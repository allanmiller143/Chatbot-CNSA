import sqlite3
import pandas as pd

conn = sqlite3.connect("chatbot_log.db")

df = pd.read_sql_query("SELECT * FROM interactions", conn)
df.to_excel("export.xlsx", index=False)

conn.close()
