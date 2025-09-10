import pandas as pd
import sqlite3

# Load dataset
df = pd.read_csv("Data/creditcard1.csv")

# Create SQLite DB
conn = sqlite3.connect("fraud.db")
df.to_sql("transactions", conn, if_exists="replace", index=False)

print("Data loaded into SQLite DB 'fraud.db'")
