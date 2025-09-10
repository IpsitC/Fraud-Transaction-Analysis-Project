CREATE TABLE if not exists transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time INTEGER,
    v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL,
    v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL,
    v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL,
    v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
    v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL,
    v26 REAL, v27 REAL, v28 REAL,
    amount REAL,
    class INTEGER
);

-- Count rows
SELECT COUNT(*) FROM transactions;

-- Fraud vs normal
SELECT Class, COUNT(*) FROM transactions GROUP BY Class;

-- Average transaction amount by fraud vs normal
SELECT Class, AVG(Amount) FROM transactions GROUP BY Class;

-- Top 5 biggest frauds
SELECT Time, Amount FROM transactions WHERE Class = 1 ORDER BY Amount DESC LIMIT 5;

