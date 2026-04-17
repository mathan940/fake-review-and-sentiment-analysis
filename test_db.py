import mysql.connector

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'mathanmani@123',
    'database': 'fake_review_db'
}

print("=== Testing DB Connection ===")
try:
    conn = mysql.connector.connect(**DB_CONFIG)
    print("[OK] Connected to fake_review_db")
except mysql.connector.Error as err:
    print(f"[FAIL] Connection error: {err}")
    exit(1)

print("\n=== Checking 'reviews' table columns ===")
cursor = conn.cursor()
cursor.execute("DESCRIBE reviews;")
rows = cursor.fetchall()
for row in rows:
    print(row)

print("\n=== Attempting test INSERT ===")
try:
    sql = """
        INSERT INTO reviews 
        (review_text, sentiment_label, sentiment_confidence, fake_label, fake_confidence, stars)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    val = ("Test input sentence", "Positive", 0.95, "Genuine", 0.85, 4.5)
    cursor.execute(sql, val)
    conn.commit()
    print("[OK] Row inserted successfully!")
except Exception as e:
    print(f"[FAIL] Insert error: {e}")

cursor.close()
conn.close()
