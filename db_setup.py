import mysql.connector
import time

def setup_db():
    try:
        # User needs to ensure MySQL is running and credentials are correct
        # Defaulting to common local dev settings
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root"
        )
        cursor = db.cursor()
        
        # Create Database
        cursor.execute("CREATE DATABASE IF NOT EXISTS crowd_management")
        cursor.execute("USE crowd_management")
        
        # Table for time-series metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                count INT,
                density FLOAT,
                velocity FLOAT,
                risk_score FLOAT,
                risk_level VARCHAR(20),
                fps FLOAT
            )
        """)
        
        # Table for incident alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                risk_level VARCHAR(20),
                risk_score FLOAT,
                count INT,
                snapshot_path VARCHAR(255),
                details TEXT
            )
        """)
        
        db.commit()
        print("[INFO] MySQL Database and tables setup successfully.")
        return True
    except mysql.connector.Error as err:
        print(f"[ERROR] MySQL Setup Failed: {err}")
        return False

if __name__ == "__main__":
    setup_db()
