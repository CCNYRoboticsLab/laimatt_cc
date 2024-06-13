import pytest
import mysql.connector  # Or psycopg2 for PostgreSQL

@pytest.fixture(scope="module")
def mysql_connection():
    conn = mysql.connector.connect(
        user='root', password='mysecretpassword', host='127.0.0.1', database='test'
    )
    yield conn
    conn.close()

def test_insert_data(mysql_connection):
    cursor = mysql_connection.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)", ("John Doe", "john@example.com"))
    mysql_connection.commit()
    cursor.execute("SELECT * FROM users WHERE name=%s", ("John Doe",))
    result = cursor.fetchone()
    assert result is not None
    # Clean up (delete the inserted row)
    cursor.execute("DELETE FROM users WHERE name=%s", ("John Doe",))
    mysql_connection.commit()

