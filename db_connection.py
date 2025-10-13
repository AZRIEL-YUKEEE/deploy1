import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",          # change to your MySQL user
        password="",          # change to your MySQL password
        database="icat_db"
    )
