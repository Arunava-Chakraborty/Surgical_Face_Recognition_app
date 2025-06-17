# App/db_utils.py

import mysql.connector
import joblib
import io
import numpy as np

# --- MySQL Connection ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123Arunava.",       # ⬅️ Replace with your MySQL password
        database="Surgical_face_recognition"    # ⬅️ Make sure this is your DB name
    )

# --- Fetch all feature vectors, labels, images ---
def fetch_all_vectors():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT label, feature_vector, image_data FROM surgical_faces")
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    vectors = []
    labels = []
    images = []

    for label, feat_blob, img_blob in results:
        vector = joblib.load(io.BytesIO(feat_blob))
        vectors.append(vector)
        labels.append(label)
        images.append(img_blob)

    return np.vstack(vectors), labels, images
