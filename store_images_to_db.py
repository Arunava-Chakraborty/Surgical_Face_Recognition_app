# App/store_images_to_db.py

import os
import cv2
import numpy as np
import joblib
import mysql.connector
import io
from skimage.feature import local_binary_pattern, hog
from dotenv import load_dotenv

load_dotenv()

# --- Feature extraction ---
def extract_features(img, size=(92, 118)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)

    lbp = local_binary_pattern(gray, 8 * 2, 2, 'uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 8*2 + 3),
                               range=(0, 8*2 + 2), density=True)
    hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    return np.concatenate((lbp_hist, hog_feat))

# --- MySQL connection ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",         # or your DB host
        user="root",              # your MySQL username
        password="123Arunava.", # your MySQL password
        database=" Surgical_face_recognition" # your database name
    )

def insert_into_db(label, stage, features, image):
    conn = get_connection()
    cursor = conn.cursor()

    feature_blob = io.BytesIO()
    joblib.dump(features, feature_blob)
    feature_blob.seek(0)

    _, img_encoded = cv2.imencode('.jpg', image)
    query = """
        INSERT INTO surgical_faces (label, stage, feature_vector, image_data)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (label, stage, feature_blob.read(), img_encoded.tobytes()))
    conn.commit()
    cursor.close()
    conn.close()

# --- Run for all images ---
def process_folder(folder_path, stage):
    for file in os.listdir(folder_path):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            full_path = os.path.join(folder_path, file)
            img = cv2.imread(full_path)
            features = extract_features(img)
            insert_into_db(label=file, stage=stage, features=features, image=img)
            print(f"✅ Inserted: {file} [{stage}]")

if __name__ == "__main__":
    process_folder("data/before", "before")
    process_folder("data/after", "after")
