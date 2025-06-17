import os
import cv2
import numpy as np
import mysql.connector
import joblib
import io
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# CONFIG
radius = 2
n_points = 8 * radius
METHOD = 'uniform'
image_size = (92, 118)
DATA_DIR = "data"
DB_CONFIG = {
    'host': "localhost",
    'user': "root",
    'password': "123Arunava.",
    'database': "Surgical_face_recognition"
}

# Load model (PCA should match training)
model = joblib.load("model/final_model.pkl")
pca = model.named_steps['pca']
scaler = model.named_steps['scaler']

def extract_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, image_size)

    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2), density=True)

    hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)

    combined = np.concatenate((lbp_hist, hog_feat)).reshape(1, -1)
    combined_scaled = scaler.transform(combined)
    return pca.transform(combined_scaled), img

def connect():
    return mysql.connector.connect(**DB_CONFIG)

def reset_database():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS surgical_faces")
    cursor.execute("""
        CREATE TABLE surgical_faces (
            id INT AUTO_INCREMENT PRIMARY KEY,
            label VARCHAR(255),
            stage VARCHAR(10),
            feature_vector LONGBLOB,
            image_data LONGBLOB
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_entry(label, stage, pca_features, img):
    conn = connect()
    cursor = conn.cursor()

    feature_blob = io.BytesIO()
    joblib.dump(pca_features, feature_blob)
    feature_blob.seek(0)

    _, img_encoded = cv2.imencode('.jpg', img)
    query = """
        INSERT INTO surgical_faces (label, stage, feature_vector, image_data)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (label, stage, feature_blob.read(), img_encoded.tobytes()))
    conn.commit()
    cursor.close()
    conn.close()

def main():
    print("🧹 Resetting and Rebuilding the Surgical Face DB")
    reset_database()

    for stage in ["before", "after"]:
        folder_path = os.path.join(DATA_DIR, stage)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(folder_path, file)
                try:
                    features, img = extract_features(filepath)
                    label = os.path.splitext(file)[0]
                    insert_entry(label, stage, features, img)
                    print(f"✅ Inserted: {label} [{stage}]")
                except Exception as e:
                    print(f"❌ Failed: {file} - {e}")

    print("🎉 Rebuild complete.")

if __name__ == "__main__":
    main()
