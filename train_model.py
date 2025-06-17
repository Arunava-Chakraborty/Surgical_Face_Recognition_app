# train_model.py

import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern

# --- CONFIG ---
radius = 2
n_points = 8 * radius
METHOD = 'uniform'
image_size = (92, 118)
DATA_DIR = "data"
MODEL_PATH = "App/model/final_model.pkl"
PCA_VEC_PATH = "App/model/pca_vectors.npy"
PCA_LABELS_PATH = "App/model/pca_visuals_labels.npy"

def extract_features(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, image_size)

    lbp = local_binary_pattern(img_gray, n_points, radius, METHOD)
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2), density=True)

    hog_feat = hog(img_gray, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), feature_vector=True)

    combined = np.concatenate((lbp_hist, hog_feat))
    return combined

def load_dataset():
    X, y, labels = [], [], []
    for label, folder in enumerate(["before", "after"]):
        folder_path = os.path.join(DATA_DIR, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                features = extract_features(img_path)
                X.append(features)
                y.append(label)
                labels.append(filename.split('.')[0])  # for matching / plotting
    return np.array(X), np.array(y), np.array(labels)

def main():
    print("📥 Loading data...")
    X, y, labels = load_dataset()

    print("🧠 Training PCA + VotingClassifier...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X_scaled)

    clf = VotingClassifier(estimators=[
        ("svm", SVC(probability=True, kernel="rbf")),
        ("rf", RandomForestClassifier(n_estimators=100))
    ], voting="soft")

    clf.fit(X_pca, y)

    # --- Save model pipeline ---
    print("💾 Saving model and PCA vectors...")
    os.makedirs("App/model", exist_ok=True)
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("clf", clf)
    ])
    joblib.dump(pipeline, MODEL_PATH)

    np.save(PCA_VEC_PATH, X_pca)
    np.save(PCA_LABELS_PATH, labels)

    print("✅ Done. Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
