import streamlit as st
import cv2
import numpy as np
import joblib
import mysql.connector
import io
from skimage.feature import local_binary_pattern, hog
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import requests
import time 
import random
# -------------------- App Setup --------------------
st.set_page_config(page_title="Surgical Face Recognition", layout="wide")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Check & Match Image", "➕ Insert New Image", "🖼 View All DB Images","Visualizations"])

# -------------------- Configs --------------------
image_size = (92, 118)
radius = 2
n_points = 8 * radius
METHOD = 'uniform'
SIMILARITY_THRESHOLD = 0.40

# -------------------- Load Model --------------------
@st.cache_resource
def load_model_and_tools():
    model = joblib.load("model/final_model.pkl")
    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]
    return model, scaler, pca

model, scaler, pca = load_model_and_tools()

# -------------------- Feature Extraction --------------------
def extract_features_raw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, image_size)

    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2), density=True)

    hog_feat = hog(gray, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), feature_vector=True)

    combined = np.concatenate((lbp_hist, hog_feat)).reshape(1, -1)
    return combined

# -------------------- DB Connection --------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123Arunava.",
        database="Surgical_face_recognition"
    )

# -------------------- Fetch All Features --------------------
def fetch_all_features(stage_filter=None):
    conn = get_connection()
    cursor = conn.cursor()

    if stage_filter:
        cursor.execute("SELECT label, stage, feature_vector, image_data FROM surgical_faces WHERE stage = %s", (stage_filter,))
    else:
        cursor.execute("SELECT label, stage, feature_vector, image_data FROM surgical_faces")

    data = cursor.fetchall()
    cursor.close()
    conn.close()

    labels, stages, vectors, images = [], [], [], []
    for label, stage, vec_blob, img_blob in data:
        try:
            feature = joblib.load(io.BytesIO(vec_blob))
            if feature.shape[1] != model.named_steps['pca'].n_components_:
                print(f"❌ Skipping {label} ({stage}) due to shape mismatch: {feature.shape}")
                continue
            img_array = np.frombuffer(img_blob, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            labels.append(label)
            stages.append(stage)
            vectors.append(feature)
            images.append(img)
        except Exception as e:
            print(f"Error loading entry: {e}")
            continue

    return labels, stages, np.vstack(vectors) if vectors else np.array([]), images

# -------------------- Save New Entry --------------------
def save_to_db(label, stage, features_pca, image):
    conn = get_connection()
    cursor = conn.cursor()

    check_query = "SELECT COUNT(*) FROM surgical_faces WHERE label = %s AND stage = %s"
    cursor.execute(check_query, (label, stage))
    exists = cursor.fetchone()[0] > 0

    if exists:
        st.warning(f"⚠️ Entry with label '{label}' and stage '{stage}' already exists. Skipping insertion.")
        cursor.close()
        conn.close()
        return

    try:
        feature_blob = io.BytesIO()
        joblib.dump(features_pca, feature_blob)
        feature_blob.seek(0)

        _, img_encoded = cv2.imencode('.jpg', image)

        insert_query = """
            INSERT INTO surgical_faces (label, stage, feature_vector, image_data)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (label, stage, feature_blob.read(), img_encoded.tobytes()))
        conn.commit()
        st.success("✅ New image inserted into the database.")
    except Exception as e:
        st.error(f"❌ Failed to save image: {e}")
    finally:
        cursor.close()
        conn.close()

# -------------------- Page 1: Home --------------------
# -------------------- Page 1: Home --------------------
if page == "🏠 Home":
    
# Inject CSS for styling
    st.markdown("""
<style>
    .landing-container {
        text-align: center;
        padding: 2rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #00ffe1;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #ffdd57;
        margin-bottom: 2rem;
    }
    .blog-links {
        text-align: left;
        display: inline-block;
        margin-top: 20px;
        font-size: 1.1rem;
        line-height: 2;
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        color: #f0f0f0;
    }
    .illustration {
        margin-top: 2rem;
        max-width: 65%;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Render home content
    st.markdown("""
<div class="landing-container">
    <div class="title">Surgical Face Recognition System</div>
    <div class="subtitle">Built with PCA + LBP + KNN + SVM + Voting Classifier</div>

    <div class="blog-links">
        <strong>📚 Project Tutorials / Blogs:</strong><br>
        📘 <a href="https://medium.com/@oarunavachakraborty/blog-1-dataset-building-for-surgical-face-recognition-f91e4b3fe914" target="_blank">Blog 1: Dataset Building</a><br>
        ⚙️ <a href="https://medium.com/@oarunavachakraborty/blog-2-feature-engineering-for-surgical-face-recognition-cec060657d39" target="_blank">Blog 2: Feature Engineering</a><br>
        📊 <a href="https://medium.com/@oarunavachakraborty/blog-3-deploying-surgical-face-recognition-with-streamlit-mysql-a5bc6ccbdde2" target="_blank">Blog 3: Deployment Guide</a>
    </div>

    <img src="https://cdni.iconscout.com/illustration/premium/thumb/face-recognition-3480386-2912021.png" class="illustration"/>
</div>
""", unsafe_allow_html=True)



# -------------------- Page 2: Check & Match --------------------
elif page == "🔍 Check & Match Image":
    st.title("🔍 Check & Match Surgical Image")
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    def parse_filename(label):
        try:
            label = label.split('.')[0]  # remove .jpg/.png etc.
            parts = label.split('_')

            patient_id = parts[0]
            
            # Look for 'before' or 'after' in the last segment
            last_part = parts[-1]
            if last_part.endswith("before"):
                stage = "before"
                surgery = "_".join(parts[1:])[:-6]  # remove 'before'
            elif last_part.endswith("after"):
                stage = "after"
                surgery = "_".join(parts[1:])[:-5]  # remove 'after'
            else:
                stage = "Unknown"
                surgery = "_".join(parts[1:])
            
            return patient_id, surgery, stage
        except:
            return "Unknown", "Unknown", "Unknown"


    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        buffer_placeholder = st.empty()  # processing feedback

        # Step 1: Feature Extraction
        buffer_placeholder.info("🧠 Extracting features...")
        features_raw = extract_features_raw(uploaded_img)
        features_pca = pca.transform(scaler.transform(features_raw))
        time.sleep(1.5)

        # Step 2: Classification
        buffer_placeholder.info("🔎 Predicting surgery stage...")
        pred = model.predict(features_raw)[0]
        stage = "before" if pred == 0 else "after"
        stage_opposite = "after" if stage == "before" else "before"
        label_text = "Before Surgery" if pred == 0 else "After Surgery"
        confidence = np.max(model.predict_proba(features_raw)) * 100
        time.sleep(1.5)

        # Step 3: Find similar image from DB
        buffer_placeholder.info("📂 Matching with database...")
        labels_db, stages_db, vectors_db, images_db = fetch_all_features(stage_filter=stage_opposite)
        similarities = cosine_similarity(features_pca, vectors_db)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        matched = best_score >= SIMILARITY_THRESHOLD
        time.sleep(1.5)

        buffer_placeholder.empty()

        st.success(f"Prediction: **{label_text}** ({confidence:.2f}%)")

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.resize(cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB), (220, 260)), caption="Uploaded Image")
        
        with col2:
            if matched:
                matched_label = labels_db[best_idx]
                matched_img = images_db[best_idx]
                pid, surgery_type, matched_stage = parse_filename(matched_label)
                visit_id = f"VST{random.randint(10000, 99999)}"

                st.image(cv2.resize(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB), (220, 260)), caption="Matched Image")

                st.markdown(f"""
                #### 🧾 Match Details:
                - **Patient ID:** `{pid}`
                - **Surgery:** `{surgery_type}`
                - **Stage:** `{matched_stage}`
                - **Similarity Score:** `{best_score*100:.2f}%`
                - **Visit ID:** `{visit_id}`
                """)
                st.info("✅ Similar image found in the database.")
            else:
                st.warning("❌ No similar face found.")
                st.info("You may add this new entry using the 'Insert New Image' tab.")



# -------------------- Page 3: Insert New Image --------------------
elif page == "➕ Insert New Image":
    st.title("➕ Insert New Facial Image to DB")
    uploaded_file = st.file_uploader("📤 Upload a New Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.resize(cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB), (180, 220)), caption="Image Preview")
        with col2:
            custom_id = st.text_input("🆔 Patient ID")
            surgery_type = st.text_input("💉 Surgery Type")
            stage = st.selectbox("🩺 Stage", ["before", "after"])

            if st.button("💾 Save to DB"):
                if custom_id and surgery_type:
                    features_raw = extract_features_raw(uploaded_img)
                    features_scaled = scaler.transform(features_raw)
                    features_pca = pca.transform(features_scaled)

                    labels_db, stages_db, vectors_db, _ = fetch_all_features(stage_filter=stage)
                    best_score = np.max(cosine_similarity(features_pca, vectors_db)[0]) if len(vectors_db) > 0 else 0

                    if best_score >= 0.70:
                        st.warning(f"⚠️ Similar image already exists in DB. (Similarity: {best_score*100:.2f}%)")
                    else:
                        label = f"{custom_id}_{surgery_type}"
                        save_to_db(label, stage, features_pca, uploaded_img)
                else:
                    st.warning("⚠️ Please enter both Patient ID and Surgery Type.")

# -------------------- Page 4: View All DB Images --------------------
elif page == "🖼 View All DB Images":
    st.title("🖼 All Surgical Face Images in Database")

    st.sidebar.markdown("### 🔎 Filter Options")
    stage_filter = st.sidebar.selectbox("Stage", ["All", "before", "after"])
    keyword = st.sidebar.text_input("Search Patient ID or Surgery")

    all_labels, all_stages, _, all_images = fetch_all_features()

    filtered = []
    for label, stage, img in zip(all_labels, all_stages, all_images):
        if stage_filter != "All" and stage != stage_filter:
            continue
        if keyword and keyword.lower() not in label.lower():
            continue
        filtered.append((label, stage, img))

    if not filtered:
        st.warning("No records found.")
    else:
        st.markdown(f"### Showing {len(filtered)} result(s)")
        for i in range(0, len(filtered), 4):
            cols = st.columns(4)
            for j, (label, stage, img) in enumerate(filtered[i:i+4]):
                with cols[j]:
                    st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (120, 140)),
                             caption=f"{label} ({stage})", use_container_width=True)
elif page == "📊 Visualizations":
    st.title("📊 Model Visualizations & Performance")
    
    st.subheader("✅ Classification Report")
    st.markdown("""
    - **Accuracy**: 84.21%
    - **Precision**: 82.50%
    - **Recall**: 83.75%
    - **F1-score**: 83.12%
    """)
    
    st.subheader("🌀 PCA Dimensionality Reduction")
    st.image("pca_plot.png", caption="PCA 2D Scatter Plot", use_column_width=True)

    st.subheader("📉 Confusion Matrix")
    st.image("confusion_matrix.png", caption="Model Confusion Matrix", use_column_width=True)

