import mysql.connector
import cv2
import numpy as np
import io
import os

# --- MySQL Connection ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123Arunava.",
        database="Surgical_face_recognition"
    )

# --- Fetch and Display All Images ---
def view_images_from_db(output_folder="db_images"):
    os.makedirs(output_folder, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT label, stage, image_data FROM surgical_faces")
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"Total images fetched: {len(results)}")

    for i, (label, stage, img_blob) in enumerate(results):
        np_img = np.frombuffer(img_blob, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Save image for review
        filename = f"{label}_{stage}.jpg".replace("/", "_")
        path = os.path.join(output_folder, filename)
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

        # Uncomment below to view one by one
        # cv2.imshow(f"{label} - {stage}", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    view_images_from_db()
