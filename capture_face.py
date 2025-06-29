import cv2
import os
import pickle
from deepface import DeepFace

EMBEDDINGS_PATH = "embeddings.pkl"
USER_NAME = input("Enter your name: ")

def capture_and_register():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print("📸 Capturing face for", USER_NAME)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image.")
        return

    cv2.imwrite("registered_face.jpg", frame)
    print("✅ Face image saved as registered_face.jpg")

    try:
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        result = DeepFace.represent(
            img_path=rgb,
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=True
        )

        embedding = result[0]['embedding']
        print("✅ Face embedding extracted.")

        # Load existing embeddings or create new
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as f:
                embeddings_dict = pickle.load(f)
        else:
            embeddings_dict = {}

        embeddings_dict.setdefault(USER_NAME, []).append(embedding)

        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings_dict, f)

        print(f"✅ Embedding saved for '{USER_NAME}'.")

    except Exception as e:
        print("⚠️ Error extracting face:", e)

if __name__ == "__main__":
    capture_and_register()
# This script captures a face image from the webcam, extracts its embedding using DeepFace,
# and saves it to a pickle file for future recognition. Make sure you have the required libraries   