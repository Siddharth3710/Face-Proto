import cv2
import os
import pickle
import numpy as np
import time
import threading
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Load known face embeddings
def load_embeddings(pkl_path="embeddings.pkl"):
    if not os.path.exists(pkl_path):
        print("❌ embeddings.pkl not found.")
        return {}
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# Extract face embeddings from a frame
def get_embeddings_from_frame(frame):
    try:
        print("🔍 Detecting face with DeepFace...")
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        start = time.time()
        results = DeepFace.represent(
            img_path=rgb,
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=False
        )
        print(f"⏱️ DeepFace took {time.time() - start:.2f} seconds")

        return [res['embedding'] for res in results if 'embedding' in res]
    except Exception as e:
        print("⚠️ DeepFace error:", e)
        return []

# Recognize face by comparing with saved embeddings
def recognize_face(embedding, known_embeddings, threshold=0.5):  # ↓ Lowered threshold
    matched_name = "Unknown"
    best_score = 0.0
    for name, emb_list in known_embeddings.items():
        for saved_emb in emb_list:
            sim = cosine_similarity([embedding], [saved_emb])[0][0]
            print(f"🔎 Comparing with {name} → Cosine: {sim:.4f}")
            if sim > threshold and sim > best_score:
                matched_name = name
                best_score = sim
    return matched_name, best_score

# Background thread for continuous recognition
class RecognizerThread(threading.Thread):
    def __init__(self, known_embeddings):
        super().__init__()
        self.known_embeddings = known_embeddings
        self.frame = None
        self.result = []
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        while self.running:
            frame_copy = None
            with self.lock:
                if self.frame is not None:
                    frame_copy = self.frame.copy()

            if frame_copy is not None:
                try:
                    print("📸 Frame received for recognition.")
                    embeddings = get_embeddings_from_frame(frame_copy)
                    temp_result = []
                    for emb in embeddings:
                        name, score = recognize_face(emb, self.known_embeddings)
                        temp_result.append((name, score))
                    self.result = temp_result
                except Exception as e:
                    print("❗ Thread error:", e)

            for _ in range(10):  # ~1s delay
                if not self.running:
                    break
                time.sleep(0.1)

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def stop(self):
        self.running = False

# Main function
def run_live_recognition():
    print("🎥 Starting threaded live face recognition...")
    known_embeddings = load_embeddings()

    print(f"🤖 Loaded users: {list(known_embeddings.keys())}")
    if "sobber" in known_embeddings:
        print(f"🧠 'sobber' has {len(known_embeddings['sobber'])} embeddings")

    if not known_embeddings:
        print("❌ No embeddings found. Please register users first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    recognizer = RecognizerThread(known_embeddings)
    recognizer.start()

    frame_saved = False
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        if not frame_saved:
            cv2.imwrite("test_frame.jpg", frame)
            print("🖼️ Saved frame as test_frame.jpg for inspection.")
            frame_saved = True

        recognizer.update_frame(frame)
        display_frame = frame.copy()

        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        for idx, (name, score) in enumerate(recognizer.result):
            label = f"✅ {name} ({score:.2f})" if name != "Unknown" else "❌ Unknown"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            y_offset = 40 + idx * 30
            cv2.putText(display_frame, label, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Live Face Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            recognizer.stop()
            recognizer.join()
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the system
if __name__ == "__main__":
    run_live_recognition()
# This script captures live video from the webcam, detects faces, extracts embeddings using DeepFace,
# and recognizes faces by comparing them with stored embeddings. It runs in a separate thread for continuous        