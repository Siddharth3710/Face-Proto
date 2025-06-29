from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
import os, cv2, pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from face_helper import get_face_embedding
from io import BytesIO
from PIL import Image


app = Flask(__name__)
camera = cv2.VideoCapture(0)

# ✅ Home Page
@app.route('/')
def index():
    return render_template('index.html')

# ✅ New User Page
@app.route('/new')
def new_user():
    return render_template('capture.html')

# ✅ Existing User Page
@app.route('/existing')
def existing_user():
    return render_template('existing.html')

# ✅ Upload View (New User Registration)
@app.route('/upload_view', methods=['POST'])
def upload_view():
    name = request.form.get('name')
    view = request.form.get('view')
    image = request.files.get('image')

    if not name or not view or not image:
        return jsonify({"success": False, "error": "Missing data"}), 400

    folder_path = os.path.join("uploads", name)
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, f"{view}.jpg")
    image.save(save_path)

    embedding = get_face_embedding(save_path)
    if embedding is None:
        return jsonify({"success": False, "error": "No face detected"}), 400

    # Save embedding
    embeddings_file = "embeddings.pkl"
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "rb") as f:
            all_embeddings = pickle.load(f)
    else:
        all_embeddings = {}

    all_embeddings.setdefault(name, []).append(embedding)

    with open(embeddings_file, "wb") as f:
        pickle.dump(all_embeddings, f)

    return jsonify({"success": True})

# ✅ Generate Video Frames for Live Stream
def generate_frames():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # 🟢 Reinitialize if released
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Draw face rectangle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ✅ Video Feed Route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ Recognize User from Live Feed
@app.route('/recognize_live', methods=['POST'])
def recognize_live():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Reinitialize camera if needed

    success, frame = camera.read()
    if not success:
        return render_template("existing.html", result="❌ Could not read from camera")

    # Convert OpenCV frame to in-memory BytesIO buffer
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = BytesIO(buffer.tobytes())

    if not os.path.exists("embeddings.pkl"):
        return render_template("existing.html", result="❌ No registered users found.")

    with open("embeddings.pkl", "rb") as f:
        db = pickle.load(f)

    try:
        # Convert to NumPy array from BytesIO and get embedding
        from PIL import Image
        img = Image.open(img_bytes).convert("RGB")
        img_np = np.array(img)

        target_embedding = DeepFace.represent(img_path=img_np, model_name="SFace")[0]['embedding']

        best_name = "Unknown"
        best_score = 0.0
        for name, embeds in db.items():
            for emb in embeds:
                sim = cosine_similarity([target_embedding], [emb])[0][0]
                if sim > 0.65 and sim > best_score:
                    best_name = name
                    best_score = sim

        result = f"{best_name} (Confidence: {best_score:.2f})"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    # 🛑 Release camera and close OpenCV windows
    if camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()

    return render_template("existing.html", result=result)

    
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
    return redirect(url_for('existing_user'))


# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
